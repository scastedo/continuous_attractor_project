import torch
import numpy as np
from abc import ABC, abstractmethod
from typing import Optional
from src.network import CANNetwork

class UpdateStrategy(ABC):
    @abstractmethod
    def update(self, network: CANNetwork, rand_index: Optional[torch.Tensor] = None,
               neuron_noise: Optional[torch.Tensor] = None) -> None:
        """
        Update the network state.
        """
        pass


class DynamicsUpdateStrategyGain(UpdateStrategy):
    """
    Update strategy using simple dynamics (sigmoidal updates with optional noise).
    """
    def update(self, network: CANNetwork, rand_index: Optional[torch.Tensor] = None,
               neuron_noise: Optional[torch.Tensor] = None) -> None:

        if rand_index is None:
            rand_index = torch.randint(0, network.num_neurons, (), device=network.device, dtype=torch.long)
        prev_value = network.state[rand_index].clone()
        epsp = network.synaptic_drive[rand_index]
        ext = network.A * network.input_bump_profile[rand_index]
        threshold = network.constrict*(network.active_count_tensor - network.target_active_tensor) * network.inv_num_neurons_tensor

        if neuron_noise is None:
            if network.sigma_eta > 0:
                neuron_noise = torch.randn((), device=network.device) * network.sigma_eta
            else:
                neuron_noise = 0.0
        activation = network.input_resistance * (
            network.ampar_conductance * epsp + ext + neuron_noise
        ) - threshold

        if network.record_diagnostics:
            network.activations.append(float(activation))

        if network.sigma_temp == 0.00:
            new_value = torch.where(
                activation < 0,
                activation.new_zeros(()),
                torch.where(
                    activation > 0,
                    activation.new_ones(()),
                    # torch.randint(
                    #     0, 2, (), device=network.device, dtype=network.state.dtype
                    # )
                    prev_value.clone()  # keep previous state if exactly at threshold
                    ,
                ),
            )
        else:
            prob = torch.sigmoid(activation / network.sigma_temp)
            new_value = torch.bernoulli(prob)

        network.state[rand_index] = new_value
        if not torch.equal(prev_value, new_value):
            delta = new_value - prev_value
            network.active_count_tensor += delta
            alpha = float(delta.item())
            network.synaptic_drive.add_(network.weights[:, rand_index], alpha=alpha)

class DynamicsUpdateStrategy(UpdateStrategy):
    """
    Update strategy using simple dynamics (sigmoidal updates with optional noise).
    """
    def update(self, network: CANNetwork, rand_index: Optional[torch.Tensor] = None,
               neuron_noise: Optional[torch.Tensor] = None) -> None:
        for _ in range(network.num_updates):
            # draw index directly on the device
            rand_index = torch.randint(0, network.num_neurons, (), device=network.device)

            # per-synapse failure mask without leaving PyTorch
            mask = (torch.rand(network.num_neurons, device=network.device) > network.syn_fail).float()
            epsp = torch.dot(network.weights[rand_index] * mask, network.state)

            dx = rand_index - network.I_dir[network.generation]
            dx = torch.abs(dx)
            dx = torch.minimum(dx, network.num_neurons - dx)
            sigma = network.field_width * network.num_neurons
            ext = network.I_str[network.generation] * torch.exp(-(dx.float() ** 2) / (2 * sigma ** 2))

            threshold = network.constrict * (
                network.state.sum() - network.num_neurons * network.fraction_active
            ) / network.num_neurons
            index_activation = epsp + network.spon_rel + ext - threshold

            network.activations.append(index_activation.item())
            network.total_activity.append(network.state.sum().item())

            if network.noise == 0:
                new_state = torch.where(
                    index_activation < 0,
                    torch.tensor(0.0, device=network.device),
                    torch.where(
                        index_activation > 0,
                        torch.tensor(1.0, device=network.device),
                        torch.randint(0, 2, (), device=network.device, dtype=torch.float32),
                    ),
                )
            else:
                prob = torch.sigmoid(index_activation / network.noise)
                new_state = torch.bernoulli(prob)

            network.state[rand_index] = new_state


class MetropolisUpdateStrategy(UpdateStrategy):
    """
    Update strategy using the Metropolis algorithm.
    """
    def calculate_energyhere(self, neuron_noise, network: CANNetwork) -> torch.Tensor:
        interaction_energy = -0.5 * network.ampar_conductance*network.input_resistance*torch.dot(network.state, network.synaptic_drive)
        external_energy = -torch.dot(
            network.state,
            network.input_resistance * (
                neuron_noise+ network.A * network.input_bump_profile
            )
        )

        total_energy = interaction_energy + external_energy 
        return total_energy
    
    def update(self, network: CANNetwork, neuron_noise: Optional[torch.Tensor] = None) -> None:
        state = network.state

        active_indices = (state == 1).nonzero(as_tuple=True)[0]
        inactive_indices = (state == 0).nonzero(as_tuple=True)[0]

        if active_indices.numel() == 0 or inactive_indices.numel() == 0:
            return  # nothing to swap

        # Random active and inactive index
        i = active_indices[torch.randint(0, active_indices.numel(), (1,), device=network.device)]
        j = inactive_indices[torch.randint(0, inactive_indices.numel(), (1,), device=network.device)]

        # Current energy
        current_energy = network.calculate_energyhere(neuron_noise, network)

        # Proposed: flip i -> 0, j -> 1
        old_i = state[i].clone()
        old_j = state[j].clone()
        state[i] = 0.0
        state[j] = 1.0

        proposed_energy = network.calculate_energyhere(neuron_noise, network)

        # Revert for now
        state[i] = old_i
        state[j] = old_j

        dE = proposed_energy - current_energy

        if getattr(network, "sigma_temp", 0.0) > 0.0:
            T = network.sigma_temp
            acceptance_prob = torch.exp(-dE / T)
            acceptance_prob = torch.clamp(acceptance_prob, max=torch.tensor(1.0, device=network.device))
        else:
            if dE < 0:
                acceptance_prob = torch.tensor(1.0, device=network.device)
            elif dE > 0:
                acceptance_prob = torch.tensor(0.0, device=network.device)
            else:
                acceptance_prob = torch.tensor(0.5, device=network.device)

        if torch.rand((), device=network.device) < acceptance_prob:
            # Accept swap: now actually commit + update synaptic_drive
            state[i] = 0.0
            state[j] = 1.0

            # synaptic_drive updates:
            # i turned off: delta = -1
            network.synaptic_drive.add_(network.weights[:, i], alpha=-1.0)
            # j turned on: delta = +1
            network.synaptic_drive.add_(network.weights[:, j], alpha=+1.0)


class MetropolisUpdateStrategy2(UpdateStrategy):
    """
    Activity-conserving Metropolis update for CANNetwork.

    Assumes:
    - network.weights is (num_neurons, num_neurons) and (approximately) symmetric.
    - network.synaptic_drive is kept in sync with network.state (W @ state).
    - network.state is a 1D tensor of 0/1 floats.
    - network.sigma_temp is the Metropolis temperature T.
    """

    def update(self, network: CANNetwork, neuron_noise: Optional[torch.Tensor] = None, x_noise: Optional[torch.Tensor] = None, synapse_noise: Optional[list] = None) -> None:
        state = network.state

        # randomise the x input normal around 
        # --- 1. Choose a random active and a random inactive neuron ---
        active_indices = (state == 1).nonzero(as_tuple=True)[0]
        inactive_indices = (state == 0).nonzero(as_tuple=True)[0]

        if active_indices.numel() == 0 or inactive_indices.numel() == 0:
            return  # nothing to swap

        # Get scalar indices i, j  (Python ints)
        idx_i = torch.randint(0, active_indices.numel(), (), device=network.device)
        idx_j = torch.randint(0, inactive_indices.numel(), (), device=network.device)

        i = active_indices[idx_i].item()
        j = inactive_indices[idx_j].item()

        # --- 2. Precompute local fields needed for ΔE ---
        g = network.ampar_conductance
        R = network.input_resistance
        A = network.A

        syn_ok, spon = synapse_noise if synapse_noise is not None else (None, None)

        syn_ok_i = 1.0 if syn_ok is None else syn_ok[i]
        syn_ok_j = 1.0 if syn_ok is None else syn_ok[j]
        spon_i   = 0.0 if spon   is None else spon[i]
        spon_j   = 0.0 if spon   is None else spon[j]
  
        # synaptic_drive = W @ state (assumed already up-to-date)
        # h_i = network.synaptic_drive[i]
        # h_j = network.synaptic_drive[j]


        a_i_before = syn_ok_i
        a_j_before = spon_j
        a_i_after  = spon_i        # after i: 1->0
        a_j_after  = syn_ok_j      # after j: 0->1

        da_i = a_i_after - a_i_before
        da_j = a_j_after - a_j_before

        h_i = network.synaptic_drive[i]
        h_j = network.synaptic_drive[j]
        W_ij = network.weights[i, j]
        c = network.ampar_conductance * network.input_resistance




        bump_i = network.input_bump_profile[i]
        bump_j = network.input_bump_profile[j]

        x_i = 0.0 if x_noise is None else x_noise[i]
        x_j = 0.0 if x_noise is None else x_noise[j]

        eta_i = 0.0 if neuron_noise is None else neuron_noise[i]
        eta_j = 0.0 if neuron_noise is None else neuron_noise[j]

        b_i = R*g*(A*bump_i + x_i) + R*eta_i
        b_j = R*g*(A*bump_j + x_j) + R*eta_j
        # b_i = (A*bump_i + x_i + eta_i)
        # b_j = (A*bump_j + x_j + eta_j)
        
        # W_ji = network.weights[j, i]
        # c = g * R

        # dE = c * (h_i - h_j + W_ji) + (b_i - b_j)
        dE = -c * (da_i*h_i + da_j*h_j + da_i*da_j*W_ij) - (b_i*da_i + b_j*da_j)


        # --- 4. Metropolis acceptance ---
        if getattr(network, "sigma_temp", 0.0) > 0.0:
            T = network.sigma_temp
            log_u = torch.log(torch.rand((), device=network.device))
            accept = (log_u < (-dE / T))
        else:
            if dE < 0:
                accept = True
            elif dE > 0:
                accept = False
            else:
                accept = bool(torch.randint(0, 2, (), device=network.device))

        # --- 5. Apply swap and update synaptic_drive if accepted ---
        if accept:
            state[i] = 0.0
            state[j] = 1.0
            network.synaptic_drive.add_(network.weights[:, i], alpha=float(da_i))
            network.synaptic_drive.add_(network.weights[:, j], alpha=float(da_j))