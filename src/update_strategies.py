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
    def update(self, network: CANNetwork, rand_index: Optional[torch.Tensor] = None,
               neuron_noise: Optional[torch.Tensor] = None) -> None:
        samples = [network.state.clone()]
        network.total_activity.append(torch.sum(network.state).item())

        for _ in range(network.num_updates):
            proposed_state = network.state.clone()
            active_indices = (proposed_state == 1).nonzero(as_tuple=True)[0]
            inactive_indices = (proposed_state == 0).nonzero(as_tuple=True)[0]

            if len(active_indices) == 0 or len(inactive_indices) == 0:
                continue  # Skip update if no valid swap exists

            idx_active = active_indices[torch.randint(0, len(active_indices), (1,)).item()].item()
            idx_inactive = inactive_indices[torch.randint(0, len(inactive_indices), (1,)).item()].item()
            proposed_state[idx_active] = 0.0
            proposed_state[idx_inactive] = 1.0

            current_energy = network.calculate_energy()
            old_state = network.state.clone()
            network.state = proposed_state.clone()
            proposed_energy = network.calculate_energy()
            network.state = old_state

            if network.noise > 0:
                acceptance_prob = min(1, np.exp((current_energy.item() - proposed_energy.item()) / network.noise))
            else:
                if (current_energy - proposed_energy) < 0:
                    acceptance_prob = 0
                elif (current_energy - proposed_energy) > 0:
                    acceptance_prob = 1
                else:
                    acceptance_prob = 0.5

            if np.random.rand() < acceptance_prob:
                network.state = proposed_state.clone()
                samples.append(proposed_state.clone())
            else:
                network.state = samples[-1].clone()
