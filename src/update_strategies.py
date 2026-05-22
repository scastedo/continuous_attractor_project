from src import network
import torch
import numpy as np
from abc import ABC, abstractmethod
from typing import Optional
from src.network import CANNetwork

class UpdateStrategy(ABC):
    @abstractmethod
    def update(self, network: CANNetwork, rand_index: Optional[torch.Tensor] = None,
               neuron_noise: Optional[torch.Tensor] = None,
               idx_i: Optional[torch.Tensor] = None,
               idx_j: Optional[torch.Tensor] = None) -> None:
        """
        Update the network state.
        """
        pass


# class DynamicsUpdateStrategyGain(UpdateStrategy):
#     """
#     Update strategy using simple dynamics (sigmoidal updates with optional noise).
#     """
#     def update(self, network: CANNetwork, rand_index: Optional[torch.Tensor] = None,
#                neuron_noise: Optional[torch.Tensor] = None) -> None:

#         if rand_index is None:
#             rand_index = torch.randint(0, network.num_neurons, (), device=network.device, dtype=torch.long)
#         prev_value = network.state[rand_index].clone()
#         epsp = network.synaptic_drive[rand_index]
#         ext = network.A * network.input_bump_profile[rand_index]
#         threshold = network.constrict*(network.active_count_tensor - network.target_active_tensor) * network.inv_num_neurons_tensor

#         if neuron_noise is None:
#             if network.sigma_eta > 0:
#                 neuron_noise = torch.randn((), device=network.device) * network.sigma_eta
#             else:
#                 neuron_noise = 0.0
#         activation = network.input_resistance * (
#             network.ampar_conductance * epsp + ext + neuron_noise
#         ) - threshold

#         if network.record_diagnostics:
#             network.activations.append(float(activation))

#         if network.sigma_temp == 0.00:
#             new_value = torch.where(
#                 activation < 0,
#                 activation.new_zeros(()),
#                 torch.where(
#                     activation > 0,
#                     activation.new_ones(()),
#                     # torch.randint(
#                     #     0, 2, (), device=network.device, dtype=network.state.dtype
#                     # )
#                     prev_value.clone()  # keep previous state if exactly at threshold
#                     ,
#                 ),
#             )
#         else:
#             prob = torch.sigmoid(activation / network.sigma_temp)
#             new_value = torch.bernoulli(prob)

#         network.state[rand_index] = new_value
#         if not torch.equal(prev_value, new_value):
#             delta = new_value - prev_value
#             network.active_count_tensor += delta
#             alpha = float(delta.item())
#             network.synaptic_drive.add_(network.weights[:, rand_index], alpha=alpha)

# class DynamicsUpdateStrategy(UpdateStrategy):
#     """
#     Update strategy using simple dynamics (sigmoidal updates with optional noise).
#     """
#     def update(self, network: CANNetwork, rand_index: Optional[torch.Tensor] = None,
#                neuron_noise: Optional[torch.Tensor] = None) -> None:
#         for _ in range(network.num_updates):
#             # draw index directly on the device
#             rand_index = torch.randint(0, network.num_neurons, (), device=network.device)

#             # per-synapse failure mask without leaving PyTorch
#             mask = (torch.rand(network.num_neurons, device=network.device) > network.syn_fail).float()
#             epsp = torch.dot(network.weights[rand_index] * mask, network.state)

#             dx = rand_index - network.I_dir[network.generation]
#             dx = torch.abs(dx)
#             dx = torch.minimum(dx, network.num_neurons - dx)
#             sigma = network.field_width * network.num_neurons
#             ext = network.I_str[network.generation] * torch.exp(-(dx.float() ** 2) / (2 * sigma ** 2))

#             threshold = network.constrict * (
#                 network.state.sum() - network.num_neurons * network.fraction_active
#             ) / network.num_neurons
#             index_activation = epsp + network.spon_rel + ext - threshold

#             network.activations.append(index_activation.item())
#             network.total_activity.append(network.state.sum().item())

#             if network.noise == 0:
#                 new_state = torch.where(
#                     index_activation < 0,
#                     torch.tensor(0.0, device=network.device),
#                     torch.where(
#                         index_activation > 0,
#                         torch.tensor(1.0, device=network.device),
#                         torch.randint(0, 2, (), device=network.device, dtype=torch.float32),
#                     ),
#                 )
#             else:
#                 prob = torch.sigmoid(index_activation / network.noise)
#                 new_state = torch.bernoulli(prob)

#             network.state[rand_index] = new_state


# class MetropolisUpdateStrategy(UpdateStrategy):
#     """
#     Update strategy using the Metropolis algorithm.
#     """
#     def calculate_energyhere(self, neuron_noise, network: CANNetwork) -> torch.Tensor:
#         interaction_energy = -0.5 * network.ampar_conductance*network.input_resistance*torch.dot(network.state, network.synaptic_drive)
#         external_energy = -torch.dot(
#             network.state,
#             network.input_resistance * (
#                 neuron_noise+ network.A * network.input_bump_profile
#             )
#         )

#         total_energy = interaction_energy + external_energy 
#         return total_energy
    
#     def update(self, network: CANNetwork, neuron_noise: Optional[torch.Tensor] = None) -> None:
#         state = network.state

#         active_indices = (state == 1).nonzero(as_tuple=True)[0]
#         inactive_indices = (state == 0).nonzero(as_tuple=True)[0]

#         if active_indices.numel() == 0 or inactive_indices.numel() == 0:
#             return  # nothing to swap

#         # Random active and inactive index
#         i = active_indices[torch.randint(0, active_indices.numel(), (1,), device=network.device)]
#         j = inactive_indices[torch.randint(0, inactive_indices.numel(), (1,), device=network.device)]

#         # Current energy
#         current_energy = network.calculate_energyhere(neuron_noise, network)

#         # Proposed: flip i -> 0, j -> 1
#         old_i = state[i].clone()
#         old_j = state[j].clone()
#         state[i] = 0.0
#         state[j] = 1.0

#         proposed_energy = network.calculate_energyhere(neuron_noise, network)

#         # Revert for now
#         state[i] = old_i
#         state[j] = old_j

#         dE = proposed_energy - current_energy

#         if getattr(network, "sigma_temp", 0.0) > 0.0:
#             T = network.sigma_temp
#             acceptance_prob = torch.exp(-dE / T)
#             acceptance_prob = torch.clamp(acceptance_prob, max=torch.tensor(1.0, device=network.device))
#         else:
#             if dE < 0:
#                 acceptance_prob = torch.tensor(1.0, device=network.device)
#             elif dE > 0:
#                 acceptance_prob = torch.tensor(0.0, device=network.device)
#             else:
#                 acceptance_prob = torch.tensor(0.5, device=network.device)

#         if torch.rand((), device=network.device) < acceptance_prob:
#             # Accept swap: now actually commit + update synaptic_drive
#             state[i] = 0.0
#             state[j] = 1.0

#             # synaptic_drive updates:
#             # i turned off: delta = -1
#             network.synaptic_drive.add_(network.weights[:, i], alpha=-1.0)
#             # j turned on: delta = +1
#             network.synaptic_drive.add_(network.weights[:, j], alpha=+1.0)

# class MetropolisUpdateStrategy2(UpdateStrategy):
#     """
#     Activity-conserving Metropolis update for CANNetwork.

#     Assumes:
#     - network.weights is (num_neurons, num_neurons) and (approximately) symmetric.
#     - network.synaptic_drive is kept in sync with network.state (W @ state).
#     - network.state is a 1D tensor of 0/1 floats.
#     - network.sigma_temp is the Metropolis temperature T.
#     """
#     def update(self, network: CANNetwork, neuron_noise: Optional[torch.Tensor] = None, x_noise: Optional[torch.Tensor] = None, synapse_noise: Optional[list] = None) -> None:
#         state = network.state

#         # randomise the x input normal around 
#         # --- 1. Choose a random active and a random inactive neuron ---
#         active_indices = (state == 1).nonzero(as_tuple=True)[0]
#         inactive_indices = (state == 0).nonzero(as_tuple=True)[0]

#         if active_indices.numel() == 0 or inactive_indices.numel() == 0:
#             return  # nothing to swap

#         # Get scalar indices i, j  (Python ints)
#         idx_i = torch.randint(0, active_indices.numel(), (), device=network.device)
#         idx_j = torch.randint(0, inactive_indices.numel(), (), device=network.device)

#         i = active_indices[idx_i].item()
#         j = inactive_indices[idx_j].item()

#         # --- 2. Precompute local fields needed for ΔE ---
#         g = network.ampar_conductance
#         R = network.input_resistance
#         A = network.A

#         syn_ok, spon = synapse_noise if synapse_noise is not None else (None, None)

#         syn_ok_i = 1.0 if syn_ok is None else syn_ok[i]
#         syn_ok_j = 1.0 if syn_ok is None else syn_ok[j]
#         spon_i   = 0.0 if spon   is None else spon[i]
#         spon_j   = 0.0 if spon   is None else spon[j]
  
#         # synaptic_drive = W @ state (assumed already up-to-date)
#         # h_i = network.synaptic_drive[i]
#         # h_j = network.synaptic_drive[j]


#         a_i_before = syn_ok_i
#         a_j_before = spon_j
#         a_i_after  = spon_i        # after i: 1->0
#         a_j_after  = syn_ok_j      # after j: 0->1

#         da_i = a_i_after - a_i_before
#         da_j = a_j_after - a_j_before

#         h_i = network.synaptic_drive[i]
#         h_j = network.synaptic_drive[j]
#         W_ij = network.weights[i, j]
#         c = network.ampar_conductance * network.input_resistance




#         bump_i = network.input_bump_profile[i]
#         bump_j = network.input_bump_profile[j]

#         x_i = 0.0 if x_noise is None else x_noise[i]
#         x_j = 0.0 if x_noise is None else x_noise[j]

#         eta_i = 0.0 if neuron_noise is None else neuron_noise[i]
#         eta_j = 0.0 if neuron_noise is None else neuron_noise[j]

#         b_i = R*g*(A*bump_i + x_i) + R*eta_i
#         b_j = R*g*(A*bump_j + x_j) + R*eta_j
#         # b_i = (A*bump_i + x_i + eta_i)
#         # b_j = (A*bump_j + x_j + eta_j)
        
#         # W_ji = network.weights[j, i]
#         # c = g * R

#         # dE = c * (h_i - h_j + W_ji) + (b_i - b_j)
#         dE = -c * (da_i*h_i + da_j*h_j + da_i*da_j*W_ij) - (b_i*da_i + b_j*da_j)


#         # --- 4. Metropolis acceptance ---
#         if getattr(network, "sigma_temp", 0.0) > 0.0:
#             T = network.sigma_temp
#             log_u = torch.log(torch.rand((), device=network.device))
#             accept = (log_u < (-dE / T))
#         else:
#             if dE < 0:
#                 accept = True
#             elif dE > 0:
#                 accept = False
#             else:
#                 accept = bool(torch.randint(0, 2, (), device=network.device))

#         # --- 5. Apply swap and update synaptic_drive if accepted ---
#         if accept:
#             state[i] = 0.0
#             state[j] = 1.0
#             network.synaptic_drive.add_(network.weights[:, i], alpha=float(da_i))
#             network.synaptic_drive.add_(network.weights[:, j], alpha=float(da_j))
class MetropolisUpdateStrategy3(UpdateStrategy):
    """
    Activity-conserving Metropolis update for CANNetwork.

    Assumes:
    - network.weights is (num_neurons, num_neurons) and (approximately) symmetric.
    - network.synaptic_drive is kept in sync with network.state (W @ state).
    - network.state is a 1D tensor of 0/1 floats.
    - network.sigma_temp is the Metropolis temperature T.
    """
    def update(
        self,
        network: CANNetwork,
        rand_index: Optional[torch.Tensor] = None,
        neuron_noise: Optional[torch.Tensor] = None,
        x_noise: Optional[torch.Tensor] = None,
        synapse_noise: Optional[list] = None,
        idx_i: Optional[torch.Tensor] = None,
        idx_j: Optional[torch.Tensor] = None,
    ) -> None:
        state = network.state
        active_pool = network.active_pool
        inactive_pool = network.inactive_pool
        active_pos = network.active_pos
        inactive_pos = network.inactive_pos

        if active_pool is None or inactive_pool is None or active_pos is None or inactive_pos is None:
            network.initialize_activity_pools()
            active_pool = network.active_pool
            inactive_pool = network.inactive_pool
            active_pos = network.active_pos
            inactive_pos = network.inactive_pos
            if active_pool is None or inactive_pool is None or active_pos is None or inactive_pos is None:
                raise ValueError("Failed to initialize activity pools")

        if active_pool.numel() == 0 or inactive_pool.numel() == 0:
            return

        if network.collect_perf_counters:
            network.sample_calls += 1
        if idx_i is None:
            idx_i = torch.randint(0, active_pool.numel(), (), device=network.device)
        if idx_j is None:
            idx_j = torch.randint(0, inactive_pool.numel(), (), device=network.device)

        i = int(active_pool[idx_i].item())
        j = int(inactive_pool[idx_j].item())
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

        # alpha = 2.0
        # eta_i = 0.0 if neuron_noise is None else neuron_noise[i] * (1.0 + alpha * torch.sigmoid(h_i))
        # eta_j = 0.0 if neuron_noise is None else neuron_noise[j] * (1.0 + alpha * torch.sigmoid(h_j))


        b_i = R*g*(A*bump_i + x_i) + R*eta_i
        b_j = R*g*(A*bump_j + x_j) + R*eta_j
        # b_i = (A*bump_i + x_i + eta_i)
        # b_j = (A*bump_j + x_j + eta_j)

        if network.energy_metrics_enabled:
            total_drive_i = R * (g * (a_i_before * h_i + A * bump_i + x_i) + eta_i)
            total_drive_j = R * (g * (a_j_before * h_j + A * bump_j + x_j) + eta_j)
            network.energy_sum_abs_total_drive_gen += torch.abs(total_drive_i) + torch.abs(total_drive_j)
            network.energy_prop_count_gen += 1
        
        # W_ji = network.weights[j, i]
        # c = g * R

        # dE = c * (h_i - h_j + W_ji) + (b_i - b_j)
        dE = -c * (da_i*h_i + da_j*h_j + da_i*da_j*W_ij) - (b_i*da_i + b_j*da_j)


        # --- 4. Metropolis acceptance ---
        if getattr(network, "sigma_temp", 0.0) > 0.0:
            if dE <= 0:
                accept = True
            else:
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
            if network.energy_metrics_enabled:
                network.energy_accept_count_gen += 1

            active_pool[idx_i] = j
            inactive_pool[idx_j] = i
            active_pos[j] = idx_i
            active_pos[i] = -1
            inactive_pos[i] = idx_j
            inactive_pos[j] = -1

            if network.collect_perf_counters:
                network.accept_calls += 1
                network.pool_swap_calls += 1
            if network.debug_validate_pools:
                network.validate_activity_pools()



class MetropolisUpdateStrategyPadamsey(UpdateStrategy):
    """
    Soft Metropolis update for CANNetwork.

    Assumes:
    - network.weights is (num_neurons, num_neurons) and (approximately) symmetric.
    - network.synaptic_drive is kept in sync with network.state (W @ state).
    - network.state is a 1D tensor of 0/1 floats.
    - network.sigma_temp is the Metropolis temperature T.
    """
    def update(
        self,
        network: CANNetwork,
        rand_index: Optional[torch.Tensor] = None,
        neuron_noise: Optional[torch.Tensor] = None,
        x_noise: Optional[torch.Tensor] = None,
        synapse_noise: Optional[list] = None,
        idx_i: Optional[torch.Tensor] = None,
        idx_j: Optional[torch.Tensor] = None,
    ) -> None:
        state = network.state

        if network.collect_perf_counters:
            network.sample_calls += 1



        i = int(torch.randint(0, network.num_neurons, (), device=network.device).item())
        prev_state = state[i]
        # turn_on = bool(prev_state < 0.5)
        new_state = 1.0 - prev_state

        # --- 2. Precompute local fields needed for ΔE ---
        # syn_ok, spon = synapse_noise if synapse_noise is not None else (None, None)
        # syn_ok_i = 1.0 if syn_ok is None else syn_ok[i]
        # spon_i   = 0.0 if spon   is None else spon[i]
        # if turn_on:
        #     a_i_before = spon_i
        #     a_i_after  = syn_ok_i        # after i: 1->0
        # else:
        #     a_i_before = syn_ok_i
        #     a_i_after  = spon_i        # after i: 1->0
        # da_i = a_i_after - a_i_before
        
        input_bump = network.A * network.input_bump_profile
  
        xi= 0.0 if x_noise is None else x_noise

        u = network.synaptic_drive + input_bump  + xi
        u_centered = u - u.mean() #Input centering inhibition


        b = network.sigma_temp*np.log(network.threshold_active_fraction / (1.0 - network.threshold_active_fraction))
        # b = -network.constrict*(network.active_count_tensor - network.target_active_tensor) * network.inv_num_neurons_tensor
        if network.input_resistance !=1.0:
            db = 0.005
        else:
            db = 0.0
        # db = 0.005
        sigma0 = 0.0

        sigma_i = sigma0 + network.input_resistance*network.sigma_eta*network.sigma_temp*torch.nn.functional.softplus(u[i]/network.sigma_temp)
        eta_i = 0.0 if neuron_noise is None else neuron_noise[i] * sigma_i

        field_i = network.ampar_conductance*network.input_resistance*u_centered[i]+b+db+eta_i
        dE = -(new_state - prev_state) * field_i


        if network.energy_metrics_enabled:
            total_drive_i =0# R * (g * (a_i_before * h_i + A * bump_i + x_i) + eta_i)
            network.energy_sum_abs_total_drive_gen += torch.abs(total_drive_i)
            network.energy_prop_count_gen += 1
        

        # --- 4. Metropolis acceptance ---
        if getattr(network, "sigma_temp", 0.0) > 0.0:
            if dE <= 0:
                accept = True
            else:
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
            state[i] = new_state
            network.active_count_tensor += (new_state - prev_state)
            network.synaptic_drive.add_(network.weights[:, i], alpha=float((new_state - prev_state)))
            if network.energy_metrics_enabled:
                network.energy_accept_count_gen += 1

            if network.collect_perf_counters:
                network.accept_calls += 1


class MetropolisUpdateStrategyMixedPadamsey(UpdateStrategy):
    """
    Mixed Metropolis update:

    - exchange move: 1,0 -> 0,1  ; helps bump diffusion
    - single flip:   s_i -> 1-s_i ; allows activity fluctuations

    Uses centered-input field:
        u = W @ s + x
        H_i = c * (u_i - mean(u)) + b + db + eta_i
    """

    def __init__(self, p_swap: float = 0.6, db_food_restricted: float = 0.008) -> None:
        if not 0.0 <= p_swap <= 1.0:
            raise ValueError("p_swap must be between 0 and 1.")
        self.p_swap = float(p_swap)
        self.db_food_restricted = float(db_food_restricted)

    def update(
        self,
        network: CANNetwork,
        rand_index: Optional[torch.Tensor] = None,
        neuron_noise: Optional[torch.Tensor] = None,
        x_noise: Optional[torch.Tensor] = None,
        synapse_noise: Optional[list] = None,
        idx_i: Optional[torch.Tensor] = None,
        idx_j: Optional[torch.Tensor] = None,
    ) -> None:
        state = network.state
        device = network.device
        T = network.sigma_temp

        if network.collect_perf_counters:
            network.sample_calls += 1

        # Main knob: probability of exchange move.
        # Larger = better bump diffusion, less activity fluctuation.
        p_swap = self.p_swap

        # Common input and field ingredients.
        x = network.A * network.input_bump_profile
        if x_noise is not None:
            x = x + x_noise

        u = network.synaptic_drive + x
        u_c = u - u.mean()

        c = network.ampar_conductance * network.input_resistance

        # Baseline bias sets mean activity for single flips.
        f = float(network.threshold_active_fraction)
        b = T * torch.log(torch.tensor(f / (1.0 - f), device=device))

        # Small FR boost, if desired.
        db = self.db_food_restricted if network.input_resistance != 1.0 else 0.0

        sigma0 = 0.0
        noise_gain = network.input_resistance

        def local_field(k):
            sigma_k = (
                sigma0
                + noise_gain
                * network.sigma_eta
                * T
                * torch.nn.functional.softplus(u[k] / T)
            )

            if neuron_noise is None:
                eta_k = sigma_k * torch.randn((), device=device)
            else:
                eta_k = sigma_k * neuron_noise[k]
            

            return c * u_c[k] + b + db + eta_k

        def accept_metropolis(dE):
            if T > 0:
                if dE <= 0:
                    return True
                return torch.log(torch.rand((), device=device)) < (-dE / T)
            else:
                return bool(dE < 0)

        # ------------------------------------------------------------
        # 1) Exchange move: choose active i and inactive j.
        # ------------------------------------------------------------
        if torch.rand((), device=device) < p_swap:
            active = torch.nonzero(state > 0.5, as_tuple=False).flatten()
            inactive = torch.nonzero(state < 0.5, as_tuple=False).flatten()

            if active.numel() == 0 or inactive.numel() == 0:
                return

            i = active[torch.randint(0, active.numel(), (), device=device)]
            j = inactive[torch.randint(0, inactive.numel(), (), device=device)]

            H_i = local_field(i)
            H_j = local_field(j)

            if network.energy_metrics_enabled:
                network.energy_sum_abs_total_drive_gen += torch.abs(c * u_c[i]) + torch.abs(c * u_c[j])

            # Swap: i: 1->0, j: 0->1
            # dE = H_i - H_j, plus pair correction.
            dE = H_i - H_j + c * network.weights[i, j]

            if accept_metropolis(dE):
                state[i] = 0.0
                state[j] = 1.0

                network.synaptic_drive.add_(network.weights[:, i], alpha=-1.0)
                network.synaptic_drive.add_(network.weights[:, j], alpha=1.0)

                if network.collect_perf_counters:
                    network.accept_calls += 1

                if network.energy_metrics_enabled:
                    network.energy_accept_count_gen += 1

        # ------------------------------------------------------------
        # 2) Single flip move: choose any neuron.
        # ------------------------------------------------------------
        else:
            i = torch.randint(0, network.num_neurons, (), device=device)

            old = state[i]
            new = 1.0 - old
            delta = new - old

            H_i = local_field(i)

            if network.energy_metrics_enabled:
                network.energy_sum_abs_total_drive_gen += torch.abs(c * u_c[i])

            # Flip energy.
            dE = -delta * H_i

            if accept_metropolis(dE):
                state[i] = new
                network.active_count_tensor += delta
                network.synaptic_drive.add_(network.weights[:, i], alpha=float(delta.item()))

                if network.collect_perf_counters:
                    network.accept_calls += 1

                if network.energy_metrics_enabled:
                    network.energy_accept_count_gen += 1

        if network.energy_metrics_enabled:
            network.energy_prop_count_gen += 1

        if network.record_diagnostics:
            network.total_activity.append(float(network.state.mean().detach().cpu()))
