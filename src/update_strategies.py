import torch
import numpy as np
from abc import ABC, abstractmethod
from src.network import CANNetwork

class UpdateStrategy(ABC):
    @abstractmethod
    def update(self, network: CANNetwork) -> None:
        """
        Update the network state.
        """
        pass

class DynamicsUpdateStrategy(UpdateStrategy):
    """
    Update strategy using simple dynamics (sigmoidal updates with optional noise).
    """
    def update(self, network: CANNetwork) -> None:
        for _ in range(network.num_updates):
            # Randomly choose a neuron to update.
            rand_index = torch.randint(0, network.num_neurons, (1,)).item()
            epsp = torch.dot(network.weights[rand_index], network.state)
            threshold = network.constrict * (torch.sum(network.state) - network.num_neurons * network.fraction_active) / network.num_neurons
            index_activation = (1 - network.syn_fail) * epsp + network.spon_rel - threshold

            network.activations.append(index_activation.item())
            network.total_activity.append(torch.sum(network.state).item())

            if network.noise == 0:
                # Deterministic update.
                if index_activation < 0:
                    network.state[rand_index] = 0.0
                elif index_activation > 0:
                    network.state[rand_index] = 1.0
                else:
                    network.state[rand_index] = torch.tensor(np.random.choice([0.0, 1.0]), device=network.device)
            else:
                # Stochastic update using a sigmoid probability.
                prob = 1 / (1 + torch.exp(-index_activation / network.noise))
                if torch.rand(1, device=network.device).item() < prob.item():
                    network.state[rand_index] = 1.0
                else:
                    network.state[rand_index] = 0.0

class MetropolisUpdateStrategy(UpdateStrategy):
    """
    Update strategy using the Metropolis algorithm.
    """
    def update(self, network: CANNetwork) -> None:
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
