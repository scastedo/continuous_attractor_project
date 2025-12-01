from collections.abc import Callable
from typing import Optional
import torch
import logging
from src.network import CANNetwork
from src.update_strategies import UpdateStrategy
torch.backends.cudnn.benchmark = True


def simulate(
        network_params: dict,
        num_generations: int,
        update_strategy: UpdateStrategy,
        progress_callback: Optional[Callable[[int], None]] = None) -> CANNetwork:
    """
    Run the CANN network simulation.

    Args:
        network_params (dict): Parameters for network initialization.
        num_generations (int): Number of simulation generations.
        runs (int): Number of independent runs.
        update_strategy (UpdateStrategy): Strategy used for updating network state.

    Returns:
        CANNetwork: The final network instance after simulation.
    """
    # device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    network = CANNetwork(**network_params)
    logging.info(f"Running on device: {network.device}")

    network.initialize_weights()
    network.initialize_state()
    # network.record_energy()
    network.synaptic_drive = torch.mv(network.weights, network.state)
    
    noise_samples = None
    if network.sigma_eta > 0:
        noise_samples = torch.randn(network.num_neurons, device=network.device) * network.sigma_eta

    for gen in range(1, num_generations):
        update_order = torch.randperm(network.num_neurons, device=network.device)
        if progress_callback and gen % (num_generations // 20) == 0:
            progress_callback(gen)
        # Generate different noise samples every 1000 generations
        if gen % 200 == 0:
            noise_samples = None
            if network.sigma_eta > 0:
                noise_samples = torch.randn(network.num_neurons, device=network.device) * network.sigma_eta

        for step, rand_index in enumerate(update_order):
            # neuron_noise = noise_samples[step] if noise_samples is not None else None
            # update_strategy.update(network, rand_index=rand_index, neuron_noise=neuron_noise)
            update_strategy.update(network,neuron_noise=noise_samples)

        network.state_history.append(network.state.clone())
        network.generation += 1
    return network