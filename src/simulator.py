from typing import Callable, Optional
import torch
import logging
from src.network import CANNetwork
from src.update_strategies import UpdateStrategy
torch.backends.cudnn.benchmark = True


def simulate(
    network_params: dict,
    num_generations: int,
    update_strategy,
    progress_callback: Optional[Callable[[int], None]] = None,
) -> "CANNetwork":
    """Run the CANN simulation for `num_generations` sweeps."""
    if num_generations <= 0:
        raise ValueError("num_generations must be > 0")

    net = CANNetwork(**network_params)
    logging.info("Running on device: %s", net.device)

    net.initialize_weights()
    net.initialize_state()
    net.synaptic_drive = net.weights @ net.state

    progress_stride = max(1, num_generations // 20)
    noise = None

    for gen in range(num_generations):
        # progress
        if progress_callback and gen % progress_stride == 0:
            progress_callback(gen)

        # refresh block-level randomness
        if gen % net.block_size == 0:
            if getattr(net, "sigma_theta_steps", 0.0) > 0.0:
                # k = int(torch.round(torch.randn((), device=net.device) * net.sigma_theta_steps).item())
                # net.input_bump_profile = torch.roll(net.base_input_bump_profile, shifts=k)
                # k is a 0-dim tensor on GPU
                k = torch.round(torch.randn((), device=net.device) * net.sigma_theta_steps).to(torch.long)
                idx = (torch.arange(net.num_neurons, device=net.device) - k) % net.num_neurons
                net.input_bump_profile = net.base_input_bump_profile[idx]

            noise = None
            if getattr(net, "sigma_eta", 0.0) > 0.0:
                noise = torch.randn(net.num_neurons, device=net.device) * net.sigma_eta

        # one sweep = N updates in random order
        for _ in torch.randperm(net.num_neurons, device=net.device):
            update_strategy.update(net, neuron_noise=noise)

        net.state_history.append(net.state.clone())
        net.generation += 1

    return net
