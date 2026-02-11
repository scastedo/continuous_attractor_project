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
    xnoise = None
    syn_ok = None    
    spon = None

    # Burn-in period
    for _ in range(1000):
        for _ in range(net.num_neurons):
            update_strategy.update(net)


    for gen in range(num_generations):
        if progress_callback and gen % progress_stride == 0:
            progress_callback(gen)
        # refresh block-level randomness
        if gen % net.block_size == 0:
            if getattr(net, "sigma_theta_steps", 0.0) > 0.0:
                z = torch.randn(net.num_neurons, device=net.device) * net.sigma_theta_steps
                xnoise = z#* net.input_bump_profile# *net.A
        
        if getattr(net, "syn_fail", 0.0) > 0.0:
            syn_ok = torch.bernoulli(torch.full((net.num_neurons,), 1.0 - net.syn_fail, device=net.device))
            spon   = torch.bernoulli(torch.full((net.num_neurons,), net.spon_rel, device=net.device))
        if getattr(net, "sigma_eta", 0.0) > 0.0:
            noise = torch.randn(net.num_neurons, device=net.device) * net.sigma_eta
        # one sweep = N updates in random order
        for _ in range(net.num_neurons):
            update_strategy.update(net, neuron_noise=noise, x_noise=xnoise, synapse_noise =(syn_ok, spon))

        net.state_history.append(net.state.clone())
        net.generation += 1

    return net
