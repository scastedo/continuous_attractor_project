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
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logging.info(f"Running on device: {device}")
    
    network = CANNetwork(device=device, **network_params)
    network.initialize_weights()
    network.initialize_state()
    network.record_energy()
    network.synaptic_drive = torch.mv(network.weights, network.state)
    network.active_count = int(torch.sum(network.state).item())

    # with torch.no_grad():
    #         eps = torch.randn((), device=network.device)
    #         network.A = network.A_mu + network.A_rho * (network.A - network.A_mu) + network.A_sigma * eps
    #         network.A.clamp_(min=0.0)
    network.A = network.A_mu
    network.A_fixed =  float(network.A.item())  # 0-D tensor on device, won’t change this gen
    network.input_fluctuations.append(network.A_fixed)

    for gen in range(1, num_generations):
        if progress_callback and gen % 100 == 0:
            progress_callback(gen)
        for _ in range(network.num_neurons):
            update_strategy.update(network)
        
        # network.total_activity.append(torch.sum(network.state).item())
        # network.record_energy()
        # network.locate_centre()
        network.state_history.append(network.state.clone())
        network.generation += 1
    return network

def simulate_tuning_curve(
        network_params: dict,
        num_generations: int,
        runs: int,
        angles: int,
        update_strategy: UpdateStrategy) -> CANNetwork:
    """
    Run the CANN network simulation to generate tuning curves.

    Args:
        network_params (dict): Parameters for network initialization.
        num_generations (int): Number of simulation generations.
        runs (int): Number of independent runs.
        update_strategy (UpdateStrategy): Strategy used for updating network state.

    Returns:
        CANNetwork: The final network instance after simulation.
    """
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logging.info(f"Running on device: {device}")
    
    final_network = None
    # Define directional input over different runs

    # I_dir_values = torch.linspace(0, network_params['num_neurons'] , runs)
    # instead make it do 10 times at same direction and over 12 different directions
    I_dir_values = torch.linspace(0, network_params['num_neurons']-angles, angles).repeat(runs)
    network = CANNetwork(device=device, **network_params)
    network.initialize_weights()
    for run in range(runs* angles):
        logging.info(f"Starting run {run+1} of {runs*angles}")
        network.I_dir = I_dir_values[run].repeat(num_generations)
        network.initialize_state()
        network.record_energy()
        for gen in range(1, num_generations):
            for _ in range(network.num_neurons):
                update_strategy.update(network)
                network.record_energy()
                network.locate_centre()
            network.state_history.append(network.state.clone())
        network.tuning_curves[I_dir_values[run]] = network.state
        final_network = network
    return final_network



def simulate_tuning_curve_parallel(
        network_params: dict,
        num_generations: int,
        I_dir_val: int,
        update_strategy: UpdateStrategy) -> CANNetwork:
    """
    Run the CANN network simulation to generate tuning curves.

    Args:
        network_params (dict): Parameters for network initialization.
        num_generations (int): Number of simulation generations.
        runs (int): Number of independent runs.
        update_strategy (UpdateStrategy): Strategy used for updating network state.

    Returns:
        CANNetwork: The final network instance after simulation.
    """
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")    
    network = CANNetwork(device=device, **network_params)
    network.initialize_weights()
    network.initialize_state()
    network.record_energy()
    network.I_dir = torch.tensor(I_dir_val).repeat(num_generations) 

    for gen in range(1, num_generations):
        for _ in range(network.num_neurons):
            update_strategy.update(network)
            network.record_energy()
            network.locate_centre()
        network.state_history.append(network.state.clone())
        network.generation += 1
    # network.record_correlations()
    final_network = network
    return final_network