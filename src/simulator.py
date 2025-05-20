import torch
import logging
from src.network import CANNetwork
from src.update_strategies import UpdateStrategy

def simulate(
        network_params: dict,
        num_generations: int,
        runs: int,
        update_strategy: UpdateStrategy) -> CANNetwork:
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
    
    final_network = None
    for run in range(runs):
        logging.info(f"Starting run {run+1} of {runs}")
        network = CANNetwork(device=device, **network_params)
        network.initialize_weights()
        network.initialize_state()
        network.record_energy()
        
        for gen in range(1, num_generations):
            for _ in range(network.num_neurons):
                update_strategy.update(network)
                network.record_energy()
                network.locate_centre()
            network.state_history.append(network.state.clone())
            network.generation += 1
        network.record_correlations()
        final_network = network
        
    return final_network

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
    I_dir_values = torch.linspace(0, network_params['num_neurons'] , angles).repeat(runs)
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