import torch
import logging
from src.network import CANNetwork
from src.update_strategies import UpdateStrategy

def simulate(network_params: dict, num_generations: int, runs: int, update_strategy: UpdateStrategy) -> CANNetwork:
    """
    Run the Hopfield network simulation.

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
        final_network = network

    return final_network
