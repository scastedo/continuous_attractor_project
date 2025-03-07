#!/usr/bin/env cann
import logging
from src import simulator, visualisation, update_strategies

def main():
    # Configure logging.
    logging.basicConfig(level=logging.INFO, 
                        format='%(asctime)s - %(levelname)s - %(message)s')
    
    # Define network parameters.
    network_params = {
        "num_neurons": 100,
        "noise": 0.0,
        "field_width": 0.05,
        "syn_fail": 0.0,
        "spon_rel": 0.0,
        "constrict": 1.0,
        "fraction_active": 0.1,
        "I_str": 0.0,
        "I_dir": 0,
        "num_updates": 1
    }
    
    num_generations = 200
    runs = 1

    # Select the update strategy.
    # update_strategy = update_strategies.DynamicsUpdateStrategy()
    update_strategy = update_strategies.MetropolisUpdateStrategy()

    # Run the simulation.
    network = simulator.simulate(network_params, num_generations, runs, update_strategy)

    # Visualize the simulation results.
    # visualisation.plot_state_history(network)
    # visualisation.plot_metrics(network)
    # visualisation.plot_lyapunov(network)
    
    visualisation.create_visualization_report(network)
    # visualisation.view_interactive_report(network)

if __name__ == "__main__":
    main()
