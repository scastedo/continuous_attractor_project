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
        "noise": 0.00000,
        "field_width": 0.05,
        "syn_fail": 0.0,
        "spon_rel": 0.0,
        "constrict": 1.0,
        "fraction_active": 0.1,
        "I_str": 0.05,
        "I_dir": 50.0,
        "num_updates": 1
    }
    
    num_generations = 40
    runs = 10

    # Select the update strategy.
    # update_strategy = update_strategies.DynamicsUpdateStrategy()
    update_strategy = update_strategies.MetropolisUpdateStrategy()

    # Run the simulation.
    # network = simulator.simulate(network_params, num_generations, runs, update_strategy)
    network = simulator.simulate_tuning_curve(network_params, num_generations, runs, update_strategy)

    # Visualize the simulation results.  
    visualisation.create_visualization_report(network)
    # visualisation.view_interactive_report(network)

if __name__ == "__main__":
    main()
