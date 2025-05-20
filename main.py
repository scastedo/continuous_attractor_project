#!/usr/bin/env cann
import logging
from src import simulator, visualisation, update_strategies
import numpy as np

def main():
    # Configure logging.
    logging.basicConfig(level=logging.INFO, 
                        format='%(asctime)s - %(levelname)s - %(message)s')
    
    num_generations = 50
    runs = 10
    
    I_dir = np.zeros(num_generations)
    # I_dir[:num_generations//4] = 50
    # I_dir[num_generations//4:num_generations//2] = 50
    # I_dir[num_generations//2:3*num_generations//4] =50
    # I_dir[3*num_generations//4:] = 80
    I_dir[:] = 30
    I_str = np.zeros(num_generations)
    # I_str[:num_generations//4] = 0.00
    # I_str[num_generations//4:num_generations//2] = 0.05
    # I_str[num_generations//2:3*num_generations//4] =0.0
    # I_str[3*num_generations//4:] = 0.05
    I_str[:] = 0.01

    # Define network parameters.
    network_params = {
        "num_neurons": 100,
        "noise": 0.0001, #0.01 for i_str 
        "field_width": 0.05,
        "syn_fail": 0.001,  # Amount of synaptic failure
        "spon_rel": 0.0,    # Spontaneous release rate
        "constrict": 1.0,   # Constriction factor... related to degree of inhibition?
        "fraction_active": 0.1,  # Fraction of neurons that are active
        "I_str": I_str,     # Strength of input....should be v small
        "I_dir": I_dir,    # Neuron index where you want input 
        "num_updates": 1   # number of trials
    }
    

    # Select the update strategy.
    update_strategy = update_strategies.DynamicsUpdateStrategy()
    # update_strategy = update_strategies.MetropolisUpdateStrategy()

    # Run the simulation.
    # network = simulator.simulate(network_params, num_generations, runs, update_strategy)
    network = simulator.simulate_tuning_curve(network_params, num_generations, runs, update_strategy)

    # Visualize the simulation results.  
    visualisation.create_visualization_report(network)
    # visualisation.view_interactive_report(network)

if __name__ == "__main__":
    main()
