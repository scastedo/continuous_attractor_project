#!/usr/bin/env cann
import logging
from src import simulator, visualisation, update_strategies
import numpy as np

def main():
    # Configure logging.
    logging.basicConfig(level=logging.INFO, 
                        format='%(asctime)s - %(levelname)s - %(message)s')
    
    num_generations = 300 # how long simulation for
    runs = 1 # how many angles to do

    # I_dir = np.zeros(num_generations)
    # I_dir[:] = 20
    # I_str = np.zeros(num_generations)
    # I_str[:] = 0.1

    I_dir = np.random.normal(loc=25, scale=2, size=num_generations)
    I_dir = np.full(num_generations, 20)  # for testing
    I_str = np.random.normal(loc=0.1, scale=0.05, size=num_generations)
    I_str = np.clip(I_str, 0,None)

    # Define network parameters.
    network_params = {
        "num_neurons": 90,
        "noise": 0, #0.1, #0.01 for i_str 
        "field_width": 0.05,
        "syn_fail": 0.000,  # Amount of synaptic failure
        "spon_rel": 0.0,    # Spontaneous release rate,
        "noise_eta": 0.15,  # Noise (temperature) parameter
        "input_resistance": 50,  # Input resistance
        "ampar_conductance": 0.3,    # AMPAR conductance
        "constrict": 1.0,   # Constriction factor... related to degree of inhibition?
        "fraction_active": 0.1,  # Fraction of neurons that are active
        "I_str": I_str,     # Strength of input....should be v small
        "I_dir": I_dir,    # Neuron index where you want input 
        "num_updates": 2   # number of trials
    }
    

    # Select the update strategy.
    update_strategy = update_strategies.DynamicsUpdateStrategyGain()
    # update_strategy = update_strategies.MetropolisUpdateStrategy()

    # Run the simulation.
    network = simulator.simulate(network_params, num_generations, runs, update_strategy)
    network.noise_covariance()
    visualisation.create_visualization_report(network)

    # network_params["I_dir"] = np.full(num_generations, 120)
    # network = simulator.simulate(network_params, num_generations, runs, update_strategy)
    # visualisation.create_visualization_report(network)

if __name__ == "__main__":
    main()
