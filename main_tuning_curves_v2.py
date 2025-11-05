#!/usr/bin/env cann
import logging
from src import simulator, visualisation, update_strategies
import numpy as np
import datetime
from pathlib import Path
import torch



def main():
    # Configure logging.
    logging.basicConfig(level=logging.INFO, 
                        format='%(asctime)s - %(levelname)s - %(message)s')
    
    num_generations = 15 # how long simulation for
    runs = 100 # how many angles to do
    angles = 6
    
    I_dir = np.zeros(num_generations)
    # I_dir[:num_generations//4] = 50
    # I_dir[num_generations//4:num_generations//2] = 50
    # I_dir[num_generations//2:3*num_generations//4] =50
    # I_dir[3*num_generations//4:] = 80
    I_dir[:] = 15
    I_str = np.zeros(num_generations)
    # I_str[:num_generations//4] = 0.00
    # I_str[num_generations//4:num_generations//2] = 0.05
    # I_str[num_generations//2:3*num_generations//4] =0.0
    # I_str[3*num_generations//4:] = 0.05
    I_str[:] = 0.2

    # Define network parameters.
    network_params = {
        "num_neurons": 150,
        "noise": 0.05, #0.01 for i_str 
        "field_width": 0.05,
        "syn_fail": 0.05,  # Amount of synaptic failure
        "spon_rel": 0.05,    # Spontaneous release rate
        "constrict": 50.0,   # Constriction factor... related to degree of inhibition?
        "fraction_active": 0.1,  # Fraction of neurons that are active
        "I_str": I_str,     # Strength of input....should be v small
        "I_dir": I_dir,    # Neuron index where you want input 
        "num_updates": 1   # number of trials
    }
    

    # Select the update strategy.
    update_strategy = update_strategies.DynamicsUpdateStrategy()
    # update_strategy = update_strategies.MetropolisUpdateStrategy()

    ### Run the normal simulation.
    # runs = 1
    # network = simulator.simulate(network_params, num_generations, runs, update_strategy)
    # visualisation.create_visualization_report(network)
    # visualisation.view_interactive_report(network)
    
    ### Run the old tuning curve simulation.

    # network = simulator.simulate_tuning_curve(network_params, num_generations, runs,angles, update_strategy)
    # visualisation.save_tuning_data(network,num_generations, angles,runs)

    
    
    
    steps_saved = 10
    
    ### Run the new tuning curve simulation.
    I_dir_values = np.linspace(0, network_params['num_neurons']-angles, angles).repeat(runs)
    save_state_history = np.zeros((network_params['num_neurons'],angles*runs))
    for run in range(runs*angles):
        logging.info(f"Starting run {run+1} of {runs*angles}")
        I_dir_val = I_dir_values[run]
        network = simulator.simulate_tuning_curve_parallel(network_params, num_generations, I_dir_val, update_strategy)
        state = network.state_history
        last_states = torch.stack(state[-steps_saved:]).cpu().numpy()
        average_state = np.mean(last_states, axis=0)
        save_state_history[:,run]  = average_state
    save_state_history = save_state_history.reshape(network_params['num_neurons'],angles,runs)
    # Generate timestamp for the filename
    timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    output_file = Path("reports") / f"state_history_{timestamp}"
    meta_data = network_params.copy()

    np.save(output_file, save_state_history)
    # Save metadata as a separate file with the same base name
    meta_output_file = output_file.with_suffix('.meta.npy')
    np.save(meta_output_file, meta_data)
    

if __name__ == "__main__":
    main()
