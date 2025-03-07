import torch
import pytest
from src.network import CANNetwork

def test_initialize_weights_symmetry():
    num_neurons = 50
    network = CANNetwork(num_neurons=num_neurons,
                              noise=0.0,
                              field_width=0.1,
                              syn_fail=0.0,
                              spon_rel=0.0,
                              constrict=1.0,
                              fraction_active=0.1,
                              I_str=0.0,
                              I_dir=0,
                              num_updates=1)
    network.initialize_weights()
    # Ensure that the weight matrix is symmetric.
    assert torch.allclose(network.weights, network.weights.T), "Weights matrix should be symmetric."

def test_initialize_state():
    num_neurons = 50
    fraction_active = 0.2
    network = CANNetwork(num_neurons=num_neurons,
                              noise=0.0,
                              field_width=0.1,
                              syn_fail=0.0,
                              spon_rel=0.0,
                              constrict=1.0,
                              fraction_active=fraction_active,
                              I_str=0.0,
                              I_dir=0,
                              num_updates=1)
    network.initialize_state()
    active_count = torch.sum(network.state).item()
    expected_active = int(fraction_active * num_neurons)
    # Allow a difference of one due to rounding.
    assert abs(active_count - expected_active) <= 1, "State does not have the expected number of active neurons."
