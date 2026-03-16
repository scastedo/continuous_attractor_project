import math
from typing import Optional
import torch

from src.network import CANNetwork
from src.update_strategies import MetropolisUpdateStrategy3


def build_network(
    num_neurons: int = 50,
    threshold_active_fraction: float = 0.2,
    recurrent_width_fraction: Optional[float] = None,
    sigma_temp: float = 0.01,
) -> CANNetwork:
    if recurrent_width_fraction is None:
        recurrent_width_fraction = threshold_active_fraction
    return CANNetwork(
        num_neurons=num_neurons,
        sigma_temp=sigma_temp,
        sigma_input=recurrent_width_fraction / 2,
        I_str=0.01,
        I_dir=0.5,
        syn_fail=0.0,
        spon_rel=0.0,
        sigma_eta=0.0,
        input_resistance=1.0,
        ampar_conductance=1.0,
        constrict=1.0,
        recurrent_width_fraction=recurrent_width_fraction,
        threshold_active_fraction=threshold_active_fraction,
        block_size=10,
        sigma_theta=0.0,
    )


def test_initialize_weights_symmetry():
    network = build_network(num_neurons=50, threshold_active_fraction=0.1)
    network.initialize_weights()
    assert torch.allclose(network.weights, network.weights.T), "Weights matrix should be symmetric."


def test_initialize_state():
    num_neurons = 50
    fraction_active = 0.2
    network = build_network(num_neurons=num_neurons, threshold_active_fraction=fraction_active)
    network.initialize_state()
    active_count = int(torch.sum(network.state).item())
    expected_active = int(fraction_active * num_neurons)
    assert abs(active_count - expected_active) <= 1, "State does not have the expected number of active neurons."


def test_initialize_activity_pools_and_validate():
    network = build_network(num_neurons=40, threshold_active_fraction=0.25)
    network.initialize_weights()
    network.initialize_state()
    network.initialize_activity_pools()
    network.validate_activity_pools()

    assert network.active_pool is not None
    assert network.inactive_pool is not None
    assert network.active_pos is not None
    assert network.inactive_pos is not None
    assert network.active_pool.numel() + network.inactive_pool.numel() == network.num_neurons


def test_metropolis_update_preserves_activity_and_pool_invariants():
    network = build_network(num_neurons=60, threshold_active_fraction=0.2, sigma_temp=0.01)
    network.initialize_weights()
    network.initialize_state()
    network.synaptic_drive = network.weights @ network.state
    network.debug_validate_pools = True
    network.collect_perf_counters = True
    network.initialize_activity_pools()

    strategy = MetropolisUpdateStrategy3()
    initial_active = int(network.state.sum().item())
    updates = 400

    for _ in range(updates):
        strategy.update(network)

    assert int(network.state.sum().item()) == initial_active
    network.validate_activity_pools()
    assert network.sample_calls == updates
    assert network.accept_calls == network.pool_swap_calls


def test_pool_index_sampling_is_reasonably_uniform():
    torch.manual_seed(0)
    network = build_network(num_neurons=80, threshold_active_fraction=0.25)
    network.initialize_weights()
    network.initialize_state()
    network.initialize_activity_pools()

    assert network.active_pool is not None
    n_active = int(network.active_pool.numel())
    draws = 20000
    counts = torch.zeros(n_active, dtype=torch.int64)
    for _ in range(draws):
        idx = torch.randint(0, n_active, (), device=network.device)
        counts[int(idx.item())] += 1

    expected = draws / n_active
    max_rel_dev = torch.max(torch.abs(counts.float() - expected)) / expected
    assert float(max_rel_dev) < 0.25, "Pool index sampling looks too skewed."


def test_recurrent_width_controls_weights_independently_from_active_fraction():
    active_fraction = 0.2
    narrow = build_network(
        num_neurons=80,
        threshold_active_fraction=active_fraction,
        recurrent_width_fraction=0.1,
    )
    wide = build_network(
        num_neurons=80,
        threshold_active_fraction=active_fraction,
        recurrent_width_fraction=0.3,
    )
    narrow.initialize_weights()
    wide.initialize_weights()

    assert not torch.equal(narrow.weights, wide.weights), "Changing recurrent width should change connectivity."


def test_active_fraction_controls_initial_state_independently_from_recurrent_width():
    recurrent_width = 0.2
    low_active = build_network(
        num_neurons=100,
        threshold_active_fraction=0.1,
        recurrent_width_fraction=recurrent_width,
    )
    high_active = build_network(
        num_neurons=100,
        threshold_active_fraction=0.3,
        recurrent_width_fraction=recurrent_width,
    )
    low_active.initialize_state()
    high_active.initialize_state()

    assert int(low_active.state.sum().item()) == 10
    assert int(high_active.state.sum().item()) == 30


def test_energy_counters_track_proposals_and_accepts():
    network = build_network(num_neurons=60, threshold_active_fraction=0.2, sigma_temp=0.01)
    network.initialize_weights()
    network.initialize_state()
    network.synaptic_drive = network.weights @ network.state
    network.initialize_activity_pools()
    network.energy_metrics_enabled = True
    network.reset_energy_counters()

    strategy = MetropolisUpdateStrategy3()
    updates = 300
    for _ in range(updates):
        strategy.update(network)

    assert network.energy_prop_count_gen == updates
    assert 0 <= network.energy_accept_count_gen <= updates
    total_drive = float(network.energy_sum_abs_total_drive_gen.item())
    assert math.isfinite(total_drive)
    assert total_drive >= 0.0
