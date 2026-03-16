from pathlib import Path

import numpy as np

from src.simulator import simulate
from src.update_strategies import MetropolisUpdateStrategy3


def _network_params(num_neurons: int) -> dict:
    return {
        "num_neurons": num_neurons,
        "sigma_temp": 0.01,
        "sigma_input": 0.05,
        "I_str": 0.01,
        "I_dir": 0.5,
        "syn_fail": 0.0,
        "spon_rel": 0.0,
        "sigma_eta": 0.0,
        "input_resistance": 1.0,
        "ampar_conductance": 1.0,
        "constrict": 1.0,
        "recurrent_width_fraction": 0.1,
        "threshold_active_fraction": 0.1,
        "block_size": 1,
        "sigma_theta": 0.0,
    }


def test_simulate_writes_energy_metrics_when_enabled(tmp_path: Path) -> None:
    num_neurons = 12
    num_generations = 4
    energy_metrics_path = tmp_path / "energy_metrics.npy"
    state_history_path = tmp_path / "state_history.npy"

    network = simulate(
        network_params=_network_params(num_neurons),
        num_generations=num_generations,
        update_strategy=MetropolisUpdateStrategy3(),
        state_history_path=state_history_path,
        energy_metrics_path=energy_metrics_path,
    )

    assert network.energy_metrics_enabled is True
    assert energy_metrics_path.exists()

    metrics = np.load(energy_metrics_path)
    assert metrics.shape == (num_generations, 4)
    assert metrics.dtype == np.float32
    assert np.all(metrics[:, 0] == num_neurons)
    assert np.all(metrics[:, 1] >= 0.0)
    assert np.all(metrics[:, 1] <= metrics[:, 0])
    assert np.all(metrics[:, 2] >= 0.0)
    assert np.allclose(metrics[:, 3], metrics[:, 2] / metrics[:, 0])


def test_simulate_skips_energy_metrics_when_disabled(tmp_path: Path) -> None:
    network = simulate(
        network_params=_network_params(12),
        num_generations=3,
        update_strategy=MetropolisUpdateStrategy3(),
    )

    assert network.energy_metrics_enabled is False
    assert network.energy_metrics_path is None
    assert network.energy_metrics_shape is None
    assert list(tmp_path.glob("energy_metrics*.npy")) == []
