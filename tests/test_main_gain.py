import json
from pathlib import Path

import pytest

from main_gain import RunSpec, parse_args, run_experiment


def _build_spec(active_fraction: float) -> RunSpec:
    return RunSpec(
        ampar=1.0,
        rin=1.0,
        idir=0.5,
        sigma_temp=0.01,
        sigma_eta=0.0,
        block_size=10,
        sigma_theta=0.0,
        trial=0,
        num_neurons=100,
        num_generations=10,
        outdir=Path("/tmp"),
        tag="",
        loglevel=20,
        i_str=0.01,
        cann_width=0.1,
        active_fraction=active_fraction,
        syn_fail=0.0,
        spon_rel=0.0,
        save_energy_metrics=False,
    )


def test_run_id_changes_when_active_fraction_changes() -> None:
    spec_a = _build_spec(active_fraction=0.1)
    spec_b = _build_spec(active_fraction=0.13)

    assert spec_a.run_id != spec_b.run_id


def test_parse_args_supports_active_fraction_sweep() -> None:
    config = parse_args(
        [
            "--threshold", "0.1",
            "--active-fraction", "0.1", "0.13",
            "--outdir", "/tmp/cann_parse_args_test",
        ]
    )
    specs = list(config.iter_specs())

    assert len(specs) == 2
    assert {spec.cann_width for spec in specs} == {0.1}
    assert sorted(spec.active_fraction for spec in specs) == pytest.approx([0.1, 0.13])


def test_parse_args_without_active_fraction_defaults_to_threshold() -> None:
    config = parse_args(
        [
            "--threshold", "0.1",
            "--outdir", "/tmp/cann_parse_args_test",
        ]
    )
    specs = list(config.iter_specs())

    assert len(specs) == 1
    assert specs[0].cann_width == pytest.approx(0.1)
    assert specs[0].active_fraction == pytest.approx(0.1)


def test_parse_args_supports_save_energy_metrics_flag() -> None:
    config = parse_args(
        [
            "--threshold", "0.1",
            "--save-energy-metrics",
            "--outdir", "/tmp/cann_parse_args_test",
        ]
    )
    specs = list(config.iter_specs())

    assert config.save_energy_metrics is True
    assert len(specs) == 1
    assert specs[0].save_energy_metrics is True


def test_parse_args_saves_energy_metrics_by_default() -> None:
    config = parse_args(
        [
            "--threshold", "0.1",
            "--outdir", "/tmp/cann_parse_args_test",
        ]
    )
    specs = list(config.iter_specs())

    assert config.save_energy_metrics is True
    assert len(specs) == 1
    assert specs[0].save_energy_metrics is True


def test_parse_args_supports_disabling_energy_metrics() -> None:
    config = parse_args(
        [
            "--threshold", "0.1",
            "--no-save-energy-metrics",
            "--outdir", "/tmp/cann_parse_args_test",
        ]
    )
    specs = list(config.iter_specs())

    assert config.save_energy_metrics is False
    assert len(specs) == 1
    assert specs[0].save_energy_metrics is False


def test_run_experiment_saves_energy_metadata_when_enabled(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    class DummyNetwork:
        state_history_dtype = "uint8"
        energy_metrics_dtype = "float32"

    monkeypatch.setattr("main_gain.simulator.simulate", lambda *args, **kwargs: DummyNetwork())
    monkeypatch.setattr("main_gain.visualisation.save_state_history", lambda *args, **kwargs: None)
    monkeypatch.setattr("main_gain.visualisation.create_visualization_report", lambda *args, **kwargs: None)

    spec = RunSpec(
        ampar=1.0,
        rin=1.0,
        idir=0.5,
        sigma_temp=0.01,
        sigma_eta=0.0,
        block_size=10,
        sigma_theta=0.0,
        trial=0,
        num_neurons=20,
        num_generations=3,
        outdir=tmp_path,
        tag="energy_meta",
        loglevel=20,
        i_str=0.01,
        cann_width=0.1,
        active_fraction=0.1,
        syn_fail=0.0,
        spon_rel=0.0,
        save_energy_metrics=True,
    )

    run_outdir = run_experiment(spec)
    metadata = json.loads((run_outdir / "run_metadata.json").read_text(encoding="utf-8"))
    network_params = metadata["network_params"]

    assert network_params["energy_metrics_enabled"] is True
    assert "energy_metrics_path" in network_params
    assert network_params["energy_metrics_dtype"] == "float32"
    assert network_params["energy_metrics_columns"] == [
        "proposal_count",
        "accepted_count",
        "sum_abs_total_drive",
        "mean_abs_total_drive",
    ]
