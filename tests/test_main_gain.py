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
        p_swap=0.6,
        db_fr=0.008,
        gain_dynamics=False,
        save_energy_metrics=False,
        seed=123,
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


def test_parse_args_derives_distinct_reproducible_run_seeds() -> None:
    args = [
        "--idir", "0.5", "0.6",
        "--trials", "2",
        "--seed", "42",
        "--outdir", "/tmp/cann_parse_args_test",
    ]
    specs = list(parse_args(args).iter_specs())
    repeated_specs = list(parse_args(args).iter_specs())

    assert len(specs) == 4
    assert len({spec.seed for spec in specs}) == 4
    assert [spec.seed for spec in specs] == [spec.seed for spec in repeated_specs]


def test_parse_args_defaults_mixed_padamsey_strategy_params() -> None:
    config = parse_args(
        [
            "--threshold", "0.1",
            "--outdir", "/tmp/cann_parse_args_test",
        ]
    )
    specs = list(config.iter_specs())

    assert config.p_swap == pytest.approx(0.6)
    assert config.db_fr == pytest.approx(0.008)
    assert len(specs) == 1
    assert specs[0].p_swap == pytest.approx(0.6)
    assert specs[0].db_fr == pytest.approx(0.008)
    assert "pswap0.6" in specs[0].run_id
    assert "dbfr0.008" in specs[0].run_id
    assert "gaindyn" not in specs[0].run_id


def test_parse_args_supports_custom_mixed_padamsey_strategy_params() -> None:
    config = parse_args(
        [
            "--pswap", "0.25",
            "--db-fr", "0.004",
            "--outdir", "/tmp/cann_parse_args_test",
        ]
    )
    specs = list(config.iter_specs())

    assert config.p_swap == pytest.approx(0.25)
    assert config.db_fr == pytest.approx(0.004)
    assert len(specs) == 1
    assert specs[0].p_swap == pytest.approx(0.25)
    assert specs[0].db_fr == pytest.approx(0.004)
    assert "pswap0.25" in specs[0].run_id
    assert "dbfr0.004" in specs[0].run_id


def test_parse_args_disables_gain_dynamics_by_default() -> None:
    config = parse_args(
        [
            "--outdir", "/tmp/cann_parse_args_test",
        ]
    )
    specs = list(config.iter_specs())

    assert config.gain_dynamics is False
    assert specs[0].gain_dynamics is False
    assert "gaindyn" not in specs[0].run_id


def test_parse_args_supports_gain_dynamics() -> None:
    config = parse_args(
        [
            "--gain-dynamics",
            "--outdir", "/tmp/cann_parse_args_test",
        ]
    )
    specs = list(config.iter_specs())

    assert config.gain_dynamics is True
    assert specs[0].gain_dynamics is True
    assert "gaindyn1" in specs[0].run_id


def test_parse_args_rejects_invalid_pswap() -> None:
    with pytest.raises(SystemExit):
        parse_args(
            [
                "--pswap", "1.1",
                "--outdir", "/tmp/cann_parse_args_test",
            ]
        )


def test_run_seed_changes_when_strategy_params_change() -> None:
    base_args = [
        "--seed", "42",
        "--outdir", "/tmp/cann_parse_args_test",
    ]
    default_spec = list(parse_args(base_args).iter_specs())[0]
    custom_spec = list(parse_args(base_args + ["--pswap", "0.25", "--db-fr", "0.004"]).iter_specs())[0]

    assert default_spec.seed != custom_spec.seed


def test_run_seed_changes_when_gain_dynamics_enabled() -> None:
    base_args = [
        "--seed", "42",
        "--outdir", "/tmp/cann_parse_args_test",
    ]
    default_spec = list(parse_args(base_args).iter_specs())[0]
    gain_dynamic_spec = list(parse_args(base_args + ["--gain-dynamics"]).iter_specs())[0]

    assert default_spec.seed != gain_dynamic_spec.seed


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
        p_swap=0.25,
        db_fr=0.004,
        gain_dynamics=True,
        save_energy_metrics=True,
        seed=123,
    )

    run_outdir = run_experiment(spec)
    metadata = json.loads((run_outdir / "run_metadata.json").read_text(encoding="utf-8"))
    network_params = metadata["network_params"]
    strategy_params = metadata["strategy_params"]
    gain_dynamics_params = metadata["gain_dynamics_params"]

    assert network_params["energy_metrics_enabled"] is True
    assert network_params["gain_dynamics"] is True
    assert "energy_metrics_path" in network_params
    assert network_params["energy_metrics_dtype"] == "float32"
    assert network_params["energy_metrics_columns"] == [
        "proposal_count",
        "accepted_count",
        "sum_abs_total_drive",
        "mean_abs_total_drive",
    ]
    assert strategy_params == {
        "strategy": "MetropolisUpdateStrategyMixedPadamsey",
        "p_swap": 0.25,
        "db_food_restricted": 0.004,
    }
    assert gain_dynamics_params["enabled"] is True
    assert gain_dynamics_params["control_start_multiplier"] == pytest.approx(1.5)
    assert gain_dynamics_params["control_end_multiplier"] == pytest.approx(1.0)
    assert gain_dynamics_params["food_restricted_start_control_ratio"] == pytest.approx(1.25)
    assert gain_dynamics_params["food_restricted_end_control_ratio"] == pytest.approx(1.10)
    assert gain_dynamics_params["food_restricted_decay_fraction"] == pytest.approx(1.0 / 3.0)
