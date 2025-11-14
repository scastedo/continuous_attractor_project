#!/usr/bin/env python3
import argparse
import itertools
import logging
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, Sequence

from src import simulator, update_strategies, visualisation


@dataclass(frozen=True)
class ExperimentConfig:
    num_neurons: int
    num_generations: int
    ampar_vals: Sequence[float]
    rin_vals: Sequence[float]
    input_direction: Sequence[float]
    trials: int
    outdir: Path
    tag: str
    loglevel: str

    @property
    def loglevel_numeric(self) -> int:
        return getattr(logging, self.loglevel.upper(), logging.INFO)

    def iter_specs(self) -> Iterable["RunSpec"]:
        combos = itertools.product(self.ampar_vals, self.rin_vals, self.input_direction)
        for ampar, rin, idir in combos:
            for trial in range(self.trials):
                yield RunSpec(
                    ampar=ampar,
                    rin=rin,
                    idir=idir,
                    trial=trial,
                    num_neurons=self.num_neurons,
                    num_generations=self.num_generations,
                    outdir=self.outdir,
                    tag=self.tag,
                    loglevel=self.loglevel_numeric,
                )


@dataclass(frozen=True)
class RunSpec:
    ampar: float
    rin: float
    idir: float
    trial: int
    num_neurons: int
    num_generations: int
    outdir: Path
    tag: str
    loglevel: int

    @property
    def run_id(self) -> str:
        base = f"g{self.ampar:.3f}_rin{self.rin:.3f}_idir{self.idir:.3f}_trial{self.trial:02d}"
        return f"{base}_{self.tag}" if self.tag else base


def parse_args() -> ExperimentConfig:
    parser = argparse.ArgumentParser(
        description="Run CAN network with CLI-overridable parameters.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--N", "--num-neurons", dest="num_neurons", type=int, default=150,
                        help="Number of neurons")
    parser.add_argument("--gens", "--num-generations", dest="num_generations", type=int, default=5000,
                        help="Simulation length (generations)")
    parser.add_argument("--ampar", "--ampar-conductance", dest="ampar_vals",
                        type=float, nargs="+", default=[1.0],
                        help="AMPAR conductance value(s)")
    parser.add_argument("--rin", "--input-resistance", dest="rin_vals",
                        type=float, nargs="+", default=[1.0],
                        help="Input resistance value(s)")
    parser.add_argument("--idir", "--input-direction", dest="input_direction", type=float, nargs="+", default=[0.5],
                        help="Input direction (neuron index fraction)")
    parser.add_argument("--tag", type=str, default="",
                        help="Optional run tag appended to output folders/files")
    parser.add_argument("--outdir", type=Path, default=Path("runs"),
                        help="Base output directory")
    parser.add_argument("--loglevel", default="INFO", choices=["DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"],
                        help="Logging level")
    parser.add_argument("--trials", type=int, default=1,
                        help="How many independent repeats to run per parameter combo.")

    args = parser.parse_args()
    return ExperimentConfig(
        num_neurons=args.num_neurons,
        num_generations=args.num_generations,
        ampar_vals=tuple(args.ampar_vals),
        rin_vals=tuple(args.rin_vals),
        input_direction=tuple(args.input_direction),
        trials=args.trials,
        outdir=args.outdir,
        tag=args.tag,
        loglevel=args.loglevel,
    )


def make_network_params(spec: RunSpec) -> dict:
    """Build the simulator kwargs for a single run."""
    return {
        "num_neurons": spec.num_neurons,
        "sigma_temp": 0.0,
        "sigma_input": 0.05,      # width of input Gaussian bump (fraction of N)
        "I_str": 0.1,             # stimulus strength
        "I_dir": spec.idir,       # neuron index fraction for the input peak
        "tau_ou": 500.0,
        "sigma_ou": 0.0,
        "syn_fail": 0.0,
        "spon_rel": 0.0,
        "sigma_eta": 0.1,
        "input_resistance": spec.rin,
        "ampar_conductance": spec.ampar,
        "constrict": 1.0,
        "threshold_active_fraction": 0.1,
    }


def progress_logger(spec: RunSpec):
    def _log(gen: int) -> None:
        pct = 100.0 * gen / spec.num_generations
        logging.info("[%s] %.1f%% complete (%d/%d generations)",
                     spec.run_id, pct, gen, spec.num_generations)
    return _log


def run_experiment(spec: RunSpec) -> Path:
    logging.info("Starting run %s", spec.run_id)
    update_strategy = update_strategies.DynamicsUpdateStrategyGain()
    network = simulator.simulate(
        make_network_params(spec),
        spec.num_generations,
        update_strategy,
        progress_callback=progress_logger(spec),
    )

    setattr(network, "run_id", spec.run_id)
    network.noise_covariance()

    run_outdir = spec.outdir / spec.run_id
    run_outdir.mkdir(parents=True, exist_ok=True)
    visualisation.save_state_history(network, run_outdir)

    try:
        visualisation.create_visualization_report(network, output_dir=run_outdir)
    except TypeError:
        visualisation.create_visualization_report(network)

    logging.info("Finished run %s", spec.run_id)
    return run_outdir


def main() -> None:
    config = parse_args()
    logging.basicConfig(level=config.loglevel_numeric,
                        format="%(asctime)s - %(levelname)s - %(message)s")
    config.outdir.mkdir(parents=True, exist_ok=True)

    for spec in config.iter_specs():
        run_experiment(spec)


if __name__ == "__main__":
    main()
