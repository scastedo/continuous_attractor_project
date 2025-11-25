#!/usr/bin/env python3
import argparse
import itertools
import logging
import math
import os
from concurrent.futures import ProcessPoolExecutor, as_completed
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
    sigma_temp_vals: Sequence[float]
    sigma_eta_vals: Sequence[float]
    trials: int
    outdir: Path
    tag: str
    loglevel: str

    @property
    def loglevel_numeric(self) -> int:
        return getattr(logging, self.loglevel.upper(), logging.INFO)

    def iter_specs(self) -> Iterable["RunSpec"]:
        combos = itertools.product(
            self.ampar_vals,
            self.rin_vals,
            self.input_direction,
            self.sigma_temp_vals,
            self.sigma_eta_vals,
        )
        for ampar, rin, idir, sigma_temp, sigma_eta in combos:
            for trial in range(self.trials):
                yield RunSpec(
                    ampar=ampar,
                    rin=rin,
                    idir=idir,
                    sigma_temp=sigma_temp,
                    sigma_eta=sigma_eta,
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
    sigma_temp: float
    sigma_eta: float
    trial: int
    num_neurons: int
    num_generations: int
    outdir: Path
    tag: str
    loglevel: int

    @property
    def run_id(self) -> str:
        base = f"g{self.ampar:.3f}_rin{self.rin:.3f}_sigmatemp{self.sigma_temp:.3f}_sigmaeta{self.sigma_eta:.3f}_idir{self.idir:.3f}_trial{self.trial:02d}"
        return f"{base}_{self.tag}" if self.tag else base


def parse_args() -> ExperimentConfig:
    parser = argparse.ArgumentParser(
        description="Run CAN network with CLI-overridable parameters.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--N", "--num-neurons", dest="num_neurons", type=int, default=200,
                        help="Number of neurons")
    parser.add_argument("--gens", "--num-generations", dest="num_generations", type=int, default=20000,
                        help="Simulation length (generations)")
    parser.add_argument("--ampar", "--ampar-conductance", dest="ampar_vals",
                        type=float, nargs="+", default=[1.0],
                        help="AMPAR conductance value(s)")
    parser.add_argument("--rin", "--input-resistance", dest="rin_vals",
                        type=float, nargs="+", default=[1.0],
                        help="Input resistance value(s)")
    parser.add_argument("--idir", "--input-direction", dest="input_direction", type=float, nargs="+", default=[0.5],
            help="Input direction (neuron index fraction)")
    parser.add_argument("--sigma-temp", dest="sigma_temp_vals", type=float, nargs="+", default=[0.0],
                        help="Sigma_temp value(s) to use for noise")
    parser.add_argument("--sigma-eta", dest="sigma_eta_vals", type=float, nargs="+", default=[0.0],
                        help="Sigma_eta value(s) to use for noise")
    parser.add_argument("--tag", type=str, default="",
                        help="Optional run tag appended to output folders/files")
    parser.add_argument("--outdir", type=Path, default=Path("runs_test_longer"),
                        help="Base output directory")
    parser.add_argument("--loglevel", default="INFO", choices=["DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"],
                        help="Logging level")
    parser.add_argument("--trials", type=int, default=20,
                        help="How many independent repeats to run per parameter combo.")

    args = parser.parse_args()
    return ExperimentConfig(
        num_neurons=args.num_neurons,
        num_generations=args.num_generations,
        ampar_vals=tuple(args.ampar_vals),
        rin_vals=tuple(args.rin_vals),
        input_direction=tuple(args.input_direction),
        sigma_temp_vals=tuple(args.sigma_temp_vals),
        sigma_eta_vals=tuple(args.sigma_eta_vals),
        trials=args.trials,
        outdir=args.outdir,
        tag=args.tag,
        loglevel=args.loglevel,
    )


def make_network_params(spec: RunSpec) -> dict:
    """Build the simulator kwargs for a single run."""
    threshold = 0.1

    return {
        "num_neurons": spec.num_neurons,
        "sigma_temp": spec.sigma_temp,                      #PARAM VARY NEEDS SCALING 
        "sigma_input": threshold/2,            #between 0 and threshold
        "I_str": 0.1,                                                                 # WHAT TO FIX PINN AS (FUNCTION OF NOISE LEVEL?) 
        "I_dir": spec.idir,                    #PARAM NO SCALE 
        "syn_fail": 0.0,                       #DONT TOUCH
        "spon_rel": 0.0,                       #DONT TOUCH
        "sigma_eta": spec.sigma_eta,                       #PARAM VARY NEEDS SCALING
        "input_resistance": spec.rin,                      #PARAM VARY NEEDS SCALING
        "ampar_conductance": spec.ampar,                   #PARAM VARY NEEDS SCALING
        "constrict": 1.0,                      #DONT TOUCH
        "threshold_active_fraction": threshold,
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


def determine_worker_count() -> int:
    cpu_count = os.cpu_count() or 1
    workers = max(1, math.floor(cpu_count / 1.1))
    return workers


def main() -> None:
    config = parse_args()
    logging.basicConfig(level=config.loglevel_numeric,
                        format="%(asctime)s - %(levelname)s - %(message)s")
    config.outdir.mkdir(parents=True, exist_ok=True)

    specs = list(config.iter_specs())
    if not specs:
        logging.warning("No parameter combinations to run.")
        return

    workers = determine_worker_count()
    logging.info("Running %d simulations across %d worker(s)", len(specs), workers)

    if workers == 1:
        for spec in specs:
            run_experiment(spec)
        return

    with ProcessPoolExecutor(max_workers=workers) as executor:
        future_to_spec = {executor.submit(run_experiment, spec): spec for spec in specs}
        for future in as_completed(future_to_spec):
            spec = future_to_spec[future]
            try:
                future.result()
            except Exception:
                logging.exception("Run %s failed", spec.run_id)


if __name__ == "__main__":
    main()
