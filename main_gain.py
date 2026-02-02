#!/usr/bin/env python3
import argparse
import itertools
import json
import logging
import math
import os
from concurrent.futures import ProcessPoolExecutor, as_completed
from dataclasses import dataclass
from datetime import datetime, timezone
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
    sigma_theta_vals: Sequence[float]
    block_size: int
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
            self.sigma_theta_vals,
        )
        for ampar, rin, idir, sigma_temp, sigma_eta, sigma_theta in combos:
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
                    sigma_theta=sigma_theta,
                    block_size=self.block_size,
                )


def format_float(value: float) -> str:
    text = f"{value:.6g}"
    return text.replace("+", "")


@dataclass(frozen=True)
class RunSpec:
    ampar: float
    rin: float
    idir: float
    sigma_temp: float
    sigma_eta: float
    block_size: int
    sigma_theta: float
    trial: int
    num_neurons: int
    num_generations: int
    outdir: Path
    tag: str
    loglevel: int

    @property
    def run_id(self) -> str:
        base = "_".join([
            f"ampar{format_float(self.ampar)}",
            f"rin{format_float(self.rin)}",
            f"sigtemp{format_float(self.sigma_temp)}",
            f"sigeta{format_float(self.sigma_eta)}",
            f"sigtheta{format_float(self.sigma_theta)}",
            f"idir{format_float(self.idir)}",
            f"trial{self.trial:02d}",
        ])
        return f"{base}_{self.tag}" if self.tag else base


def parse_args() -> ExperimentConfig:
    parser = argparse.ArgumentParser(
        description="Run CAN network with CLI-overridable parameters.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--N", "--num-neurons", dest="num_neurons", type=int, default=400,
                        help="Number of neurons")
    parser.add_argument("--gens", "--num-generations", dest="num_generations", type=int, default=30000,
                        help="Simulation length (generations)")
    parser.add_argument("--ampar", "--ampar-conductance", dest="ampar_vals",
                        type=float, nargs="+", default=[1.0],
                        help="AMPAR conductance value(s)")
    parser.add_argument("--rin", "--input-resistance", dest="rin_vals",
                        type=float, nargs="+", default=[1.0],
                        help="Input resistance value(s)")
    parser.add_argument("--idir", "--input-direction", dest="input_direction", type=float, nargs="+", default=[0.5],
            help="Input direction (neuron index fraction)")
    parser.add_argument("--sigma-temp", dest="sigma_temp_vals", type=float, nargs="+", default=[0.01],
                        help="Sigma_temp value(s) to use for noise")
    parser.add_argument("--sigma-eta", dest="sigma_eta_vals", type=float, nargs="+", default=[0.01],
                        help="Sigma_eta value(s) to use for noise")
    parser.add_argument("--tag", type=str, default="",
                        help="Optional run tag appended to output folders/files")
    parser.add_argument("--outdir", type=Path, default=Path("/data/scastedo/runs_test_fast3"),
    # parser.add_argument("--outdir", type=Path, default=Path("runs/test_runs"),
                        help="Base output directory")
    parser.add_argument("--loglevel", default="INFO", choices=["DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"],
                        help="Logging level")
    parser.add_argument("--trials", type=int, default=20,
                        help="How many independent repeats to run per parameter combo.")
    parser.add_argument("--block-size", type=int, default=500,dest="block_size",
                        help="Block size for updates")
    parser.add_argument("--sigma-theta", type=float, default=[0.01], dest="sigma_theta_vals", nargs="+",
                        help="Sigma theta value for bump shift noise")

    args = parser.parse_args()
    return ExperimentConfig(
        num_neurons=args.num_neurons,
        num_generations=args.num_generations,
        ampar_vals=tuple(args.ampar_vals),
        rin_vals=tuple(args.rin_vals),
        input_direction=tuple(args.input_direction),
        sigma_temp_vals=tuple(args.sigma_temp_vals),
        sigma_eta_vals=tuple(args.sigma_eta_vals),
        sigma_theta_vals=tuple(args.sigma_theta_vals),
        block_size = args.block_size,
        trials=args.trials,
        outdir=args.outdir,
        tag=args.tag,
        loglevel=args.loglevel,
    )


def make_network_params(spec: RunSpec) -> dict:
    """Build the simulator kwargs for a single run."""
    threshold = 0.2

    return {
        "num_neurons": spec.num_neurons,
        "sigma_temp": spec.sigma_temp,                      #PARAM VARY NEEDS SCALING 
        "sigma_input": threshold/2,            #between 0 and threshold
        "I_str": 0.03,                                               # WHAT TO FIX PINN AS (FUNCTION OF NOISE LEVEL?) 
        "I_dir": spec.idir,                    #PARAM NO SCALE 
        "syn_fail": 0.0,                       #DONT TOUCH
        "spon_rel": 0.0,                       #DONT TOUCH
        "sigma_eta": spec.sigma_eta,                       #PARAM VARY NEEDS SCALING
        "input_resistance": spec.rin,                      #PARAM VARY NEEDS SCALING
        "ampar_conductance": spec.ampar,                   #PARAM VARY NEEDS SCALING
        "constrict": 1.0,                      #DONT TOUCH
        "threshold_active_fraction": threshold,
        "block_size": spec.block_size,                        #DONT TOUCH
        "sigma_theta": spec.sigma_theta,   #PARAM VARY NEEDS SCALING
    }


def save_run_metadata(spec: RunSpec, network_params: dict, run_outdir: Path) -> Path:
    payload = {
        "run_id": spec.run_id,
        "run_dir": str(run_outdir),
        "created_at": datetime.now(timezone.utc).isoformat(),
        "spec": {
            "ampar": spec.ampar,
            "rin": spec.rin,
            "idir": spec.idir,
            "sigma_temp": spec.sigma_temp,
            "sigma_eta": spec.sigma_eta,
            "sigma_theta": spec.sigma_theta,
            "block_size": spec.block_size,
            "trial": spec.trial,
            "num_neurons": spec.num_neurons,
            "num_generations": spec.num_generations,
            "outdir": str(spec.outdir),
            "tag": spec.tag,
            "loglevel": spec.loglevel,
        },
        "network_params": network_params,
    }
    metadata_path = run_outdir / "run_metadata.json"
    with metadata_path.open("w", encoding="utf-8") as handle:
        json.dump(payload, handle, indent=2, sort_keys=True)
    return metadata_path


def progress_logger(spec: RunSpec):
    def _log(gen: int) -> None:
        pct = 100.0 * gen / spec.num_generations
        logging.info("[%s] %.1f%% complete (%d/%d generations)",
                     spec.run_id, pct, gen, spec.num_generations)
    return _log


def run_experiment(spec: RunSpec) -> Path:
    logging.info("Starting run %s", spec.run_id)
    update_strategy = update_strategies.MetropolisUpdateStrategy2()
    network_params = make_network_params(spec)
    network = simulator.simulate(
        network_params,
        spec.num_generations,
        update_strategy,
        progress_callback=progress_logger(spec),
    )


    setattr(network, "run_id", spec.run_id)
    network.noise_covariance()

    run_outdir = spec.outdir / spec.run_id
    run_outdir.mkdir(parents=True, exist_ok=True)
    save_run_metadata(spec, network_params, run_outdir)
    visualisation.save_state_history(network, run_outdir)

    try:
        visualisation.create_visualization_report(network, output_dir=run_outdir)
    except TypeError:
        visualisation.create_visualization_report(network)

    logging.info("Finished run %s", spec.run_id)
    return run_outdir


def determine_worker_count() -> int:
    cpu_count = os.cpu_count() or 1
    workers = max(1, math.floor(cpu_count ))
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
