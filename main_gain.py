#!/usr/bin/env python3
import logging
import argparse
import itertools
import os
from pathlib import Path
import random
from src import simulator, visualisation, update_strategies


def parse_args():
    p = argparse.ArgumentParser(
        description="Run CAN network with CLI-overridable parameters.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    p.add_argument("--N", "--num-neurons", dest="num_neurons", type=int, default=150,
                   help="Number of neurons")
    p.add_argument("--gens", "--num-generations", dest="num_generations", type=int, default=5000,
                   help="Simulation length (generations)")
    # Allow one or many values; we’ll run all combinations.
    p.add_argument("--ampar", "--ampar-conductance", dest="ampar_vals",
                   type=float, nargs="+", default=[1],
                   help="AMPAR conductance value(s)")
    p.add_argument("--rin", "--input-resistance", dest="rin_vals",
                   type=float, nargs="+", default=[1],
                   help="Input resistance value(s)")
    p.add_argument("--idir", "--input-direction", dest="input_direction", type=float, nargs="+", default=[0.5],
                   help="Input direction (neuron index fraction)")
    p.add_argument("--tag", type=str, default="",
                   help="Optional run tag appended to output folders/files")
    p.add_argument("--outdir", type=Path, default=Path("runs_novaryx_13_nov"),
                   help="Base output directory (if your visualisation supports it)")
    p.add_argument("--loglevel", default="INFO", choices=["DEBUG","INFO","WARNING","ERROR","CRITICAL"],
                   help="Logging level")
    p.add_argument("--trials", type=int, default=1,
                   help="How many independent repeats to run per parameter combo.")
  
    return p.parse_args()


def make_network_params(num_neurons, rin, ampar, idir):
    return {
        "num_neurons": num_neurons,
        "sigma_temp": 0.000,

        "sigma_input": 0.05,   # width of x input gaussian bump. Indep of N
        "I_str": 0.1,          # strength of input
        "I_dir": idir,          # neuron index where you want input
        "tau_ou": 500.0,
        "sigma_ou": 0.0,

        "syn_fail": 0.000,     # synaptic failure
        "spon_rel": 0.0,       # spontaneous release rate
        "sigma_eta": 0.1,      # noise (temperature)

        "input_resistance": rin,
        "ampar_conductance": ampar,

        "constrict": 1.0,      # constriction factor (inhibition-ish)
        "threshold_active_fraction": 0.1
    }

def run_condition(spec: dict) -> str:
    logging.basicConfig(level=spec["loglevel"],
                        format="%(asctime)s - %(levelname)s - %(message)s")

    update_strategy = update_strategies.DynamicsUpdateStrategyGain()

    network_params = make_network_params(
        spec["num_neurons"], spec["rin"], spec["ampar"], spec["idir"]
    )

    total_gens = spec["num_generations"]
    run_id = f"g{spec['ampar']:.3f}_rin{spec['rin']:.3f}_idir{spec['idir']:.3f}"

    run_id = (
            f"g{spec['ampar']:.3f}_rin{spec['rin']:.3f}_idir{spec['idir']:.3f}"
            f"_trial{spec['trial']:02d}"
        )
    if spec["tag"]:
        run_id += f"_{spec['tag']}"


    def log_progress(gen: int) -> None:
        pct = 100.0 * gen / total_gens
        logging.info("[%s] %.1f%% complete (%d/%d generations)",
                        run_id, pct, gen, total_gens)
    network = simulator.simulate(network_params, spec["num_generations"], update_strategy, progress_callback=log_progress)

    setattr(network, "run_id", run_id)

    network.noise_covariance()
    outdir = Path(spec["outdir"]) / run_id
    visualisation.save_state_history(network, outdir)

    try:
        visualisation.create_visualization_report(network, output_dir=outdir)
    except TypeError:
        visualisation.create_visualization_report(network)

    return run_id
def main():
    args = parse_args()
    logging.basicConfig(level=getattr(logging, args.loglevel),
                        format="%(asctime)s - %(levelname)s - %(message)s")
    args.outdir.mkdir(parents=True, exist_ok=True)

    idir_vals = (
        args.input_direction
        if isinstance(args.input_direction, (list, tuple))
        else [args.input_direction]
    )

    loglevel_numeric = getattr(logging, args.loglevel)
    specs = []
    for ampar, rin, idir in itertools.product(args.ampar_vals, args.rin_vals, idir_vals):
        for trial in range(args.trials):
            specs.append({
                "ampar": ampar,
                "rin": rin,
                "idir": idir,
                "trial": trial,
                "num_neurons": args.num_neurons,
                "num_generations": args.num_generations,
                "tag": args.tag,
                "outdir": str(args.outdir),
                "loglevel": loglevel_numeric,
            })

    for spec in specs:
        run_condition(spec)



if __name__ == "__main__":
    main()
