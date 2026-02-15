#!/usr/bin/env python3
import argparse
import cProfile
import json
import pstats
import sys
import time
from pathlib import Path

import torch

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from src import simulator, update_strategies


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Profile CAN simulation runtime for a given update strategy.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--neurons", type=int, default=250, help="Number of neurons")
    parser.add_argument("--gens", type=int, default=1000, help="Simulation generations")
    parser.add_argument("--seed", type=int, default=0, help="RNG seed")
    parser.add_argument(
        "--strategy",
        type=str,
        default="MetropolisUpdateStrategy3",
        help="Update strategy class from src.update_strategies",
    )
    parser.add_argument(
        "--profile-out",
        type=Path,
        default=Path("/tmp/cann_profile_current.out"),
        help="cProfile output path",
    )
    parser.add_argument(
        "--summary-out",
        type=Path,
        default=Path("/tmp/cann_profile_summary.json"),
        help="JSON summary output path",
    )
    parser.add_argument(
        "--state-out",
        type=Path,
        default=Path("/tmp/cann_profile_state.npy"),
        help="State history output path",
    )
    parser.add_argument("--top", type=int, default=15, help="How many top cProfile rows to print")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    torch.manual_seed(args.seed)

    strategy_cls = getattr(update_strategies, args.strategy, None)
    if strategy_cls is None:
        raise ValueError(f"Unknown strategy class: {args.strategy}")
    strategy = strategy_cls()

    network_params = {
        "num_neurons": args.neurons,
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
        "threshold_active_fraction": 0.1,
        "block_size": args.neurons,
        "sigma_theta": 0.0,
    }

    args.profile_out.parent.mkdir(parents=True, exist_ok=True)
    args.summary_out.parent.mkdir(parents=True, exist_ok=True)
    args.state_out.parent.mkdir(parents=True, exist_ok=True)

    pr = cProfile.Profile()
    t0 = time.perf_counter()
    pr.enable()
    network = simulator.simulate(
        network_params,
        num_generations=args.gens,
        update_strategy=strategy,
        progress_callback=None,
        state_history_path=args.state_out,
        state_write_chunk=256,
        keep_state_history_in_memory=False,
    )
    pr.disable()
    elapsed = time.perf_counter() - t0
    pr.dump_stats(str(args.profile_out))

    summary = {
        "strategy": args.strategy,
        "elapsed_sec": elapsed,
        "num_generations": args.gens,
        "num_neurons": args.neurons,
        "profile_path": str(args.profile_out),
        "state_path": str(args.state_out),
        "sample_calls": int(getattr(network, "sample_calls", 0)),
        "accept_calls": int(getattr(network, "accept_calls", 0)),
        "pool_swap_calls": int(getattr(network, "pool_swap_calls", 0)),
    }
    args.summary_out.write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(json.dumps(summary, indent=2))

    stats = pstats.Stats(str(args.profile_out))
    print("\nTOP_BY_CUMULATIVE")
    stats.sort_stats("cumulative").print_stats(args.top)
    print("\nTOP_BY_TOTTIME")
    stats.sort_stats("tottime").print_stats(args.top)


if __name__ == "__main__":
    main()
