"""
run_batch.py -- Overnight batch runner for DynaMag data collection.

Runs N trials with randomized start/target positions across the valid
workspace region, collects all results into a single CSV and generates
per-trial plots.

Usage:
    python run_batch.py                  # 50 trials, default settings
    python run_batch.py --trials 200     # 200 trials
    python run_batch.py --trials 100 --steps 2000 --no-plots
"""

import os
import sys
# Add parent directory to path to allow importing dynamag_simulation when run directly
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

import time
import argparse
import json

import numpy as np

from dynamag_simulation.core.constants import DEFAULT_TARGET_TOLERANCE, HALF
from dynamag_simulation.core.profiles import MICROSPHERE_500UM
from dynamag_simulation.core.magnets import ElectromagnetArray3D
from dynamag_simulation.core.reporting import print_summary, save_results_csv
from dynamag_simulation.core.plotting import save_all_plots
from simulation_3d_mpc import simulate_trial_mpc


DEFAULT_EDGE = 0.04
MIN_SEPARATION = 0.02  # reject start/target pairs closer than 20 mm


def random_point(rng, edge):
    """Uniform random point in [-edge, edge]^3."""
    return rng.uniform(-edge, edge, size=3)


def make_trials(n_trials, edge, base_seed):
    """Generate n_trials random start/target pairs with guaranteed minimum separation."""
    rng = np.random.default_rng(base_seed)
    trials = []
    for i in range(n_trials):
        while True:
            start = random_point(rng, edge)
            target = random_point(rng, edge)
            if np.linalg.norm(target - start) >= MIN_SEPARATION:
                break
        seed = int(rng.integers(0, 2**31))
        trials.append({
            "name": f"trial_{i+1:04d}",
            "start": tuple(start),
            "target": tuple(target),
            "seed": seed,
        })
    return trials


def run_batch(args):
    batch_start = time.time()
    output_dir = args.output_dir
    os.makedirs(output_dir, exist_ok=True)

    trials = make_trials(args.trials, args.edge, args.base_seed)

    # Save trial plan so the batch is fully reproducible
    plan_path = os.path.join(output_dir, "trial_plan.json")
    with open(plan_path, "w") as f:
        json.dump({
            "n_trials": args.trials,
            "base_seed": args.base_seed,
            "edge": args.edge,
            "dt": args.dt,
            "max_steps": args.steps,
            "tolerance": args.tolerance,
            "profile": MICROSPHERE_500UM.name,
            "trials": trials,
        }, f, indent=2)
    print(f"Trial plan saved to {plan_path}")

    mag = ElectromagnetArray3D() if not args.no_plots else None
    results = []
    completed = 0
    failed = 0

    for i, trial in enumerate(trials):
        t0 = time.time()
        pct = (i / args.trials) * 100
        dist_mm = np.linalg.norm(np.array(trial["target"]) - np.array(trial["start"])) * 1000

        print(f"\n[{pct:5.1f}%] {trial['name']}  "
              f"dist={dist_mm:.1f}mm  seed={trial['seed']}")

        try:
            result = simulate_trial_mpc(
                start=trial["start"],
                target=trial["target"],
                trial_name=trial["name"],
                steps=args.steps,
                dt=args.dt,
                tolerance=args.tolerance,
                rng_seed=trial["seed"],
            )
            results.append(result)
            completed += 1

            status = "REACHED" if result.reached else "TIMEOUT"
            elapsed = time.time() - t0
            print(f"        {status}  "
                  f"steps={result.steps_taken}  "
                  f"err={result.final_error*1000:.2f}mm  "
                  f"max_Re={result.max_reynolds:.1f}  "
                  f"low_Re={result.low_re_valid}  "
                  f"wall={elapsed:.1f}s")

            if not args.no_plots:
                save_all_plots(result, output_dir, magnet_array=mag)

            # Write CSV incrementally so partial results are saved if interrupted
            if (completed % 10 == 0) or (i == len(trials) - 1):
                save_results_csv(results, output_dir)

        except Exception as e:
            failed += 1
            print(f"        FAILED: {e}")

    # Final CSV write
    save_results_csv(results, output_dir)

    batch_elapsed = time.time() - batch_start

    # Summary statistics
    if results:
        reached = [r for r in results if r.reached]
        re_valid = [r for r in results if r.low_re_valid]
        all_errors = np.array([r.final_error * 1000 for r in results])
        all_steps = np.array([r.steps_taken for r in results])
        all_wall = np.array([r.wall_time_s for r in results])
        all_max_re = np.array([r.max_reynolds for r in results])
        all_peak_v = np.array([np.max(r.velocity_log) * 1000 for r in results])

        summary = {
            "total_trials": len(results),
            "reached_target": len(reached),
            "reach_rate_pct": len(reached) / len(results) * 100,
            "low_re_valid_count": len(re_valid),
            "final_error_mm_mean": float(np.mean(all_errors)),
            "final_error_mm_std": float(np.std(all_errors)),
            "final_error_mm_median": float(np.median(all_errors)),
            "steps_mean": float(np.mean(all_steps)),
            "steps_std": float(np.std(all_steps)),
            "wall_time_s_mean": float(np.mean(all_wall)),
            "wall_time_s_total": float(np.sum(all_wall)),
            "max_reynolds_mean": float(np.mean(all_max_re)),
            "max_reynolds_max": float(np.max(all_max_re)),
            "peak_velocity_mm_s_mean": float(np.mean(all_peak_v)),
            "peak_velocity_mm_s_max": float(np.max(all_peak_v)),
            "batch_wall_time_s": batch_elapsed,
            "failed_trials": failed,
        }

        summary_path = os.path.join(output_dir, "batch_summary.json")
        with open(summary_path, "w") as f:
            json.dump(summary, f, indent=2)

        print(f"\n{'=' * 60}")
        print(f"  BATCH COMPLETE")
        print(f"{'=' * 60}")
        print(f"  Trials run      : {len(results)}")
        print(f"  Reached target  : {len(reached)}/{len(results)} ({summary['reach_rate_pct']:.1f}%)")
        print(f"  Low-Re valid    : {len(re_valid)}/{len(results)}")
        print(f"  Final error     : {summary['final_error_mm_mean']:.2f} +/- {summary['final_error_mm_std']:.2f} mm (mean +/- std)")
        print(f"  Steps           : {summary['steps_mean']:.0f} +/- {summary['steps_std']:.0f}")
        print(f"  Peak velocity   : {summary['peak_velocity_mm_s_mean']:.1f} mm/s mean, {summary['peak_velocity_mm_s_max']:.1f} mm/s max")
        print(f"  Max Reynolds    : {summary['max_reynolds_mean']:.1f} mean, {summary['max_reynolds_max']:.1f} max")
        print(f"  Wall time       : {batch_elapsed:.0f}s ({batch_elapsed/60:.1f} min)")
        print(f"  Failed trials   : {failed}")
        print(f"\n  Results: {output_dir}/trial_summary.csv")
        print(f"  Summary: {summary_path}")
        print(f"  Plan:    {plan_path}")


def main():
    parser = argparse.ArgumentParser(
        description="DynaMag overnight batch runner for data collection.",
    )
    parser.add_argument("--trials", type=int, default=50,
                        help="Number of random trials (default: 50)")
    parser.add_argument("--steps", type=int, default=1500,
                        help="Max steps per trial (default: 1500)")
    parser.add_argument("--dt", type=float, default=0.02,
                        help="Timestep in seconds (default: 0.02)")
    parser.add_argument("--tolerance", type=float, default=DEFAULT_TARGET_TOLERANCE,
                        help="Target tolerance in meters (default: 0.0005)")
    parser.add_argument("--edge", type=float, default=DEFAULT_EDGE,
                        help="Start/target range in meters (default: 0.04)")
    parser.add_argument("--base-seed", type=int, default=2024,
                        help="Master seed for reproducibility (default: 2024)")
    parser.add_argument("--output-dir", type=str, default="outputs/batch_results",
                        help="Output directory (default: outputs/batch_results)")
    parser.add_argument("--no-plots", action="store_true",
                        help="Skip plot generation (faster)")

    args = parser.parse_args()
    run_batch(args)


if __name__ == "__main__":
    main()
