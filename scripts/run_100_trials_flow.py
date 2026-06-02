"""
run_100_trials_flow.py
======================
Run 100 3D MPC trials with background flow enabled.
Saves plots and per-trial data (trajectory, logs) for each run.
"""

import os
import sys
# Add parent directory to path to allow importing dynamag_simulation when run directly
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

import json
import time

import numpy as np

from dynamag_simulation.core.constants import HALF, TEST_RANGE
from dynamag_simulation.core.profiles import MICROSPHERE_500UM
from dynamag_simulation.core.workspace import Workspace3D
from dynamag_simulation.core.magnets import ElectromagnetArray3D
from dynamag_simulation.core.flow import UniformFlow, PoiseuilleFlow
from dynamag_simulation.core.reporting import TrialResult, print_summary, save_results_csv
from dynamag_simulation.core.plotting import save_all_plots

from simulation_3d_mpc import simulate_trial_mpc


# Safe inner half-extent so start/target stay > TEST_RANGE from walls (and coil radius)
INNER = max(0.01, HALF - TEST_RANGE)


def _random_position(rng):
    """One random 3D position in [-INNER, INNER]."""
    return tuple(float(rng.uniform(-INNER, INNER)) for _ in range(3))


def save_trial_data(result: TrialResult, output_dir: str) -> None:
    """Save per-trial arrays (trajectory, logs) to a .npz file."""
    base = result.trial_name
    path = os.path.join(output_dir, f"{base}_data.npz")
    np.savez(
        path,
        start=result.start,
        target=result.target,
        final_position=result.final_position,
        trajectory=result.trajectory,
        distance_log=result.distance_log,
        force_log=result.force_log,
        velocity_log=np.array(result.velocity_log),
        field_log=result.field_log,
        reynolds_log=result.reynolds_log,
        current_log=result.current_log,
        cost_log=result.cost_log,
        step_time_log=result.step_time_log,
        reached=result.reached,
        steps_taken=result.steps_taken,
        final_error=result.final_error,
        dt=result.dt,
        workspace_half=result.workspace_half,
    )


def run_100_trials_with_flow(
    output_dir: str = "outputs/flow_100_trials",
    flow_type: str = "uniform",
    n_trials: int = 100,
    seed: int = 42,
    steps: int = 3000,
    dt: float = 0.02,
    tolerance: float = 0.003,
    verbose: bool = True,
):
    """
    Run n_trials (default 100) 3D MPC trials with flow enabled.
    Saves for each trial: all plots and a _data.npz; at end a batch CSV + summary JSON.
    """
    os.makedirs(output_dir, exist_ok=True)
    rng = np.random.default_rng(seed)

    # Flow: uniform along +x (typical vessel phantom) or Poiseuille along x through center
    if flow_type == "uniform":
        # ~1–2 mm/s so flow matters but isn't dominant
        flow = UniformFlow(velocity=[0.002, 0.0, 0.0])
    elif flow_type == "poiseuille":
        flow = PoiseuilleFlow(
            center=(0.0, 0.0, 0.0),
            axis=[1.0, 0.0, 0.0],
            radius=0.04,
            v_max=0.004,
        )
    else:
        raise ValueError(f"Unknown flow_type: {flow_type}")

    # Pre-generate start/target pairs
    starts = [_random_position(rng) for _ in range(n_trials)]
    targets = [_random_position(rng) for _ in range(n_trials)]

    magnets_for_plots = ElectromagnetArray3D(
        coil_radius=0.03, coil_length=0.12, turns=1500, max_current=8.0
    )

    results = []
    batch_start = time.perf_counter()

    for i in range(n_trials):
        trial_name = f"trial_{i + 1:04d}"
        start = starts[i]
        target = targets[i]
        if verbose:
            print(f"\n>>> [{i + 1}/{n_trials}] {trial_name}  start={start}  target={target}")

        result = simulate_trial_mpc(
            start=start,
            target=target,
            trial_name=trial_name,
            dt=dt,
            steps=steps,
            tolerance=tolerance,
            profile=MICROSPHERE_500UM,
            rng_seed=seed + i,
            flow=flow,
        )
        result.params.update(flow.params_dict())
        results.append(result)

        save_all_plots(result, output_dir, magnet_array=magnets_for_plots)
        save_trial_data(result, output_dir)

        if verbose and (i + 1) % 10 == 0:
            print_summary(result)

    batch_elapsed = time.perf_counter() - batch_start

    save_results_csv(results, output_dir, filename="trial_summary.csv")

    summary = {
        "n_trials": n_trials,
        "flow_type": flow_type,
        "flow_params": flow.params_dict(),
        "seed": seed,
        "dt": dt,
        "steps": steps,
        "tolerance": tolerance,
        "batch_wall_time_s": round(batch_elapsed, 2),
        "reached_count": sum(1 for r in results if r.reached),
        "trial_names": [r.trial_name for r in results],
    }
    summary_path = os.path.join(output_dir, "batch_summary.json")
    with open(summary_path, "w") as f:
        json.dump(summary, f, indent=2)

    if verbose:
        print(f"\n{'=' * 60}")
        print(f"  Batch complete: {n_trials} trials with flow={flow_type}")
        print(f"  Reached target: {summary['reached_count']}/{n_trials}")
        print(f"  Wall time: {batch_elapsed:.1f} s")
        print(f"  Outputs: {output_dir}")
        print(f"  - trial_summary.csv, batch_summary.json")
        print(f"  - per trial: *_trajectory_3d.png, *_distance.png, *_physics.png,")
        print(f"    *_currents.png, *_fieldmap_z0.png, *_data.npz")
        print(f"{'=' * 60}")

    return results


if __name__ == "__main__":
    run_100_trials_with_flow(
        output_dir="outputs/flow_100_trials",
        flow_type="uniform",
        n_trials=100,
        seed=42,
        steps=3000,
        dt=0.02,
        tolerance=0.003,
        verbose=True,
    )
