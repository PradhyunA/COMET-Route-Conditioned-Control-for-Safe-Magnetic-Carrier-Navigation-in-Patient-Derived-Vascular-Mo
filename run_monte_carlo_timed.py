"""
run_monte_carlo_timed.py
========================
Time-limited Monte Carlo data collection for the DynaMag 50 µm simulation.

Runs as many randomized MPC trials as possible within a wall-clock budget
(default 1.5 hours = 5400 s).  Skips per-trial plot generation to maximize
throughput (plots are generated afterwards by generate_mc_analysis.py).

Each trial is saved as an .npz file + incremental CSV so partial results
are preserved if the process is interrupted.

Usage:
    .venv/bin/python3 run_monte_carlo_timed.py --time-limit 5400
"""

import os
import sys
import json
import time
import argparse

import numpy as np

from dynamag_simulation.core.constants import HALF, TEST_RANGE
from dynamag_simulation.core.profiles import MICROSPHERE_50UM
from dynamag_simulation.core.magnets import ElectromagnetArray3D
from dynamag_simulation.core.flow import UniformFlow
from dynamag_simulation.core.reporting import save_results_csv

from simulation_3d_mpc import simulate_trial_mpc

INNER = max(0.01, HALF - TEST_RANGE)


def _random_pos(rng):
    return tuple(float(rng.uniform(-INNER, INNER)) for _ in range(3))


def compute_trial_analytics(result):
    """Compute extra per-trial metrics from the TrialResult arrays."""
    traj = result.trajectory
    target = result.target
    start = result.start
    dt = result.dt

    straight_dist = np.linalg.norm(target - start)
    segments = np.diff(traj, axis=0)
    path_length = float(np.sum(np.linalg.norm(segments, axis=1)))
    path_efficiency = straight_dist / path_length if path_length > 0 else 0.0

    # Energy proxy: sum of I^2 * dt over all steps and coils
    if len(result.current_log) > 0:
        energy_proxy = float(np.sum(result.current_log ** 2) * dt)
        peak_total_current = float(np.max(np.sum(np.abs(result.current_log), axis=1)))
        mean_total_current = float(np.mean(np.sum(np.abs(result.current_log), axis=1)))
    else:
        energy_proxy = 0.0
        peak_total_current = 0.0
        mean_total_current = 0.0

    # Convergence rate: slope of distance curve in first half
    if len(result.distance_log) > 10:
        n_half = len(result.distance_log) // 2
        t_half = np.arange(n_half) * dt
        d_half = result.distance_log[:n_half]
        if n_half > 1:
            slope = float(np.polyfit(t_half, d_half, 1)[0])
        else:
            slope = 0.0
    else:
        slope = 0.0

    # Velocity stats
    if len(result.velocity_log) > 0:
        mean_vel = float(np.mean(result.velocity_log))
        std_vel = float(np.std(result.velocity_log))
        max_vel = float(np.max(result.velocity_log))
    else:
        mean_vel = 0.0
        std_vel = 0.0
        max_vel = 0.0

    # Distance curve smoothness
    if len(result.distance_log) > 1:
        dist_diffs = np.diff(result.distance_log)
        monotonic_fraction = float(np.mean(dist_diffs < 0))
        dist_smoothness = float(np.std(dist_diffs))
    else:
        monotonic_fraction = 0.0
        dist_smoothness = 0.0

    # Force stats
    if len(result.force_log) > 0:
        mean_force = float(np.mean(result.force_log))
        max_force = float(np.max(result.force_log))
    else:
        mean_force = 0.0
        max_force = 0.0

    # Field stats
    if len(result.field_log) > 0:
        mean_field = float(np.mean(result.field_log))
        max_field = float(np.max(result.field_log))
    else:
        mean_field = 0.0
        max_field = 0.0

    # Cost stats
    if len(result.cost_log) > 0:
        mean_cost = float(np.mean(result.cost_log))
        final_cost = float(result.cost_log[-1])
        initial_cost = float(result.cost_log[0])
        cost_reduction = 1.0 - (final_cost / initial_cost) if initial_cost > 0 else 0.0
    else:
        mean_cost = 0.0
        final_cost = 0.0
        initial_cost = 0.0
        cost_reduction = 0.0

    # Current derivative stats (control smoothness)
    if len(result.current_log) > 1:
        dI = np.diff(result.current_log, axis=0) / dt
        jerk_norms = np.linalg.norm(dI, axis=1)
        mean_jerk = float(np.mean(jerk_norms))
        max_jerk = float(np.max(jerk_norms))
    else:
        mean_jerk = 0.0
        max_jerk = 0.0

    # Step time stats
    if len(result.step_time_log) > 0:
        mean_step_time = float(np.mean(result.step_time_log))
        max_step_time = float(np.max(result.step_time_log))
    else:
        mean_step_time = 0.0
        max_step_time = 0.0

    return {
        "straight_distance_m": straight_dist,
        "path_length_m": path_length,
        "path_efficiency": path_efficiency,
        "energy_proxy_A2s": energy_proxy,
        "peak_total_current_A": peak_total_current,
        "mean_total_current_A": mean_total_current,
        "convergence_slope_m_per_s": slope,
        "mean_velocity_m_s": mean_vel,
        "std_velocity_m_s": std_vel,
        "max_velocity_m_s": max_vel,
        "monotonic_fraction": monotonic_fraction,
        "distance_smoothness": dist_smoothness,
        "mean_force_N": mean_force,
        "max_force_N": max_force,
        "mean_field_T": mean_field,
        "max_field_T": max_field,
        "mean_cost": mean_cost,
        "final_cost": final_cost,
        "initial_cost": initial_cost,
        "cost_reduction": cost_reduction,
        "mean_jerk_A_per_s": mean_jerk,
        "max_jerk_A_per_s": max_jerk,
        "mean_step_time_s": mean_step_time,
        "max_step_time_s": max_step_time,
    }


def save_trial_data(result, analytics, output_dir):
    """Save all per-trial arrays plus computed analytics."""
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
        # Analytics
        straight_distance=analytics["straight_distance_m"],
        path_length=analytics["path_length_m"],
        path_efficiency=analytics["path_efficiency"],
        energy_proxy=analytics["energy_proxy_A2s"],
        convergence_slope=analytics["convergence_slope_m_per_s"],
        monotonic_fraction=analytics["monotonic_fraction"],
        mean_jerk=analytics["mean_jerk_A_per_s"],
        max_jerk=analytics["max_jerk_A_per_s"],
        mean_cost=analytics["mean_cost"],
        cost_reduction=analytics["cost_reduction"],
    )


def run_timed(
    output_dir="mc_data_run",
    time_limit_s=5400,       # 1.5 hours
    max_trials=500,          # safety cap
    seed=12345,
    dt=1.0,
    steps=3000,
    tolerance=0.003,
    flow_speed=0.0002,
    max_current=13.0,
):
    """Run as many trials as possible within the time limit."""
    output_dir = os.path.abspath(output_dir)
    os.makedirs(output_dir, exist_ok=True)
    rng = np.random.default_rng(seed)

    flow = UniformFlow(velocity=[flow_speed, 0.0, 0.0])
    profile = MICROSPHERE_50UM

    results = []
    all_analytics = []
    batch_start = time.perf_counter()

    # Pre-generate a pool of start/target positions
    starts = [_random_pos(rng) for _ in range(max_trials)]
    targets = [_random_pos(rng) for _ in range(max_trials)]

    completed = 0
    skipped_time = 0

    # Save run config
    config = {
        "time_limit_s": time_limit_s,
        "max_trials": max_trials,
        "seed": seed,
        "dt": dt,
        "steps_budget": steps,
        "tolerance_m": tolerance,
        "flow_speed_m_s": flow_speed,
        "max_current_A": max_current,
        "particle_profile": profile.name,
        "particle_diameter_um": profile.diameter * 1e6,
        "chi_eff": profile.chi_eff,
        "flow_params": flow.params_dict(),
        "start_time": time.strftime("%Y-%m-%d %H:%M:%S"),
    }
    with open(os.path.join(output_dir, "run_config.json"), "w") as f:
        json.dump(config, f, indent=2)

    print(f"\n{'='*70}")
    print(f"  DynaMag Monte Carlo — Time-Limited Run")
    print(f"  Budget: {time_limit_s/3600:.1f} h  |  Max trials: {max_trials}")
    print(f"  Profile: {profile.name} ({profile.diameter*1e6:.0f} µm)")
    print(f"  Flow: {flow_speed*1e3:.1f} mm/s +x  |  I_max: {max_current} A")
    print(f"  Output: {output_dir}")
    print(f"{'='*70}\n")

    for i in range(max_trials):
        elapsed = time.perf_counter() - batch_start
        remaining = time_limit_s - elapsed

        # Estimate time for next trial based on running average
        if completed > 0:
            avg_trial_time = elapsed / completed
            est_next = avg_trial_time * 1.2  # 20% safety margin
        else:
            est_next = 200  # conservative first estimate

        # Stop if not enough time for another trial
        if remaining < est_next and completed > 2:
            print(f"\n⏱️  Time budget nearly exhausted ({remaining:.0f}s remaining, "
                  f"est. {est_next:.0f}s/trial). Stopping.")
            break

        trial_name = f"trial_{i + 1:04d}"
        dist = np.linalg.norm(np.array(targets[i]) - np.array(starts[i]))
        pct_time = (elapsed / time_limit_s) * 100

        print(f"\n>>> [{i+1}/{max_trials}] {trial_name}  "
              f"dist={dist*1000:.1f} mm  "
              f"time: {elapsed/60:.1f}/{time_limit_s/60:.0f} min ({pct_time:.0f}%)")

        trial_start = time.perf_counter()

        result = simulate_trial_mpc(
            start=starts[i],
            target=targets[i],
            trial_name=trial_name,
            dt=dt,
            steps=steps,
            tolerance=tolerance,
            max_current=max_current,
            profile=profile,
            rng_seed=seed + i,
            flow=flow,
        )
        result.params.update(flow.params_dict())
        result.params["particle_diameter_um"] = profile.diameter * 1e6

        trial_elapsed = time.perf_counter() - trial_start

        analytics = compute_trial_analytics(result)
        results.append(result)
        all_analytics.append(analytics)
        completed += 1

        # Save trial data (no plots — we generate those later)
        save_trial_data(result, analytics, output_dir)

        # Save CSV incrementally every 5 trials
        if completed % 5 == 0 or i == 0:
            save_results_csv(results, output_dir, filename="trial_summary.csv")

        status = "REACHED" if result.reached else f"NOT reached (err={result.final_error*1000:.1f}mm)"
        print(f"    -> {status}  steps={result.steps_taken}  "
              f"Re_max={result.max_reynolds:.2e}  "
              f"eff={analytics['path_efficiency']:.2f}  "
              f"energy={analytics['energy_proxy_A2s']:.1f} A²·s  "
              f"wall={trial_elapsed:.1f}s")

    # Final outputs
    batch_elapsed = time.perf_counter() - batch_start
    save_results_csv(results, output_dir, filename="trial_summary.csv")

    # Compute aggregate statistics
    reached_count = sum(1 for r in results if r.reached)
    all_steps = np.array([r.steps_taken for r in results])
    all_errors = np.array([r.final_error for r in results])
    all_re = np.array([r.max_reynolds for r in results])
    all_eff = np.array([a["path_efficiency"] for a in all_analytics])
    all_energy = np.array([a["energy_proxy_A2s"] for a in all_analytics])
    all_mono = np.array([a["monotonic_fraction"] for a in all_analytics])
    all_vel = np.array([a["mean_velocity_m_s"] for a in all_analytics])
    all_force = np.array([a["max_force_N"] for a in all_analytics])
    all_jerk = np.array([a["mean_jerk_A_per_s"] for a in all_analytics])
    all_cost_red = np.array([a["cost_reduction"] for a in all_analytics])

    summary = {
        "n_trials": completed,
        "particle_profile": profile.name,
        "particle_diameter_um": profile.diameter * 1e6,
        "chi_eff": profile.chi_eff,
        "max_current_A": max_current,
        "flow_type": "uniform",
        "flow_params": flow.params_dict(),
        "dt": dt,
        "steps_budget": steps,
        "tolerance_m": tolerance,
        "seed": seed,
        "batch_wall_time_s": round(batch_elapsed, 2),
        "batch_wall_time_h": round(batch_elapsed / 3600, 3),
        "reached_count": reached_count,
        "success_rate": reached_count / completed if completed > 0 else 0,
        "avg_trial_time_s": round(batch_elapsed / completed, 2) if completed > 0 else 0,
        "end_time": time.strftime("%Y-%m-%d %H:%M:%S"),
        "stats": {
            "steps_to_target": {
                "median": float(np.median(all_steps)),
                "mean": float(np.mean(all_steps)),
                "std": float(np.std(all_steps)),
                "min": int(np.min(all_steps)),
                "max": int(np.max(all_steps)),
                "q25": float(np.percentile(all_steps, 25)),
                "q75": float(np.percentile(all_steps, 75)),
            },
            "final_error_mm": {
                "median": float(np.median(all_errors) * 1000),
                "mean": float(np.mean(all_errors) * 1000),
                "std": float(np.std(all_errors) * 1000),
                "max": float(np.max(all_errors) * 1000),
            },
            "max_reynolds": {
                "median": float(np.median(all_re)),
                "mean": float(np.mean(all_re)),
                "max": float(np.max(all_re)),
            },
            "path_efficiency": {
                "median": float(np.median(all_eff)),
                "mean": float(np.mean(all_eff)),
                "std": float(np.std(all_eff)),
            },
            "energy_proxy_A2s": {
                "median": float(np.median(all_energy)),
                "mean": float(np.mean(all_energy)),
                "std": float(np.std(all_energy)),
            },
            "monotonic_convergence": {
                "median": float(np.median(all_mono)),
                "mean": float(np.mean(all_mono)),
            },
            "velocity_m_s": {
                "mean": float(np.mean(all_vel)),
                "std": float(np.std(all_vel)),
                "max_of_means": float(np.max(all_vel)),
            },
            "max_force_N": {
                "median": float(np.median(all_force)),
                "mean": float(np.mean(all_force)),
                "max": float(np.max(all_force)),
            },
            "control_smoothness_A_per_s": {
                "median_mean_jerk": float(np.median(all_jerk)),
                "mean_mean_jerk": float(np.mean(all_jerk)),
            },
            "mpc_cost_reduction": {
                "median": float(np.median(all_cost_red)),
                "mean": float(np.mean(all_cost_red)),
            },
        },
    }

    summary_path = os.path.join(output_dir, "batch_summary.json")
    with open(summary_path, "w") as f:
        json.dump(summary, f, indent=2)

    print(f"\n{'='*70}")
    print(f"  MONTE CARLO RUN COMPLETE")
    print(f"{'='*70}")
    print(f"  Trials completed : {completed}")
    print(f"  Success rate     : {reached_count}/{completed} "
          f"({reached_count/completed*100:.0f}%)" if completed > 0 else "")
    print(f"  Median steps     : {np.median(all_steps):.0f} "
          f"(sim time {np.median(all_steps)*dt:.0f} s)")
    print(f"  Median error     : {np.median(all_errors)*1000:.2f} mm")
    print(f"  Max Reynolds     : {np.max(all_re):.3e}  "
          f"(low-Re valid: {np.max(all_re) < 0.1})")
    print(f"  Path efficiency  : {np.median(all_eff):.2f} (1.0 = straight line)")
    print(f"  Avg trial time   : {batch_elapsed/completed:.1f} s" if completed > 0 else "")
    print(f"  Wall time        : {batch_elapsed/3600:.2f} h")
    print(f"  Output           : {output_dir}/")
    print(f"{'='*70}")

    return results, all_analytics


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Time-limited Monte Carlo data collection for DynaMag."
    )
    parser.add_argument("--time-limit", type=int, default=5400,
                        help="Wall-clock time budget in seconds (default: 5400 = 1.5h)")
    parser.add_argument("--max-trials", type=int, default=500,
                        help="Maximum number of trials (default: 500)")
    parser.add_argument("--seed", type=int, default=12345,
                        help="Master seed (default: 12345, different from prior runs)")
    parser.add_argument("--output-dir", type=str, default="mc_data_run",
                        help="Output directory (default: mc_data_run)")
    args = parser.parse_args()

    run_timed(
        output_dir=args.output_dir,
        time_limit_s=args.time_limit,
        max_trials=args.max_trials,
        seed=args.seed,
    )
