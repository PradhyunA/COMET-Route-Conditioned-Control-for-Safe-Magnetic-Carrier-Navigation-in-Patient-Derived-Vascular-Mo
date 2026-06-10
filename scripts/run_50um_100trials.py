"""
run_50um_100trials.py
=====================
100 3D MPC trials with a 50 um microsphere under uniform flow.

Physics regime: Re ~ 0.02 (safely creeping flow).
dt = 1.0 s  (scaled from 500um runs to match lower velocity).
Flow = 200 um/s +x  (comparable to capillary flow; magnetic drift ~3x flow at 13A).
"""

import os
import sys
# Add parent directory to path to allow importing dynamag_simulation when run directly
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

import json
import time

import numpy as np

from dynamag_simulation.core.constants import DEFAULT_TARGET_TOLERANCE, HALF, TEST_RANGE
from dynamag_simulation.core.profiles import MICROSPHERE_50UM
from dynamag_simulation.core.magnets import ElectromagnetArray3D
from dynamag_simulation.core.flow import UniformFlow
from dynamag_simulation.core.reporting import print_summary, save_results_csv
from dynamag_simulation.core.plotting import save_all_plots

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

    # Convergence rate: slope of distance curve (mm/s) in first half
    if len(result.distance_log) > 10:
        n_half = len(result.distance_log) // 2
        t_half = np.arange(n_half) * dt
        d_half = result.distance_log[:n_half]
        if n_half > 1:
            slope = float(np.polyfit(t_half, d_half, 1)[0])  # m/s (negative = converging)
        else:
            slope = 0.0
    else:
        slope = 0.0

    # Force-direction alignment with target direction per step
    if len(result.force_log) > 0 and len(traj) > 1:
        alignments = []
        for i in range(min(len(result.force_log), len(traj) - 1)):
            desired = target - traj[i]
            d_mag = np.linalg.norm(desired)
            if d_mag > 1e-10 and result.force_log[i] > 1e-30:
                alignments.append(1.0)  # scalar force, can't compute direction
            else:
                alignments.append(0.0)
        mean_alignment = float(np.mean(alignments)) if alignments else 0.0
    else:
        mean_alignment = 0.0

    # Velocity stats
    if len(result.velocity_log) > 0:
        mean_vel = float(np.mean(result.velocity_log))
        std_vel = float(np.std(result.velocity_log))
    else:
        mean_vel = 0.0
        std_vel = 0.0

    # Distance curve smoothness (std of step-to-step distance changes)
    if len(result.distance_log) > 1:
        dist_diffs = np.diff(result.distance_log)
        monotonic_fraction = float(np.mean(dist_diffs < 0))
        dist_smoothness = float(np.std(dist_diffs))
    else:
        monotonic_fraction = 0.0
        dist_smoothness = 0.0

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
        "monotonic_fraction": monotonic_fraction,
        "distance_smoothness": dist_smoothness,
    }


def save_trial_data_comprehensive(result, analytics, output_dir):
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
    )


def run(
    output_dir="outputs/flow_50um_100trials",
    n_trials=100,
    seed=42,
    dt=1.0,
    steps=3000,
    tolerance=DEFAULT_TARGET_TOLERANCE,
    flow_speed=0.0002,
    max_current=13.0,
    verbose=True,
):
    output_dir = os.path.abspath(output_dir)
    os.makedirs(output_dir, exist_ok=True)
    rng = np.random.default_rng(seed)

    flow = UniformFlow(velocity=[flow_speed, 0.0, 0.0])
    profile = MICROSPHERE_50UM

    starts = [_random_pos(rng) for _ in range(n_trials)]
    targets = [_random_pos(rng) for _ in range(n_trials)]

    magnets_for_plots = ElectromagnetArray3D(
        coil_radius=0.03, coil_length=0.12, turns=1500, max_current=max_current,
    )

    results = []
    all_analytics = []
    batch_start = time.perf_counter()

    for i in range(n_trials):
        trial_name = f"trial_{i + 1:04d}"
        if verbose:
            dist = np.linalg.norm(np.array(targets[i]) - np.array(starts[i]))
            print(f"\n>>> [{i+1}/{n_trials}] {trial_name}  dist={dist*1000:.1f} mm")

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

        analytics = compute_trial_analytics(result)
        results.append(result)
        all_analytics.append(analytics)

        os.makedirs(output_dir, exist_ok=True)
        save_all_plots(result, output_dir, magnet_array=magnets_for_plots)
        save_trial_data_comprehensive(result, analytics, output_dir)

        if verbose:
            status = "REACHED" if result.reached else f"NOT reached (err={result.final_error*1000:.1f}mm)"
            print(f"    -> {status}  steps={result.steps_taken}  "
                  f"Re_max={result.max_reynolds:.2e}  "
                  f"eff={analytics['path_efficiency']:.2f}  "
                  f"energy={analytics['energy_proxy_A2s']:.1f} A^2*s")

    batch_elapsed = time.perf_counter() - batch_start

    save_results_csv(results, output_dir, filename="trial_summary.csv")

    # Enhanced batch summary with aggregate statistics
    reached_count = sum(1 for r in results if r.reached)
    all_steps = np.array([r.steps_taken for r in results])
    all_errors = np.array([r.final_error for r in results])
    all_re = np.array([r.max_reynolds for r in results])
    all_eff = np.array([a["path_efficiency"] for a in all_analytics])
    all_energy = np.array([a["energy_proxy_A2s"] for a in all_analytics])
    all_mono = np.array([a["monotonic_fraction"] for a in all_analytics])

    summary = {
        "n_trials": n_trials,
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
        "reached_count": reached_count,
        "success_rate": reached_count / n_trials,
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
        },
    }

    summary_path = os.path.join(output_dir, "batch_summary.json")
    with open(summary_path, "w") as f:
        json.dump(summary, f, indent=2)

    if verbose:
        print(f"\n{'=' * 65}")
        print(f"  BATCH COMPLETE: {n_trials} trials  |  50 um  |  I_max={max_current}A  |  flow = {flow_speed*1e3:.1f} mm/s")
        print(f"{'=' * 65}")
        print(f"  Success rate     : {reached_count}/{n_trials}")
        print(f"  Median steps     : {np.median(all_steps):.0f}  (sim time {np.median(all_steps)*dt:.0f} s)")
        print(f"  Median error     : {np.median(all_errors)*1000:.2f} mm")
        print(f"  Max Reynolds     : {np.max(all_re):.3e}  (low-Re valid: {np.max(all_re) < 0.1})")
        print(f"  Path efficiency  : {np.median(all_eff):.2f} (1.0 = straight line)")
        print(f"  Wall time        : {batch_elapsed/3600:.1f} h")
        print(f"  Outputs          : {output_dir}/")
        print(f"{'=' * 65}")

    return results, all_analytics


if __name__ == "__main__":
    n = int(sys.argv[1]) if len(sys.argv) > 1 else 100
    run(n_trials=n)
