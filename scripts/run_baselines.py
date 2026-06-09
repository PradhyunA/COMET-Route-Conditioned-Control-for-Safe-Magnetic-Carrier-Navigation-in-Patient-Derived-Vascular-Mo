"""
run_baselines.py
================
Baseline comparison datasets for the DynaMag MPC steering paper.

Two modes, each running 30 trials with the same start/target pairs
(seed=42) and identical 50 µm microsphere + uniform flow physics:

  1. Random Control  – all coil currents = 0 every step.
     Particle drifts with flow + Brownian only.  (No steering.)

  2. Static Field    – coil currents set once at trial start based on
     the start→target direction, then held constant.  (Non-adaptive.)

Both store the same metrics/files as the MPC batch runner for direct
comparison (trial_summary.csv, batch_summary.json, per-trial npz, plots).

Usage:
    python run_baselines.py              # both baselines, 30 trials each
    python run_baselines.py 10           # 10 trials each (quick test)
"""

import os
import sys
# Add parent directory to path to allow importing dynamag_simulation when run directly
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

import json
import time

import numpy as np

from dynamag_simulation.core.constants import HALF, TEST_RANGE
from dynamag_simulation.core.profiles import MICROSPHERE_50UM
from dynamag_simulation.core.magnets import ElectromagnetArray3D
from dynamag_simulation.core.particle import MagneticParticle3D
from dynamag_simulation.core.workspace import Workspace3D
from dynamag_simulation.core.integrator import adaptive_step
from dynamag_simulation.core.flow import UniformFlow
from dynamag_simulation.core.validation import compute_reynolds, check_low_re_validity
from dynamag_simulation.core.reporting import TrialResult, print_summary, save_results_csv
from dynamag_simulation.core.plotting import save_all_plots

# ── shared configuration (matches run_50um_100trials.py) ───────────
INNER = max(0.01, HALF - TEST_RANGE)
DEFAULT_DT = 1.0
DEFAULT_STEPS = 3000
DEFAULT_TOL = 0.003       # 3 mm
DEFAULT_FLOW_SPEED = 0.0002   # 200 µm/s  +x
DEFAULT_MAX_CURRENT = 13.0
DEFAULT_SEED = 42


def _random_pos(rng):
    return tuple(float(rng.uniform(-INNER, INNER)) for _ in range(3))


# ── generic baseline trial (no MPC) ───────────────────────────────
def simulate_trial_baseline(
    start,
    target,
    trial_name,
    mode,          # "random" or "static"
    dt=DEFAULT_DT,
    steps=DEFAULT_STEPS,
    tolerance=DEFAULT_TOL,
    max_current=DEFAULT_MAX_CURRENT,
    profile=MICROSPHERE_50UM,
    rng_seed=42,
    flow=None,
):
    """
    Run a single baseline trial.

    mode="random": currents = 0 every step  (no steering)
    mode="static": currents fixed once per trial toward target  (non-adaptive)
    """
    workspace = Workspace3D()
    magnets = ElectromagnetArray3D(
        coil_radius=0.03, coil_length=0.12, turns=1500, max_current=max_current,
    )
    particle = MagneticParticle3D(
        position=start, profile=profile, rng_seed=rng_seed,
    )

    start_arr = np.asarray(start, dtype=float)
    target_arr = np.asarray(target, dtype=float)

    # ── decide fixed currents ──────────────────────────────────
    if mode == "random":
        fixed_currents = np.zeros(magnets.n_magnets, dtype=float)
    elif mode == "static":
        # Project each coil orientation onto start→target direction
        desired = target_arr - start_arr
        dist = np.linalg.norm(desired)
        if dist > 1e-10:
            desired_hat = desired / dist
        else:
            desired_hat = np.array([1.0, 0.0, 0.0])
        proj = magnets.orientations @ desired_hat          # (8,)
        fixed_currents = max_current * proj / (np.max(np.abs(proj)) + 1e-30)
    else:
        raise ValueError(f"Unknown mode: {mode!r}")

    # Apply the constant currents once
    magnets.set_currents(fixed_currents)

    trajectory = [particle.pos.copy()]
    distance_log = []
    current_log = []
    field_log = []
    force_log = []
    velocity_log = []
    reynolds_log = []
    cost_log = []            # always 0 for baselines (no optimizer)
    step_time_log = []

    reached = False
    wall_start = time.perf_counter()

    for step in range(steps):
        sim_time = step * dt
        t0 = time.perf_counter()

        # Compute force at current position with fixed currents
        magnets.set_currents(fixed_currents)   # idempotent
        B = magnets.field_at(particle.pos)
        grad_Bsq = magnets.grad_B_squared(particle.pos)
        force = particle.magnetic_force(B, grad_Bsq)

        t1 = time.perf_counter()

        # Integrate (with flow + Brownian)
        realized_disp, n_subs = adaptive_step(
            particle, force, dt, workspace,
            flow=flow, sim_time=sim_time,
        )

        mag_vel = np.linalg.norm(particle.vel)
        re = compute_reynolds(
            mag_vel, profile.diameter, profile.fluid_density, profile.fluid_viscosity,
        )

        trajectory.append(particle.pos.copy())
        distance = np.linalg.norm(particle.pos - target_arr)

        distance_log.append(distance)
        current_log.append(fixed_currents.copy())
        field_log.append(np.linalg.norm(B))
        force_log.append(np.linalg.norm(force))
        vel_mag = np.linalg.norm(particle.realized_vel)
        velocity_log.append(vel_mag)
        reynolds_log.append(re)
        cost_log.append(0.0)
        step_time_log.append(t1 - t0)

        if distance < tolerance:
            reached = True
            break

        if (step + 1) % 200 == 0:
            print(
                f"  step {step + 1:>4d}/{steps}  "
                f"dist={distance:.5f} m  "
                f"|B|={np.linalg.norm(B):.3e} T  "
                f"|F|={np.linalg.norm(force):.3e} N  "
                f"Re={re:.2e}"
            )

    wall_end = time.perf_counter()

    trajectory = np.array(trajectory)
    distance_log = np.array(distance_log)
    current_log = np.array(current_log)
    field_log = np.array(field_log)
    force_log = np.array(force_log)
    velocity_log = np.array(velocity_log)
    reynolds_log = np.array(reynolds_log)
    cost_log = np.array(cost_log)
    step_time_log = np.array(step_time_log)

    re_stats = check_low_re_validity(reynolds_log)

    return TrialResult(
        trial_name=trial_name,
        start=start_arr,
        target=target_arr,
        final_position=trajectory[-1],
        trajectory=trajectory,
        distance_log=distance_log,
        current_log=current_log,
        field_log=field_log,
        force_log=force_log,
        velocity_log=velocity_log,
        reynolds_log=reynolds_log,
        cost_log=cost_log,
        step_time_log=step_time_log,
        reached=reached,
        steps_taken=len(trajectory) - 1,
        final_error=float(np.linalg.norm(trajectory[-1] - target_arr)),
        max_current_used=float(np.max(np.abs(current_log))) if len(current_log) > 0 else 0.0,
        mean_reynolds=re_stats["mean_reynolds"],
        max_reynolds=re_stats["max_reynolds"],
        low_re_valid=re_stats["low_re_valid"],
        stokes_regime=re_stats["stokes_regime"],
        finite_re_drag_valid=re_stats["finite_re_drag_valid"],
        drag_model=re_stats["drag_model"],
        drag_regime_label=re_stats["drag_regime_label"],
        wall_time_s=wall_end - wall_start,
        avg_control_step_time_s=float(np.mean(step_time_log)) if len(step_time_log) > 0 else 0.0,
        max_control_step_time_s=float(np.max(step_time_log)) if len(step_time_log) > 0 else 0.0,
        seed=rng_seed,
        profile_name=profile.name,
        dt=dt,
        workspace_half=HALF,
        magnets=magnets.positions.copy(),
        orientations=magnets.orientations.copy(),
        params={
            "dt": dt,
            "steps": steps,
            "tolerance": tolerance,
            "coil_radius": 0.03,
            "coil_length": 0.12,
            "turns": 1500,
            "max_current": max_current,
            "profile": profile.name,
            "baseline_mode": mode,
        },
    )


# ── per-trial analytics (identical to run_50um_100trials.py) ──────
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

    if len(result.current_log) > 0:
        energy_proxy = float(np.sum(result.current_log ** 2) * dt)
        peak_total_current = float(np.max(np.sum(np.abs(result.current_log), axis=1)))
        mean_total_current = float(np.mean(np.sum(np.abs(result.current_log), axis=1)))
    else:
        energy_proxy = 0.0
        peak_total_current = 0.0
        mean_total_current = 0.0

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

    mean_alignment = 0.0  # can't compute direction from scalar force log

    if len(result.velocity_log) > 0:
        mean_vel = float(np.mean(result.velocity_log))
        std_vel = float(np.std(result.velocity_log))
    else:
        mean_vel = 0.0
        std_vel = 0.0

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
        straight_distance=analytics["straight_distance_m"],
        path_length=analytics["path_length_m"],
        path_efficiency=analytics["path_efficiency"],
        energy_proxy=analytics["energy_proxy_A2s"],
        convergence_slope=analytics["convergence_slope_m_per_s"],
        monotonic_fraction=analytics["monotonic_fraction"],
    )


# ── batch runner ──────────────────────────────────────────────────
def run_baseline(
    mode,
    output_dir,
    n_trials=30,
    seed=DEFAULT_SEED,
    dt=DEFAULT_DT,
    steps=DEFAULT_STEPS,
    tolerance=DEFAULT_TOL,
    flow_speed=DEFAULT_FLOW_SPEED,
    max_current=DEFAULT_MAX_CURRENT,
    verbose=True,
):
    """Run *n_trials* baseline trials of the given *mode*."""
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
            print(f"\n>>> [{i+1}/{n_trials}] {trial_name}  dist={dist*1000:.1f} mm  mode={mode}")

        result = simulate_trial_baseline(
            start=starts[i],
            target=targets[i],
            trial_name=trial_name,
            mode=mode,
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

    # ── aggregate stats ───────────────────────────────────────
    reached_count = sum(1 for r in results if r.reached)
    all_steps = np.array([r.steps_taken for r in results])
    all_errors = np.array([r.final_error for r in results])
    all_re = np.array([r.max_reynolds for r in results])
    all_eff = np.array([a["path_efficiency"] for a in all_analytics])
    all_energy = np.array([a["energy_proxy_A2s"] for a in all_analytics])
    all_mono = np.array([a["monotonic_fraction"] for a in all_analytics])

    summary = {
        "baseline_mode": mode,
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
        label = "RANDOM CONTROL (no steering)" if mode == "random" else "STATIC FIELD  (non-adaptive)"
        print(f"\n{'=' * 65}")
        print(f"  {label}")
        print(f"  {n_trials} trials  |  50 µm  |  I_max={max_current}A  |  flow = {flow_speed*1e3:.1f} mm/s")
        print(f"{'=' * 65}")
        print(f"  Success rate     : {reached_count}/{n_trials}")
        print(f"  Median steps     : {np.median(all_steps):.0f}  (sim time {np.median(all_steps)*dt:.0f} s)")
        print(f"  Median error     : {np.median(all_errors)*1000:.2f} mm")
        print(f"  Max Reynolds     : {np.max(all_re):.3e}  (low-Re valid: {np.max(all_re) < 0.1})")
        print(f"  Path efficiency  : {np.median(all_eff):.2f} (1.0 = straight line)")
        print(f"  Wall time        : {batch_elapsed:.1f} s")
        print(f"  Outputs          : {output_dir}/")
        print(f"{'=' * 65}")

    return results, all_analytics


# ── main ──────────────────────────────────────────────────────────
if __name__ == "__main__":
    sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
    n = int(sys.argv[1]) if len(sys.argv) > 1 else 30

    print("\n" + "=" * 65)
    print("  BASELINE 1/2:  Random Control (no steering)")
    print("=" * 65)
    run_baseline(mode="random", output_dir="outputs/baseline_random_30trials", n_trials=n)

    print("\n\n" + "=" * 65)
    print("  BASELINE 2/2:  Static Field (non-adaptive)")
    print("=" * 65)
    run_baseline(mode="static", output_dir="outputs/baseline_static_30trials", n_trials=n)

    print("\n\nBoth baselines complete.")
