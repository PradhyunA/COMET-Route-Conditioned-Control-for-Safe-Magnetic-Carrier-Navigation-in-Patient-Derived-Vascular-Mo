"""
simulation_3d_mpc.py
====================
3D MPC magnetic particle steering — real vessel network edition.
"""

import os
import time
import warnings

import numpy as np

from dynamag_simulation.core.profiles import PhysicalProfile, MICROSPHERE_500UM
from dynamag_simulation.core.magnets import ElectromagnetArray3D
from dynamag_simulation.core.particle import MagneticParticle3D
from dynamag_simulation.core.integrator import adaptive_step
from dynamag_simulation.core.controllers import continuous_mpc_controller
from dynamag_simulation.core.validation import compute_reynolds, check_low_re_validity
from dynamag_simulation.core.reporting import TrialResult, print_summary, save_results_csv
from dynamag_simulation.core.plotting import save_all_plots
from dynamag_simulation.core.real_vessel_network import RealVesselNetwork
from dynamag_simulation.core.workspace import WorkspaceVessel


def simulate_trial_mpc(
    start,
    target,
    network,
    trial_name: str = "trial",
    dt: float = 0.02,
    steps: int = 500,
    tolerance: float = 0.003,
    coil_radius: float = 0.03,
    coil_length: float = 0.12,
    turns: float = 1500,
    max_current: float = 8.0,
    profile: PhysicalProfile = MICROSPHERE_500UM,
    max_substep_disp: float = 1e-3,
    rng_seed: int = 42,
    w_progress: float = 80.0,
    w_align: float = 8.0,
    w_dist: float = 2.0,
    w_curr: float = 1e-5,
    w_smooth: float = 1e-4,
    w_edge: float = 20.0,
    w_overshoot: float = 2.0,
) -> TrialResult:

    workspace = WorkspaceVessel(network)
    flow = network

    magnets = ElectromagnetArray3D(
        coil_radius=coil_radius,
        coil_length=coil_length,
        turns=turns,
        max_current=max_current,
    )

    start = workspace.clip(np.asarray(start, dtype=float))
    target = workspace.clip(np.asarray(target, dtype=float))

    particle = MagneticParticle3D(
        position=start,
        profile=profile,
        rng_seed=rng_seed,
    )

    navigator = network.waypoint_navigator(start, target, arrival_tol=tolerance)

    print(f"    Waypoints planned: {navigator.total_hops} hops")
    for i, wp in enumerate(navigator.waypoints):
        print(f"      wp{i + 1}: {np.round(wp * 1000, 1)} mm")

    trajectory = [particle.pos.copy()]
    distance_log = []
    current_log = []
    field_log = []
    force_log = []
    velocity_log = []
    reynolds_log = []
    cost_log = []
    step_time_log = []

    reached = False
    prev_currents = np.zeros(magnets.n_magnets, dtype=float)
    wall_start = time.perf_counter()

    for step in range(steps):
        sim_time = step * dt
        t0 = time.perf_counter()

        current_target = navigator(particle.pos)

        best_currents, best_force, best_cost, converged = continuous_mpc_controller(
            particle=particle,
            particle_pos=particle.pos,
            target_pos=current_target,
            magnet_array=magnets,
            workspace=workspace,
            dt=dt,
            max_current=max_current,
            initial_guess=prev_currents,
            w_progress=w_progress,
            w_align=w_align,
            w_dist=w_dist,
            w_curr=w_curr,
            w_smooth=w_smooth,
            w_edge=w_edge,
            w_overshoot=w_overshoot,
            flow=flow,
            sim_time=sim_time,
        )

        t1 = time.perf_counter()

        if not converged:
            warnings.warn(
                f"Step {step}: no SLSQP start converged; using best-so-far",
                RuntimeWarning,
                stacklevel=1,
            )

        prev_currents = best_currents.copy()
        magnets.set_currents(best_currents)
        B = magnets.field_at(particle.pos)

        adaptive_step(
            particle,
            best_force,
            dt,
            workspace,
            max_substep_disp=max_substep_disp,
            flow=flow,
            sim_time=sim_time,
        )

        mag_vel = np.linalg.norm(particle.vel)
        re = compute_reynolds(
            mag_vel,
            profile.diameter,
            profile.fluid_density,
            profile.fluid_viscosity,
        )

        trajectory.append(particle.pos.copy())
        distance = np.linalg.norm(particle.pos - target)

        distance_log.append(distance)
        current_log.append(magnets.currents.copy())
        field_log.append(np.linalg.norm(B))
        force_log.append(np.linalg.norm(best_force))
        velocity_log.append(np.linalg.norm(particle.realized_vel))
        reynolds_log.append(re)
        cost_log.append(best_cost)
        step_time_log.append(t1 - t0)

        if distance < tolerance:
            reached = True
            break

        if (step + 1) % 10 == 0:
            print(
                f"  step {step + 1:>4d}/{steps}  "
                f"dist={distance * 1000:.1f} mm  "
                f"wp={getattr(navigator, 'current_idx', 0) + 1}/{navigator.total_hops}  "
                f"|I|max={np.max(np.abs(best_currents)):.3f} A  "
                f"|B|={np.linalg.norm(B):.3e} T  "
                f"|F|={np.linalg.norm(best_force):.3e} N  "
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
        start=start,
        target=target,
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
        final_error=float(np.linalg.norm(trajectory[-1] - target)),
        max_current_used=float(np.max(np.abs(current_log))) if len(current_log) > 0 else 0.0,
        mean_reynolds=re_stats["mean_reynolds"],
        max_reynolds=re_stats["max_reynolds"],
        low_re_valid=re_stats["low_re_valid"],
        wall_time_s=wall_end - wall_start,
        avg_control_step_time_s=float(np.mean(step_time_log)) if len(step_time_log) > 0 else 0.0,
        max_control_step_time_s=float(np.max(step_time_log)) if len(step_time_log) > 0 else 0.0,
        seed=rng_seed,
        profile_name=profile.name,
        dt=dt,
        workspace_half=network.half,
        magnets=magnets.positions.copy(),
        orientations=magnets.orientations.copy(),
        params={
            "dt": dt,
            "steps": steps,
            "tolerance": tolerance,
            "coil_radius": coil_radius,
            "coil_length": coil_length,
            "turns": turns,
            "max_current": max_current,
            "profile": profile.name,
            "max_substep_disp": max_substep_disp,
            "vessel_network": network.params_dict(),
        },
    )


def run_all_trials():
    network = RealVesselNetwork("real_vessel_segments.pkl")
    print(network.segment_summary())

    rng = np.random.default_rng(42)

    inlet = network.inlet_start()
    outlet = network.outlet_end()
    mid = network.random_point_in_vessel(rng)

    trials = {
        "trial_1_inlet_to_outlet": {
            "start": inlet,
            "target": outlet,
        },
        "trial_2_mid_to_outlet": {
            "start": mid,
            "target": outlet,
        },
        "trial_3_inlet_to_mid": {
            "start": inlet,
            "target": network.random_point_in_vessel(rng),
        },
    }

    output_dir = "simulation_outputs_3d_mpc"
    os.makedirs(output_dir, exist_ok=True)
    results = []

    magnets_for_plots = ElectromagnetArray3D(
        coil_radius=0.03,
        coil_length=0.12,
        turns=1500,
        max_current=8.0,
    )

    for trial_name, cfg in trials.items():
        print(f"\n>>> Running {trial_name}")
        print(f"    start:  {cfg['start']}")
        print(f"    target: {cfg['target']}")

        result = simulate_trial_mpc(
            start=cfg["start"],
            target=cfg["target"],
            network=network,
            trial_name=trial_name,
        )

        results.append(result)

        print_summary(result)
        save_all_plots(result, output_dir, magnet_array=magnets_for_plots)

    save_results_csv(results, output_dir)
    print(f"\nResults saved to: {output_dir}/")
    return results


if __name__ == "__main__":
    run_all_trials()