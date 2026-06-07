"""
simulation_3d_mpc.py
====================
3D ACBC magnetic nanoparticle steering — real vessel network edition.

This runner uses:

    ACBC high-level supervisor
        +
    existing continuous MPC current allocator

ACBC is the main algorithmic contribution.
The MPC allocator is only the low-level coil-current solver.
"""

import os
import sys
import time
import warnings
import argparse
import csv

# Add parent directory to path to allow importing dynamag_simulation when run directly
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

import numpy as np

from dynamag_simulation.core.profiles import PhysicalProfile, SPION_CARRIER_10UM
from dynamag_simulation.core.magnets import ElectromagnetArray3D
from dynamag_simulation.core.particle import MagneticParticle3D
from dynamag_simulation.core.integrator import adaptive_step
from dynamag_simulation.core.validation import compute_reynolds, check_low_re_validity
from dynamag_simulation.core.reporting import TrialResult, print_summary, save_results_csv
from dynamag_simulation.core.plotting import save_all_plots
from dynamag_simulation.core.real_vessel_network import RealVesselNetwork
from dynamag_simulation.core.workspace import WorkspaceVessel
from dynamag_simulation.core.acbc_controller import ACBCController


DEFAULT_VESSEL_FILE = "real_vessel_segments_cm_corrected.pkl"
DEFAULT_MAGNET_CLEARANCE = 0.04
DEFAULT_COIL_RADIUS_FRACTION = 0.5
DEFAULT_COIL_LENGTH_FACTOR = 2.0


def magnet_setup_for_network(
    network,
    clearance: float = DEFAULT_MAGNET_CLEARANCE,
    placement_radius: float | None = None,
):
    """
    Size and center the triangular-prism magnet array around the vessel tree.
    """
    bounds_min, bounds_max = network.bounds(include_radius=True)
    center = 0.5 * (bounds_min + bounds_max)
    span = bounds_max - bounds_min

    if placement_radius is None:
        required = max(
            0.5 * span[0],
            0.5 * np.sqrt(3.0) * span[1],
            0.5 * span[2],
        )
        placement_radius = required + clearance

    placement_radius = float(placement_radius)

    return {
        "center": center,
        "placement_radius": placement_radius,
        "coil_radius": DEFAULT_COIL_RADIUS_FRACTION * placement_radius,
        "coil_length": DEFAULT_COIL_LENGTH_FACTOR * placement_radius,
        "bounds_min": bounds_min,
        "bounds_max": bounds_max,
    }


def _vec_or_nan(v):
    if v is None:
        return [np.nan, np.nan, np.nan]
    v = np.asarray(v, dtype=float)
    return [float(v[0]), float(v[1]), float(v[2])]


def save_acbc_diagnostics(results, output_dir, filename="acbc_diagnostics.csv"):
    """
    Save per-step ACBC state-machine diagnostics.

    This is important for the paper:
        - state transitions
        - branch commitment events
        - recovery events
        - authority margin over time
        - bifurcation distance over time
    """
    path = os.path.join(output_dir, filename)

    header = [
        "trial_name",
        "step",
        "time_s",
        "state",
        "previous_state",
        "transition",
        "near_bifurcation",
        "bifurcation_distance_m",
        "selected_branch_seg_id",
        "selected_authority_margin",
        "selected_score",
        "target_branch_capturable",
        "num_candidates",
        "control_target_x",
        "control_target_y",
        "control_target_z",
        "nominal_target_x",
        "nominal_target_y",
        "nominal_target_z",
        "recovery_target_x",
        "recovery_target_y",
        "recovery_target_z",
    ]

    with open(path, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(header)

        for result in results:
            log = getattr(result, "acbc_log", [])

            for row in log:
                writer.writerow([
                    result.trial_name,
                    row["step"],
                    row["time_s"],
                    row["state"],
                    row["previous_state"],
                    row["transition"],
                    row["near_bifurcation"],
                    row["bifurcation_distance_m"],
                    row["selected_branch_seg_id"],
                    row["selected_authority_margin"],
                    row["selected_score"],
                    row["target_branch_capturable"],
                    row["num_candidates"],
                    *row["control_target"],
                    *row["nominal_target"],
                    *row["recovery_target"],
                ])

    print(f"ACBC diagnostics saved to: {path}")


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
    placement_radius: float = 0.06,
    magnet_center=None,
    profile: PhysicalProfile = SPION_CARRIER_10UM,
    max_substep_disp: float = 1e-3,
    rng_seed: int = 42,
    w_progress: float = 120.0,
    w_align: float = 8.0,
    w_dist: float = 1.0,
    w_curr: float = 1e-5,
    w_smooth: float = 1e-4,
    w_edge: float = 50.0,
    w_overshoot: float = 2.0,
) -> TrialResult:

    workspace = WorkspaceVessel(network)
    flow = network

    magnets = ElectromagnetArray3D(
        coil_radius=coil_radius,
        coil_length=coil_length,
        turns=turns,
        max_current=max_current,
        placement_radius=placement_radius,
        center=magnet_center,
    )

    start = workspace.clip(np.asarray(start, dtype=float))
    target = workspace.clip(np.asarray(target, dtype=float))

    particle = MagneticParticle3D(
        position=start,
        profile=profile,
        rng_seed=rng_seed,
    )

    navigator = network.waypoint_navigator(
        start,
        target,
        arrival_tol=0.001,
        waypoint_spacing=0.001,
        lookahead=1,
    )

    acbc = ACBCController(
        network=network,
        magnet_array=magnets,
        workspace=workspace,
        max_current=max_current,
        branch_detection_radius=None,
        branch_lookahead=None,
        authority_threshold=1.05,
        recovery_threshold=0.85,
        min_wall_safety_fraction=0.05,
    )

    print(
        f"    Raw waypoints: "
        f"{getattr(navigator, 'raw_waypoint_count', 'unknown')}, "
        f"sparse waypoints: {navigator.total_hops}"
    )

    for i, wp in enumerate(navigator.waypoints[:10]):
        print(f"      wp{i + 1}: {np.round(wp * 1000, 1)} mm")

    if navigator.total_hops > 10:
        print("      ...")
        print(
            f"      wp{navigator.total_hops}: "
            f"{np.round(navigator.waypoints[-1] * 1000, 1)} mm"
        )

    print(
        f"    ACBC: detection_radius={acbc.branch_detection_radius * 1000:.1f} mm, "
        f"branch_lookahead={acbc.branch_lookahead * 1000:.1f} mm"
    )

    trajectory = [particle.pos.copy()]
    distance_log = []
    current_log = []
    field_log = []
    force_log = []
    velocity_log = []
    reynolds_log = []
    cost_log = []
    step_time_log = []

    acbc_log = []

    reached = False
    prev_currents = np.zeros(magnets.n_magnets, dtype=float)
    wall_start = time.perf_counter()

    converged_count = 0

    for step in range(steps):

        sim_time = step * dt
        t0 = time.perf_counter()

        best_currents, best_force, best_cost, converged, acbc_info = acbc.step(
            particle=particle,
            final_target=target,
            navigator=navigator,
            dt=dt,
            flow=flow,
            sim_time=sim_time,
            initial_guess=prev_currents,
            w_progress=w_progress,
            w_align=w_align,
            w_dist=w_dist,
            w_curr=w_curr,
            w_smooth=w_smooth,
            w_edge=w_edge,
            w_overshoot=w_overshoot,
        )

        t1 = time.perf_counter()

        if converged:
            converged_count += 1
        else:
            warnings.warn(
                f"Step {step}: low-level current allocator did not converge; using best-so-far",
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

        acbc_log.append({
            "step": step,
            "time_s": sim_time,
            "state": acbc_info.state,
            "previous_state": acbc_info.previous_state,
            "transition": acbc_info.transition,
            "near_bifurcation": acbc_info.near_bifurcation,
            "bifurcation_distance_m": acbc_info.bifurcation_distance_m,
            "selected_branch_seg_id": acbc_info.selected_branch_seg_id,
            "selected_authority_margin": acbc_info.selected_authority_margin,
            "selected_score": acbc_info.selected_score,
            "target_branch_capturable": acbc_info.target_branch_capturable,
            "num_candidates": acbc_info.num_candidates,
            "control_target": _vec_or_nan(acbc_info.control_target),
            "nominal_target": _vec_or_nan(acbc_info.nominal_target),
            "recovery_target": _vec_or_nan(acbc_info.recovery_target),
        })

        if distance < tolerance:
            reached = True
            break

        if (step + 1) % 10 == 0:

            flow_v = flow.velocity_at(particle.pos, sim_time)
            flow_speed = np.linalg.norm(flow_v)

            particle_speed = np.linalg.norm(particle.realized_vel)

            seg, _ = network.find_segment(particle.pos)

            if seg is not None:
                _, _, radial_dist, _ = seg.closest_point_and_radial(
                    particle.pos
                )
                wall_clearance = seg.radius - radial_dist
            else:
                wall_clearance = np.nan

            print(
                f"  step {step + 1:>4d}/{steps}  "
                f"state={acbc_info.state}  "
                f"final_dist={distance * 1000:.1f} mm  "
                f"wp={getattr(navigator, 'current_idx', 0) + 1}/"
                f"{navigator.total_hops}  "
                f"auth={acbc_info.selected_authority_margin:.2f}  "
                f"capturable={acbc_info.target_branch_capturable}  "
                f"cands={acbc_info.num_candidates}  "
                f"|I|max={np.max(np.abs(best_currents)):.3f} A  "
                f"|F|={np.linalg.norm(best_force):.3e} N  "
                f"particle_v={particle_speed * 1000:.2f} mm/s  "
                f"flow_v={flow_speed * 1000:.2f} mm/s  "
                f"wall_clear={wall_clearance * 1000:.2f} mm  "
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

    acbc_summary = acbc.summary_dict()
    low_level_convergence_rate = (
        converged_count / max(1, len(step_time_log))
    )

    result = TrialResult(
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
        max_current_used=float(np.max(np.abs(current_log)))
        if len(current_log) > 0
        else 0.0,
        mean_reynolds=re_stats["mean_reynolds"],
        max_reynolds=re_stats["max_reynolds"],
        low_re_valid=re_stats["low_re_valid"],
        wall_time_s=wall_end - wall_start,
        avg_control_step_time_s=float(np.mean(step_time_log))
        if len(step_time_log) > 0
        else 0.0,
        max_control_step_time_s=float(np.max(step_time_log))
        if len(step_time_log) > 0
        else 0.0,
        seed=rng_seed,
        profile_name=profile.name,
        dt=dt,
        workspace_half=network.half,
        magnets=magnets.positions.copy(),
        orientations=magnets.orientations.copy(),
        params={
            "controller": "ACBC",
            "low_level_allocator": "continuous_mpc_controller",
            "dt": dt,
            "steps": steps,
            "tolerance": tolerance,
            "coil_radius": coil_radius,
            "coil_length": coil_length,
            "turns": turns,
            "max_current": max_current,
            "magnet_center": magnets.center.tolist(),
            "placement_radius": placement_radius,
            "profile": profile.name,
            "max_substep_disp": max_substep_disp,
            "vessel_network": network.params_dict(),
            "low_level_convergence_rate": low_level_convergence_rate,
            **acbc_summary,
        },
    )

    result.acbc_log = acbc_log

    print(
        f"  ACBC summary: transitions={acbc_summary['acbc_transitions']}, "
        f"commitments={acbc_summary['acbc_commitments']}, "
        f"recoveries={acbc_summary['acbc_recoveries']}, "
        f"low-level convergence={low_level_convergence_rate:.2%}"
    )

    return result


def run_all_trials(
    vessel_file: str = DEFAULT_VESSEL_FILE,
    magnet_clearance: float = DEFAULT_MAGNET_CLEARANCE,
    placement_radius: float | None = None,
):

    flow_scale = 1.0

    network = RealVesselNetwork(
        vessel_file,
        flow_scale=flow_scale,
    )

    print(network.segment_summary())

    magnet_setup = magnet_setup_for_network(
        network,
        clearance=magnet_clearance,
        placement_radius=placement_radius,
    )

    placement_radius = magnet_setup["placement_radius"]
    coil_radius = magnet_setup["coil_radius"]
    coil_length = magnet_setup["coil_length"]
    magnet_center = magnet_setup["center"]

    print(
        "Magnet setup: "
        f"center={np.round(magnet_center, 4)} m, "
        f"placement_radius={placement_radius:.3f} m, "
        f"coil_radius={coil_radius:.3f} m, "
        f"coil_length={coil_length:.3f} m"
    )

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

    output_dir = "outputs/simulation_outputs_3d_acbc"

    os.makedirs(output_dir, exist_ok=True)

    results = []

    magnets_for_plots = ElectromagnetArray3D(
        coil_radius=coil_radius,
        coil_length=coil_length,
        turns=1500,
        max_current=8.0,
        placement_radius=placement_radius,
        center=magnet_center,
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
            coil_radius=coil_radius,
            coil_length=coil_length,
            placement_radius=placement_radius,
            magnet_center=magnet_center,
        )

        results.append(result)

        print_summary(result)

        save_all_plots(
            result,
            output_dir,
            magnet_array=magnets_for_plots,
            network=network,
        )

    save_results_csv(results, output_dir)
    save_acbc_diagnostics(results, output_dir)

    print(f"\nResults saved to: {output_dir}/")

    return results


if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--vessel-file",
        default=DEFAULT_VESSEL_FILE,
        help=(
            "Vessel segment pickle to load. Defaults to the cm-corrected "
            "aorta geometry in data/."
        ),
    )

    parser.add_argument(
        "--magnet-clearance",
        type=float,
        default=DEFAULT_MAGNET_CLEARANCE,
        help="Extra clearance in meters between vessel tube bounds and magnet layout.",
    )

    parser.add_argument(
        "--placement-radius",
        type=float,
        default=None,
        help="Override computed magnet placement radius in meters.",
    )

    args = parser.parse_args()

    run_all_trials(
        vessel_file=args.vessel_file,
        magnet_clearance=args.magnet_clearance,
        placement_radius=args.placement_radius,
    )