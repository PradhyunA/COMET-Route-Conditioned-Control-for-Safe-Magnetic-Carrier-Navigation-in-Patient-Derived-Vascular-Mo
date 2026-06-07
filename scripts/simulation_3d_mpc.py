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

Main correction in this version:
    The magnetic device geometry is fixed and the low-level controller is
    tuned for smoother, more physically defensible motion.

Previous behavior:
    coil size was derived from the full vessel bounds.

Problem:
    This created huge coils such as:
        coil_radius ≈ 12 cm
        coil_length ≈ 48 cm

New fixed device:
    placement_radius = 6.0 cm
    coil_radius      = 1.8 cm
    coil_diameter    = 3.6 cm
    coil_length      = 7.0 cm
    turns            = 3000
    max_current      = 12 A

Optional:
    The simulated device can follow the particle using a local moving-frame
    model. This represents a local steering stage/gantry and prevents the
    compact fixed device from being unrealistically centered over the entire
    40 cm vascular path.
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

# ----------------------------------------------------------------------
# Fixed electromagnetic steering device
# ----------------------------------------------------------------------
# These values define the simulated steering hardware.
# They intentionally do NOT scale with vessel-tree size.
# ----------------------------------------------------------------------

FIXED_PLACEMENT_RADIUS = 0.060   # m, coil centers 6 cm from device center
FIXED_COIL_RADIUS = 0.018        # m, 3.6 cm coil diameter
FIXED_COIL_LENGTH = 0.070        # m, 7.0 cm coil/core length
FIXED_CORE_RADIUS = 0.008        # m, 1.6 cm core diameter
FIXED_TURNS = 3000.0

# Tuned down from 16 A.
# 16 A proved the architecture could move, but it produced aggressive motion,
# frequent saturation, high velocities, and Low-Re validity failures.
FIXED_MAX_CURRENT = 12.0         # A, simulated pulsed/peak current

# Effective steel-core model.
FIXED_CORE_MU_R_EFFECTIVE = 60.0
FIXED_CORE_SATURATION_T = 1.8

# If True, the compact steering array follows the particle.
# This is the best simulation model for a local steering stage.
# If False, the compact device stays centered at the route midpoint.
FOLLOW_PARTICLE_DEVICE = True

# Keep this CLI argument for compatibility, but it no longer controls coil size.
DEFAULT_MAGNET_CLEARANCE = 0.04


def magnet_setup_for_network(
    network,
    clearance: float = DEFAULT_MAGNET_CLEARANCE,
    placement_radius: float | None = None,
):
    """
    Build fixed-size magnetic device settings.

    The network is used only to compute a default anatomical center and bounds.
    Coil radius/length are NOT derived from the network bounds.
    """
    bounds_min, bounds_max = network.bounds(include_radius=True)
    center = 0.5 * (bounds_min + bounds_max)

    if placement_radius is None:
        placement_radius = FIXED_PLACEMENT_RADIUS

    placement_radius = float(placement_radius)

    return {
        "center": center,
        "placement_radius": placement_radius,
        "coil_radius": FIXED_COIL_RADIUS,
        "coil_length": FIXED_COIL_LENGTH,
        "core_radius": FIXED_CORE_RADIUS,
        "turns": FIXED_TURNS,
        "max_current": FIXED_MAX_CURRENT,
        "core_relative_permeability": FIXED_CORE_MU_R_EFFECTIVE,
        "core_saturation_tesla": FIXED_CORE_SATURATION_T,
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
                writer.writerow(
                    [
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
                    ]
                )

    print(f"ACBC diagnostics saved to: {path}")


def simulate_trial_mpc(
    start,
    target,
    network,
    trial_name: str = "trial",
    dt: float = 0.02,
    steps: int = 1600,
    tolerance: float = 0.003,
    coil_radius: float = FIXED_COIL_RADIUS,
    coil_length: float = FIXED_COIL_LENGTH,
    core_radius: float = FIXED_CORE_RADIUS,
    turns: float = FIXED_TURNS,
    max_current: float = FIXED_MAX_CURRENT,
    placement_radius: float = FIXED_PLACEMENT_RADIUS,
    magnet_center=None,
    core_relative_permeability: float = FIXED_CORE_MU_R_EFFECTIVE,
    core_saturation_tesla: float = FIXED_CORE_SATURATION_T,
    follow_particle_device: bool = FOLLOW_PARTICLE_DEVICE,
    profile: PhysicalProfile = SPION_CARRIER_10UM,
    max_substep_disp: float = 1e-3,
    rng_seed: int = 42,

    # ------------------------------------------------------------------
    # Smoother MPC weights
    # ------------------------------------------------------------------
    # Previous aggressive values:
    #     w_progress = 300
    #     w_align    = 20
    #     w_dist     = 0.5
    #     w_curr     = 1e-6
    #     w_smooth   = 1e-5
    #     w_edge     = 25
    #     w_overshoot= 1
    #
    # Those proved the system could move, but caused:
    #     high speeds,
    #     frequent current saturation,
    #     Low-Re validity failures,
    #     many optimizer non-convergence warnings.
    # ------------------------------------------------------------------

    w_progress: float = 170.0,
    w_align: float = 10.0,
    w_dist: float = 1.2,
    w_curr: float = 8e-5,
    w_smooth: float = 8e-4,
    w_edge: float = 120.0,
    w_overshoot: float = 6.0,
) -> TrialResult:
    workspace = WorkspaceVessel(network)
    flow = network

    start = workspace.clip(np.asarray(start, dtype=float))
    target = workspace.clip(np.asarray(target, dtype=float))

    # For a fixed compact device, route midpoint is a better static default
    # than the full vessel-tree center.
    route_midpoint = 0.5 * (start + target)

    if magnet_center is None:
        magnet_center = route_midpoint

    magnets = ElectromagnetArray3D(
        coil_radius=coil_radius,
        coil_length=coil_length,
        turns=turns,
        max_current=max_current,
        placement_radius=placement_radius,
        center=magnet_center,
        core_radius=core_radius,
        core_relative_permeability=core_relative_permeability,
        core_saturation_tesla=core_saturation_tesla,
    )

    particle = MagneticParticle3D(
        position=start,
        profile=profile,
        rng_seed=rng_seed,
    )

    # If using the local moving-frame device, begin centered around the particle.
    if follow_particle_device:
        magnets.recenter(particle.pos)

    navigator = network.waypoint_navigator(
        start,
        target,
        arrival_tol=0.004,
        waypoint_spacing=0.0015,
        lookahead=8,
        forward_search_points=35,
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
        absolute_wall_clearance_threshold=0.00075,
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

        # Local moving-frame steering stage.
        # This keeps the compact coil array close enough to generate useful
        # gradients along a long 40 cm anatomical path.
        if follow_particle_device:
            magnets.recenter(particle.pos)

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
                (
                    f"Step {step}: low-level current allocator did not converge; "
                    f"using best-so-far"
                ),
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

        acbc_log.append(
            {
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
            }
        )

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
                f"lookahead={getattr(navigator, 'lookahead_idx', 0) + 1}/"
                f"{navigator.total_hops}  "
                f"wp_dist={getattr(navigator, 'current_wp_dist', np.nan) * 1000:.1f} mm  "
                f"auth={acbc_info.selected_authority_margin:.2f}  "
                f"capturable={acbc_info.target_branch_capturable}  "
                f"cands={acbc_info.num_candidates}  "
                f"|I|max={np.max(np.abs(best_currents)):.3f} A  "
                f"|F|={np.linalg.norm(best_force):.3e} N  "
                f"|B|={np.linalg.norm(B):.3e} T  "
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
    low_level_convergence_rate = converged_count / max(1, len(step_time_log))

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
            "coil_diameter": 2.0 * coil_radius,
            "coil_length": coil_length,
            "core_radius": core_radius,
            "turns": turns,
            "max_current": max_current,
            "magnet_center": magnets.center.tolist(),
            "placement_radius": placement_radius,
            "core_relative_permeability": core_relative_permeability,
            "core_saturation_tesla": core_saturation_tesla,
            "follow_particle_device": follow_particle_device,
            "profile": profile.name,
            "max_substep_disp": max_substep_disp,
            "vessel_network": network.params_dict(),
            "low_level_convergence_rate": low_level_convergence_rate,
            "navigator_arrival_tol": getattr(navigator, "arrival_tol", None),
            "navigator_waypoint_spacing": getattr(navigator, "waypoint_spacing", None),
            "navigator_lookahead": getattr(navigator, "lookahead", None),
            "navigator_forward_search_points": getattr(
                navigator,
                "forward_search_points",
                None,
            ),
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
    core_radius = magnet_setup["core_radius"]
    turns = magnet_setup["turns"]
    max_current = magnet_setup["max_current"]
    core_relative_permeability = magnet_setup["core_relative_permeability"]
    core_saturation_tesla = magnet_setup["core_saturation_tesla"]

    # This is only the default static center.
    # For FOLLOW_PARTICLE_DEVICE=True, each trial recenters the device around
    # the particle during the control loop.
    magnet_center = magnet_setup["center"]

    print(
        "Fixed magnet setup: "
        f"default_center={np.round(magnet_center, 4)} m, "
        f"placement_radius={placement_radius:.3f} m, "
        f"coil_radius={coil_radius:.3f} m, "
        f"coil_diameter={2.0 * coil_radius:.3f} m, "
        f"coil_length={coil_length:.3f} m, "
        f"core_radius={core_radius:.3f} m, "
        f"turns={turns:.0f}, "
        f"max_current={max_current:.1f} A, "
        f"follow_particle_device={FOLLOW_PARTICLE_DEVICE}"
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
        turns=turns,
        max_current=max_current,
        placement_radius=placement_radius,
        center=magnet_center,
        core_radius=core_radius,
        core_relative_permeability=core_relative_permeability,
        core_saturation_tesla=core_saturation_tesla,
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
            core_radius=core_radius,
            turns=turns,
            max_current=max_current,
            placement_radius=placement_radius,
            magnet_center=magnet_center,
            core_relative_permeability=core_relative_permeability,
            core_saturation_tesla=core_saturation_tesla,
            follow_particle_device=FOLLOW_PARTICLE_DEVICE,
        )

        results.append(result)

        print_summary(result)

        # Plot using the default static device geometry.
        # If follow_particle_device=True, the trajectory/control data are still valid;
        # the plotted magnet geometry is representative rather than time-varying.
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
        help=(
            "Kept for compatibility. Fixed-device mode does not use this to "
            "scale coil dimensions."
        ),
    )

    parser.add_argument(
        "--placement-radius",
        type=float,
        default=None,
        help=(
            "Override fixed magnet placement radius in meters. "
            "Default is 0.060 m."
        ),
    )

    args = parser.parse_args()

    run_all_trials(
        vessel_file=args.vessel_file,
        magnet_clearance=args.magnet_clearance,
        placement_radius=args.placement_radius,
    )