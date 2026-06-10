"""
simulation_3d_acbc.py — v2
==========================

3D ACBC magnetic carrier steering — real vessel network edition.

ACBC high-level supervisor + continuous MPC current allocator.

Recommended current setting:
    profile = SPION_CARRIER_1MM

Important:
    A 1 mm carrier is NOT a nanoparticle. It should be described as a
    magnetic milliparticle / large-vessel magnetic carrier.

Device modes:
    moving_internal_upper_bound
    moving_linear_external_gantry
    moving_linear_enclosing_gantry
    moving_adaptive_enclosing_gantry   ← recommended / default
    fixed_enclosing_stress_test

Key settings:
    - Hardware-constrained body-scale scanner constants
    - 1 mm magnetic carrier by default
    - --flow-scale default 5.0, making max segment mean flow ~100 mm/s
    - Bidirectional current allocation assumed in controllers.py
"""

import os
import sys
import time
import warnings
import argparse
import csv

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

import numpy as np

from dynamag_simulation.core.profiles import (
    PhysicalProfile,
    SPION_CARRIER_1MM,
)
from dynamag_simulation.core.constants import DEFAULT_TARGET_TOLERANCE
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
# Hardware-constrained body-scale scanner parameters
# ----------------------------------------------------------------------
# These are intentionally much more plausible than the old:
#     20,000 turns / 60 A / 6 T saturation
#
# They match a strong, MRI-like research-scanner authority benchmark:
#     - large body-scale placement radius
#     - hundreds of turns, not tens of thousands
#     - moderate/high peak current
#     - realistic soft core saturation
#     - hardware limits handled inside magnets.py
# ----------------------------------------------------------------------

FIXED_PLACEMENT_RADIUS = 0.300       # m
FIXED_COIL_RADIUS = 0.070            # m
FIXED_COIL_LENGTH = 0.160            # m
FIXED_CORE_RADIUS = 0.030            # m
FIXED_TURNS = 600.0
FIXED_MAX_CURRENT = 16.0             # A peak command limit
FIXED_CORE_MU_R_EFFECTIVE = 70.0
FIXED_CORE_SATURATION_T = 1.8        # T

VALID_DEVICE_MODES = {
    "moving_internal_upper_bound",
    "moving_linear_external_gantry",
    "moving_linear_enclosing_gantry",
    "moving_adaptive_enclosing_gantry",
    "fixed_enclosing_stress_test",
}

DEVICE_MODE = "moving_adaptive_enclosing_gantry"
GANTRY_STANDOFF = 0.180
MAX_LATERAL_SHIFT = 0.120
LATERAL_GAIN = 0.85
DEFAULT_MAGNET_CLEARANCE = 0.04

# flow_scale=5 means current baseline max v_mean 20 mm/s becomes ~100 mm/s
DEFAULT_FLOW_SCALE = 1.0


def magnet_setup_for_network(
    network,
    clearance: float = DEFAULT_MAGNET_CLEARANCE,
    placement_radius: float | None = None,
):
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


def _unit_vector(v, fallback=None):
    v = np.asarray(v, dtype=float)
    n = float(np.linalg.norm(v))

    if n < 1e-12:
        if fallback is None:
            return np.array([0.0, 0.0, 1.0], dtype=float)
        return _unit_vector(np.asarray(fallback, dtype=float))

    return v / n


def _external_normal_for_axis(axis):
    axis = _unit_vector(axis)
    preferred = np.array([0.0, 1.0, 0.0], dtype=float)
    normal = preferred - np.dot(preferred, axis) * axis

    if np.linalg.norm(normal) < 1e-6:
        preferred = np.array([1.0, 0.0, 0.0], dtype=float)
        normal = preferred - np.dot(preferred, axis) * axis

    return _unit_vector(normal, fallback=np.array([1.0, 0.0, 0.0]))


def _route_axis_projection(particle_pos, start, target):
    particle_pos = np.asarray(particle_pos, dtype=float)
    start = np.asarray(start, dtype=float)
    target = np.asarray(target, dtype=float)

    route_axis = _unit_vector(target - start)
    route_midpoint = 0.5 * (start + target)

    s_start = float(np.dot(start - route_midpoint, route_axis))
    s_target = float(np.dot(target - route_midpoint, route_axis))
    s_min = min(s_start, s_target)
    s_max = max(s_start, s_target)

    s_particle = float(np.dot(particle_pos - route_midpoint, route_axis))
    s_particle = float(np.clip(s_particle, s_min, s_max))

    axis_point = route_midpoint + s_particle * route_axis

    return axis_point, route_axis


def _moving_linear_external_gantry_center(particle_pos, start, target, standoff):
    axis_point, route_axis = _route_axis_projection(
        particle_pos=particle_pos,
        start=start,
        target=target,
    )
    external_normal = _external_normal_for_axis(route_axis)
    gantry_center = axis_point + float(standoff) * external_normal
    return gantry_center, axis_point, external_normal, route_axis


def _moving_linear_enclosing_gantry_center(particle_pos, start, target):
    axis_point, route_axis = _route_axis_projection(
        particle_pos=particle_pos,
        start=start,
        target=target,
    )
    gantry_center = axis_point.copy()
    return gantry_center, axis_point, route_axis


def _moving_adaptive_enclosing_gantry_center(
    particle_pos,
    start,
    target,
    max_lateral_shift,
    lateral_gain,
):
    particle_pos = np.asarray(particle_pos, dtype=float)
    axis_point, route_axis = _route_axis_projection(
        particle_pos=particle_pos,
        start=start,
        target=target,
    )

    lateral_error = particle_pos - axis_point
    lateral_error = lateral_error - np.dot(lateral_error, route_axis) * route_axis

    desired_shift = float(lateral_gain) * lateral_error
    shift_norm = float(np.linalg.norm(desired_shift))
    max_shift = max(0.0, float(max_lateral_shift))

    if shift_norm > max_shift and shift_norm > 1e-12:
        lateral_shift = desired_shift * (max_shift / shift_norm)
    else:
        lateral_shift = desired_shift

    gantry_center = axis_point + lateral_shift

    return gantry_center, axis_point, lateral_shift, route_axis


def _initial_magnet_center_for_mode(device_mode, start, target, network, gantry_standoff):
    if device_mode == "moving_internal_upper_bound":
        return np.asarray(start, dtype=float)

    if device_mode == "moving_linear_external_gantry":
        center, _, _, _ = _moving_linear_external_gantry_center(
            particle_pos=start,
            start=start,
            target=target,
            standoff=gantry_standoff,
        )
        return center

    if device_mode in {
        "moving_linear_enclosing_gantry",
        "moving_adaptive_enclosing_gantry",
    }:
        center, _, _ = _moving_linear_enclosing_gantry_center(
            particle_pos=start,
            start=start,
            target=target,
        )
        return center

    if device_mode == "fixed_enclosing_stress_test":
        bounds_min, bounds_max = network.bounds(include_radius=True)
        return 0.5 * (bounds_min + bounds_max)

    raise ValueError(f"Unknown device_mode: {device_mode}")


def _apply_device_motion(
    magnets,
    device_mode,
    particle_pos,
    start,
    target,
    network,
    gantry_standoff,
    max_lateral_shift=MAX_LATERAL_SHIFT,
    lateral_gain=LATERAL_GAIN,
):
    particle_pos = np.asarray(particle_pos, dtype=float)

    if device_mode == "moving_internal_upper_bound":
        magnets.recenter(particle_pos)
        return {
            "gantry_center": magnets.center.copy(),
            "gantry_axis_point": particle_pos.copy(),
            "gantry_external_normal": np.array([np.nan, np.nan, np.nan]),
            "gantry_route_axis": _unit_vector(target - start),
            "gantry_particle_dist": 0.0,
            "gantry_axis_dist": 0.0,
            "gantry_lateral_shift": np.zeros(3, dtype=float),
        }

    if device_mode == "moving_linear_external_gantry":
        center, axis_point, external_normal, route_axis = (
            _moving_linear_external_gantry_center(
                particle_pos=particle_pos,
                start=start,
                target=target,
                standoff=gantry_standoff,
            )
        )
        magnets.recenter(center)
        return {
            "gantry_center": center.copy(),
            "gantry_axis_point": axis_point.copy(),
            "gantry_external_normal": external_normal.copy(),
            "gantry_route_axis": route_axis.copy(),
            "gantry_particle_dist": float(np.linalg.norm(center - particle_pos)),
            "gantry_axis_dist": float(np.linalg.norm(center - axis_point)),
            "gantry_lateral_shift": np.zeros(3, dtype=float),
        }

    if device_mode == "moving_linear_enclosing_gantry":
        center, axis_point, route_axis = _moving_linear_enclosing_gantry_center(
            particle_pos=particle_pos,
            start=start,
            target=target,
        )
        magnets.recenter(center)
        return {
            "gantry_center": center.copy(),
            "gantry_axis_point": axis_point.copy(),
            "gantry_external_normal": np.array([np.nan, np.nan, np.nan]),
            "gantry_route_axis": route_axis.copy(),
            "gantry_particle_dist": float(np.linalg.norm(center - particle_pos)),
            "gantry_axis_dist": 0.0,
            "gantry_lateral_shift": np.zeros(3, dtype=float),
        }

    if device_mode == "moving_adaptive_enclosing_gantry":
        center, axis_point, lateral_shift, route_axis = (
            _moving_adaptive_enclosing_gantry_center(
                particle_pos=particle_pos,
                start=start,
                target=target,
                max_lateral_shift=max_lateral_shift,
                lateral_gain=lateral_gain,
            )
        )
        magnets.recenter(center)
        return {
            "gantry_center": center.copy(),
            "gantry_axis_point": axis_point.copy(),
            "gantry_external_normal": np.array([np.nan, np.nan, np.nan]),
            "gantry_route_axis": route_axis.copy(),
            "gantry_particle_dist": float(np.linalg.norm(center - particle_pos)),
            "gantry_axis_dist": float(np.linalg.norm(center - axis_point)),
            "gantry_lateral_shift": lateral_shift.copy(),
        }

    if device_mode == "fixed_enclosing_stress_test":
        bounds_min, bounds_max = network.bounds(include_radius=True)
        axis_point = 0.5 * (bounds_min + bounds_max)
        return {
            "gantry_center": magnets.center.copy(),
            "gantry_axis_point": axis_point.copy(),
            "gantry_external_normal": np.array([np.nan, np.nan, np.nan]),
            "gantry_route_axis": _unit_vector(target - start),
            "gantry_particle_dist": float(np.linalg.norm(magnets.center - particle_pos)),
            "gantry_axis_dist": float(np.linalg.norm(magnets.center - axis_point)),
            "gantry_lateral_shift": np.zeros(3, dtype=float),
        }

    raise ValueError(f"Unknown device_mode: {device_mode}")


def _vec_or_nan(v):
    if v is None:
        return [np.nan, np.nan, np.nan]

    v = np.asarray(v, dtype=float)

    if v.shape[0] < 3:
        return [np.nan, np.nan, np.nan]

    return [float(v[0]), float(v[1]), float(v[2])]


def _set_currents_with_optional_thermal(magnets, currents, dt=None):
    """
    Compatibility helper.

    New hardware-constrained magnets.py accepts set_currents(currents, dt=...).
    Older magnets.py accepts only set_currents(currents).
    """
    try:
        magnets.set_currents(currents, dt=dt)
    except TypeError:
        magnets.set_currents(currents)


def save_acbc_diagnostics(results, output_dir, filename="acbc_diagnostics.csv"):
    path = os.path.join(output_dir, filename)

    header = [
        "trial_name", "step", "time_s", "state", "previous_state", "transition",
        "near_bifurcation", "bifurcation_distance_m",
        "selected_branch_seg_id", "selected_authority_margin",
        "selected_score", "target_branch_capturable", "num_candidates",
        "best_route_idx", "stagnant_steps",
        "wall_clearance_m", "wall_fraction",
        "control_target_x", "control_target_y", "control_target_z",
        "nominal_target_x", "nominal_target_y", "nominal_target_z",
        "recovery_target_x", "recovery_target_y", "recovery_target_z",
        "escape_target_x", "escape_target_y", "escape_target_z",
        "gantry_center_x", "gantry_center_y", "gantry_center_z",
        "gantry_axis_point_x", "gantry_axis_point_y", "gantry_axis_point_z",
        "gantry_particle_dist_m", "gantry_axis_dist_m",
        "gantry_lateral_shift_x", "gantry_lateral_shift_y", "gantry_lateral_shift_z",
    ]

    with open(path, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(header)

        for result in results:
            log = getattr(result, "acbc_log", [])

            for row in log:
                writer.writerow([
                    result.trial_name,
                    row["step"], row["time_s"],
                    row["state"], row["previous_state"], row["transition"],
                    row["near_bifurcation"], row["bifurcation_distance_m"],
                    row["selected_branch_seg_id"], row["selected_authority_margin"],
                    row["selected_score"], row["target_branch_capturable"],
                    row["num_candidates"],
                    row["best_route_idx"], row["stagnant_steps"],
                    row["wall_clearance_m"], row["wall_fraction"],
                    *row["control_target"],
                    *row["nominal_target"],
                    *row["recovery_target"],
                    *row["escape_target"],
                    *row["gantry_center"],
                    *row["gantry_axis_point"],
                    row["gantry_particle_dist_m"], row["gantry_axis_dist_m"],
                    *row["gantry_lateral_shift"],
                ])

    print(f"ACBC diagnostics saved to: {path}")


def simulate_trial_acbc(
    start,
    target,
    network,
    trial_name: str = "trial",
    dt: float = 0.02,
    steps: int = 4000,
    tolerance: float = DEFAULT_TARGET_TOLERANCE,
    coil_radius: float = FIXED_COIL_RADIUS,
    coil_length: float = FIXED_COIL_LENGTH,
    core_radius: float = FIXED_CORE_RADIUS,
    turns: float = FIXED_TURNS,
    max_current: float = FIXED_MAX_CURRENT,
    placement_radius: float = FIXED_PLACEMENT_RADIUS,
    magnet_center=None,
    core_relative_permeability: float = FIXED_CORE_MU_R_EFFECTIVE,
    core_saturation_tesla: float = FIXED_CORE_SATURATION_T,
    device_mode: str = DEVICE_MODE,
    gantry_standoff: float = GANTRY_STANDOFF,
    max_lateral_shift: float = MAX_LATERAL_SHIFT,
    lateral_gain: float = LATERAL_GAIN,
    profile: PhysicalProfile = SPION_CARRIER_1MM,
    max_substep_disp: float = 1e-3,
    rng_seed: int = 42,
    wall_clearance_threshold_m: float = 0.0007,
    # MPC weights — aggressive authority, moderate route-tangent
    w_progress: float = 450.0,
    w_align: float = 35.0,
    w_dist: float = 2.0,
    w_curr: float = 1e-7,
    w_smooth: float = 1e-6,
    w_edge: float = 250.0,
    w_overshoot: float = 8.0,
    w_route: float = 15.0,
    enable_branch_commitment: bool = True,
    enable_authority_assessment: bool = True,
    enable_recovery: bool = True,
    enable_stall_escape: bool = True,
) -> TrialResult:
    if device_mode not in VALID_DEVICE_MODES:
        raise ValueError(
            f"device_mode must be one of {sorted(VALID_DEVICE_MODES)}, "
            f"got {device_mode}"
        )

    workspace = WorkspaceVessel(network)
    flow = network

    start = workspace.clip(np.asarray(start, dtype=float))
    target = workspace.clip(np.asarray(target, dtype=float))

    if magnet_center is None:
        magnet_center = _initial_magnet_center_for_mode(
            device_mode=device_mode,
            start=start,
            target=target,
            network=network,
            gantry_standoff=gantry_standoff,
        )

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

    initial_device_diag = _apply_device_motion(
        magnets=magnets,
        device_mode=device_mode,
        particle_pos=particle.pos,
        start=start,
        target=target,
        network=network,
        gantry_standoff=gantry_standoff,
        max_lateral_shift=max_lateral_shift,
        lateral_gain=lateral_gain,
    )

    navigator = network.waypoint_navigator(
        start,
        target,
        arrival_tol=DEFAULT_TARGET_TOLERANCE,
        waypoint_spacing=0.0015,
        lookahead=5,
        forward_search_points=50,
    )

    acbc = ACBCController(
        network=network,
        magnet_array=magnets,
        workspace=workspace,
        max_current=max_current,

        branch_detection_radius=None,
        branch_lookahead=None,
        authority_threshold=1.02,
        recovery_threshold=0.65,

        wall_clearance_threshold_m=wall_clearance_threshold_m,
        min_wall_safety_fraction=0.0,
        absolute_wall_clearance_threshold=wall_clearance_threshold_m,
        surface_clearance_target_m=0.0,
        recovery_enter_wall_clearance=wall_clearance_threshold_m,
        recovery_exit_wall_clearance=wall_clearance_threshold_m + 0.00020,
        near_wall_edge_multiplier=6.0,
        stall_escape_min_wall_clearance=wall_clearance_threshold_m,
        stall_escape_min_wall_fraction_multiplier=1.00,

        recovery_center_fraction=0.55,
        recovery_route_fraction=0.10,
        recovery_route_tangent_steps=3,
        max_recovery_steps=40,
        recovery_release_clearance=wall_clearance_threshold_m,

        stall_window_steps=15,
        stall_escape_lookahead=12,
        stall_escape_release_hops=3,
        stall_escape_max_steps=80,

        recovery_escape_lockout_steps=0,
        enable_wall_safety_current_filter=False,
        enable_branch_commitment=enable_branch_commitment,
        enable_authority_assessment=enable_authority_assessment,
        enable_recovery=enable_recovery,
        enable_stall_escape=enable_stall_escape,
    )

    print(
        f"    Profile: {profile.name}, "
        f"diameter={profile.diameter * 1000:.3f} mm"
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

    print(
        "    Device motion: "
        f"mode={device_mode}, "
        f"placement_radius={placement_radius * 1000:.1f} mm, "
        f"initial_center={np.round(initial_device_diag['gantry_center'] * 1000, 1)} mm"
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
    gantry_center_log = []
    gantry_particle_dist_log = []
    gantry_axis_dist_log = []
    gantry_lateral_shift_log = []
    acbc_log = []

    reached = False
    prev_currents = np.zeros(magnets.n_magnets, dtype=float)
    wall_start = time.perf_counter()
    converged_count = 0

    for step in range(steps):
        sim_time = step * dt
        t0 = time.perf_counter()

        device_diag = _apply_device_motion(
            magnets=magnets,
            device_mode=device_mode,
            particle_pos=particle.pos,
            start=start,
            target=target,
            network=network,
            gantry_standoff=gantry_standoff,
            max_lateral_shift=max_lateral_shift,
            lateral_gain=lateral_gain,
        )

        nav_waypoints = getattr(navigator, "waypoints", None)
        route_tangent = None

        if nav_waypoints is not None and len(nav_waypoints) > 1:
            nav_idx = getattr(navigator, "current_idx", 0)
            tidx = min(nav_idx, len(nav_waypoints) - 1)
            tidx_next = min(tidx + 3, len(nav_waypoints) - 1)

            if tidx_next > tidx:
                rt = (
                    np.asarray(nav_waypoints[tidx_next], dtype=float)
                    - np.asarray(nav_waypoints[tidx], dtype=float)
                )
                rtn = np.linalg.norm(rt)
                if rtn > 1e-10:
                    route_tangent = rt / rtn

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
            w_route=w_route,
            route_tangent=route_tangent,
        )

        t1 = time.perf_counter()

        if converged:
            converged_count += 1
        else:
            warnings.warn(
                f"Step {step}: low-level current allocator did not converge; "
                f"using best-so-far",
                RuntimeWarning,
                stacklevel=1,
            )

        prev_currents = best_currents.copy()

        _set_currents_with_optional_thermal(magnets, best_currents, dt=dt)
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
        gantry_center_log.append(device_diag["gantry_center"].copy())
        gantry_particle_dist_log.append(device_diag["gantry_particle_dist"])
        gantry_axis_dist_log.append(device_diag["gantry_axis_dist"])
        gantry_lateral_shift_log.append(device_diag["gantry_lateral_shift"].copy())

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
            "best_route_idx": acbc_info.best_route_idx,
            "stagnant_steps": acbc_info.stagnant_steps,
            "wall_clearance_m": acbc_info.wall_clearance_m,
            "wall_fraction": acbc_info.wall_fraction,
            "control_target": _vec_or_nan(acbc_info.control_target),
            "nominal_target": _vec_or_nan(acbc_info.nominal_target),
            "recovery_target": _vec_or_nan(acbc_info.recovery_target),
            "escape_target": _vec_or_nan(acbc_info.stall_escape_target),
            "gantry_center": _vec_or_nan(device_diag["gantry_center"]),
            "gantry_axis_point": _vec_or_nan(device_diag["gantry_axis_point"]),
            "gantry_particle_dist_m": float(device_diag["gantry_particle_dist"]),
            "gantry_axis_dist_m": float(device_diag["gantry_axis_dist"]),
            "gantry_lateral_shift": _vec_or_nan(device_diag["gantry_lateral_shift"]),
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
                _, _, radial_dist, _ = seg.closest_point_and_radial(particle.pos)
                wall_clearance = seg.radius - radial_dist
            else:
                wall_clearance = np.nan

            lateral_shift_norm = np.linalg.norm(device_diag["gantry_lateral_shift"])
            nav_curr_idx = getattr(navigator, "current_idx", 0)
            nav_best_idx = getattr(navigator, "best_idx", 0)

            print(
                f"  step {step + 1:>4d}/{steps}  "
                f"state={acbc_info.state:<20s}  "
                f"dist={distance * 1000:.1f}mm  "
                f"wp={nav_curr_idx + 1}/{navigator.total_hops}  "
                f"best_wp={nav_best_idx + 1}/{navigator.total_hops}  "
                f"acbc_best={acbc_info.best_route_idx + 1}/{navigator.total_hops}  "
                f"stag={acbc_info.stagnant_steps}  "
                f"wall={wall_clearance * 1000:.2f}mm  "
                f"wall_frac={acbc_info.wall_fraction:.3f}  "
                f"|F|={np.linalg.norm(best_force):.2e}N  "
                f"|B|={np.linalg.norm(B):.2e}T  "
                f"|I|={np.max(np.abs(best_currents)):.1f}A  "
                f"v={particle_speed * 1000:.2f}mm/s  "
                f"flow={flow_speed * 1000:.2f}mm/s  "
                f"Re={re:.2e}  "
                f"lat={lateral_shift_norm * 1000:.1f}mm"
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
    gantry_center_log = np.array(gantry_center_log)
    gantry_particle_dist_log = np.array(gantry_particle_dist_log)
    gantry_axis_dist_log = np.array(gantry_axis_dist_log)
    gantry_lateral_shift_log = np.array(gantry_lateral_shift_log)

    re_stats = check_low_re_validity(reynolds_log)
    acbc_summary = acbc.summary_dict()
    low_level_convergence_rate = converged_count / max(1, len(step_time_log))

    hardware_diag = {}
    if hasattr(magnets, "hardware_diagnostics"):
        hardware_diag = magnets.hardware_diagnostics()

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
        if len(current_log) > 0 else 0.0,
        mean_reynolds=re_stats["mean_reynolds"],
        max_reynolds=re_stats["max_reynolds"],
        low_re_valid=re_stats["low_re_valid"],
        stokes_regime=re_stats["stokes_regime"],
        finite_re_drag_valid=re_stats["finite_re_drag_valid"],
        drag_model=re_stats["drag_model"],
        drag_regime_label=re_stats["drag_regime_label"],
        wall_time_s=wall_end - wall_start,
        avg_control_step_time_s=float(np.mean(step_time_log))
        if len(step_time_log) > 0 else 0.0,
        max_control_step_time_s=float(np.max(step_time_log))
        if len(step_time_log) > 0 else 0.0,
        seed=rng_seed,
        profile_name=profile.name,
        dt=dt,
        workspace_half=network.half,
        magnets=magnets.positions.copy(),
        orientations=magnets.orientations.copy(),
        params={
            "controller": "ACBC_v2",
            "low_level_allocator": "continuous_mpc_controller_v2",
            "device_mode": device_mode,
            "dt": dt,
            "steps": steps,
            "tolerance": tolerance,
            "coil_radius": coil_radius,
            "coil_diameter": 2.0 * coil_radius,
            "coil_length": coil_length,
            "core_radius": core_radius,
            "turns": turns,
            "max_current": max_current,
            "magnet_center_final": magnets.center.tolist(),
            "placement_radius": placement_radius,
            "gantry_standoff_m": gantry_standoff,
            "max_lateral_shift_m": max_lateral_shift,
            "lateral_gain": lateral_gain,
            "gantry_mean_particle_dist_m": float(np.mean(gantry_particle_dist_log))
            if len(gantry_particle_dist_log) > 0 else np.nan,
            "gantry_mean_axis_dist_m": float(np.mean(gantry_axis_dist_log))
            if len(gantry_axis_dist_log) > 0 else np.nan,
            "gantry_mean_lateral_shift_m": float(
                np.mean(np.linalg.norm(gantry_lateral_shift_log, axis=1))
            ) if len(gantry_lateral_shift_log) > 0 else np.nan,
            "gantry_max_lateral_shift_m": float(
                np.max(np.linalg.norm(gantry_lateral_shift_log, axis=1))
            ) if len(gantry_lateral_shift_log) > 0 else np.nan,
            "core_relative_permeability": core_relative_permeability,
            "core_saturation_tesla": core_saturation_tesla,
            "follow_particle_device": device_mode == "moving_internal_upper_bound",
            "profile": profile.name,
            "particle_diameter_m": profile.diameter,
            "particle_diameter_mm": profile.diameter * 1000.0,
            "max_substep_disp": max_substep_disp,
            "wall_clearance_threshold_m": wall_clearance_threshold_m,
            "vessel_network": network.params_dict(),
            "low_level_convergence_rate": low_level_convergence_rate,
            "navigator_arrival_tol": getattr(navigator, "arrival_tol", None),
            "navigator_waypoint_spacing": getattr(navigator, "waypoint_spacing", None),
            "navigator_lookahead": getattr(navigator, "lookahead", None),
            "navigator_forward_search_points": getattr(
                navigator, "forward_search_points", None,
            ),
            "w_route": w_route,
            "enable_branch_commitment": enable_branch_commitment,
            "enable_authority_assessment": enable_authority_assessment,
            "enable_recovery": enable_recovery,
            "enable_stall_escape": enable_stall_escape,
            **hardware_diag,
            **acbc_summary,
        },
    )

    result.acbc_log = acbc_log
    result.gantry_center_log = gantry_center_log
    result.gantry_particle_dist_log = gantry_particle_dist_log
    result.gantry_axis_dist_log = gantry_axis_dist_log
    result.gantry_lateral_shift_log = gantry_lateral_shift_log

    print(
        f"  ACBC summary: transitions={acbc_summary['acbc_transitions']}, "
        f"commitments={acbc_summary['acbc_commitments']}, "
        f"recoveries={acbc_summary['acbc_recoveries']}, "
        f"escapes={acbc_summary['acbc_stall_escapes']}, "
        f"low-level convergence={low_level_convergence_rate:.2%}"
    )

    if hardware_diag:
        print(
            f"  Hardware: max_actual_I={hardware_diag.get('max_actual_current_A', np.nan):.2f}A, "
            f"max_V={hardware_diag.get('max_voltage_V', np.nan):.1f}V, "
            f"max_P={hardware_diag.get('max_power_W', np.nan):.1f}W, "
            f"max_temp={hardware_diag.get('max_coil_temp_C', np.nan):.1f}C"
        )

    if len(gantry_axis_dist_log) > 0:
        lateral_norms = np.linalg.norm(gantry_lateral_shift_log, axis=1)
        print(
            f"  Gantry: mean_particle_dist={np.mean(gantry_particle_dist_log) * 1000:.1f}mm, "
            f"max_lateral={np.max(lateral_norms) * 1000:.1f}mm"
        )

    return result


def run_all_trials(
    vessel_file: str = DEFAULT_VESSEL_FILE,
    magnet_clearance: float = DEFAULT_MAGNET_CLEARANCE,
    placement_radius: float | None = None,
    device_mode: str = DEVICE_MODE,
    gantry_standoff: float = GANTRY_STANDOFF,
    max_lateral_shift: float = MAX_LATERAL_SHIFT,
    lateral_gain: float = LATERAL_GAIN,
    only_trial: str | None = None,
    flow_scale: float = DEFAULT_FLOW_SCALE,
):
    if device_mode not in VALID_DEVICE_MODES:
        raise ValueError(
            f"device_mode must be one of {sorted(VALID_DEVICE_MODES)}, "
            f"got {device_mode}"
        )

    network = RealVesselNetwork(vessel_file, flow_scale=flow_scale)
    print(network.segment_summary())
    print(f"Flow scale: {flow_scale:.2f}x")

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
    anatomical_center = magnet_setup["center"]

    print(
        f"Magnet: mode={device_mode}, placement_r={placement_radius:.3f}m, "
        f"coil_r={coil_radius:.3f}m, turns={turns:.0f}, "
        f"max_I={max_current:.1f}A, "
        f"mu_r={core_relative_permeability:.0f}, "
        f"B_sat={core_saturation_tesla:.1f}T"
    )

    rng = np.random.default_rng(42)
    inlet = network.inlet_start()
    outlet = network.outlet_end()
    mid = network.random_point_in_vessel(rng)

    trials = {
        "trial_1_inlet_to_outlet": {"start": inlet, "target": outlet},
        "trial_2_mid_to_outlet": {"start": mid, "target": outlet},
        "trial_3_inlet_to_mid": {
            "start": inlet,
            "target": network.random_point_in_vessel(rng),
        },
    }

    if only_trial is not None:
        if only_trial not in trials:
            raise ValueError(
                f"--only-trial must be one of {sorted(trials)}, got {only_trial}"
            )
        trials = {only_trial: trials[only_trial]}

    output_dirs = {
        "moving_linear_external_gantry":
            "outputs/simulation_outputs_3d_acbc_external_gantry",
        "moving_linear_enclosing_gantry":
            "outputs/simulation_outputs_3d_acbc_enclosing_gantry",
        "moving_adaptive_enclosing_gantry":
            "outputs/simulation_outputs_3d_acbc_adaptive_enclosing_gantry",
        "moving_internal_upper_bound":
            "outputs/simulation_outputs_3d_acbc_moving_internal",
        "fixed_enclosing_stress_test":
            "outputs/simulation_outputs_3d_acbc_fixed_stress",
    }

    output_dir = output_dirs[device_mode]
    os.makedirs(output_dir, exist_ok=True)

    results = []

    if device_mode == "fixed_enclosing_stress_test":
        plot_center = anatomical_center
    else:
        first_trial_cfg = next(iter(trials.values()))
        plot_center = _initial_magnet_center_for_mode(
            device_mode=device_mode,
            start=first_trial_cfg["start"],
            target=first_trial_cfg["target"],
            network=network,
            gantry_standoff=gantry_standoff,
        )

    magnets_for_plots = ElectromagnetArray3D(
        coil_radius=coil_radius,
        coil_length=coil_length,
        turns=turns,
        max_current=max_current,
        placement_radius=placement_radius,
        center=plot_center,
        core_radius=core_radius,
        core_relative_permeability=core_relative_permeability,
        core_saturation_tesla=core_saturation_tesla,
    )

    for trial_name, cfg in trials.items():
        print(f"\n>>> Running {trial_name}")
        print(f"    start:  {cfg['start']}")
        print(f"    target: {cfg['target']}")

        result = simulate_trial_acbc(
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
            magnet_center=None,
            core_relative_permeability=core_relative_permeability,
            core_saturation_tesla=core_saturation_tesla,
            device_mode=device_mode,
            gantry_standoff=gantry_standoff,
            max_lateral_shift=max_lateral_shift,
            lateral_gain=lateral_gain,
            profile=SPION_CARRIER_1MM,
        )

        result.params["flow_scale"] = flow_scale

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

    parser.add_argument("--vessel-file", default=DEFAULT_VESSEL_FILE)
    parser.add_argument("--magnet-clearance", type=float, default=DEFAULT_MAGNET_CLEARANCE)
    parser.add_argument("--placement-radius", type=float, default=None)
    parser.add_argument(
        "--device-mode",
        default=DEVICE_MODE,
        choices=sorted(VALID_DEVICE_MODES),
    )
    parser.add_argument("--gantry-standoff", type=float, default=GANTRY_STANDOFF)
    parser.add_argument("--max-lateral-shift", type=float, default=MAX_LATERAL_SHIFT)
    parser.add_argument("--lateral-gain", type=float, default=LATERAL_GAIN)
    parser.add_argument("--flow-scale", type=float, default=DEFAULT_FLOW_SCALE)
    parser.add_argument(
        "--only-trial",
        default=None,
        choices=[
            "trial_1_inlet_to_outlet",
            "trial_2_mid_to_outlet",
            "trial_3_inlet_to_mid",
        ],
    )

    args = parser.parse_args()

    run_all_trials(
        vessel_file=args.vessel_file,
        magnet_clearance=args.magnet_clearance,
        placement_radius=args.placement_radius,
        device_mode=args.device_mode,
        gantry_standoff=args.gantry_standoff,
        max_lateral_shift=args.max_lateral_shift,
        lateral_gain=args.lateral_gain,
        only_trial=args.only_trial,
        flow_scale=args.flow_scale,
    )
