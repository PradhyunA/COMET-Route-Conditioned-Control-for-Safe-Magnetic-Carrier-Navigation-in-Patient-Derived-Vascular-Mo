"""
simulation_3d_mpc.py
====================

3D baseline MPC magnetic carrier steering — real vessel network edition.

Purpose:
    This script is the fair baseline controller for comparison against ACBC.

Important:
    This file intentionally does NOT include ACBC supervisory logic:
        - no branch commitment state
        - no recovery mode
        - no stall escape state
        - no ACBC authority-margin decision layer

    It uses the same physical setup as the ACBC script:
        - SPION_CARRIER_1MM
        - realistic body-scale scanner constants
        - same magnetic array geometry
        - same flow_scale command-line option
        - same trial definitions
        - same final-target tolerance

Baseline behavior:
    MPC tracks local route waypoints using continuous_mpc_controller().
    Route-resolution arguments are exposed so MPC and COMET can be
    compared under the same waypoint discretization.
    ACBC should be compared against this under the same physics.

Recommended run:
    python -u scripts/simulation_3d_mpc.py \
      --only-trial trial_2_mid_to_outlet \
      --flow-scale 5 \
      --waypoint-spacing 0.002 \
      --waypoint-arrival-tol 0.0015 \
      --forward-search-points 25 \
      2>&1 | tee saved_runs/MPC_route_resolution_2mm_flow5x.log
"""

import os
import sys
import time
import warnings
import argparse

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

import numpy as np

from dynamag_simulation.core.constants import DEFAULT_TARGET_TOLERANCE
from dynamag_simulation.core.profiles import PhysicalProfile, SPION_CARRIER_1MM
from dynamag_simulation.core.magnets import ElectromagnetArray3D
from dynamag_simulation.core.particle import MagneticParticle3D
from dynamag_simulation.core.integrator import adaptive_step
from dynamag_simulation.core.controllers import continuous_mpc_controller
from dynamag_simulation.core.validation import compute_reynolds, check_low_re_validity
from dynamag_simulation.core.reporting import TrialResult, print_summary, save_results_csv
from dynamag_simulation.core.plotting import save_all_plots
from dynamag_simulation.core.real_vessel_network import RealVesselNetwork
from dynamag_simulation.core.workspace import WorkspaceVessel


DEFAULT_VESSEL_FILE = "real_vessel_segments_cm_corrected.pkl"

# ----------------------------------------------------------------------
# Hardware-constrained body-scale scanner parameters
# ----------------------------------------------------------------------
# These should match the ACBC script for fair comparison.
# They are intentionally more plausible than the old:
#     20,000 turns / 60 A / 6 T saturation
# ----------------------------------------------------------------------

FIXED_PLACEMENT_RADIUS = 0.300       # m
FIXED_COIL_RADIUS = 0.070            # m
FIXED_COIL_LENGTH = 0.160            # m
FIXED_CORE_RADIUS = 0.030            # m
FIXED_TURNS = 600.0
FIXED_MAX_CURRENT = 16.0             # A peak command limit
FIXED_CORE_MU_R_EFFECTIVE = 70.0
FIXED_CORE_SATURATION_T = 1.8        # T

DEFAULT_MAGNET_CLEARANCE = 0.04

# flow_scale=5 makes baseline max v_mean 20 mm/s become ~100 mm/s.
DEFAULT_FLOW_SCALE = 5.0

# Fair route-resolution defaults used for route-discretization sensitivity.
# These are intentionally coarser than the old 1.0 mm / 50-search setting,
# but still give every controller the same planned centerline route.
DEFAULT_WAYPOINT_SPACING = 0.002          # 2.0 mm
DEFAULT_WAYPOINT_ARRIVAL_TOL = 0.0015    # 1.5 mm
DEFAULT_MPC_WAYPOINT_LOOKAHEAD = 1
DEFAULT_FORWARD_SEARCH_POINTS = 25
DEFAULT_EARLY_STOP_WINDOW_STEPS = 200
DEFAULT_EARLY_STOP_IMPROVEMENT_THRESHOLD_M = 0.002

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


# ----------------------------------------------------------------------
# Shared gantry/device motion helpers
# ----------------------------------------------------------------------

def magnet_setup_for_network(
    network,
    clearance: float = DEFAULT_MAGNET_CLEARANCE,
    placement_radius: float | None = None,
):
    """
    Use fixed scanner geometry centered on the vessel network.

    The clearance argument is kept for interface compatibility, but this
    realistic scanner baseline does not auto-scale the magnets from vessel size.
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
            "gantry_route_axis": _unit_vector(target - start),
            "gantry_particle_dist": 0.0,
            "gantry_axis_dist": 0.0,
            "gantry_lateral_shift": np.zeros(3, dtype=float),
        }

    if device_mode == "moving_linear_external_gantry":
        center, axis_point, _, route_axis = _moving_linear_external_gantry_center(
            particle_pos=particle_pos,
            start=start,
            target=target,
            standoff=gantry_standoff,
        )
        magnets.recenter(center)
        return {
            "gantry_center": center.copy(),
            "gantry_axis_point": axis_point.copy(),
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
            "gantry_route_axis": _unit_vector(target - start),
            "gantry_particle_dist": float(np.linalg.norm(magnets.center - particle_pos)),
            "gantry_axis_dist": float(np.linalg.norm(magnets.center - axis_point)),
            "gantry_lateral_shift": np.zeros(3, dtype=float),
        }

    raise ValueError(f"Unknown device_mode: {device_mode}")


def _set_currents_with_optional_thermal(magnets, currents, dt=None):
    """
    Compatibility helper.

    Hardware-constrained magnets.py may accept set_currents(currents, dt=...).
    Older magnets.py accepts only set_currents(currents).
    """
    try:
        magnets.set_currents(currents, dt=dt)
    except TypeError:
        magnets.set_currents(currents)


# ----------------------------------------------------------------------
# MPC simulation
# ----------------------------------------------------------------------

def simulate_trial_mpc(
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
    # Baseline MPC weights. These are intentionally simpler than ACBC.
    w_progress: float = 120.0,
    w_align: float = 8.0,
    w_dist: float = 1.0,
    w_curr: float = 1e-5,
    w_smooth: float = 1e-4,
    w_edge: float = 50.0,
    w_overshoot: float = 2.0,
    w_route: float = 0.0,
    # Dense local waypoint baseline.
    waypoint_arrival_tol: float = DEFAULT_WAYPOINT_ARRIVAL_TOL,
    waypoint_spacing: float = DEFAULT_WAYPOINT_SPACING,
    waypoint_lookahead: int = DEFAULT_MPC_WAYPOINT_LOOKAHEAD,
    forward_search_points: int = DEFAULT_FORWARD_SEARCH_POINTS,
    early_stop_enabled: bool = False,
    early_stop_window_steps: int = DEFAULT_EARLY_STOP_WINDOW_STEPS,
    early_stop_improvement_threshold_m: float = (
        DEFAULT_EARLY_STOP_IMPROVEMENT_THRESHOLD_M
    ),
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
        arrival_tol=waypoint_arrival_tol,
        waypoint_spacing=waypoint_spacing,
        lookahead=waypoint_lookahead,
        forward_search_points=forward_search_points,
    )

    print(
        f"    Profile: {profile.name}, "
        f"diameter={profile.diameter * 1000:.3f} mm",
        flush=True,
    )

    print(
        f"    Raw waypoints: "
        f"{getattr(navigator, 'raw_waypoint_count', 'unknown')}, "
        f"sparse waypoints: {navigator.total_hops}",
        flush=True,
    )

    for i, wp in enumerate(navigator.waypoints[:10]):
        print(f"      wp{i + 1}: {np.round(wp * 1000, 1)} mm", flush=True)

    if navigator.total_hops > 10:
        print("      ...", flush=True)
        print(
            f"      wp{navigator.total_hops}: "
            f"{np.round(navigator.waypoints[-1] * 1000, 1)} mm",
            flush=True,
        )

    print(
        "    Baseline MPC navigator: "
        f"arrival_tol={waypoint_arrival_tol * 1000:.2f} mm, "
        f"spacing={waypoint_spacing * 1000:.2f} mm, "
        f"lookahead={waypoint_lookahead}, "
        f"forward_search_points={forward_search_points}",
        flush=True,
    )

    print(
        "    Device motion: "
        f"mode={device_mode}, "
        f"placement_radius={placement_radius * 1000:.1f} mm, "
        f"initial_center={np.round(initial_device_diag['gantry_center'] * 1000, 1)} mm",
        flush=True,
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

    reached = False
    early_stopped = False
    early_stop_reason = ""
    early_stop_measured_improvement_m = np.nan
    early_stop_window_steps = int(early_stop_window_steps)
    early_stop_improvement_threshold_m = float(early_stop_improvement_threshold_m)
    best_final_target_distance = float(np.linalg.norm(particle.pos - target))
    best_distance_history = [best_final_target_distance]
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

        current_target = navigator(particle.pos)

        route_tangent = None
        nav_waypoints = getattr(navigator, "waypoints", None)
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
            w_route=w_route,
            route_tangent=route_tangent,
            wall_buffer_m=wall_clearance_threshold_m,
            flow=flow,
            sim_time=sim_time,
        )

        t1 = time.perf_counter()

        if converged:
            converged_count += 1
        else:
            warnings.warn(
                f"Step {step}: no SLSQP start converged; using best-so-far",
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

        best_final_target_distance = min(best_final_target_distance, float(distance))
        best_distance_history.append(best_final_target_distance)

        if distance < tolerance:
            reached = True
            break

        if (
            early_stop_enabled
            and early_stop_window_steps > 0
            and len(best_distance_history) > early_stop_window_steps
        ):
            previous_best = best_distance_history[-early_stop_window_steps - 1]
            improvement = previous_best - best_final_target_distance
            if improvement < early_stop_improvement_threshold_m:
                early_stopped = True
                early_stop_measured_improvement_m = float(improvement)
                early_stop_reason = (
                    "best final-target distance improved by "
                    f"{improvement * 1000.0:.3f} mm over "
                    f"{early_stop_window_steps} completed steps "
                    f"(< {early_stop_improvement_threshold_m * 1000.0:.3f} mm)"
                )
                print(
                    f"  Early stopping: {early_stop_reason}",
                    flush=True,
                )
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
                f"final_dist={distance * 1000:.1f}mm  "
                f"wp={nav_curr_idx + 1}/{navigator.total_hops}  "
                f"best_wp={nav_best_idx + 1}/{navigator.total_hops}  "
                f"lookahead={getattr(navigator, 'lookahead_idx', 0) + 1}/"
                f"{navigator.total_hops}  "
                f"wp_dist={getattr(navigator, 'current_wp_dist', np.nan) * 1000:.2f}mm  "
                f"wall={wall_clearance * 1000:.2f}mm  "
                f"|F|={np.linalg.norm(best_force):.2e}N  "
                f"|B|={np.linalg.norm(B):.2e}T  "
                f"|I|={np.max(np.abs(best_currents)):.1f}A  "
                f"v={particle_speed * 1000:.2f}mm/s  "
                f"flow={flow_speed * 1000:.2f}mm/s  "
                f"Re={re:.2e}  "
                f"lat={lateral_shift_norm * 1000:.1f}mm",
                flush=True,
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
        early_stopped=early_stopped,
        early_stop_reason=early_stop_reason,
        early_stop_window_steps=early_stop_window_steps,
        early_stop_improvement_threshold_m=early_stop_improvement_threshold_m,
        early_stop_measured_improvement_m=early_stop_measured_improvement_m,
        seed=rng_seed,
        profile_name=profile.name,
        dt=dt,
        workspace_half=network.half,
        magnets=magnets.positions.copy(),
        orientations=magnets.orientations.copy(),
        params={
            "controller": "baseline_dense_waypoint_MPC",
            "low_level_allocator": "continuous_mpc_controller_v2",
            "device_mode": device_mode,
            "dt": dt,
            "steps": steps,
            "tolerance": tolerance,
            "early_stop_enabled": bool(early_stop_enabled),
            "early_stopped": bool(early_stopped),
            "early_stop_reason": early_stop_reason,
            "early_stop_window_steps": early_stop_window_steps,
            "early_stop_improvement_threshold_m": early_stop_improvement_threshold_m,
            "early_stop_improvement_threshold_mm": (
                early_stop_improvement_threshold_m * 1000.0
            ),
            "early_stop_measured_improvement_m": early_stop_measured_improvement_m,
            "early_stop_measured_improvement_mm": (
                early_stop_measured_improvement_m * 1000.0
            ),
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
            "w_progress": w_progress,
            "w_align": w_align,
            "w_dist": w_dist,
            "w_curr": w_curr,
            "w_smooth": w_smooth,
            "w_edge": w_edge,
            "w_overshoot": w_overshoot,
            "w_route": w_route,
            **hardware_diag,
        },
    )

    result.gantry_center_log = gantry_center_log
    result.gantry_particle_dist_log = gantry_particle_dist_log
    result.gantry_axis_dist_log = gantry_axis_dist_log
    result.gantry_lateral_shift_log = gantry_lateral_shift_log

    print(
        f"  Baseline MPC summary: "
        f"low-level convergence={low_level_convergence_rate:.2%}",
        flush=True,
    )

    if hardware_diag:
        print(
            f"  Hardware: max_actual_I={hardware_diag.get('max_actual_current_A', np.nan):.2f}A, "
            f"max_V={hardware_diag.get('max_voltage_V', np.nan):.1f}V, "
            f"max_P={hardware_diag.get('max_power_W', np.nan):.1f}W, "
            f"max_temp={hardware_diag.get('max_coil_temp_C', np.nan):.1f}C",
            flush=True,
        )

    if len(gantry_axis_dist_log) > 0:
        lateral_norms = np.linalg.norm(gantry_lateral_shift_log, axis=1)
        print(
            f"  Gantry: mean_particle_dist={np.mean(gantry_particle_dist_log) * 1000:.1f}mm, "
            f"max_lateral={np.max(lateral_norms) * 1000:.1f}mm",
            flush=True,
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
    waypoint_spacing: float = DEFAULT_WAYPOINT_SPACING,
    waypoint_arrival_tol: float = DEFAULT_WAYPOINT_ARRIVAL_TOL,
    waypoint_lookahead: int = DEFAULT_MPC_WAYPOINT_LOOKAHEAD,
    forward_search_points: int = DEFAULT_FORWARD_SEARCH_POINTS,
):
    if device_mode not in VALID_DEVICE_MODES:
        raise ValueError(
            f"device_mode must be one of {sorted(VALID_DEVICE_MODES)}, "
            f"got {device_mode}"
        )

    network = RealVesselNetwork(vessel_file, flow_scale=flow_scale)
    print(network.segment_summary(), flush=True)
    print(f"Flow scale: {flow_scale:.2f}x", flush=True)

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
        f"B_sat={core_saturation_tesla:.1f}T",
        flush=True,
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

    output_dir = "outputs/simulation_outputs_3d_mpc"
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
        print(f"\n>>> Running {trial_name}", flush=True)
        print(f"    start:  {cfg['start']}", flush=True)
        print(f"    target: {cfg['target']}", flush=True)

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
            magnet_center=None,
            core_relative_permeability=core_relative_permeability,
            core_saturation_tesla=core_saturation_tesla,
            device_mode=device_mode,
            gantry_standoff=gantry_standoff,
            max_lateral_shift=max_lateral_shift,
            lateral_gain=lateral_gain,
            profile=SPION_CARRIER_1MM,
            waypoint_arrival_tol=waypoint_arrival_tol,
            waypoint_spacing=waypoint_spacing,
            waypoint_lookahead=waypoint_lookahead,
            forward_search_points=forward_search_points,
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

    print(f"\nResults saved to: {output_dir}/", flush=True)

    return results


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Run fair baseline dense-waypoint MPC on the real vessel network."
    )

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
    parser.add_argument("--waypoint-spacing", type=float, default=DEFAULT_WAYPOINT_SPACING)
    parser.add_argument("--waypoint-arrival-tol", type=float, default=DEFAULT_WAYPOINT_ARRIVAL_TOL)
    parser.add_argument("--waypoint-lookahead", type=int, default=DEFAULT_MPC_WAYPOINT_LOOKAHEAD)
    parser.add_argument("--forward-search-points", type=int, default=DEFAULT_FORWARD_SEARCH_POINTS)
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
        waypoint_spacing=args.waypoint_spacing,
        waypoint_arrival_tol=args.waypoint_arrival_tol,
        waypoint_lookahead=args.waypoint_lookahead,
        forward_search_points=args.forward_search_points,
    )
