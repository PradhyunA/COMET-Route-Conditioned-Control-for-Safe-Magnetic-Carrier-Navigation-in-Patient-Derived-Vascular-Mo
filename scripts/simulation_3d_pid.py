"""
Route-aware PID baseline for 3D magnetic carrier steering.

This is a classical-control baseline for publication comparisons. It uses the
same physics, vessel waypoint navigator, moving gantry modes, and actuator
limits as the MPC/ACBC/COMET benchmark harnesses, but the high-level control
law is only PID feedback on the current route waypoint.
"""

from __future__ import annotations

import argparse
import csv
import os
import sys
import time
import warnings

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

import numpy as np

from dynamag_simulation.core.constants import DEFAULT_TARGET_TOLERANCE
from dynamag_simulation.core.integrator import adaptive_step
from dynamag_simulation.core.magnets import ElectromagnetArray3D
from dynamag_simulation.core.particle import MagneticParticle3D
from dynamag_simulation.core.pid_controller import PIDConfig, PIDController
from dynamag_simulation.core.smc_dob_controller import (
    SMCDOBInspiredConfig,
    SMCDOBInspiredController,
)
from dynamag_simulation.core.profiles import PhysicalProfile, SPION_CARRIER_1MM
from dynamag_simulation.core.real_vessel_network import RealVesselNetwork
from dynamag_simulation.core.reporting import TrialResult, print_summary, save_results_csv
from dynamag_simulation.core.validation import check_low_re_validity, compute_reynolds
from dynamag_simulation.core.workspace import WorkspaceVessel

DEFAULT_VESSEL_FILE = "real_vessel_segments_cm_corrected.pkl"

FIXED_PLACEMENT_RADIUS = 0.300
FIXED_COIL_RADIUS = 0.070
FIXED_COIL_LENGTH = 0.160
FIXED_CORE_RADIUS = 0.030
FIXED_TURNS = 600.0
FIXED_MAX_CURRENT = 16.0
FIXED_CORE_MU_R_EFFECTIVE = 70.0
FIXED_CORE_SATURATION_T = 1.8

DEFAULT_MAGNET_CLEARANCE = 0.04
DEFAULT_WAYPOINT_SPACING = 0.002
DEFAULT_WAYPOINT_ARRIVAL_TOL = 0.0015
DEFAULT_MPC_WAYPOINT_LOOKAHEAD = 1
DEFAULT_FORWARD_SEARCH_POINTS = 25

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
    return axis_point.copy(), axis_point, route_axis


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
    lateral_shift = (
        desired_shift * (max_shift / shift_norm)
        if shift_norm > max_shift and shift_norm > 1e-12
        else desired_shift
    )
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
    if device_mode in {"moving_linear_enclosing_gantry", "moving_adaptive_enclosing_gantry"}:
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
        center, axis_point, lateral_shift, route_axis = _moving_adaptive_enclosing_gantry_center(
            particle_pos=particle_pos,
            start=start,
            target=target,
            max_lateral_shift=max_lateral_shift,
            lateral_gain=lateral_gain,
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
    try:
        magnets.set_currents(currents, dt=dt)
    except TypeError:
        magnets.set_currents(currents)


def _vec_or_nan(v):
    if v is None:
        return [np.nan, np.nan, np.nan]
    v = np.asarray(v, dtype=float)
    if v.shape[0] < 3:
        return [np.nan, np.nan, np.nan]
    return [float(v[0]), float(v[1]), float(v[2])]


def save_pid_diagnostics(results, output_dir, filename="pid_diagnostics.csv"):
    path = os.path.join(output_dir, filename)
    header = [
        "trial_name", "step", "time_s",
        "waypoint_index", "waypoint_distance_m", "position_error_norm_m",
        "p_norm", "i_norm", "d_norm",
        "flow_compensation_norm", "wall_repulsion_norm",
        "desired_velocity_norm", "current_norm", "max_abs_current",
        "saturated", "integral_norm", "wall_clearance_m",
        "controller_step_time_s",
        "target_x", "target_y", "target_z",
        "gantry_center_x", "gantry_center_y", "gantry_center_z",
        "gantry_axis_point_x", "gantry_axis_point_y", "gantry_axis_point_z",
        "gantry_particle_dist_m", "gantry_axis_dist_m",
        "gantry_lateral_shift_x", "gantry_lateral_shift_y", "gantry_lateral_shift_z",
    ]

    with open(path, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(header)
        for result in results:
            for row in getattr(result, "pid_log", []):
                writer.writerow([
                    result.trial_name,
                    row["step"], row["time_s"],
                    row["waypoint_index"], row["waypoint_distance_m"],
                    row["position_error_norm_m"],
                    row["p_norm"], row["i_norm"], row["d_norm"],
                    row["flow_compensation_norm"], row["wall_repulsion_norm"],
                    row["desired_velocity_norm"], row["current_norm"],
                    row["max_abs_current"], row["saturated"],
                    row["integral_norm"], row["wall_clearance_m"],
                    row["controller_step_time_s"],
                    *row["target"],
                    *row["gantry_center"],
                    *row["gantry_axis_point"],
                    row["gantry_particle_dist_m"], row["gantry_axis_dist_m"],
                    *row["gantry_lateral_shift"],
                ])

    print(f"PID diagnostics saved to: {path}", flush=True)


def save_smc_dob_diagnostics(
    results,
    output_dir,
    filename="smc_dob_inspired_diagnostics.csv",
):
    """Write paper-structure diagnostics for the adapted SMC-DOB baseline."""
    path = os.path.join(output_dir, filename)
    header = [
        "trial_name", "step", "time_s",
        "waypoint_index", "waypoint_distance_m", "position_error_norm_m",
        "velocity_error_norm_m_s", "filtered_velocity_error_norm_m_s",
        "sliding_surface_norm", "equivalent_velocity_norm_m_s",
        "switching_velocity_norm_m_s", "disturbance_residual_norm_m_s",
        "disturbance_estimate_norm_m_s", "flow_compensation_norm_m_s",
        "wall_repulsion_norm_m_s", "reference_velocity_norm_m_s",
        "desired_velocity_norm_m_s", "current_norm_A", "max_abs_current_A",
        "saturated", "wall_clearance_m", "controller_step_time_s",
        "target_x", "target_y", "target_z",
        "gantry_center_x", "gantry_center_y", "gantry_center_z",
        "gantry_axis_point_x", "gantry_axis_point_y", "gantry_axis_point_z",
        "gantry_particle_dist_m", "gantry_axis_dist_m",
        "gantry_lateral_shift_x", "gantry_lateral_shift_y", "gantry_lateral_shift_z",
    ]

    with open(path, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(header)
        for result in results:
            for row in getattr(result, "smc_dob_log", []):
                writer.writerow([
                    result.trial_name,
                    row["step"], row["time_s"],
                    row["waypoint_index"], row["waypoint_distance_m"],
                    row["position_error_norm_m"],
                    row["velocity_error_norm_m_s"],
                    row["filtered_velocity_error_norm_m_s"],
                    row["sliding_surface_norm"],
                    row["equivalent_velocity_norm_m_s"],
                    row["switching_velocity_norm_m_s"],
                    row["disturbance_residual_norm_m_s"],
                    row["disturbance_estimate_norm_m_s"],
                    row["flow_compensation_norm_m_s"],
                    row["wall_repulsion_norm_m_s"],
                    row["reference_velocity_norm_m_s"],
                    row["desired_velocity_norm_m_s"],
                    row["current_norm_A"], row["max_abs_current_A"],
                    row["saturated"], row["wall_clearance_m"],
                    row["controller_step_time_s"],
                    *row["target"],
                    *row["gantry_center"],
                    *row["gantry_axis_point"],
                    row["gantry_particle_dist_m"], row["gantry_axis_dist_m"],
                    *row["gantry_lateral_shift"],
                ])

    print(f"SMC-DOB-inspired diagnostics saved to: {path}", flush=True)


def simulate_trial_pid(
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
    ampere_turn_scale: float = 1.0,
    device_mode: str = DEVICE_MODE,
    gantry_standoff: float = GANTRY_STANDOFF,
    max_lateral_shift: float = MAX_LATERAL_SHIFT,
    lateral_gain: float = LATERAL_GAIN,
    profile: PhysicalProfile = SPION_CARRIER_1MM,
    max_substep_disp: float = 1e-3,
    rng_seed: int = 42,
    wall_clearance_threshold_m: float = 0.0007,
    waypoint_arrival_tol: float = DEFAULT_WAYPOINT_ARRIVAL_TOL,
    waypoint_spacing: float = DEFAULT_WAYPOINT_SPACING,
    waypoint_lookahead: int = DEFAULT_MPC_WAYPOINT_LOOKAHEAD,
    forward_search_points: int = DEFAULT_FORWARD_SEARCH_POINTS,
    kp: float = PIDConfig.kp,
    ki: float = PIDConfig.ki,
    kd: float = PIDConfig.kd,
    flow_compensation_gain: float = PIDConfig.flow_compensation_gain,
    wall_repulsion_gain: float = PIDConfig.wall_repulsion_gain,
    max_integral_norm: float = PIDConfig.max_integral_norm,
    derivative_filter_alpha: float = PIDConfig.derivative_filter_alpha,
    max_control_norm: float = PIDConfig.max_control_norm,
    debug: bool = False,
    controller_mode: str = "pid",
    smc_dob_kwargs: dict | None = None,
) -> TrialResult:
    if controller_mode not in {"pid", "smc_dob_inspired"}:
        raise ValueError(
            "controller_mode must be 'pid' or 'smc_dob_inspired', "
            f"got {controller_mode!r}"
        )
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
        ampere_turn_scale=ampere_turn_scale,
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

    if controller_mode == "smc_dob_inspired":
        smc_params = dict(smc_dob_kwargs or {})
        smc_params.update({
            "wall_clearance_threshold": wall_clearance_threshold_m,
            "waypoint_lookahead_distance": (
                waypoint_spacing * max(1, waypoint_lookahead)
            ),
            "waypoint_reached_radius": waypoint_arrival_tol,
            "max_current": max_current,
            "debug": debug,
        })
        controller_cfg = SMCDOBInspiredConfig(**smc_params)
        controller = SMCDOBInspiredController(
            network=network,
            magnet_array=magnets,
            workspace=workspace,
            config=controller_cfg,
        )
    else:
        controller_cfg = PIDConfig(
            kp=kp,
            ki=ki,
            kd=kd,
            flow_compensation_gain=flow_compensation_gain,
            wall_repulsion_gain=wall_repulsion_gain,
            wall_clearance_threshold=wall_clearance_threshold_m,
            waypoint_lookahead_distance=waypoint_spacing * max(1, waypoint_lookahead),
            waypoint_reached_radius=waypoint_arrival_tol,
            max_integral_norm=max_integral_norm,
            derivative_filter_alpha=derivative_filter_alpha,
            max_control_norm=max_control_norm,
            max_current=max_current,
            debug=debug,
        )
        controller = PIDController(
            network=network,
            magnet_array=magnets,
            workspace=workspace,
            config=controller_cfg,
        )

    print(
        f"    Profile: {profile.name}, "
        f"diameter={profile.diameter * 1000:.3f} mm",
        flush=True,
    )
    print(
        f"    Raw waypoints: {getattr(navigator, 'raw_waypoint_count', 'unknown')}, "
        f"sparse waypoints: {navigator.total_hops}",
        flush=True,
    )
    if controller_mode == "smc_dob_inspired":
        print(
            "    Published-style adapted baseline: SMC-DOB-inspired "
            "(Bhattacharjee et al., 2026; not an exact reproduction)",
            flush=True,
        )
        print(
            "    SMC-DOB-inspired: "
            f"gamma={controller_cfg.sliding_gamma:.3g}, "
            f"lambda={controller_cfg.sliding_lambda:.3g}, "
            f"Ksw_v={controller_cfg.switching_gain:.3g}m/s, "
            f"eta={controller_cfg.disturbance_time_constant:.3g}s, "
            f"phi_v={controller_cfg.boundary_layer:.3g}m/s, "
            f"route_speed={controller_cfg.route_speed * 1000:.1f}mm/s",
            flush=True,
        )
        print(
            "    Allocation: bounded bidirectional SLSQP velocity allocator; "
            "no COMET logic and no CEM",
            flush=True,
        )
    else:
        print(
            "    PID: "
            f"kp={kp:.3g}, ki={ki:.3g}, kd={kd:.3g}, "
            f"flow_gain={flow_compensation_gain:.3g}, "
            f"max_v={max_control_norm * 1000:.1f}mm/s",
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
    pid_log = []
    smc_dob_log = []

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

        currents, force, cost, converged, info = controller.step(
            particle=particle,
            final_target=target,
            navigator=navigator,
            dt=dt,
            flow=flow,
            sim_time=sim_time,
            initial_guess=prev_currents,
        )

        t1 = time.perf_counter()

        if converged:
            converged_count += 1
        else:
            warnings.warn(
                f"Step {step}: {controller_mode} current allocator did not "
                "converge; using best-so-far",
                RuntimeWarning,
                stacklevel=1,
            )

        prev_currents = currents.copy()

        _set_currents_with_optional_thermal(magnets, currents, dt=dt)
        B = magnets.field_at(particle.pos)

        adaptive_step(
            particle,
            force,
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
        force_log.append(np.linalg.norm(force))
        velocity_log.append(np.linalg.norm(particle.realized_vel))
        reynolds_log.append(re)
        cost_log.append(cost)
        step_time_log.append(t1 - t0)
        gantry_center_log.append(device_diag["gantry_center"].copy())
        gantry_particle_dist_log.append(device_diag["gantry_particle_dist"])
        gantry_axis_dist_log.append(device_diag["gantry_axis_dist"])
        gantry_lateral_shift_log.append(device_diag["gantry_lateral_shift"].copy())

        pid_log.append({
            "step": step,
            "time_s": sim_time,
            "waypoint_index": info.waypoint_index,
            "waypoint_distance_m": info.waypoint_distance_m,
            "position_error_norm_m": info.position_error_norm_m,
            "p_norm": info.p_norm,
            "i_norm": info.i_norm,
            "d_norm": info.d_norm,
            "flow_compensation_norm": info.flow_compensation_norm,
            "wall_repulsion_norm": info.wall_repulsion_norm,
            "desired_velocity_norm": info.desired_velocity_norm,
            "current_norm": info.current_norm,
            "max_abs_current": info.max_abs_current,
            "saturated": info.saturated,
            "integral_norm": info.integral_norm,
            "wall_clearance_m": info.wall_clearance_m,
            "controller_step_time_s": info.controller_step_time_s,
            "target": _vec_or_nan(info.target),
            "gantry_center": _vec_or_nan(device_diag["gantry_center"]),
            "gantry_axis_point": _vec_or_nan(device_diag["gantry_axis_point"]),
            "gantry_particle_dist_m": float(device_diag["gantry_particle_dist"]),
            "gantry_axis_dist_m": float(device_diag["gantry_axis_dist"]),
            "gantry_lateral_shift": _vec_or_nan(device_diag["gantry_lateral_shift"]),
        })
        if controller_mode == "smc_dob_inspired":
            smc_dob_log.append({
                "step": step,
                "time_s": sim_time,
                "waypoint_index": info.waypoint_index,
                "waypoint_distance_m": info.waypoint_distance_m,
                "position_error_norm_m": info.position_error_norm_m,
                "velocity_error_norm_m_s": info.velocity_error_norm_m_s,
                "filtered_velocity_error_norm_m_s": (
                    info.filtered_velocity_error_norm_m_s
                ),
                "sliding_surface_norm": info.sliding_surface_norm,
                "equivalent_velocity_norm_m_s": (
                    info.equivalent_velocity_norm_m_s
                ),
                "switching_velocity_norm_m_s": (
                    info.switching_velocity_norm_m_s
                ),
                "disturbance_residual_norm_m_s": (
                    info.disturbance_residual_norm_m_s
                ),
                "disturbance_estimate_norm_m_s": (
                    info.disturbance_estimate_norm_m_s
                ),
                "flow_compensation_norm_m_s": info.flow_compensation_norm_m_s,
                "wall_repulsion_norm_m_s": info.wall_repulsion_norm_m_s,
                "reference_velocity_norm_m_s": info.reference_velocity_norm_m_s,
                "desired_velocity_norm_m_s": info.desired_velocity_norm,
                "current_norm_A": info.current_norm,
                "max_abs_current_A": info.max_abs_current,
                "saturated": info.saturated,
                "wall_clearance_m": info.wall_clearance_m,
                "controller_step_time_s": info.controller_step_time_s,
                "target": _vec_or_nan(info.target),
                "gantry_center": _vec_or_nan(device_diag["gantry_center"]),
                "gantry_axis_point": _vec_or_nan(device_diag["gantry_axis_point"]),
                "gantry_particle_dist_m": float(device_diag["gantry_particle_dist"]),
                "gantry_axis_dist_m": float(device_diag["gantry_axis_dist"]),
                "gantry_lateral_shift": _vec_or_nan(
                    device_diag["gantry_lateral_shift"]
                ),
            })

        if distance < tolerance:
            reached = True
            break

        if debug and (step + 1) % 10 == 0:
            flow_v = flow.velocity_at(particle.pos, sim_time)
            print(
                f"  step {step + 1:>4d}/{steps}  "
                f"final_dist={distance * 1000:.1f}mm  "
                f"wp={info.waypoint_index + 1}/{navigator.total_hops}  "
                f"err={info.position_error_norm_m * 1000:.1f}mm  "
                f"wall={info.wall_clearance_m * 1000:.2f}mm  "
                f"|F|={np.linalg.norm(force):.2e}N  "
                f"|B|={np.linalg.norm(B):.2e}T  "
                f"|I|={np.max(np.abs(currents)):.1f}A  "
                f"v={np.linalg.norm(particle.realized_vel) * 1000:.2f}mm/s  "
                f"flow={np.linalg.norm(flow_v) * 1000:.2f}mm/s",
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
        seed=rng_seed,
        profile_name=profile.name,
        dt=dt,
        workspace_half=network.half,
        magnets=magnets.positions.copy(),
        orientations=magnets.orientations.copy(),
        params={
            "controller": (
                "smc_dob_inspired_route_following"
                if controller_mode == "smc_dob_inspired"
                else "route_aware_PID"
            ),
            "controller_adaptation_note": (
                "Adapted published-style SMC-DOB baseline based on "
                "Bhattacharjee et al. (2026), not an exact reproduction; "
                "the source paper used a 2-D in-vitro cardiac phantom and "
                "different actuation/localization hardware."
                if controller_mode == "smc_dob_inspired"
                else ""
            ),
            "low_level_allocator": "bounded_bidirectional_slsqp_velocity_allocator",
            "uses_cem": False,
            "uses_comet_branch_probability": False,
            "uses_comet_downstream_value": False,
            "uses_comet_terminal_precision": False,
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
            "ampere_turn_scale": ampere_turn_scale,
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
            **(
                {
                    f"smc_dob_{key}": value
                    for key, value in vars(controller_cfg).items()
                    if isinstance(value, (bool, int, float, str)) or value is None
                }
                if controller_mode == "smc_dob_inspired"
                else {
                    "pid_kp": kp,
                    "pid_ki": ki,
                    "pid_kd": kd,
                    "pid_flow_compensation_gain": flow_compensation_gain,
                    "pid_wall_repulsion_gain": wall_repulsion_gain,
                    "pid_max_integral_norm": max_integral_norm,
                    "pid_derivative_filter_alpha": derivative_filter_alpha,
                    "pid_max_control_norm": max_control_norm,
                }
            ),
            **hardware_diag,
        },
    )

    result.gantry_center_log = gantry_center_log
    result.gantry_particle_dist_log = gantry_particle_dist_log
    result.gantry_axis_dist_log = gantry_axis_dist_log
    result.gantry_lateral_shift_log = gantry_lateral_shift_log
    result.pid_log = pid_log
    result.smc_dob_log = smc_dob_log

    print(
        f"  {controller_mode} summary: "
        f"low-level convergence={low_level_convergence_rate:.2%}",
        flush=True,
    )

    return result


def main():
    parser = argparse.ArgumentParser(description="Run the route-aware PID baseline.")
    parser.add_argument("--vessel-file", default=DEFAULT_VESSEL_FILE)
    parser.add_argument("--output-dir", default="outputs/simulation_outputs_3d_pid")
    parser.add_argument("--flow-scale", type=float, default=1.0)
    parser.add_argument("--steps", type=int, default=400)
    parser.add_argument("--dt", type=float, default=0.02)
    parser.add_argument("--debug", action="store_true")
    args = parser.parse_args()

    network = RealVesselNetwork(args.vessel_file, flow_scale=args.flow_scale)
    start = network.inlet_start()
    target = network.outlet_end()
    result = simulate_trial_pid(
        start=start,
        target=target,
        network=network,
        trial_name="pid_inlet_to_outlet",
        dt=args.dt,
        steps=args.steps,
        debug=args.debug,
    )

    os.makedirs(args.output_dir, exist_ok=True)
    print_summary(result)
    save_results_csv([result], args.output_dir)
    save_pid_diagnostics([result], args.output_dir)
    try:
        from dynamag_simulation.core.plotting import save_all_plots
    except ModuleNotFoundError:
        save_all_plots = None
    if save_all_plots is not None:
        save_all_plots([result], args.output_dir)


if __name__ == "__main__":
    main()
