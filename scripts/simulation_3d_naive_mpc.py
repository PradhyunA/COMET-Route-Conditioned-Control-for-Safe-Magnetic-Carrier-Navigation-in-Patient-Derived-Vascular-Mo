"""
simulation_3d_naive_mpc.py
==========================

Safety-matched route-aware MPC baseline for DynaMag.

The filename remains simulation_3d_naive_mpc.py for compatibility, but this file
now implements the fair MPC baseline you want:

    - same real vessel network
    - same magnetic hardware model
    - same adaptive gantry capability
    - same MagneticParticle3D plant
    - same adaptive_step integrator
    - same continuous_mpc_controller allocator
    - same ACBC-tuned low-level weights by default
    - same wall-buffer clearance requirement
    - same post-MPC wall-safety current filter style

It does NOT include ACBC method logic:

    - no ACBC state machine
    - no branch commitment
    - no authority assessment
    - no recovery mode
    - no stall escape
    - no state dwell/cooldown logic

Default behavior:
    safety_matched_route_aware_mpc

Recommended run:
---------------
python -u scripts/simulation_3d_naive_mpc.py \
  --device-mode moving_adaptive_enclosing_gantry \
  --flow-scale 1 \
  --only-trial trial_2_mid_to_outlet \
  --profile spion_carrier_500um \
  --target-mode local_waypoint \
  --weight-preset acbc_tuned \
  --enable-wall-safety-filter
"""

from __future__ import annotations

import argparse
import os
import sys
import time
import warnings

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

import numpy as np

from dynamag_simulation.core import profiles as profiles_module
from dynamag_simulation.core.constants import DEFAULT_TARGET_TOLERANCE
from dynamag_simulation.core.controllers import (
    continuous_mpc_controller,
    _simulate_one_step,
)
from dynamag_simulation.core.integrator import adaptive_step
from dynamag_simulation.core.magnets import ElectromagnetArray3D
from dynamag_simulation.core.particle import MagneticParticle3D
from dynamag_simulation.core.plotting import save_all_plots
from dynamag_simulation.core.profiles import PhysicalProfile, SPION_CARRIER_1MM
from dynamag_simulation.core.real_vessel_network import RealVesselNetwork
from dynamag_simulation.core.reporting import TrialResult, print_summary, save_results_csv
from dynamag_simulation.core.validation import check_low_re_validity, compute_reynolds
from dynamag_simulation.core.workspace import WorkspaceVessel


DEFAULT_VESSEL_FILE = "real_vessel_segments_cm_corrected.pkl"


# ----------------------------------------------------------------------
# Hardware-constrained body-scale scanner parameters
# ----------------------------------------------------------------------

FIXED_PLACEMENT_RADIUS = 0.300
FIXED_COIL_RADIUS = 0.070
FIXED_COIL_LENGTH = 0.160
FIXED_CORE_RADIUS = 0.030
FIXED_TURNS = 600.0
FIXED_MAX_CURRENT = 16.0
FIXED_CORE_MU_R_EFFECTIVE = 70.0
FIXED_CORE_SATURATION_T = 1.8

DEFAULT_MAGNET_CLEARANCE = 0.04
DEFAULT_FLOW_SCALE = 1.0


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


VALID_TARGET_MODES = {
    "local_waypoint",
    "final_target",
}

DEFAULT_TARGET_MODE = "local_waypoint"


# ----------------------------------------------------------------------
# Publication weight presets
# ----------------------------------------------------------------------

MPC_DEFAULT_WEIGHTS = {
    "w_progress": 120.0,
    "w_align": 8.0,
    "w_dist": 1.0,
    "w_curr": 1e-5,
    "w_smooth": 1e-4,
    "w_edge": 50.0,
    "w_overshoot": 2.0,
    "w_route": 0.0,
}

ACBC_TUNED_WEIGHTS = {
    "w_progress": 450.0,
    "w_align": 35.0,
    "w_dist": 2.0,
    "w_curr": 1e-7,
    "w_smooth": 1e-6,
    "w_edge": 250.0,
    "w_overshoot": 8.0,
    "w_route": 15.0,
}

WEIGHT_PRESETS = {
    "acbc_tuned": ACBC_TUNED_WEIGHTS,
    "mpc_default": MPC_DEFAULT_WEIGHTS,
}

DEFAULT_WEIGHT_PRESET = "acbc_tuned"

DEFAULT_W_PROGRESS = ACBC_TUNED_WEIGHTS["w_progress"]
DEFAULT_W_ALIGN = ACBC_TUNED_WEIGHTS["w_align"]
DEFAULT_W_DIST = ACBC_TUNED_WEIGHTS["w_dist"]
DEFAULT_W_CURR = ACBC_TUNED_WEIGHTS["w_curr"]
DEFAULT_W_SMOOTH = ACBC_TUNED_WEIGHTS["w_smooth"]
DEFAULT_W_EDGE = ACBC_TUNED_WEIGHTS["w_edge"]
DEFAULT_W_OVERSHOOT = ACBC_TUNED_WEIGHTS["w_overshoot"]
DEFAULT_W_ROUTE = ACBC_TUNED_WEIGHTS["w_route"]

DEFAULT_SURFACE_CLEARANCE_TARGET_M = 0.00050
DEFAULT_ENABLE_WALL_SAFETY_FILTER = True


# ----------------------------------------------------------------------
# Profile helper
# ----------------------------------------------------------------------

def _normalize_profile_key(name: str) -> str:
    return (
        str(name)
        .strip()
        .lower()
        .replace("-", "_")
        .replace(" ", "_")
        .replace("__", "_")
    )


def resolve_profile(profile_name: str) -> PhysicalProfile:
    key = _normalize_profile_key(profile_name)

    aliases = {
        "1mm": [
            "SPION_CARRIER_1MM",
            "SPION_CARRIER_1000UM",
        ],
        "spion_carrier_1mm": [
            "SPION_CARRIER_1MM",
            "SPION_CARRIER_1000UM",
        ],
        "500um": [
            "SPION_CARRIER_500UM",
            "SPION_CARRIER_500_UM",
            "SPION_CARRIER_500um",
            "MICROSPHERE_500UM",
            "microsphere_500um",
        ],
        "spion_carrier_500um": [
            "SPION_CARRIER_500UM",
            "SPION_CARRIER_500_UM",
            "SPION_CARRIER_500um",
            "MICROSPHERE_500UM",
            "microsphere_500um",
        ],
        "microsphere_500um": [
            "MICROSPHERE_500UM",
            "microsphere_500um",
            "SPION_CARRIER_500UM",
            "SPION_CARRIER_500_UM",
        ],
    }

    direct_candidates = [
        profile_name,
        profile_name.upper(),
        profile_name.lower(),
        key,
        key.upper(),
    ]

    candidate_names = direct_candidates + aliases.get(key, [])

    for attr_name in candidate_names:
        if hasattr(profiles_module, attr_name):
            candidate = getattr(profiles_module, attr_name)
            if isinstance(candidate, PhysicalProfile):
                return candidate

    available = []
    for attr_name in dir(profiles_module):
        value = getattr(profiles_module, attr_name)
        if isinstance(value, PhysicalProfile):
            available.append(attr_name)

    raise ValueError(
        f"Could not resolve profile '{profile_name}'. "
        f"Available PhysicalProfile objects in profiles.py: {available}"
    )


# ----------------------------------------------------------------------
# Device / gantry helpers
# ----------------------------------------------------------------------

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
    try:
        magnets.set_currents(currents, dt=dt)
    except TypeError:
        magnets.set_currents(currents)


# ----------------------------------------------------------------------
# Target helpers
# ----------------------------------------------------------------------

def _make_optional_navigator(
    network,
    start,
    target,
    target_mode,
    waypoint_arrival_tol,
    waypoint_spacing,
    waypoint_lookahead,
    forward_search_points,
):
    if target_mode == "final_target":
        return None

    if target_mode == "local_waypoint":
        return network.waypoint_navigator(
            start,
            target,
            arrival_tol=waypoint_arrival_tol,
            waypoint_spacing=waypoint_spacing,
            lookahead=waypoint_lookahead,
            forward_search_points=forward_search_points,
        )

    raise ValueError(f"Unknown target_mode: {target_mode}")


def _select_control_target(
    particle_pos,
    final_target,
    target_mode,
    navigator,
):
    if target_mode == "final_target":
        return np.asarray(final_target, dtype=float)

    if target_mode == "local_waypoint":
        if navigator is None:
            raise RuntimeError("target_mode=local_waypoint requires navigator")
        return np.asarray(navigator(particle_pos), dtype=float)

    raise ValueError(f"Unknown target_mode: {target_mode}")


def _route_tangent_from_navigator(navigator, particle_pos):
    if navigator is None:
        return None

    waypoints = getattr(navigator, "waypoints", None)
    if waypoints is None or len(waypoints) < 2:
        return None

    idx = getattr(navigator, "current_idx", None)
    if idx is None:
        pts = np.asarray(waypoints, dtype=float)
        idx = int(np.argmin(np.linalg.norm(pts - np.asarray(particle_pos), axis=1)))
    else:
        idx = int(idx)

    idx = int(np.clip(idx, 0, len(waypoints) - 1))
    idx_next = min(idx + 3, len(waypoints) - 1)

    if idx_next <= idx:
        return None

    tangent = np.asarray(waypoints[idx_next], dtype=float) - np.asarray(
        waypoints[idx],
        dtype=float,
    )
    norm = float(np.linalg.norm(tangent))

    if norm < 1e-12:
        return None

    return tangent / norm


def _apply_weight_preset(weight_preset: str):
    if weight_preset not in WEIGHT_PRESETS:
        raise ValueError(
            f"weight_preset must be one of {sorted(WEIGHT_PRESETS)}, "
            f"got {weight_preset}"
        )
    return dict(WEIGHT_PRESETS[weight_preset])


# ----------------------------------------------------------------------
# Safety filter helpers
# ----------------------------------------------------------------------

def _wall_safety_at_position(network, particle_pos, particle):
    seg, _ = network.find_segment(particle_pos)

    if seg is None:
        return -np.inf, -np.inf, -np.inf

    _, _, radial_dist, _ = seg.closest_point_and_radial(particle_pos)

    clearance = float(seg.radius - radial_dist)
    margin = float(clearance - particle.radius)
    fraction = float(margin / (seg.radius + 1e-12))

    return clearance, margin, fraction


def _required_wall_clearance(particle, surface_clearance_target_m):
    return float(particle.radius + float(surface_clearance_target_m))


def _filter_wall_unsafe_currents_mpc(
    particle,
    particle_pos,
    currents,
    magnet_array,
    workspace,
    network,
    dt,
    flow,
    sim_time,
    required_wall_clearance_m,
    min_wall_safety_fraction=0.015,
):
    """
    Safety-matched post-MPC current filter.

    This mirrors ACBC's safety burden without adding ACBC state logic.
    It does not choose recovery targets, stall targets, or branch targets.
    It only reduces current magnitude if the one-step prediction violates
    the required wall-clearance band.
    """
    currents = np.asarray(currents, dtype=float)

    def predict(candidate):
        next_pos, force, _ = _simulate_one_step(
            particle=particle,
            particle_pos=particle_pos,
            currents=candidate,
            magnet_array=magnet_array,
            workspace=workspace,
            dt=dt,
            flow=flow,
            sim_time=sim_time,
        )

        clearance, _, fraction = _wall_safety_at_position(
            network=network,
            particle_pos=next_pos,
            particle=particle,
        )

        safe = bool(
            clearance >= required_wall_clearance_m
            and fraction >= min_wall_safety_fraction
        )

        return safe, clearance, force

    safe, _, force = predict(currents)
    if safe:
        return currents, force, False

    zero = np.zeros_like(currents)
    zero_safe, _, zero_force = predict(zero)

    best_currents = zero if zero_safe else currents
    best_force = zero_force if zero_safe else force

    if zero_safe:
        lo = 0.0
        hi = 1.0

        for _ in range(10):
            alpha = 0.5 * (lo + hi)
            candidate = alpha * currents
            candidate_safe, _, candidate_force = predict(candidate)

            if candidate_safe:
                lo = alpha
                best_currents = candidate
                best_force = candidate_force
            else:
                hi = alpha

    return best_currents, best_force, True


# ----------------------------------------------------------------------
# Simulation
# ----------------------------------------------------------------------

def simulate_trial_naive_mpc(
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
    target_mode: str = DEFAULT_TARGET_MODE,
    waypoint_arrival_tol: float = DEFAULT_TARGET_TOLERANCE,
    waypoint_spacing: float = 0.001,
    waypoint_lookahead: int = 1,
    forward_search_points: int = 30,
    wall_clearance_threshold_m: float = 0.0007,
    surface_clearance_target_m: float = DEFAULT_SURFACE_CLEARANCE_TARGET_M,
    enable_wall_safety_filter: bool = False,
    min_wall_safety_fraction: float = 0.015,
    w_progress: float = DEFAULT_W_PROGRESS,
    w_align: float = DEFAULT_W_ALIGN,
    w_dist: float = DEFAULT_W_DIST,
    w_curr: float = DEFAULT_W_CURR,
    w_smooth: float = DEFAULT_W_SMOOTH,
    w_edge: float = DEFAULT_W_EDGE,
    w_overshoot: float = DEFAULT_W_OVERSHOOT,
    w_route: float = DEFAULT_W_ROUTE,
) -> TrialResult:
    if device_mode not in VALID_DEVICE_MODES:
        raise ValueError(
            f"device_mode must be one of {sorted(VALID_DEVICE_MODES)}, "
            f"got {device_mode}"
        )

    if target_mode not in VALID_TARGET_MODES:
        raise ValueError(
            f"target_mode must be one of {sorted(VALID_TARGET_MODES)}, "
            f"got {target_mode}"
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

    navigator = _make_optional_navigator(
        network=network,
        start=start,
        target=target,
        target_mode=target_mode,
        waypoint_arrival_tol=waypoint_arrival_tol,
        waypoint_spacing=waypoint_spacing,
        waypoint_lookahead=waypoint_lookahead,
        forward_search_points=forward_search_points,
    )

    wall_buffer_m = float(wall_clearance_threshold_m)
    implied_surface_clearance_target_m = wall_buffer_m - particle.radius

    print("    Controller: safety-matched route-aware MPC baseline", flush=True)
    print(
        "    Controlled comparison: shared continuous_mpc_controller=True, "
        "shared plant/integrator=True, shared hardware/gantry=True",
        flush=True,
    )
    print(
        "    Shared safety: wall_buffer_m=True, post_mpc_wall_safety_filter="
        f"{enable_wall_safety_filter}",
        flush=True,
    )
    print(
        "    Excluded ACBC logic: branch_commitment=False, "
        "authority_assessment=False, recovery=False, stall_escape=False, "
        "state_machine=False, dwell_time=False",
        flush=True,
    )
    print(f"    Target mode: {target_mode}", flush=True)
    print(
        f"    Profile: {profile.name}, "
        f"diameter={profile.diameter * 1000:.3f} mm",
        flush=True,
    )
    print(
        f"    Wall buffer: particle_radius={particle.radius * 1000:.3f} mm, "
        f"implied_surface_clearance={implied_surface_clearance_target_m * 1000:.3f} mm, "
        f"wall_buffer={wall_buffer_m * 1000:.3f} mm",
        flush=True,
    )

    if navigator is not None:
        print(
            f"    Raw waypoints: "
            f"{getattr(navigator, 'raw_waypoint_count', 'unknown')}, "
            f"sparse waypoints: {navigator.total_hops}",
            flush=True,
        )
    else:
        print(
            "    No waypoint navigator instantiated. "
            "Controller sees only the final target.",
            flush=True,
        )

    print(
        "    MPC weights: "
        f"w_progress={w_progress}, "
        f"w_align={w_align}, "
        f"w_dist={w_dist}, "
        f"w_curr={w_curr}, "
        f"w_smooth={w_smooth}, "
        f"w_edge={w_edge}, "
        f"w_overshoot={w_overshoot}, "
        f"w_route={w_route}",
        flush=True,
    )

    print(
        "    Device motion: "
        f"mode={device_mode}, "
        f"placement_radius={placement_radius * 1000:.1f} mm, "
        f"max_lateral_shift={max_lateral_shift * 1000:.1f} mm, "
        f"lateral_gain={lateral_gain:.2f}, "
        f"initial_center={np.round(initial_device_diag['gantry_center'] * 1000, 1)} mm",
        flush=True,
    )

    trajectory = [particle.pos.copy()]
    distance_log = []
    control_target_distance_log = []
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
    wall_clearance_log = []
    waypoint_index_log = []
    control_target_log = []
    safety_filter_log = []

    reached = False
    prev_currents = np.zeros(magnets.n_magnets, dtype=float)
    wall_start = time.perf_counter()
    converged_count = 0
    safety_filter_count = 0

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

        control_target = _select_control_target(
            particle_pos=particle.pos,
            final_target=target,
            target_mode=target_mode,
            navigator=navigator,
        )

        route_tangent = _route_tangent_from_navigator(
            navigator=navigator,
            particle_pos=particle.pos,
        )

        best_currents, best_force, best_cost, converged = continuous_mpc_controller(
            particle=particle,
            particle_pos=particle.pos,
            target_pos=control_target,
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
            wall_buffer_m=wall_buffer_m,
            flow=flow,
            sim_time=sim_time,
        )

        if enable_wall_safety_filter:
            best_currents, best_force, filter_active = _filter_wall_unsafe_currents_mpc(
                particle=particle,
                particle_pos=particle.pos,
                currents=best_currents,
                magnet_array=magnets,
                workspace=workspace,
                network=network,
                dt=dt,
                flow=flow,
                sim_time=sim_time,
                required_wall_clearance_m=wall_buffer_m,
                min_wall_safety_fraction=min_wall_safety_fraction,
            )
        else:
            filter_active = False

        if filter_active:
            safety_filter_count += 1

        t1 = time.perf_counter()

        if converged:
            converged_count += 1
        else:
            warnings.warn(
                f"Step {step}: safety-matched MPC allocator did not converge; "
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

        distance = float(np.linalg.norm(particle.pos - target))
        control_target_distance = float(np.linalg.norm(particle.pos - control_target))

        seg, _ = network.find_segment(particle.pos)
        if seg is not None:
            _, _, radial_dist, _ = seg.closest_point_and_radial(particle.pos)
            wall_clearance = float(seg.radius - radial_dist)
        else:
            wall_clearance = np.nan

        if navigator is not None:
            nav_curr_idx = int(getattr(navigator, "current_idx", 0))
        else:
            nav_curr_idx = -1

        distance_log.append(distance)
        control_target_distance_log.append(control_target_distance)
        current_log.append(magnets.currents.copy())
        field_log.append(float(np.linalg.norm(B)))
        force_log.append(float(np.linalg.norm(best_force)))
        velocity_log.append(float(np.linalg.norm(particle.realized_vel)))
        reynolds_log.append(float(re))
        cost_log.append(float(best_cost))
        step_time_log.append(float(t1 - t0))
        gantry_center_log.append(device_diag["gantry_center"].copy())
        gantry_particle_dist_log.append(float(device_diag["gantry_particle_dist"]))
        gantry_axis_dist_log.append(float(device_diag["gantry_axis_dist"]))
        gantry_lateral_shift_log.append(device_diag["gantry_lateral_shift"].copy())
        wall_clearance_log.append(float(wall_clearance))
        waypoint_index_log.append(nav_curr_idx)
        control_target_log.append(control_target.copy())
        safety_filter_log.append(bool(filter_active))

        if distance < tolerance:
            reached = True
            break

        if (step + 1) % 10 == 0:
            flow_v = flow.velocity_at(particle.pos, sim_time)
            flow_speed = np.linalg.norm(flow_v)
            particle_speed = np.linalg.norm(particle.realized_vel)
            lateral_shift_norm = np.linalg.norm(device_diag["gantry_lateral_shift"])

            if navigator is not None:
                nav_best_idx = getattr(navigator, "best_idx", 0)
                current_wp_dist = getattr(navigator, "current_wp_dist", np.nan)
                lookahead_idx = getattr(navigator, "lookahead_idx", nav_curr_idx)
                nav_text = (
                    f"wp={nav_curr_idx + 1}/{navigator.total_hops}  "
                    f"best_wp={nav_best_idx + 1}/{navigator.total_hops}  "
                    f"lookahead={lookahead_idx + 1}/{navigator.total_hops}  "
                    f"wp_dist={current_wp_dist * 1000:.2f}mm  "
                )
            else:
                nav_text = "wp=none  "

            print(
                f"  step {step + 1:>4d}/{steps}  "
                f"target_mode={target_mode:<13s}  "
                f"final_dist={distance * 1000:.1f}mm  "
                f"{nav_text}"
                f"ctrl_dist={control_target_distance * 1000:.1f}mm  "
                f"wall={wall_clearance * 1000:.2f}mm  "
                f"|F|={np.linalg.norm(best_force):.2e}N  "
                f"|B|={np.linalg.norm(B):.2e}T  "
                f"|I|={np.max(np.abs(best_currents)):.1f}A  "
                f"v={particle_speed * 1000:.2f}mm/s  "
                f"flow={flow_speed * 1000:.2f}mm/s  "
                f"Re={re:.2e}  "
                f"lat={lateral_shift_norm * 1000:.1f}mm  "
                f"safety_filter={int(filter_active)}",
                flush=True,
            )

    wall_end = time.perf_counter()

    trajectory = np.array(trajectory)
    distance_log = np.array(distance_log)
    control_target_distance_log = np.array(control_target_distance_log)
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
    wall_clearance_log = np.array(wall_clearance_log)
    waypoint_index_log = np.array(waypoint_index_log)
    control_target_log = np.array(control_target_log)
    safety_filter_log = np.array(safety_filter_log, dtype=bool)

    re_stats = check_low_re_validity(reynolds_log)
    low_level_convergence_rate = converged_count / max(1, len(step_time_log))
    safety_filter_fraction = safety_filter_count / max(1, len(step_time_log))

    hardware_diag = {}
    if hasattr(magnets, "hardware_diagnostics"):
        hardware_diag = magnets.hardware_diagnostics()

    initial_error = float(np.linalg.norm(start - target))
    final_error = float(np.linalg.norm(trajectory[-1] - target))

    if len(distance_log) > 0:
        route_error_reduction_m = initial_error - final_error
        route_error_reduction_fraction = (
            route_error_reduction_m / initial_error
            if initial_error > 1e-12 else 0.0
        )
    else:
        route_error_reduction_m = 0.0
        route_error_reduction_fraction = 0.0

    if len(waypoint_index_log) > 0 and navigator is not None:
        final_waypoint_idx = int(waypoint_index_log[-1])
        total_waypoints = int(navigator.total_hops)
        waypoint_progress_fraction = final_waypoint_idx / max(1, total_waypoints - 1)
    else:
        final_waypoint_idx = -1
        total_waypoints = 0
        waypoint_progress_fraction = np.nan

    actual_weights = {
        "w_progress": w_progress,
        "w_align": w_align,
        "w_dist": w_dist,
        "w_curr": w_curr,
        "w_smooth": w_smooth,
        "w_edge": w_edge,
        "w_overshoot": w_overshoot,
        "w_route": w_route,
    }
    if actual_weights == ACBC_TUNED_WEIGHTS:
        weight_preset = "acbc_tuned"
    elif actual_weights == MPC_DEFAULT_WEIGHTS:
        weight_preset = "mpc_default"
    else:
        weight_preset = "custom"

    if target_mode == "local_waypoint":
        controller_name = "safety_matched_route_aware_mpc"
        baseline_role = "primary_safety_matched_mpc_baseline"
    else:
        controller_name = "safety_matched_final_target_mpc"
        baseline_role = "secondary_final_target_lower_bound"

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
        final_error=final_error,
        max_current_used=float(np.max(np.abs(current_log))) if len(current_log) else 0.0,
        mean_reynolds=re_stats["mean_reynolds"],
        max_reynolds=re_stats["max_reynolds"],
        low_re_valid=re_stats["low_re_valid"],
        stokes_regime=re_stats["stokes_regime"],
        finite_re_drag_valid=re_stats["finite_re_drag_valid"],
        drag_model=re_stats["drag_model"],
        drag_regime_label=re_stats["drag_regime_label"],
        wall_time_s=wall_end - wall_start,
        avg_control_step_time_s=float(np.mean(step_time_log)) if len(step_time_log) else 0.0,
        max_control_step_time_s=float(np.max(step_time_log)) if len(step_time_log) else 0.0,
        seed=rng_seed,
        profile_name=profile.name,
        dt=dt,
        workspace_half=network.half,
        magnets=magnets.positions.copy(),
        orientations=magnets.orientations.copy(),
        params={
            "controller": controller_name,
            "baseline_role": baseline_role,
            "target_mode": target_mode,
            "weight_preset": weight_preset,
            "low_level_allocator": "continuous_mpc_controller",
            "uses_continuous_mpc_controller": True,
            "uses_same_weights_as_acbc_tuned": weight_preset == "acbc_tuned",
            "uses_same_physics_as_acbc": True,
            "uses_same_gantry_as_acbc": True,
            "uses_same_hardware_as_acbc": True,
            "uses_same_plant_integrator_as_acbc": True,
            "uses_route_tangent": bool(target_mode == "local_waypoint"),
            "uses_waypoint_navigator": bool(navigator is not None),
            "uses_final_target_only": bool(target_mode == "final_target"),
            "uses_wall_buffer_m": True,
            "wall_buffer_m": wall_buffer_m,
            "surface_clearance_target_m": implied_surface_clearance_target_m,
            "wall_clearance_threshold_m": wall_clearance_threshold_m,
            "particle_radius_m": particle.radius,
            "enable_wall_safety_filter": bool(enable_wall_safety_filter),
            "wall_safety_filter_count": int(safety_filter_count),
            "wall_safety_filter_fraction": float(safety_filter_fraction),
            "min_wall_safety_fraction": float(min_wall_safety_fraction),
            "supervisory_logic": "none",
            "branch_commitment": False,
            "authority_assessment": False,
            "recovery_mode": False,
            "stall_escape": False,
            "state_machine": False,
            "state_dwell_time": False,
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
            "profile": profile.name,
            "particle_diameter_m": profile.diameter,
            "particle_diameter_mm": profile.diameter * 1000.0,
            "max_substep_disp": max_substep_disp,
            "vessel_network": network.params_dict(),
            "low_level_convergence_rate": low_level_convergence_rate,
            "navigator_arrival_tol": getattr(navigator, "arrival_tol", None)
            if navigator is not None else None,
            "navigator_waypoint_spacing": getattr(navigator, "waypoint_spacing", None)
            if navigator is not None else None,
            "navigator_lookahead": getattr(navigator, "lookahead", None)
            if navigator is not None else None,
            "navigator_forward_search_points": getattr(
                navigator,
                "forward_search_points",
                None,
            ) if navigator is not None else None,
            "final_waypoint_idx": final_waypoint_idx,
            "total_waypoints": total_waypoints,
            "waypoint_progress_fraction": float(waypoint_progress_fraction)
            if np.isfinite(waypoint_progress_fraction) else np.nan,
            "initial_error_m": initial_error,
            "route_error_reduction_m": float(route_error_reduction_m),
            "route_error_reduction_fraction": float(route_error_reduction_fraction),
            "min_wall_clearance_m": float(np.nanmin(wall_clearance_log))
            if len(wall_clearance_log) > 0 else np.nan,
            "mean_wall_clearance_m": float(np.nanmean(wall_clearance_log))
            if len(wall_clearance_log) > 0 else np.nan,
            "mean_control_target_distance_m": float(np.mean(control_target_distance_log))
            if len(control_target_distance_log) > 0 else np.nan,
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
    result.wall_clearance_log = wall_clearance_log
    result.waypoint_index_log = waypoint_index_log
    result.control_target_log = control_target_log
    result.control_target_distance_log = control_target_distance_log
    result.safety_filter_log = safety_filter_log

    print(
        f"  Safety-matched MPC summary: "
        f"target_mode={target_mode}, "
        f"reached={reached}, "
        f"steps={result.steps_taken}, "
        f"final_error={final_error * 1000:.2f}mm, "
        f"error_reduction={route_error_reduction_fraction:.2%}, "
        f"low-level convergence={low_level_convergence_rate:.2%}, "
        f"safety_filter_fraction={safety_filter_fraction:.2%}",
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
    profile_name: str = "spion_carrier_1mm",
    dt: float = 0.02,
    steps: int = 4000,
    max_substep_disp: float = 1e-3,
    target_mode: str = DEFAULT_TARGET_MODE,
    waypoint_spacing: float = 0.001,
    waypoint_arrival_tol: float = DEFAULT_TARGET_TOLERANCE,
    waypoint_lookahead: int = 1,
    forward_search_points: int = 30,
    surface_clearance_target_m: float = DEFAULT_SURFACE_CLEARANCE_TARGET_M,
    enable_wall_safety_filter: bool = DEFAULT_ENABLE_WALL_SAFETY_FILTER,
    min_wall_safety_fraction: float = 0.015,
    weight_preset: str = DEFAULT_WEIGHT_PRESET,
    w_progress: float | None = None,
    w_align: float | None = None,
    w_dist: float | None = None,
    w_curr: float | None = None,
    w_smooth: float | None = None,
    w_edge: float | None = None,
    w_overshoot: float | None = None,
    w_route: float | None = None,
):
    if device_mode not in VALID_DEVICE_MODES:
        raise ValueError(
            f"device_mode must be one of {sorted(VALID_DEVICE_MODES)}, "
            f"got {device_mode}"
        )

    if target_mode not in VALID_TARGET_MODES:
        raise ValueError(
            f"target_mode must be one of {sorted(VALID_TARGET_MODES)}, "
            f"got {target_mode}"
        )

    weights = _apply_weight_preset(weight_preset)

    if w_progress is not None:
        weights["w_progress"] = float(w_progress)
    if w_align is not None:
        weights["w_align"] = float(w_align)
    if w_dist is not None:
        weights["w_dist"] = float(w_dist)
    if w_curr is not None:
        weights["w_curr"] = float(w_curr)
    if w_smooth is not None:
        weights["w_smooth"] = float(w_smooth)
    if w_edge is not None:
        weights["w_edge"] = float(w_edge)
    if w_overshoot is not None:
        weights["w_overshoot"] = float(w_overshoot)
    if w_route is not None:
        weights["w_route"] = float(w_route)

    profile = resolve_profile(profile_name)

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

    print(
        f"Profile selected: {profile.name}, "
        f"diameter={profile.diameter * 1000:.3f}mm",
        flush=True,
    )

    print(
        f"MPC baseline role: "
        f"{'safety-matched route-aware primary baseline' if target_mode == 'local_waypoint' else 'safety-matched final-target lower bound'}",
        flush=True,
    )

    print(
        f"Weight preset: {weight_preset} | "
        f"w_progress={weights['w_progress']}, "
        f"w_align={weights['w_align']}, "
        f"w_dist={weights['w_dist']}, "
        f"w_curr={weights['w_curr']}, "
        f"w_smooth={weights['w_smooth']}, "
        f"w_edge={weights['w_edge']}, "
        f"w_overshoot={weights['w_overshoot']}, "
        f"w_route={weights['w_route']}",
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

    if target_mode == "local_waypoint":
        output_dir = "outputs/simulation_outputs_3d_safety_matched_route_aware_mpc"
    else:
        output_dir = "outputs/simulation_outputs_3d_safety_matched_final_target_mpc"

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

        result = simulate_trial_naive_mpc(
            start=cfg["start"],
            target=cfg["target"],
            network=network,
            trial_name=trial_name,
            dt=dt,
            steps=steps,
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
            profile=profile,
            max_substep_disp=max_substep_disp,
            target_mode=target_mode,
            waypoint_arrival_tol=waypoint_arrival_tol,
            waypoint_spacing=waypoint_spacing,
            waypoint_lookahead=waypoint_lookahead,
            forward_search_points=forward_search_points,
            surface_clearance_target_m=surface_clearance_target_m,
            enable_wall_safety_filter=enable_wall_safety_filter,
            min_wall_safety_fraction=min_wall_safety_fraction,
            **weights,
        )

        result.params["flow_scale"] = flow_scale
        result.params["weight_preset"] = weight_preset
        result.params["uses_same_weights_as_acbc_tuned"] = bool(
            weight_preset == "acbc_tuned"
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

    print(f"\nResults saved to: {output_dir}/", flush=True)

    return results


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Run safety-matched route-aware MPC baseline."
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
    parser.add_argument("--profile", default="spion_carrier_1mm")

    parser.add_argument("--dt", type=float, default=0.02)
    parser.add_argument("--steps", type=int, default=4000)
    parser.add_argument("--max-substep-disp", type=float, default=1e-3)

    parser.add_argument(
        "--target-mode",
        default=DEFAULT_TARGET_MODE,
        choices=sorted(VALID_TARGET_MODES),
        help=(
            "local_waypoint = safety-matched route-aware MPC baseline. "
            "final_target = safety-matched final-target lower bound."
        ),
    )

    parser.add_argument("--waypoint-spacing", type=float, default=0.001)
    parser.add_argument("--waypoint-arrival-tol", type=float, default=DEFAULT_TARGET_TOLERANCE)
    parser.add_argument("--waypoint-lookahead", type=int, default=1)
    parser.add_argument("--forward-search-points", type=int, default=30)

    parser.add_argument(
        "--surface-clearance-target-m",
        type=float,
        default=DEFAULT_SURFACE_CLEARANCE_TARGET_M,
    )
    parser.add_argument("--min-wall-safety-fraction", type=float, default=0.015)

    wall_filter_group = parser.add_mutually_exclusive_group()
    wall_filter_group.add_argument(
        "--enable-wall-safety-filter",
        action="store_true",
        default=DEFAULT_ENABLE_WALL_SAFETY_FILTER,
    )
    wall_filter_group.add_argument(
        "--disable-wall-safety-filter",
        action="store_false",
        dest="enable_wall_safety_filter",
    )

    parser.add_argument(
        "--weight-preset",
        default=DEFAULT_WEIGHT_PRESET,
        choices=sorted(WEIGHT_PRESETS),
        help=(
            "acbc_tuned uses the same tuned low-level MPC weights as ACBC-tuned. "
            "mpc_default uses standard MPC-default weights."
        ),
    )

    parser.add_argument("--w-progress", type=float, default=None)
    parser.add_argument("--w-align", type=float, default=None)
    parser.add_argument("--w-dist", type=float, default=None)
    parser.add_argument("--w-curr", type=float, default=None)
    parser.add_argument("--w-smooth", type=float, default=None)
    parser.add_argument("--w-edge", type=float, default=None)
    parser.add_argument("--w-overshoot", type=float, default=None)
    parser.add_argument("--w-route", type=float, default=None)

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
        profile_name=args.profile,
        dt=args.dt,
        steps=args.steps,
        max_substep_disp=args.max_substep_disp,
        target_mode=args.target_mode,
        waypoint_spacing=args.waypoint_spacing,
        waypoint_arrival_tol=args.waypoint_arrival_tol,
        waypoint_lookahead=args.waypoint_lookahead,
        forward_search_points=args.forward_search_points,
        surface_clearance_target_m=args.surface_clearance_target_m,
        enable_wall_safety_filter=args.enable_wall_safety_filter,
        min_wall_safety_fraction=args.min_wall_safety_fraction,
        weight_preset=args.weight_preset,
        w_progress=args.w_progress,
        w_align=args.w_align,
        w_dist=args.w_dist,
        w_curr=args.w_curr,
        w_smooth=args.w_smooth,
        w_edge=args.w_edge,
        w_overshoot=args.w_overshoot,
        w_route=args.w_route,
    )
