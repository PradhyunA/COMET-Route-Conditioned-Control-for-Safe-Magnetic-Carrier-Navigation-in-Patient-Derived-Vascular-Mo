"""
simulation_3d_acbc.py
====================

3D ACBC magnetic nanoparticle steering — real vessel network edition.

This runner uses:

    ACBC high-level supervisor
        +
    existing continuous MPC current allocator

ACBC is the main algorithmic contribution.
The MPC allocator is only the low-level coil-current solver.

This version implements multiple physical device modes.

Device modes:
    moving_internal_upper_bound:
        Idealized upper-bound model.
        The magnetic array center follows the particle directly.
        This is NOT physically realistic for a human body, but useful as a
        maximum-authority control benchmark.

    moving_linear_external_gantry:
        One-sided external gantry stress test.
        The magnetic array moves along one straight line parallel to the
        start-target/body/aorta axis while staying outside the route axis by a
        fixed standoff distance.

    moving_linear_enclosing_gantry:
        Scanner/ring-like moving gantry model.
        The magnetic array surrounds the body/route axis and slides in a
        straight line along the body/aorta axis. The gantry center stays on the
        projected route axis, and placement_radius determines how far the coil
        centers are from the body/route axis.

    moving_adaptive_enclosing_gantry:
        Recommended usable external scanner model.
        The array remains physically constrained by the body/scanner axis, but
        it may apply a bounded lateral repositioning offset, like a robotic
        scanner head or patient bed adjustment. It does NOT track the curved
        aorta exactly. The lateral offset is capped by max_lateral_shift.

    fixed_enclosing_stress_test:
        Device stays fixed at the vessel-tree anatomical center.
        This is a hard physical authority stress test.

Current default:
    DEVICE_MODE = "moving_adaptive_enclosing_gantry"

Recommended final working run:
    python -u scripts/simulation_3d_acbc.py

Interpretation:
    placement_radius = ring radius from scanner centerline to coil centers.
    max_lateral_shift = maximum external scanner/bed lateral offset from the
    straight body axis. This makes the system usable without unrealistically
    centering the ring exactly around the curved aorta.
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
# Electromagnetic steering hardware
# ----------------------------------------------------------------------
# These values define the simulated steering hardware.
# They intentionally do NOT scale with vessel-tree size.
# ----------------------------------------------------------------------

# Default scanner/ring radius. Override with --placement-radius.
# Body-scale scanner/ring radius.
# 0.300 m means coil centers are 30 cm from the gantry/body axis.
# This is large enough to represent a human-body-scale enclosing scanner bore.
# ----------------------------------------------------------------------
# Electromagnetic steering hardware
# ----------------------------------------------------------------------
# These values define the simulated steering hardware.
# They intentionally do NOT scale with vessel-tree size.
# ----------------------------------------------------------------------

# Body-scale scanner/ring radius.
# 0.300 m means coil centers are 30 cm from the gantry/body axis.
# This is large enough to represent a human-body-scale enclosing scanner bore.
FIXED_PLACEMENT_RADIUS = 0.300   # m, coil centers 30 cm from gantry/body axis

# High-authority large-bore scanner coil model.
# These are simulation authority parameters, not a solved thermal/electrical build.
FIXED_COIL_RADIUS = 0.080        # m, 16 cm coil diameter
FIXED_COIL_LENGTH = 0.280        # m, 28 cm coil/core length
FIXED_CORE_RADIUS = 0.040        # m, 8 cm core diameter
FIXED_TURNS = 20000.0
FIXED_MAX_CURRENT = 60.0         # A, simulated pulsed/peak current

# Compromise core model.
# 35 / 8.0 was too weak and barely progressed.
# 120 / 2.2 produced strong movement but repeatedly flattened gradients near saturation.
# 90 / 6.0 is the next compromise to restore authority without the old low saturation cap.
FIXED_CORE_MU_R_EFFECTIVE = 90.0
FIXED_CORE_SATURATION_T = 6.0

VALID_DEVICE_MODES = {
    "moving_internal_upper_bound",
    "moving_linear_external_gantry",
    "moving_linear_enclosing_gantry",
    "moving_adaptive_enclosing_gantry",
    "fixed_enclosing_stress_test",
}

DEVICE_MODE = "moving_adaptive_enclosing_gantry"

# Only used for moving_linear_external_gantry.
# Ignored for moving_linear_enclosing_gantry and moving_adaptive_enclosing_gantry.
GANTRY_STANDOFF = 0.180  # m

# Adaptive scanner defaults. This is the recommended usable system model.
# It remains body-axis constrained, but allows limited lateral scanner/bed
# repositioning to keep the target region within controllable magnetic authority.
# Larger body-scale scanner can apply a larger bounded table/scanner offset.
# Adaptive scanner defaults. This is the recommended usable system model.
# It remains body-axis constrained, but allows limited lateral scanner/bed
# repositioning to keep the target region within controllable magnetic authority.
MAX_LATERAL_SHIFT = 0.120  # m, max 12 cm lateral offset from body axis
LATERAL_GAIN = 0.85        # fraction of particle lateral error corrected

# Kept for CLI compatibility only. Fixed-device mode does not use this to scale coils.
DEFAULT_MAGNET_CLEARANCE = 0.04


def magnet_setup_for_network(
    network,
    clearance: float = DEFAULT_MAGNET_CLEARANCE,
    placement_radius: float | None = None,
):
    """
    Build fixed-size magnetic device settings.

    The network is used only to compute anatomical bounds/center.
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


def _unit_vector(v, fallback=None):
    v = np.asarray(v, dtype=float)
    n = float(np.linalg.norm(v))

    if n < 1e-12:
        if fallback is None:
            return np.array([0.0, 0.0, 1.0], dtype=float)
        return _unit_vector(np.asarray(fallback, dtype=float))

    return v / n


def _external_normal_for_axis(axis):
    """
    Choose a stable external normal direction perpendicular to route/body axis.

    The VTP coordinate frame is not guaranteed to match anatomical axes, so we
    build a normal from the route axis itself.
    """
    axis = _unit_vector(axis)

    preferred = np.array([0.0, 1.0, 0.0], dtype=float)
    normal = preferred - np.dot(preferred, axis) * axis

    if np.linalg.norm(normal) < 1e-6:
        preferred = np.array([1.0, 0.0, 0.0], dtype=float)
        normal = preferred - np.dot(preferred, axis) * axis

    return _unit_vector(normal, fallback=np.array([1.0, 0.0, 0.0]))


def _route_axis_projection(
    particle_pos,
    start,
    target,
):
    """
    Project particle progress onto the straight start-target route/body axis.

    This defines the straight-line rail/gantry coordinate. The gantry moves only
    along this axis, not in full 3D with the particle.
    """
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


def _moving_linear_external_gantry_center(
    particle_pos,
    start,
    target,
    standoff: float,
):
    """
    One-sided external gantry model.

    The gantry does NOT center on the internal particle.

    It projects the particle position onto the start-target route axis, then
    places the magnetic device center on a parallel external line offset by
    `standoff`.

    This approximates a one-sided external rail/applicator. It is useful as a
    stress test, but it is NOT the ring/scanner-like model.
    """
    axis_point, route_axis = _route_axis_projection(
        particle_pos=particle_pos,
        start=start,
        target=target,
    )

    external_normal = _external_normal_for_axis(route_axis)
    gantry_center = axis_point + float(standoff) * external_normal

    return gantry_center, axis_point, external_normal, route_axis


def _moving_linear_enclosing_gantry_center(
    particle_pos,
    start,
    target,
):
    """
    Moving scanner/ring-like enclosing gantry model.

    The gantry center moves along the straight route/body axis.
    The body/aorta route passes through the scanner centerline.
    The electromagnets surround that centerline at `placement_radius`.

    There is no side standoff offset here.
    """
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
    max_lateral_shift: float,
    lateral_gain: float,
):
    """
    Physically constrained adaptive enclosing gantry model.

    The scanner primarily travels along the straight body/scanner axis, but it
    can apply a bounded lateral repositioning offset. This represents a robotic
    external scanner head or patient bed adjustment, not a ring magically
    centered on the curved aorta.

    The center is:
        body_axis_projection + clipped(lateral_gain * lateral_error)

    where lateral_error is the particle's offset from the body axis.
    """
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


def _initial_magnet_center_for_mode(
    device_mode,
    start,
    target,
    network,
    gantry_standoff,
):
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
    """
    Apply device motion rule and return diagnostic geometry.
    """
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


def save_acbc_diagnostics(results, output_dir, filename="acbc_diagnostics.csv"):
    """
    Save per-step ACBC state-machine and device-motion diagnostics.
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
        "gantry_center_x",
        "gantry_center_y",
        "gantry_center_z",
        "gantry_axis_point_x",
        "gantry_axis_point_y",
        "gantry_axis_point_z",
        "gantry_particle_dist_m",
        "gantry_axis_dist_m",
        "gantry_lateral_shift_x",
        "gantry_lateral_shift_y",
        "gantry_lateral_shift_z",
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
                        *row["gantry_center"],
                        *row["gantry_axis_point"],
                        row["gantry_particle_dist_m"],
                        row["gantry_axis_dist_m"],
                        *row["gantry_lateral_shift"],
                    ]
                )

    print(f"ACBC diagnostics saved to: {path}")


def simulate_trial_acbc(
    start,
    target,
    network,
    trial_name: str = "trial",
    dt: float = 0.02,
    steps: int = 3200,
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
    device_mode: str = DEVICE_MODE,
    gantry_standoff: float = GANTRY_STANDOFF,
    max_lateral_shift: float = MAX_LATERAL_SHIFT,
    lateral_gain: float = LATERAL_GAIN,
    profile: PhysicalProfile = SPION_CARRIER_10UM,
    max_substep_disp: float = 1e-3,
    rng_seed: int = 42,

    # High-authority weights.
    # These are intentionally aggressive because weak penalties previously made
    # the current allocator avoid using the available 12 A authority.
    w_progress: float = 450.0,
    w_align: float = 35.0,
    w_dist: float = 2.0,
    w_curr: float = 1e-7,
    w_smooth: float = 1e-6,
    w_edge: float = 250.0,
    w_overshoot: float = 8.0,
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
        authority_threshold=1.02,
        recovery_threshold=0.65,

        # More conservative wall guard.
        min_wall_safety_fraction=0.015,
        absolute_wall_clearance_threshold=0.00075,
        stall_escape_min_wall_clearance=0.00100,
        stall_escape_min_wall_fraction_multiplier=1.00,

        # Pure recentering recovery.
        recovery_center_fraction=1.00,
        recovery_route_fraction=0.00,
        max_recovery_steps=60,

        # Stall escape only when safely away from wall.
        stall_window_steps=10,
        stall_escape_lookahead=90,
        stall_escape_release_hops=18,
        stall_escape_max_steps=260,

        recovery_escape_lockout_steps=0,
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
        f"gantry_standoff={gantry_standoff * 1000:.1f} mm, "
        f"max_lateral_shift={max_lateral_shift * 1000:.1f} mm, "
        f"lateral_gain={lateral_gain:.2f}, "
        f"initial_center={np.round(initial_device_diag['gantry_center'] * 1000, 1)} mm, "
        f"initial_axis_dist={initial_device_diag['gantry_axis_dist'] * 1000:.1f} mm, "
        f"initial_particle_dist={initial_device_diag['gantry_particle_dist'] * 1000:.1f} mm"
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
        gantry_center_log.append(device_diag["gantry_center"].copy())
        gantry_particle_dist_log.append(device_diag["gantry_particle_dist"])
        gantry_axis_dist_log.append(device_diag["gantry_axis_dist"])
        gantry_lateral_shift_log.append(device_diag["gantry_lateral_shift"].copy())

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
                "gantry_center": _vec_or_nan(device_diag["gantry_center"]),
                "gantry_axis_point": _vec_or_nan(device_diag["gantry_axis_point"]),
                "gantry_particle_dist_m": float(device_diag["gantry_particle_dist"]),
                "gantry_axis_dist_m": float(device_diag["gantry_axis_dist"]),
                "gantry_lateral_shift": _vec_or_nan(device_diag["gantry_lateral_shift"]),
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
                _, _, radial_dist, _ = seg.closest_point_and_radial(particle.pos)
                wall_clearance = seg.radius - radial_dist
            else:
                wall_clearance = np.nan

            lateral_shift_norm = np.linalg.norm(device_diag["gantry_lateral_shift"])

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
                f"Re={re:.2e}  "
                f"gantry_axis_dist={device_diag['gantry_axis_dist'] * 1000:.1f} mm  "
                f"gantry_particle_dist={device_diag['gantry_particle_dist'] * 1000:.1f} mm  "
                f"lateral_shift={lateral_shift_norm * 1000:.1f} mm"
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
            if len(gantry_particle_dist_log) > 0
            else np.nan,
            "gantry_mean_axis_dist_m": float(np.mean(gantry_axis_dist_log))
            if len(gantry_axis_dist_log) > 0
            else np.nan,
            "gantry_mean_lateral_shift_m": float(
                np.mean(np.linalg.norm(gantry_lateral_shift_log, axis=1))
            )
            if len(gantry_lateral_shift_log) > 0
            else np.nan,
            "gantry_max_lateral_shift_m": float(
                np.max(np.linalg.norm(gantry_lateral_shift_log, axis=1))
            )
            if len(gantry_lateral_shift_log) > 0
            else np.nan,
            "core_relative_permeability": core_relative_permeability,
            "core_saturation_tesla": core_saturation_tesla,
            "follow_particle_device": device_mode == "moving_internal_upper_bound",
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
    result.gantry_center_log = gantry_center_log
    result.gantry_particle_dist_log = gantry_particle_dist_log
    result.gantry_axis_dist_log = gantry_axis_dist_log
    result.gantry_lateral_shift_log = gantry_lateral_shift_log

    print(
        f"  ACBC summary: transitions={acbc_summary['acbc_transitions']}, "
        f"commitments={acbc_summary['acbc_commitments']}, "
        f"recoveries={acbc_summary['acbc_recoveries']}, "
        f"low-level convergence={low_level_convergence_rate:.2%}"
    )

    if len(gantry_axis_dist_log) > 0:
        lateral_norms = np.linalg.norm(gantry_lateral_shift_log, axis=1)

        print(
            f"  Gantry summary: "
            f"mean_axis_dist={np.mean(gantry_axis_dist_log) * 1000:.1f} mm, "
            f"mean_particle_dist={np.mean(gantry_particle_dist_log) * 1000:.1f} mm, "
            f"min_particle_dist={np.min(gantry_particle_dist_log) * 1000:.1f} mm, "
            f"max_particle_dist={np.max(gantry_particle_dist_log) * 1000:.1f} mm, "
            f"mean_lateral_shift={np.mean(lateral_norms) * 1000:.1f} mm, "
            f"max_lateral_shift={np.max(lateral_norms) * 1000:.1f} mm"
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
):
    if device_mode not in VALID_DEVICE_MODES:
        raise ValueError(
            f"device_mode must be one of {sorted(VALID_DEVICE_MODES)}, "
            f"got {device_mode}"
        )

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
    anatomical_center = magnet_setup["center"]

    print(
        "Magnet setup: "
        f"device_mode={device_mode}, "
        f"anatomical_center={np.round(anatomical_center, 4)} m, "
        f"placement_radius={placement_radius:.3f} m, "
        f"coil_radius={coil_radius:.3f} m, "
        f"coil_diameter={2.0 * coil_radius:.3f} m, "
        f"coil_length={coil_length:.3f} m, "
        f"core_radius={core_radius:.3f} m, "
        f"turns={turns:.0f}, "
        f"max_current={max_current:.1f} A, "
        f"gantry_standoff={gantry_standoff:.3f} m, "
        f"max_lateral_shift={max_lateral_shift:.3f} m, "
        f"lateral_gain={lateral_gain:.2f}"
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

    if only_trial is not None:
        if only_trial not in trials:
            raise ValueError(
                f"--only-trial must be one of {sorted(trials)}, got {only_trial}"
            )
        trials = {only_trial: trials[only_trial]}

    if device_mode == "moving_linear_external_gantry":
        output_dir = "outputs/simulation_outputs_3d_acbc_external_gantry"
    elif device_mode == "moving_linear_enclosing_gantry":
        output_dir = "outputs/simulation_outputs_3d_acbc_enclosing_gantry"
    elif device_mode == "moving_adaptive_enclosing_gantry":
        output_dir = "outputs/simulation_outputs_3d_acbc_adaptive_enclosing_gantry"
    elif device_mode == "moving_internal_upper_bound":
        output_dir = "outputs/simulation_outputs_3d_acbc_moving_internal"
    elif device_mode == "fixed_enclosing_stress_test":
        output_dir = "outputs/simulation_outputs_3d_acbc_fixed_stress"
    else:
        raise ValueError(f"Unknown device_mode: {device_mode}")

    os.makedirs(output_dir, exist_ok=True)

    results = []

    # Representative plotting array.
    # For moving modes, this is a representative static snapshot only.
    # If --only-trial is used, use the first remaining trial rather than
    # assuming trial_1_inlet_to_outlet still exists.
    if device_mode == "fixed_enclosing_stress_test":
        plot_center = anatomical_center
    else:
        first_trial_cfg = next(iter(trials.values()))
        first_start = first_trial_cfg["start"]
        first_target = first_trial_cfg["target"]

        plot_center = _initial_magnet_center_for_mode(
            device_mode=device_mode,
            start=first_start,
            target=first_target,
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
            "Override magnet placement radius in meters. "
            "Default is 0.3 m."
        ),
    )

    parser.add_argument(
        "--device-mode",
        default=DEVICE_MODE,
        choices=sorted(VALID_DEVICE_MODES),
        help="Device motion model to run.",
    )

    parser.add_argument(
        "--gantry-standoff",
        type=float,
        default=GANTRY_STANDOFF,
        help=(
            "Distance from route/body axis to moving external gantry center "
            "in meters. Used only for moving_linear_external_gantry. "
            "Ignored for moving_linear_enclosing_gantry."
        ),
    )

    parser.add_argument(
        "--max-lateral-shift",
        type=float,
        default=MAX_LATERAL_SHIFT,
        help=(
            "Maximum lateral scanner/bed offset from the straight body axis, "
            "in meters. Used for moving_adaptive_enclosing_gantry."
        ),
    )

    parser.add_argument(
        "--lateral-gain",
        type=float,
        default=LATERAL_GAIN,
        help=(
            "Fraction of particle lateral error corrected by the adaptive "
            "gantry, before clipping to --max-lateral-shift."
        ),
    )

    parser.add_argument(
        "--only-trial",
        default=None,
        choices=[
            "trial_1_inlet_to_outlet",
            "trial_2_mid_to_outlet",
            "trial_3_inlet_to_mid",
        ],
        help="Run only one named trial. Useful for tuning Trial 2 quickly.",
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
    )