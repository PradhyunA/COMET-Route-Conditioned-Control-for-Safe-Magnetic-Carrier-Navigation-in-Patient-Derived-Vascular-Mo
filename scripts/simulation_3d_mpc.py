"""
simulation_3d_mpc.py
====================
3D MPC magnetic particle steering — real vessel network edition.

Intended run command:
    .venv/bin/python scripts/simulation_3d_mpc.py
"""

import os
import sys
import time
import warnings
import argparse

# Add parent directory to path to allow importing dynamag_simulation when run directly
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

import numpy as np

from dynamag_simulation.core.profiles import PhysicalProfile, MICROSPHERE_500UM
from dynamag_simulation.core.magnets import ElectromagnetArray3D
from dynamag_simulation.core.particle import MagneticParticle3D
from dynamag_simulation.core.integrator import adaptive_step
from dynamag_simulation.core.controllers import continuous_mpc_controller
from dynamag_simulation.core.validation import compute_reynolds, check_low_re_validity
from dynamag_simulation.core.reporting import TrialResult, print_summary, save_results_csv
from dynamag_simulation.core.real_vessel_network import RealVesselNetwork
from dynamag_simulation.core.workspace import WorkspaceVessel


DEFAULT_VESSEL_FILE = "data/real_vessel_segments_cm_corrected.pkl"
DEFAULT_MAGNET_CLEARANCE = 0.04
DEFAULT_COIL_RADIUS_FRACTION = 0.5
DEFAULT_COIL_LENGTH_FACTOR = 2.0


def vessel_geometry_stats(network):
    """Return tube-inclusive span and diameter stats in millimeters."""
    bounds_min, bounds_max = network.bounds(include_radius=True)
    diameters_mm = np.array([2.0 * seg.radius * 1000.0 for seg in network.segments])

    return {
        "bounds_min": bounds_min,
        "bounds_max": bounds_max,
        "span_mm": (bounds_max - bounds_min) * 1000.0,
        "diameter_min_mm": float(np.min(diameters_mm)),
        "diameter_median_mm": float(np.median(diameters_mm)),
        "diameter_max_mm": float(np.max(diameters_mm)),
    }


def print_geometry_sanity_check(network):
    """Print a compact scale check for the real-vessel pickle."""
    stats = vessel_geometry_stats(network)
    span_mm = stats["span_mm"]
    d_min = stats["diameter_min_mm"]
    d_med = stats["diameter_median_mm"]
    d_max = stats["diameter_max_mm"]

    print(
        "Geometry sanity: "
        f"source={network.source_file}, "
        f"tube_span={np.round(span_mm, 1)} mm, "
        f"diameter min/median/max={d_min:.1f}/{d_med:.1f}/{d_max:.1f} mm",
        flush=True,
    )

    if d_max < 10.0:
        warnings.warn(
            "Vessel diameters look pre-scaled/small; expected corrected "
            "aorta diameters up to roughly 30 mm.",
            RuntimeWarning,
            stacklevel=2,
        )
    elif d_max > 80.0:
        warnings.warn(
            "Vessel diameters look double-scaled/large; expected corrected "
            "aorta diameters well below 80 mm.",
            RuntimeWarning,
            stacklevel=2,
        )


def magnet_setup_for_network(
    network,
    clearance: float = DEFAULT_MAGNET_CLEARANCE,
    placement_radius: float | None = None,
):
    """
    Size and center the triangular-prism magnet array around the vessel tree.

    ``placement_radius`` is the local magnet layout scale ``d``. For this
    prism, x and z extents are +/- d and y extents are [-d/sqrt(3), 2d/sqrt(3)].
    """
    bounds_min, bounds_max = network.bounds(include_radius=True)
    center = 0.5 * (bounds_min + bounds_max)
    span = bounds_max - bounds_min

    required = max(
        0.5 * span[0],
        0.5 * np.sqrt(3.0) * span[1],
        0.5 * span[2],
    )

    if placement_radius is None:
        placement_radius = required + clearance
    elif placement_radius < required:
        raise ValueError(
            "placement_radius is too small for this vessel geometry: "
            f"{placement_radius:.3f} m < required {required:.3f} m "
            "(tube-inclusive bounds would not fit inside the magnet layout)."
        )
    elif placement_radius < required + clearance:
        warnings.warn(
            "placement_radius fits the vessel bounds but provides less than "
            f"the requested clearance ({clearance:.3f} m).",
            RuntimeWarning,
            stacklevel=2,
        )

    placement_radius = float(placement_radius)

    return {
        "center": center,
        "placement_radius": placement_radius,
        "coil_radius": DEFAULT_COIL_RADIUS_FRACTION * placement_radius,
        "coil_length": DEFAULT_COIL_LENGTH_FACTOR * placement_radius,
        "bounds_min": bounds_min,
        "bounds_max": bounds_max,
        "required_radius": required,
    }


def resolve_magnet_setup(
    network,
    *,
    magnet_clearance: float = DEFAULT_MAGNET_CLEARANCE,
    placement_radius: float | None = None,
    coil_radius: float | None = None,
    coil_length: float | None = None,
    magnet_center=None,
):
    """
    Fill in any omitted magnet geometry from the vessel-centered setup.

    This keeps direct ``simulate_trial_mpc()`` calls from falling back to the
    obsolete origin-centered 6 cm cubic-workspace layout.
    """
    setup = magnet_setup_for_network(
        network,
        clearance=magnet_clearance,
        placement_radius=placement_radius,
    )

    return {
        "center": setup["center"]
        if magnet_center is None
        else np.asarray(magnet_center, dtype=float),
        "placement_radius": setup["placement_radius"],
        "coil_radius": setup["coil_radius"] if coil_radius is None else float(coil_radius),
        "coil_length": setup["coil_length"] if coil_length is None else float(coil_length),
        "bounds_min": setup["bounds_min"],
        "bounds_max": setup["bounds_max"],
        "required_radius": setup["required_radius"],
    }


def simulate_trial_mpc(
    start,
    target,
    network,
    trial_name: str = "trial",
    dt: float = 0.02,
    steps: int = 500,
    tolerance: float = 0.003,
    coil_radius: float | None = None,
    coil_length: float | None = None,
    turns: float = 1500,
    max_current: float = 8.0,
    placement_radius: float | None = None,
    magnet_center=None,
    magnet_clearance: float = DEFAULT_MAGNET_CLEARANCE,
    profile: PhysicalProfile = MICROSPHERE_500UM,
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

    magnet_setup = resolve_magnet_setup(
        network,
        magnet_clearance=magnet_clearance,
        placement_radius=placement_radius,
        coil_radius=coil_radius,
        coil_length=coil_length,
        magnet_center=magnet_center,
    )
    coil_radius = magnet_setup["coil_radius"]
    coil_length = magnet_setup["coil_length"]
    placement_radius = magnet_setup["placement_radius"]
    magnet_center = magnet_setup["center"]

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
                f"final_dist={distance * 1000:.1f} mm  "
                f"wp={getattr(navigator, 'current_idx', 0) + 1}/"
                f"{navigator.total_hops}  "
                f"lookahead={getattr(navigator, 'lookahead_idx', 0) + 1}  "
                f"wp_dist={getattr(navigator, 'current_wp_dist', np.nan) * 1000:.2f} mm  "
                f"|I|max={np.max(np.abs(best_currents)):.3f} A  "
                f"|F|={np.linalg.norm(best_force):.3e} N  "
                f"particle_v={particle_speed * 1000:.2f} mm/s  "
                f"flow_v={flow_speed * 1000:.2f} mm/s  "
                f"wall_clear={wall_clearance * 1000:.2f} mm  "
                f"Re={re:.2e}",
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
        },
    )


def run_all_trials(
    vessel_file: str = DEFAULT_VESSEL_FILE,
    magnet_clearance: float = DEFAULT_MAGNET_CLEARANCE,
    placement_radius: float | None = None,
    flow_scale: float = 1.0,
):

    network = RealVesselNetwork(vessel_file,
                                flow_scale=flow_scale)

    print_geometry_sanity_check(network)
    print(network.segment_summary(), flush=True)

    magnet_setup = resolve_magnet_setup(
        network,
        magnet_clearance=magnet_clearance,
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
        f"coil_length={coil_length:.3f} m",
        flush=True,
    )
    print(
        "Vessel tube bounds: "
        f"min={np.round(magnet_setup['bounds_min'], 4)} m, "
        f"max={np.round(magnet_setup['bounds_max'], 4)} m",
        flush=True,
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

    output_dir = "outputs/simulation_outputs_3d_mpc"

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
    print(
        "Magnet layout bounds: "
        f"min={np.round(magnets_for_plots.positions.min(axis=0), 4)} m, "
        f"max={np.round(magnets_for_plots.positions.max(axis=0), 4)} m",
        flush=True,
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
            placement_radius=placement_radius,
            magnet_center=magnet_center,
            magnet_clearance=magnet_clearance,
        )

        results.append(result)

        print_summary(result)

        from dynamag_simulation.core.plotting import save_all_plots

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
        description="Run real-vessel MPC steering on the corrected Stanford VMR aorta geometry."
    )
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
    parser.add_argument(
        "--flow-scale",
        type=float,
        default=1.0,
        help="Global multiplier for Murray's-law flow speeds; geometry is unchanged.",
    )
    args = parser.parse_args()
    run_all_trials(
        vessel_file=args.vessel_file,
        magnet_clearance=args.magnet_clearance,
        placement_radius=args.placement_radius,
        flow_scale=args.flow_scale,
    )