"""
simulation_3d_comet.py
======================


3D magnetic carrier steering with COMET
(Commitment-Oriented Measure-Transport Control).


This wrapper mirrors simulation_3d_mpc.py / simulation_3d_acbc.py:
   - identical physical setup (scanner constants, gantry motion, profile)
   - identical trial bookkeeping (TrialResult, logs, CSV/NPZ outputs)
   - the only difference is the controller: COMETController instead of
     the MPC current allocator or the ACBC supervisor.


Recommended run:
   python -u scripts/simulation_3d_comet.py --flow-scale 5
"""


import os
import sys
import csv
import time
import argparse


sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))


import numpy as np


from dynamag_simulation.core.constants import DEFAULT_TARGET_TOLERANCE
from dynamag_simulation.core.profiles import PhysicalProfile, SPION_CARRIER_1MM
from dynamag_simulation.core.magnets import ElectromagnetArray3D
from dynamag_simulation.core.particle import MagneticParticle3D
from dynamag_simulation.core.integrator import adaptive_step
from dynamag_simulation.core.comet_controller import COMETConfig, COMETController
from dynamag_simulation.core.validation import compute_reynolds, check_low_re_validity
from dynamag_simulation.core.reporting import TrialResult, print_summary, save_results_csv
from dynamag_simulation.core.real_vessel_network import (
   RealVesselNetwork,
   RoutePlanningOptions,
)
from dynamag_simulation.core.workspace import WorkspaceVessel


from simulation_3d_pid import (
   DEFAULT_VESSEL_FILE,
   DEVICE_MODE,
   FIXED_COIL_LENGTH,
   FIXED_COIL_RADIUS,
   FIXED_CORE_MU_R_EFFECTIVE,
   FIXED_CORE_RADIUS,
   FIXED_CORE_SATURATION_T,
   FIXED_MAX_CURRENT,
   FIXED_PLACEMENT_RADIUS,
   FIXED_TURNS,
   GANTRY_STANDOFF,
   LATERAL_GAIN,
   MAX_LATERAL_SHIFT,
   VALID_DEVICE_MODES,
   _apply_device_motion,
   _initial_magnet_center_for_mode,
   _set_currents_with_optional_thermal,
)

DEFAULT_FLOW_SCALE = 5.0


def magnet_setup_for_network(network, clearance: float = 0.04, placement_radius=None):
   bounds_min, bounds_max = network.bounds(include_radius=True)
   center = 0.5 * (bounds_min + bounds_max)
   if placement_radius is None:
      placement_radius = FIXED_PLACEMENT_RADIUS
   return {
      "center": center,
      "placement_radius": float(placement_radius),
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




def save_comet_diagnostics(results, output_dir, filename="comet_diagnostics.csv"):
   """Per-step COMET intent/commitment diagnostics for a batch of trials."""
   path = os.path.join(output_dir, filename)


   header = [
       "trial_name", "step", "time_s", "mode", "previous_mode", "transition",
       "dist_to_bifurcation_m", "selected_branch_seg_id", "selected_reserve_m",
       "selected_capture_probability", "temperature", "intent_entropy",
       "num_candidates", "stagnant_steps", "wall_clearance_m",
       "route_arc_m", "control_norm_A", "commit_reason",
       "safety_filter_active", "q_json", "terminal_mode_active",
       "terminal_stage", "target_distance_m", "target_flow_x_m_s",
       "target_flow_y_m_s", "target_flow_z_m_s",
       "terminal_correction_x_m_s", "terminal_correction_y_m_s",
       "terminal_correction_z_m_s", "current_saturation",
       "final_hold_steps", "overshoot_count", "direct_terminal_feasible",
   ]


   with open(path, "w", newline="") as f:
       writer = csv.writer(f)
       writer.writerow(header)


       for result in results:
           for row in getattr(result, "comet_log", []):
               writer.writerow([result.trial_name, *row])


   print(f"COMET diagnostics saved to: {path}")




def simulate_trial_comet(
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
   comet_config: COMETConfig | None = None,
   verbose_every: int = 25,
   **comet_overrides,
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


   _apply_device_motion(
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


   config = comet_config or COMETConfig()
   for key, value in comet_overrides.items():
       if not hasattr(config, key):
           raise TypeError(f"Unknown COMET config field: {key}")
       setattr(config, key, value)

   particle_radius = 0.5 * profile.diameter
   planning_options = RoutePlanningOptions(
       wall_clearance_m=wall_clearance_threshold_m,
       particle_radius_m=particle_radius,
       minimum_surface_clearance_m=max(
           wall_clearance_threshold_m - particle_radius, 0.0,
       ),
       magnetic_authority_m_s=config.cruise_speed,
       minimum_authority_reserve_m_s=0.0,
   )
   navigator = network.waypoint_navigator(
       start,
       target,
       arrival_tol=DEFAULT_TARGET_TOLERANCE,
       waypoint_spacing=0.0015,
       lookahead=3,
       forward_search_points=50,
       planning_options=planning_options,
   )
   if not navigator.route_plan.feasible:
       raise ValueError(
           f"No feasible COMET route: {navigator.route_plan.reason}"
       )


   comet = COMETController(
       network=network,
       magnet_array=magnets,
       workspace=workspace,
       max_current=max_current,
       config=config,
       wall_clearance_threshold_m=wall_clearance_threshold_m,
   )


   print(
       f"    Profile: {profile.name}, diameter={profile.diameter * 1000:.3f} mm",
       flush=True,
   )
   print(
       f"    COMET: route waypoints={navigator.total_hops}, "
       f"window={config.bifurcation_window * 1000:.0f}mm, "
       f"commit_dist={config.commitment_distance_threshold * 1000:.0f}mm, "
       f"T_far={config.entropy_temperature_far:.2f}, "
       f"T_near={config.entropy_temperature_near:.2f}",
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
   comet_log = []
   transition_log = []


   reached = False
   prev_currents = np.zeros(magnets.n_magnets, dtype=float)
   wall_start = time.perf_counter()


   import json


   for step in range(steps):
       sim_time = step * dt
       t0 = time.perf_counter()


       _apply_device_motion(
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


       best_currents, best_force, best_cost, converged, info = comet.step(
           particle=particle,
           final_target=target,
           navigator=navigator,
           dt=dt,
           flow=flow,
           sim_time=sim_time,
           initial_guess=prev_currents,
       )


       t1 = time.perf_counter()
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


       comet_log.append([
           step, sim_time, info.mode, info.previous_mode, info.transition,
           info.dist_to_bifurcation, info.selected_branch_seg_id,
           info.selected_reserve, info.selected_capture_probability,
           info.temperature, info.entropy, info.num_candidates,
           info.stagnant_steps, info.wall_clearance_m, info.route_arc_m,
           info.control_norm, info.commit_reason,
           info.safety_filter_active,
           json.dumps({str(k): round(v, 4) for k, v in info.q.items()}),
           info.terminal_mode_active, info.terminal_stage,
           info.target_distance_m, *info.predicted_flow_at_target,
           *info.terminal_correction, info.current_saturation,
           info.final_hold_steps, info.overshoot_count,
           info.direct_terminal_feasible,
       ])
       # Compatibility log so the shared benchmark CSV can count
       # mode transitions for COMET like it does for ACBC.
       transition_log.append({
           "state": info.mode,
           "transition": bool(info.transition),
       })


       if distance < tolerance:
           reached = True
           break


       if verbose_every and (step + 1) % verbose_every == 0:
           print(
               f"  step {step + 1:>4d}/{steps}  "
               f"mode={info.mode:<14s}  "
               f"dist={distance * 1000:.1f}mm  "
               f"bif={info.dist_to_bifurcation * 1000:.1f}mm  "
               f"H={info.entropy:.2f}  "
               f"res={info.selected_reserve * 1000 if np.isfinite(info.selected_reserve) else np.nan:.1f}mm  "
               f"wall={info.wall_clearance_m * 1000:.2f}mm  "
               f"|I|={np.max(np.abs(best_currents)):.1f}A  "
               f"v={np.linalg.norm(particle.realized_vel) * 1000:.2f}mm/s",
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
           "controller": "COMET_measure_transport",
           "low_level_allocator": "comet_local_surrogate_cem",
           "device_mode": device_mode,
           "dt": dt,
           "steps": steps,
           "tolerance": tolerance,
           "coil_radius": coil_radius,
           "coil_length": coil_length,
           "core_radius": core_radius,
           "turns": turns,
           "max_current": max_current,
           "placement_radius": placement_radius,
           "core_relative_permeability": core_relative_permeability,
           "core_saturation_tesla": core_saturation_tesla,
           "profile": profile.name,
           "particle_diameter_m": profile.diameter,
           "particle_diameter_mm": profile.diameter * 1000.0,
           "max_substep_disp": max_substep_disp,
           "wall_clearance_threshold_m": wall_clearance_threshold_m,
           "vessel_network": network.params_dict(),
           **comet.summary_dict(),
           **hardware_diag,
       },
   )


   result.comet_log = comet_log
   # Shared paired-CSV compatibility: per-step state/transition rows.
   result.acbc_log = transition_log


   print(
       f"  COMET summary: hard_commits={comet.hard_commit_count}, "
       f"transitions={comet.transition_count}, "
       f"emergency_switches={comet.emergency_switch_count}, "
       f"avg_ctrl={result.avg_control_step_time_s * 1000:.1f}ms",
       flush=True,
   )


   return result




def run_all_trials(
   vessel_file: str = DEFAULT_VESSEL_FILE,
   device_mode: str = DEVICE_MODE,
   only_trial: str | None = None,
   flow_scale: float = DEFAULT_FLOW_SCALE,
):
   network = RealVesselNetwork(vessel_file, flow_scale=flow_scale)
   print(network.segment_summary(), flush=True)
   print(f"Flow scale: {flow_scale:.2f}x", flush=True)


   magnet_setup = magnet_setup_for_network(network)


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


   output_dir = "outputs/simulation_outputs_3d_comet"
   os.makedirs(output_dir, exist_ok=True)


   results = []


   for trial_name, cfg in trials.items():
       print(f"\n>>> Running {trial_name}", flush=True)
       result = simulate_trial_comet(
           start=cfg["start"],
           target=cfg["target"],
           network=network,
           trial_name=trial_name,
           coil_radius=magnet_setup["coil_radius"],
           coil_length=magnet_setup["coil_length"],
           core_radius=magnet_setup["core_radius"],
           turns=magnet_setup["turns"],
           max_current=magnet_setup["max_current"],
           placement_radius=magnet_setup["placement_radius"],
           core_relative_permeability=magnet_setup["core_relative_permeability"],
           core_saturation_tesla=magnet_setup["core_saturation_tesla"],
           device_mode=device_mode,
           profile=SPION_CARRIER_1MM,
       )
       result.params["flow_scale"] = flow_scale
       results.append(result)
       print_summary(result)


   save_results_csv(results, output_dir)
   save_comet_diagnostics(results, output_dir)
   print(f"\nResults saved to: {output_dir}/", flush=True)
   return results




if __name__ == "__main__":
   parser = argparse.ArgumentParser(
       description="Run COMET on the real vessel network."
   )
   parser.add_argument("--vessel-file", default=DEFAULT_VESSEL_FILE)
   parser.add_argument(
       "--device-mode", default=DEVICE_MODE, choices=sorted(VALID_DEVICE_MODES),
   )
   parser.add_argument("--flow-scale", type=float, default=DEFAULT_FLOW_SCALE)
   parser.add_argument("--only-trial", default=None)
   args = parser.parse_args()


   run_all_trials(
       vessel_file=args.vessel_file,
       device_mode=args.device_mode,
       only_trial=args.only_trial,
       flow_scale=args.flow_scale,
   )
