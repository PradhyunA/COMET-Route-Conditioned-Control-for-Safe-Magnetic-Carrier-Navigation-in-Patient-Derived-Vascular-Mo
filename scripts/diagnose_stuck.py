"""
Investigate why particle gets stuck at wp8 — the force collapses from 68nN to 2.8nN.
"""
import os, sys
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

import numpy as np
from dynamag_simulation.core.profiles import MICROSPHERE_500UM
from dynamag_simulation.core.magnets import ElectromagnetArray3D
from dynamag_simulation.core.particle import MagneticParticle3D
from dynamag_simulation.core.real_vessel_network import RealVesselNetwork
from dynamag_simulation.core.workspace import WorkspaceVessel
from dynamag_simulation.core.controllers import _simulate_one_step, continuous_mpc_controller
from dynamag_simulation.core.integrator import adaptive_step
from dynamag_simulation.core.constants import DEFAULT_TARGET_TOLERANCE

network = RealVesselNetwork("real_vessel_segments.pkl", flow_scale=1.0)
workspace = WorkspaceVessel(network)
magnets = ElectromagnetArray3D(coil_radius=0.03, coil_length=0.12, turns=1500, max_current=8.0, placement_radius=0.06)

inlet = network.inlet_start()
outlet = network.outlet_end()

# Get waypoints
nav = network.waypoint_navigator(inlet, outlet, arrival_tol=DEFAULT_TARGET_TOLERANCE, waypoint_spacing=0.0015, lookahead=3)
wps = nav.waypoints

print("=== Waypoints ===")
for i, wp in enumerate(wps[:15]):
    seg, r = network.find_segment(wp)
    in_vessel = network.in_any_vessel(wp)
    print(f"wp{i+1}: {wp*1000} mm, in_vessel={in_vessel}, seg={seg.seg_id if seg else None}, r/R={r/seg.radius:.3f}" if seg else f"wp{i+1}: {wp*1000} mm, NO SEGMENT")

# The stuck position from the log is near wp8
# From log: step 60 position ≈ the final stuck position
# final_position = [ 0.00361945 -0.00092925 -0.03738115]
stuck_pos = np.array([0.00361945, -0.00092925, -0.03738115])
print(f"\n=== At stuck position {stuck_pos*1000} mm ===")

seg, r = network.find_segment(stuck_pos)
in_vessel = network.in_any_vessel(stuck_pos)
print(f"In vessel: {in_vessel}")
print(f"Nearest segment: {seg.seg_id}, radius={seg.radius*1000:.3f} mm")
print(f"Radial dist: {r*1000:.4f} mm, r/R={r/seg.radius:.3f}")
print(f"Wall clearance: {(seg.radius - r)*1000:.4f} mm")
print(f"Segment start: {seg.start*1000}, end: {seg.end*1000}")
print(f"Segment length: {seg._length*1000:.4f} mm")
print(f"Segment axis: {seg._axis}")
print(f"v_mean: {seg.v_mean*1000:.3f} mm/s")

# Check flow at stuck position
flow_v = network.velocity_at(stuck_pos, 0.0)
print(f"Flow at stuck pos: {np.linalg.norm(flow_v)*1000:.3f} mm/s")

# Check magnetic force at stuck position
particle = MagneticParticle3D(position=stuck_pos, profile=MICROSPHERE_500UM, rng_seed=42)
all_max = np.full(8, 8.0)
magnets.set_currents(all_max)
B = magnets.field_at(stuck_pos)
grad_Bsq = magnets.grad_B_squared(stuck_pos)
force = particle.magnetic_force(B, grad_Bsq)
print(f"\n|B| at stuck pos: {np.linalg.norm(B)*1e3:.4f} mT")
print(f"|F| at stuck pos (all 8A): {np.linalg.norm(force):.3e} N")

# Check MPC at stuck position
print(f"\n=== MPC at stuck position ===")
target_wp = wps[10]  # wp11 = lookahead target
print(f"Target (wp11): {target_wp*1000} mm")
print(f"Distance to target: {np.linalg.norm(stuck_pos - target_wp)*1000:.2f} mm")

best_curr, best_force, best_cost, converged = continuous_mpc_controller(
    particle=particle, particle_pos=stuck_pos, target_pos=target_wp,
    magnet_array=magnets, workspace=workspace, dt=0.02, max_current=8.0,
    fast_mode=False, flow=network, sim_time=0.0,
    w_progress=120.0, w_align=8.0, w_dist=1.0, w_curr=1e-5,
    w_smooth=1e-4, w_edge=50.0, w_overshoot=2.0,
)
print(f"|F|: {np.linalg.norm(best_force):.3e} N")
print(f"Currents: {best_curr}")
print(f"Converged: {converged}, Cost: {best_cost:.4f}")

# Simulate one step
next_pos, force_out, step_vec = _simulate_one_step(
    particle, stuck_pos, best_curr, magnets, workspace, 0.02, flow=network, sim_time=0.0
)
print(f"Step vec: {step_vec*1000} mm, |step|={np.linalg.norm(step_vec)*1000:.4f} mm")
print(f"Next pos: {next_pos*1000} mm")
print(f"In vessel after step: {network.in_any_vessel(next_pos)}")
print(f"In vessel at unclipped: {network.in_any_vessel(stuck_pos + step_vec)}")

# Now simulate step-by-step FROM inlet to see where exactly it gets stuck
print(f"\n=== Step-by-step simulation from inlet ===")
particle2 = MagneticParticle3D(position=inlet.copy(), profile=MICROSPHERE_500UM, rng_seed=42)
nav2 = network.waypoint_navigator(inlet, outlet, arrival_tol=DEFAULT_TARGET_TOLERANCE, waypoint_spacing=0.0015, lookahead=3)

for step in range(80):
    target = nav2(particle2.pos)
    
    best_curr, best_force, best_cost, converged = continuous_mpc_controller(
        particle=particle2, particle_pos=particle2.pos, target_pos=target,
        magnet_array=magnets, workspace=workspace, dt=0.02, max_current=8.0,
        fast_mode=True, flow=network, sim_time=step*0.02,
        w_progress=120.0, w_align=8.0, w_dist=1.0, w_curr=1e-5,
        w_smooth=1e-4, w_edge=50.0, w_overshoot=2.0,
    )
    
    magnets.set_currents(best_curr)
    B = magnets.field_at(particle2.pos)
    
    adaptive_step(particle2, best_force, 0.02, workspace, flow=network, sim_time=step*0.02)
    
    seg, r = network.find_segment(particle2.pos)
    in_v = network.in_any_vessel(particle2.pos)
    wall_cl = (seg.radius - r) if seg else float('nan')
    
    if step % 5 == 0 or step >= 55:
        dist = np.linalg.norm(particle2.pos - outlet)
        print(f"step {step:3d}: pos={particle2.pos*1000}, dist={dist*1000:.1f}mm, "
              f"|F|={np.linalg.norm(best_force):.2e}N, "
              f"in_vessel={in_v}, wall_cl={wall_cl*1000:.3f}mm, "
              f"wp={nav2.current_idx+1}/{nav2.total_hops}")
