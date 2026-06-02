"""
Test MPC navigation with tuned waypoint parameters to avoid corner-cutting.
"""
import os, sys
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

import numpy as np
from dynamag_simulation.core.profiles import MICROSPHERE_500UM
from dynamag_simulation.core.magnets import ElectromagnetArray3D
from dynamag_simulation.core.particle import MagneticParticle3D
from dynamag_simulation.core.real_vessel_network import RealVesselNetwork
from dynamag_simulation.core.workspace import WorkspaceVessel
from dynamag_simulation.core.controllers import continuous_mpc_controller, _simulate_one_step

network = RealVesselNetwork("real_vessel_segments.pkl", flow_scale=1.0)
workspace = WorkspaceVessel(network)
magnets = ElectromagnetArray3D(
    coil_radius=0.03, coil_length=0.12, turns=1500, max_current=8.0, placement_radius=0.06
)

inlet = network.inlet_start()
outlet = network.outlet_end()

particle = MagneticParticle3D(position=inlet, profile=MICROSPHERE_500UM, rng_seed=42)

# Use strict waypoint following
nav = network.waypoint_navigator(
    inlet, outlet, 
    arrival_tol=0.001,      # 1 mm to reach a waypoint
    waypoint_spacing=0.001, # 1 mm spacing
    lookahead=1             # Look 1 waypoint ahead (1 mm)
)

print(f"Waypoints: {nav.total_hops}")
print("Running 100 steps...")

for step in range(100):
    target = nav(particle.pos)
    
    best_curr, best_force, best_cost, converged = continuous_mpc_controller(
        particle=particle, particle_pos=particle.pos, target_pos=target,
        magnet_array=magnets, workspace=workspace, dt=0.02, max_current=8.0,
        fast_mode=True, flow=network, sim_time=step*0.02,
        w_progress=120.0, w_align=8.0, w_dist=1.0, w_curr=1e-5,
        w_smooth=1e-4, w_edge=50.0, w_overshoot=2.0,
    )
    
    next_pos, force, step_vec = _simulate_one_step(
        particle, particle.pos, best_curr, magnets, workspace, 0.02,
        flow=network, sim_time=step*0.02
    )
    
    particle.pos = next_pos
    
    if step % 10 == 0:
        dist = np.linalg.norm(particle.pos - outlet)
        print(f"step {step:3d}: dist={dist*1000:.1f}mm, |F|={np.linalg.norm(best_force):.2e}N, "
              f"wp={nav.current_idx+1}/{nav.total_hops}")

dist = np.linalg.norm(particle.pos - outlet)
print(f"Final: dist={dist*1000:.1f}mm, wp={nav.current_idx+1}/{nav.total_hops}")
