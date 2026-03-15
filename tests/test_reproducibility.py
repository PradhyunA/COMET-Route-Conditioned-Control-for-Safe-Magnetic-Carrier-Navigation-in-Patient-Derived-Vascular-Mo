"""Tests for seed-based reproducibility."""

import numpy as np
from dynamag_simulation.core.profiles import MICROSPHERE_500UM
from dynamag_simulation.core.workspace import Workspace3D
from dynamag_simulation.core.magnets import ElectromagnetArray3D
from dynamag_simulation.core.particle import MagneticParticle3D
from dynamag_simulation.core.integrator import adaptive_step
from dynamag_simulation.core.controllers import continuous_mpc_controller


def _run_short_trial(seed, steps=5):
    """Run a short trial and return the trajectory."""
    ws = Workspace3D()
    mag = ElectromagnetArray3D()
    p = MagneticParticle3D(
        position=[0.02, 0.02, 0.02],
        profile=MICROSPHERE_500UM,
        rng_seed=seed,
    )
    target = np.array([0.05, 0.05, 0.05])
    prev = np.zeros(8)
    traj = [p.pos.copy()]

    for _ in range(steps):
        currents, force, _, _ = continuous_mpc_controller(
            particle=p, particle_pos=p.pos, target_pos=target,
            magnet_array=mag, workspace=ws, dt=0.02,
            initial_guess=prev,
        )
        prev = currents.copy()
        mag.set_currents(currents)
        adaptive_step(p, force, 0.02, ws)
        traj.append(p.pos.copy())

    return np.array(traj)


def test_same_seed_bitwise_reproducible():
    traj_a = _run_short_trial(seed=42)
    traj_b = _run_short_trial(seed=42)
    np.testing.assert_array_equal(traj_a, traj_b)


def test_different_seed_gives_different_trajectory():
    traj_a = _run_short_trial(seed=42)
    traj_b = _run_short_trial(seed=99)
    assert not np.array_equal(traj_a, traj_b), \
        "Different seeds should produce different Brownian paths"
