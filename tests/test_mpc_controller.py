"""Tests for the MPC controller."""

import numpy as np
import pytest

from dynamag_simulation.core.profiles import PhysicalProfile
from dynamag_simulation.core.workspace import Workspace3D
from dynamag_simulation.core.magnets import ElectromagnetArray3D
from dynamag_simulation.core.particle import MagneticParticle3D
from dynamag_simulation.core.controllers import continuous_mpc_controller


@pytest.fixture
def setup():
    profile = PhysicalProfile(
        name="test", diameter=500e-6, chi_eff=3.0, M_sat=4.5e5,
        fluid_viscosity=8.9e-4, fluid_density=997.0,
        temperature=298.0, enable_brownian=False,
    )
    ws = Workspace3D()
    mag = ElectromagnetArray3D()
    p = MagneticParticle3D(position=[0.02, 0.02, 0.02], profile=profile)
    return ws, mag, p


def test_mpc_returns_bounded_currents(setup):
    ws, mag, p = setup
    target = np.array([0.05, 0.05, 0.05])
    max_i = 8.0

    currents, force, cost, converged = continuous_mpc_controller(
        particle=p, particle_pos=p.pos, target_pos=target,
        magnet_array=mag, workspace=ws, dt=0.02, max_current=max_i,
    )

    assert np.all(np.abs(currents) <= max_i + 1e-10)
    assert currents.shape == (8,)


def test_mpc_makes_progress_toward_target(setup):
    ws, mag, p = setup
    target = np.array([0.05, 0.05, 0.05])

    dist_before = np.linalg.norm(p.pos - target)

    currents, force, cost, _ = continuous_mpc_controller(
        particle=p, particle_pos=p.pos, target_pos=target,
        magnet_array=mag, workspace=ws, dt=0.02,
    )

    step = (force / p.drag) * 0.02
    next_pos = ws.clip(p.pos + step)
    dist_after = np.linalg.norm(next_pos - target)

    assert dist_after < dist_before, "MPC should reduce distance to target"


def test_mpc_at_target_returns_zero(setup):
    ws, mag, p = setup
    p.pos = np.array([0.05, 0.05, 0.05])
    target = p.pos.copy()

    currents, force, cost, converged = continuous_mpc_controller(
        particle=p, particle_pos=p.pos, target_pos=target,
        magnet_array=mag, workspace=ws, dt=0.02,
    )

    assert np.allclose(currents, 0.0)
    assert cost == 0.0


def test_mpc_returns_convergence_flag(setup):
    ws, mag, p = setup
    target = np.array([0.05, 0.05, 0.05])

    _, _, _, converged = continuous_mpc_controller(
        particle=p, particle_pos=p.pos, target_pos=target,
        magnet_array=mag, workspace=ws, dt=0.02,
    )

    assert isinstance(converged, bool)
