"""Tests for velocity logging consistency.

The key invariant: particle.realized_vel must equal
the actual displacement divided by dt.
"""

import numpy as np
import pytest

from dynamag_simulation.core.profiles import PhysicalProfile
from dynamag_simulation.core.workspace import Workspace3D
from dynamag_simulation.core.magnets import ElectromagnetArray3D
from dynamag_simulation.core.particle import MagneticParticle3D
from dynamag_simulation.core.integrator import adaptive_step


@pytest.fixture
def deterministic_profile():
    """Profile with Brownian off so velocity is purely deterministic."""
    return PhysicalProfile(
        name="test_deterministic",
        diameter=500e-6,
        chi_eff=3.0,
        M_sat=4.5e5,
        magnetic_loading_fraction=1.0,
        hydrodynamic_drag_multiplier=1.0,
        fluid_viscosity=8.9e-4,
        fluid_density=997.0,
        temperature=298.0,
        enable_brownian=False,
    )


def test_realized_vel_matches_displacement(deterministic_profile):
    ws = Workspace3D()
    mag = ElectromagnetArray3D()
    p = MagneticParticle3D(
        position=[0.03, 0.02, 0.01],
        profile=deterministic_profile,
    )

    currents = np.zeros(mag.n_magnets)
    currents[:8] = [8, 0, -4, 0, 4, 0, 0, -8]
    mag.set_currents(currents)
    B = mag.field_at(p.pos)
    grad = mag.grad_B_squared(p.pos)
    F = p.magnetic_force(B, grad)

    dt = 0.02
    pos_before = p.pos.copy()
    adaptive_step(p, F, dt, ws)
    pos_after = p.pos.copy()

    actual_disp = pos_after - pos_before
    expected_vel = actual_disp / dt

    np.testing.assert_allclose(
        p.realized_vel, expected_vel, rtol=1e-10,
        err_msg="realized_vel must match (pos_new - pos_old) / dt",
    )


def test_realized_vel_nonzero_under_force(deterministic_profile):
    ws = Workspace3D()
    mag = ElectromagnetArray3D()
    p = MagneticParticle3D(
        position=[0.03, 0.02, 0.01],
        profile=deterministic_profile,
    )

    currents = np.zeros(mag.n_magnets)
    currents[0] = 8
    mag.set_currents(currents)
    B = mag.field_at(p.pos)
    grad = mag.grad_B_squared(p.pos)
    F = p.magnetic_force(B, grad)

    assert np.linalg.norm(F) > 0, "Force should be nonzero off-center"

    adaptive_step(p, F, 0.02, ws)
    assert np.linalg.norm(p.realized_vel) > 0


def test_zero_force_brownian_off_gives_zero_vel(deterministic_profile):
    ws = Workspace3D()
    p = MagneticParticle3D(
        position=[0.0, 0.0, 0.0],
        profile=deterministic_profile,
    )

    F = np.zeros(3)
    adaptive_step(p, F, 0.02, ws)

    np.testing.assert_array_equal(p.realized_vel, np.zeros(3))
