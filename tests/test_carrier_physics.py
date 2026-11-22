"""Tests for carrier loading and drag profile fields."""

from dataclasses import replace

import numpy as np
import pytest

from dynamag_simulation.core.particle import MagneticParticle3D
from dynamag_simulation.core.profiles import PhysicalProfile


def _profile(name, loading=1.0, drag_multiplier=1.0):
    return PhysicalProfile(
        name=name,
        diameter=1e-3,
        chi_eff=10.0,
        M_sat=4.5e5,
        magnetic_loading_fraction=loading,
        hydrodynamic_drag_multiplier=drag_multiplier,
        fluid_viscosity=3.5e-3,
        fluid_density=1060.0,
        temperature=310.15,
        enable_brownian=False,
    )


def test_hydrodynamic_drag_multiplier_scales_stokes_drag():
    base = MagneticParticle3D([0, 0, 0], profile=_profile("base"))
    scaled = MagneticParticle3D(
        [0, 0, 0],
        profile=_profile("scaled_drag", drag_multiplier=1.5),
    )

    assert scaled.drag_stokes == pytest.approx(1.5 * base.drag_stokes)


def test_magnetic_loading_fraction_scales_force_prefactors():
    full = MagneticParticle3D([0, 0, 0], profile=_profile("full", loading=1.0))
    loaded = MagneticParticle3D(
        [0, 0, 0],
        profile=_profile("loaded", loading=0.6),
    )

    assert loaded.magnetic_volume == pytest.approx(0.6 * full.volume)
    assert loaded.force_prefactor_linear == pytest.approx(
        0.6 * full.force_prefactor_linear
    )
    assert loaded.force_prefactor_sat == pytest.approx(0.6 * full.force_prefactor_sat)


def test_gravity_is_off_by_default():
    particle = MagneticParticle3D([0, 0, 0], profile=_profile("no_gravity"))

    assert np.allclose(particle.body_force, 0.0)


def test_body_force_is_weight_minus_buoyancy():
    profile = replace(
        _profile("dense"), particle_density=3542.0, gravity_m_s2=(0.0, -9.81, 0.0),
    )
    particle = MagneticParticle3D([0, 0, 0], profile=profile)

    expected = (3542.0 - 1060.0) * particle.volume * 9.81
    assert particle.body_force == pytest.approx([0.0, -expected, 0.0])


def test_neutrally_buoyant_carrier_has_no_body_force():
    profile = replace(
        _profile("neutral"), particle_density=1060.0, gravity_m_s2=(0.0, 0.0, -9.81),
    )
    particle = MagneticParticle3D([0, 0, 0], profile=profile)

    assert np.allclose(particle.body_force, 0.0)


def test_update_settles_at_drag_balanced_speed_without_magnetic_force():
    profile = replace(
        _profile("settling"), particle_density=3542.0, gravity_m_s2=(0.0, 0.0, -9.81),
    )
    particle = MagneticParticle3D([0, 0, 0], profile=profile)

    particle.update(np.zeros(3), dt=0.01)

    gamma_eff, speed, _ = particle._effective_drag(np.linalg.norm(particle.body_force))
    assert particle.vel == pytest.approx([0.0, 0.0, -speed])
    assert gamma_eff > particle.drag_stokes  # finite-Re correction is active


def test_body_force_combines_with_magnetic_force_before_drag():
    profile = replace(
        _profile("combined"), particle_density=3542.0, gravity_m_s2=(0.0, 0.0, -9.81),
    )
    particle = MagneticParticle3D([0, 0, 0], profile=profile)
    magnetic = -particle.body_force  # exactly cancels the net weight

    particle.update(magnetic, dt=0.01)

    assert np.allclose(particle.vel, 0.0)


def test_negative_particle_density_is_rejected():
    with pytest.raises(ValueError):
        replace(_profile("bad"), particle_density=-1.0)
