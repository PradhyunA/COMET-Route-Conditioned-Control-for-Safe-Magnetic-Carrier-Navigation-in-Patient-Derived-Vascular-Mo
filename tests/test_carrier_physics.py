"""Tests for carrier loading and drag profile fields."""

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
