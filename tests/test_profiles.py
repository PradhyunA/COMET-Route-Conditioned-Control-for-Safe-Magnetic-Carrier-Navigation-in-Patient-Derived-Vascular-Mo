"""Tests for physical profiles and parameter validation."""

import pytest
from dynamag_simulation.core.profiles import (
    PhysicalProfile, MICROSPHERE_500UM, NANOPARTICLE_500NM,
)


def test_microsphere_defaults():
    p = MICROSPHERE_500UM
    assert p.name == "microsphere_500um"
    assert p.diameter == 500e-6
    assert p.chi_eff == 3.0
    assert p.M_sat == 4.5e5
    assert p.fluid_viscosity == pytest.approx(8.9e-4)
    assert p.temperature == 298.0


def test_nanoparticle_defaults():
    p = NANOPARTICLE_500NM
    assert p.name == "nanoparticle_500nm"
    assert p.diameter == 500e-9
    assert p.diameter < MICROSPHERE_500UM.diameter


def test_profile_immutability():
    with pytest.raises(AttributeError):
        MICROSPHERE_500UM.diameter = 1e-3


def test_invalid_diameter_rejected():
    with pytest.raises(ValueError, match="diameter"):
        PhysicalProfile(
            name="bad", diameter=-1e-6, chi_eff=3.0, M_sat=4.5e5,
            fluid_viscosity=8.9e-4, fluid_density=997.0,
            temperature=298.0, enable_brownian=False,
        )


def test_invalid_viscosity_rejected():
    with pytest.raises(ValueError, match="fluid_viscosity"):
        PhysicalProfile(
            name="bad", diameter=500e-6, chi_eff=3.0, M_sat=4.5e5,
            fluid_viscosity=0.0, fluid_density=997.0,
            temperature=298.0, enable_brownian=False,
        )


def test_invalid_temperature_rejected():
    with pytest.raises(ValueError, match="temperature"):
        PhysicalProfile(
            name="bad", diameter=500e-6, chi_eff=3.0, M_sat=4.5e5,
            fluid_viscosity=8.9e-4, fluid_density=997.0,
            temperature=-10.0, enable_brownian=False,
        )
