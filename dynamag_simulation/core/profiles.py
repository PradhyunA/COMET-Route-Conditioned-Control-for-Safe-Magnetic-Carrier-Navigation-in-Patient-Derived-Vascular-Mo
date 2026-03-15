"""
Named physical profiles for particle/fluid combinations.

Each profile bundles every parameter needed to instantiate a
MagneticParticle3D so that simulations are reproducible and
unambiguous about what is being modelled.
"""

from dataclasses import dataclass

from .constants import WATER_VISCOSITY_25C


@dataclass(frozen=True)
class PhysicalProfile:
    """Immutable descriptor for a particle-in-fluid system."""
    name: str
    diameter: float          # m
    chi_eff: float           # effective magnetic susceptibility (dimensionless)
    M_sat: float             # saturation magnetisation (A/m)
    fluid_viscosity: float   # Pa*s
    fluid_density: float     # kg/m^3
    temperature: float       # K
    enable_brownian: bool

    def __post_init__(self):
        if self.diameter <= 0:
            raise ValueError(f"diameter must be > 0, got {self.diameter}")
        if self.fluid_viscosity <= 0:
            raise ValueError(f"fluid_viscosity must be > 0, got {self.fluid_viscosity}")
        if self.temperature <= 0:
            raise ValueError(f"temperature must be > 0, got {self.temperature}")


MICROSPHERE_500UM = PhysicalProfile(
    name="microsphere_500um",
    diameter=500e-6,
    chi_eff=3.0,
    M_sat=4.5e5,
    fluid_viscosity=WATER_VISCOSITY_25C,
    fluid_density=997.0,
    temperature=298.0,
    enable_brownian=True,
)

NANOPARTICLE_500NM = PhysicalProfile(
    name="nanoparticle_500nm",
    diameter=500e-9,
    chi_eff=3.0,
    M_sat=4.5e5,
    fluid_viscosity=WATER_VISCOSITY_25C,
    fluid_density=997.0,
    temperature=298.0,
    enable_brownian=True,
)
