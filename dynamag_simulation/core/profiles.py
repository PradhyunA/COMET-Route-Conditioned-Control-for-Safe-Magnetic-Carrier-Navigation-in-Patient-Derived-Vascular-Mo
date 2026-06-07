"""
Named physical profiles for particle/fluid combinations.

Each profile bundles every parameter needed to instantiate a
MagneticParticle3D so that simulations are reproducible and
unambiguous about what is being modelled.
"""

from dataclasses import dataclass


# -------------------------------------------------------------------
# Fluid properties
# -------------------------------------------------------------------

# Water @ 25 C
WATER_VISCOSITY_25C = 8.9e-4      # Pa*s
WATER_DENSITY_25C = 997.0         # kg/m^3

# Human blood @ 37 C
BLOOD_VISCOSITY_37C = 3.5e-3      # Pa*s
BLOOD_DENSITY_37C = 1060.0        # kg/m^3


@dataclass(frozen=True)
class PhysicalProfile:
    """Immutable descriptor for a particle-in-fluid system."""

    name: str
    diameter: float          # m
    chi_eff: float           # effective magnetic susceptibility
    M_sat: float             # saturation magnetisation (A/m)

    fluid_viscosity: float   # Pa*s
    fluid_density: float     # kg/m^3

    temperature: float       # K
    enable_brownian: bool

    def __post_init__(self):
        if self.diameter <= 0:
            raise ValueError(
                f"diameter must be > 0, got {self.diameter}"
            )

        if self.fluid_viscosity <= 0:
            raise ValueError(
                f"fluid_viscosity must be > 0, got {self.fluid_viscosity}"
            )

        if self.temperature <= 0:
            raise ValueError(
                f"temperature must be > 0, got {self.temperature}"
            )


# -------------------------------------------------------------------
# Legacy microsphere profiles (water)
# -------------------------------------------------------------------

MICROSPHERE_500UM = PhysicalProfile(
    name="microsphere_500um",

    diameter=500e-6,

    chi_eff=3.0,
    M_sat=4.5e5,

    fluid_viscosity=WATER_VISCOSITY_25C,
    fluid_density=WATER_DENSITY_25C,

    temperature=298.0,
    enable_brownian=True,
)

MICROSPHERE_50UM = PhysicalProfile(
    name="microsphere_50um",

    diameter=50e-6,

    chi_eff=3.0,
    M_sat=4.5e5,

    fluid_viscosity=WATER_VISCOSITY_25C,
    fluid_density=WATER_DENSITY_25C,

    temperature=298.0,
    enable_brownian=True,
)


# -------------------------------------------------------------------
# Biomedical nanoparticle profile (blood)
# -------------------------------------------------------------------

NANOPARTICLE_500NM = PhysicalProfile(
    name="nanoparticle_500nm_blood",

    diameter=500e-9,   # 500 nm

    chi_eff=3.0,
    M_sat=4.5e5,

    fluid_viscosity=BLOOD_VISCOSITY_37C,
    fluid_density=BLOOD_DENSITY_37C,

    temperature=310.15,   # 37 C

    enable_brownian=True,
)