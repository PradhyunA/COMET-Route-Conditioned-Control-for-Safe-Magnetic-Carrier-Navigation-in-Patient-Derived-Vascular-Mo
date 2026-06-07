"""
Named physical profiles for particle/fluid combinations.

Each profile bundles every parameter needed to instantiate a
MagneticParticle3D so that simulations are reproducible and
unambiguous about what is being modelled.

Updated for:
    - realistic blood physics
    - magnetic nanoparticle carriers
    - SPION-loaded aggregates
    - hydrodynamic drag scaling
    - magnetic loading fraction
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


# -------------------------------------------------------------------
# Physical profile definition
# -------------------------------------------------------------------

@dataclass(frozen=True)
class PhysicalProfile:
    """
    Immutable descriptor for a particle-in-fluid system.

    Parameters
    ----------
    diameter:
        Effective hydrodynamic diameter [m]

    chi_eff:
        Effective magnetic susceptibility

    M_sat:
        Saturation magnetisation [A/m]

    magnetic_loading_fraction:
        Fraction of carrier volume occupied by magnetic material.
        Represents SPION packing/loading density.

    hydrodynamic_drag_multiplier:
        Additional drag scaling for non-ideal carrier geometry,
        roughness, protein corona, porous structures, etc.
    """

    name: str

    # Geometry
    diameter: float                  # m

    # Magnetic properties
    chi_eff: float
    M_sat: float                     # A/m

    # Carrier realism
    magnetic_loading_fraction: float
    hydrodynamic_drag_multiplier: float

    # Fluid properties
    fluid_viscosity: float           # Pa*s
    fluid_density: float             # kg/m^3

    # Thermal / stochastic
    temperature: float               # K
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

        if not (0.0 < self.magnetic_loading_fraction <= 1.0):
            raise ValueError(
                "magnetic_loading_fraction must be in (0,1]"
            )

        if self.hydrodynamic_drag_multiplier <= 0:
            raise ValueError(
                "hydrodynamic_drag_multiplier must be > 0"
            )


# -------------------------------------------------------------------
# Legacy microsphere profiles (water)
# -------------------------------------------------------------------

MICROSPHERE_500UM = PhysicalProfile(
    name="microsphere_500um",

    diameter=500e-6,

    chi_eff=3.0,
    M_sat=4.5e5,

    magnetic_loading_fraction=1.0,
    hydrodynamic_drag_multiplier=1.0,

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

    magnetic_loading_fraction=1.0,
    hydrodynamic_drag_multiplier=1.0,

    fluid_viscosity=WATER_VISCOSITY_25C,
    fluid_density=WATER_DENSITY_25C,

    temperature=298.0,
    enable_brownian=True,
)


# -------------------------------------------------------------------
# Single nanoparticle profile (blood)
# -------------------------------------------------------------------

NANOPARTICLE_500NM = PhysicalProfile(
    name="nanoparticle_500nm_blood",

    diameter=500e-9,   # 500 nm

    chi_eff=3.0,
    M_sat=4.5e5,

    magnetic_loading_fraction=1.0,
    hydrodynamic_drag_multiplier=1.0,

    fluid_viscosity=BLOOD_VISCOSITY_37C,
    fluid_density=BLOOD_DENSITY_37C,

    temperature=310.15,
    enable_brownian=True,
)


# -------------------------------------------------------------------
# Realistic SPION-loaded carrier profiles
# -------------------------------------------------------------------

SPION_CARRIER_5UM = PhysicalProfile(
    name="spion_carrier_5um_blood",

    diameter=5e-6,     # 5 µm

    chi_eff=6.0,
    M_sat=4.5e5,

    magnetic_loading_fraction=0.35,
    hydrodynamic_drag_multiplier=1.15,

    fluid_viscosity=BLOOD_VISCOSITY_37C,
    fluid_density=BLOOD_DENSITY_37C,

    temperature=310.15,
    enable_brownian=True,
)

SPION_CARRIER_10UM = PhysicalProfile(
    name="spion_carrier_10um_blood",

    diameter=10e-6,    # 10 µm

    chi_eff=8.0,
    M_sat=4.5e5,

    magnetic_loading_fraction=0.45,
    hydrodynamic_drag_multiplier=1.20,

    fluid_viscosity=BLOOD_VISCOSITY_37C,
    fluid_density=BLOOD_DENSITY_37C,

    temperature=310.15,
    enable_brownian=True,
)

SPION_CARRIER_20UM = PhysicalProfile(
    name="spion_carrier_20um_blood",

    diameter=20e-6,    # 20 µm

    chi_eff=10.0,
    M_sat=4.5e5,

    magnetic_loading_fraction=0.55,
    hydrodynamic_drag_multiplier=1.30,

    fluid_viscosity=BLOOD_VISCOSITY_37C,
    fluid_density=BLOOD_DENSITY_37C,

    temperature=310.15,
    enable_brownian=True,
)