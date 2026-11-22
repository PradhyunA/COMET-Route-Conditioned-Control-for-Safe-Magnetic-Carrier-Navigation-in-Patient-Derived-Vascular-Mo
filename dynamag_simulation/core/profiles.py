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
    - larger magnetic carrier / milliparticle profiles
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
    name:
        Human-readable profile name.

    diameter:
        Effective hydrodynamic diameter [m].

    chi_eff:
        Effective magnetic susceptibility.

    M_sat:
        Saturation magnetisation [A/m].

    magnetic_loading_fraction:
        Fraction of carrier volume occupied by magnetic material.
        Represents SPION packing/loading density.

    hydrodynamic_drag_multiplier:
        Additional drag scaling for non-ideal carrier geometry,
        roughness, protein corona, porous structures, etc.

    fluid_viscosity:
        Dynamic viscosity [Pa*s].

    fluid_density:
        Fluid density [kg/m^3].

    temperature:
        Temperature [K].

    enable_brownian:
        Whether Brownian motion is included.

    particle_density:
        Carrier density [kg/m^3]. None disables gravity and buoyancy.

    gravity_m_s2:
        Gravitational acceleration vector [m/s^2] in vessel coordinates.
        Only used when particle_density is set.
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

    # Gravity / buoyancy (off by default to preserve existing benchmarks)
    particle_density: float | None = None    # kg/m^3
    gravity_m_s2: tuple = (0.0, 0.0, 0.0)    # m/s^2

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

        if self.particle_density is not None and self.particle_density <= 0:
            raise ValueError(
                f"particle_density must be > 0, got {self.particle_density}"
            )

        if len(self.gravity_m_s2) != 3:
            raise ValueError("gravity_m_s2 must have 3 components")


# -------------------------------------------------------------------
# Legacy microsphere profiles (water)
# -------------------------------------------------------------------

MICROSPHERE_500UM = PhysicalProfile(
    name="microsphere_500um",

    diameter=500e-6,  # 500 µm

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

    diameter=50e-6,  # 50 µm

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
# SPION-loaded microcarrier profiles in blood
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


# -------------------------------------------------------------------
# Larger magnetic carrier / milliparticle profiles in blood
# -------------------------------------------------------------------
#
# These are NOT nanoparticles.
#
# They are included for large-vessel magnetic steering feasibility tests
# where a body-scale scanner must steer a much larger magnetic carrier.
# Use language such as:
#     "magnetic carrier"
#     "magnetic milliparticle"
#     "large-vessel carrier"
#
# Do not describe these as nanoparticles.
# -------------------------------------------------------------------

SPION_CARRIER_50UM = PhysicalProfile(
    name="spion_carrier_50um_blood",

    diameter=50e-6,    # 50 µm

    chi_eff=10.0,
    M_sat=4.5e5,

    magnetic_loading_fraction=0.55,
    hydrodynamic_drag_multiplier=1.30,

    fluid_viscosity=BLOOD_VISCOSITY_37C,
    fluid_density=BLOOD_DENSITY_37C,

    temperature=310.15,
    enable_brownian=True,
)

SPION_CARRIER_100UM = PhysicalProfile(
    name="spion_carrier_100um_blood",

    diameter=100e-6,   # 100 µm

    chi_eff=10.0,
    M_sat=4.5e5,

    magnetic_loading_fraction=0.55,
    hydrodynamic_drag_multiplier=1.35,

    fluid_viscosity=BLOOD_VISCOSITY_37C,
    fluid_density=BLOOD_DENSITY_37C,

    temperature=310.15,
    enable_brownian=False,
)

SPION_CARRIER_250UM = PhysicalProfile(
    name="spion_carrier_250um_blood",

    diameter=250e-6,   # 250 µm

    chi_eff=10.0,
    M_sat=4.5e5,

    magnetic_loading_fraction=0.60,
    hydrodynamic_drag_multiplier=1.40,

    fluid_viscosity=BLOOD_VISCOSITY_37C,
    fluid_density=BLOOD_DENSITY_37C,

    temperature=310.15,
    enable_brownian=False,
)

SPION_CARRIER_500UM = PhysicalProfile(
    name="spion_carrier_500um_blood",

    diameter=500e-6,   # 500 µm

    chi_eff=10.0,
    M_sat=4.5e5,

    magnetic_loading_fraction=0.60,
    hydrodynamic_drag_multiplier=1.45,

    fluid_viscosity=BLOOD_VISCOSITY_37C,
    fluid_density=BLOOD_DENSITY_37C,

    temperature=310.15,
    enable_brownian=False,
)

SPION_CARRIER_1MM = PhysicalProfile(
    name="spion_carrier_1mm_blood",

    diameter=1e-3,     # 1 mm

    chi_eff=10.0,
    M_sat=4.5e5,

    magnetic_loading_fraction=0.60,
    hydrodynamic_drag_multiplier=1.50,

    fluid_viscosity=BLOOD_VISCOSITY_37C,
    fluid_density=BLOOD_DENSITY_37C,

    temperature=310.15,
    enable_brownian=False,
)