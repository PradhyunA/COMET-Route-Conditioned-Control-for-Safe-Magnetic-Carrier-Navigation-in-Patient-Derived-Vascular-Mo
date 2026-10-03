"""
DynaMag Simulation Core
========================
Shared physics engine for all DynaMag simulation variants.
"""

from .constants import MU_0, K_B, PI, INCH_TO_METER, SIDE, HALF, EPS
from .profiles import PhysicalProfile, MICROSPHERE_500UM, MICROSPHERE_50UM, NANOPARTICLE_500NM
from .workspace import Workspace2D, Workspace3D, WorkspaceVessel
from .magnets import ElectromagnetArray3D
from .particle import MagneticParticle3D
from .integrator import adaptive_step
from .smc_dob_controller import SMCDOBInspiredConfig, SMCDOBInspiredController
try:
    from .controllers import continuous_mpc_controller
    from .cem_mpc_controller import (
        ICEMConfig,
        ICEMMPCController,
        PETSCEMConfig,
        PETSCEMMPCController,
    )
    from .constrained_cem_controller import ConstrainedCEMConfig, ConstrainedCEMMPCController
except ModuleNotFoundError as exc:
    if exc.name != "scipy":
        raise
    continuous_mpc_controller = None
    PETSCEMConfig = None
    PETSCEMMPCController = None
    ICEMConfig = None
    ICEMMPCController = None
    ConstrainedCEMConfig = None
    ConstrainedCEMMPCController = None
from .validation import compute_reynolds, check_low_re_validity, classify_reynolds_regime
from .reporting import TrialResult, save_results_csv, print_summary
from .flow import NoFlow, UniformFlow, PoiseuilleFlow, PulsatileFlow
from .vessel_network import VesselNetwork, VesselSegment
from .workspace import WorkspaceVessel

__all__ = [
    "MU_0", "K_B", "PI", "INCH_TO_METER", "SIDE", "HALF", "EPS",
    "PhysicalProfile", "MICROSPHERE_500UM", "MICROSPHERE_50UM", "NANOPARTICLE_500NM",
    "Workspace2D", "Workspace3D",
    "ElectromagnetArray3D",
    "MagneticParticle3D",
    "adaptive_step",
    "SMCDOBInspiredConfig", "SMCDOBInspiredController",
    "continuous_mpc_controller",
    "PETSCEMConfig", "PETSCEMMPCController", "ICEMConfig", "ICEMMPCController",
    "ConstrainedCEMConfig", "ConstrainedCEMMPCController",
    "compute_reynolds", "check_low_re_validity", "classify_reynolds_regime",
    "TrialResult", "save_results_csv", "print_summary",
    "NoFlow", "UniformFlow", "PoiseuilleFlow", "PulsatileFlow",
]
