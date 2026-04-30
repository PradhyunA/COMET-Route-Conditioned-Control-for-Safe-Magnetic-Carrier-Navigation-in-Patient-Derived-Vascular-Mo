"""
DynaMag Simulation Core
========================
Shared physics engine for all DynaMag simulation variants.
"""

from .constants import MU_0, K_B, PI, INCH_TO_METER, SIDE, HALF, EPS
from .profiles import PhysicalProfile, MICROSPHERE_500UM, MICROSPHERE_50UM, NANOPARTICLE_500NM
from .workspace import Workspace2D, Workspace3D
from .magnets import ElectromagnetArray3D
from .particle import MagneticParticle3D
from .integrator import adaptive_step
from .controllers import continuous_mpc_controller
from .validation import compute_reynolds, check_low_re_validity
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
    "continuous_mpc_controller",
    "compute_reynolds", "check_low_re_validity",
    "TrialResult", "save_results_csv", "print_summary",
    "NoFlow", "UniformFlow", "PoiseuilleFlow", "PulsatileFlow",
]
