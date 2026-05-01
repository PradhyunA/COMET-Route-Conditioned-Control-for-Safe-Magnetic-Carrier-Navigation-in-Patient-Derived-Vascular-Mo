"""
Workspace geometry: 2D/3D bounding boxes AND vessel-network boundary.

WorkspaceVessel wraps a VesselNetwork and satisfies the same
interface as Workspace3D (clip, half) so the integrator, MPC
controller, and plotting code need no changes.
"""

import numpy as np
from .constants import HALF


class Workspace2D:
    """Square workspace with position clamping."""
    def __init__(self, half_side: float = HALF):
        self.half = float(half_side)

    def clip(self, p):
        p = np.asarray(p, dtype=float).copy()
        return np.clip(p, -self.half, self.half)


class Workspace3D:
    """Cubic workspace with position clamping."""
    def __init__(self, half_side: float = HALF):
        self.half = float(half_side)

    def clip(self, p):
        return np.clip(np.asarray(p, dtype=float), -self.half, self.half)


class WorkspaceVessel:
    """
    Vessel-network workspace.

    Delegates clip() and half to a VesselNetwork instance so that
    all existing simulation code (integrator, MPC, plotting) works
    without modification — they all call workspace.clip(pos) and
    read workspace.half.

    Parameters
    ----------
    network : VesselNetwork
        The vascular geometry object.
    """

    def __init__(self, network):
        self._network = network

    @property
    def half(self) -> float:
        """Bounding half-extent of the vessel tree (for plotting)."""
        return self._network.half

    def clip(self, p: np.ndarray) -> np.ndarray:
        """
        Clamp position to vessel interior.
        If p is inside any vessel segment, return unchanged.
        Otherwise project back to nearest vessel wall.
        """
        return self._network.clip(np.asarray(p, dtype=float))

    @property
    def network(self):
        return self._network
