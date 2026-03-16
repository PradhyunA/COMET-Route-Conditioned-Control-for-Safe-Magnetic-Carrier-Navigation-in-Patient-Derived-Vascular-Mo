"""
Workspace geometry: 2D and 3D bounding regions with position clamping.
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
