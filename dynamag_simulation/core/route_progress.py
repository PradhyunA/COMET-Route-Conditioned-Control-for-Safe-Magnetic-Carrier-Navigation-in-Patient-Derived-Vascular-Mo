"""Remaining distance along a planned route, for stagnation early stopping."""

import numpy as np

from .constants import EPS


class RouteRemaining:
    """
    Route length left from the closest point of a fixed polyline to pos.

    Straight-line distance to the target can stay flat for many steps while
    a carrier makes steady progress around a winding route, so stagnation
    tests that only watch it stop live trials. This measures progress along
    the route instead.
    """

    def __init__(self, points):
        pts = np.asarray(points, dtype=float).reshape(-1, 3)
        if len(pts) < 2:
            pts = np.vstack([pts, pts])
        self._a = pts[:-1]
        self._ab = pts[1:] - pts[:-1]
        self._len_sq = np.maximum(np.einsum("ij,ij->i", self._ab, self._ab), EPS)
        self._cum = np.concatenate([[0.0], np.cumsum(np.sqrt(self._len_sq))])

    def __call__(self, pos):
        pos = np.asarray(pos, dtype=float)
        t = np.clip(np.einsum("ij,ij->i", pos[None, :] - self._a, self._ab) / self._len_sq, 0.0, 1.0)
        d = np.linalg.norm(pos[None, :] - (self._a + t[:, None] * self._ab), axis=1)
        j = int(np.argmin(d))
        return float(self._cum[-1] - (self._cum[j] + t[j] * np.sqrt(self._len_sq[j])))
