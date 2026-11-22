"""Steady SI CFD samples with bounded inverse-distance interpolation."""

from functools import lru_cache
import json
from pathlib import Path

import numpy as np
from scipy.spatial import cKDTree


class CFDFieldData:
    """Immutable source data, shared between flow scales within a process."""

    def __init__(self, path):
        self.path = str(Path(path).resolve())
        with np.load(self.path, allow_pickle=False) as data:
            self.points = data["points_m"]
            self.velocity = data["velocity_m_s"]
            self.radius = data["support_radius_m"]
            self.metadata = json.loads(str(data["metadata_json"]))
        if (self.points.ndim != 2 or self.points.shape[1] != 3
                or self.velocity.shape != self.points.shape
                or self.radius.shape != (len(self.points),) or not len(self.points)
                or not np.all(np.isfinite(self.points))
                or not np.all(np.isfinite(self.velocity))
                or not np.all(np.isfinite(self.radius)) or np.any(self.radius <= 0)):
            raise ValueError("Invalid CFD sample arrays")
        for array in (self.points, self.velocity, self.radius):
            array.flags.writeable = False
        self.tree = cKDTree(self.points)


@lru_cache(maxsize=2)
def load_cfd_data(path):
    return CFDFieldData(path)


class SteadyCFDFlowField:
    def __init__(self, npz_path, flow_scale=1.0):
        self.source_file = str(Path(npz_path).resolve())
        self.data = load_cfd_data(self.source_file)
        self.metadata = self.data.metadata
        self.flow_scale = float(flow_scale)
        if not np.isfinite(self.flow_scale) or self.flow_scale <= 0:
            raise ValueError("CFD velocity scale must be finite and positive")
        self.query_count = 0
        self.max_nearest_distance_m = 0.0

    def velocity_at_many(self, positions):
        positions = np.asarray(positions, dtype=float).reshape(-1, 3)
        if not np.all(np.isfinite(positions)):
            raise ValueError("CFD query coordinates must be finite")
        distance, index = self.data.tree.query(positions, k=min(8, len(self.data.points)))
        if distance.ndim == 1:
            distance, index = distance[:, None], index[:, None]
        # The old tubular wall model can differ slightly from the CFD surface.
        # Bound this local extrapolation; this is not exact mesh containment.
        supported = distance[:, 0] <= 2.0 * self.data.radius[index[:, 0]]
        if not np.all(supported):
            row = int(np.flatnonzero(~supported)[0])
            raise ValueError(
                f"CFD query outside local sample support: {positions[row].tolist()} "
                f"({distance[row, 0] * 1000:.3f} mm from nearest sample). "
                "No analytical fallback is allowed."
            )
        self.query_count += len(positions)
        if len(positions):
            self.max_nearest_distance_m = max(self.max_nearest_distance_m, float(distance[:, 0].max()))
        weights = 1.0 / np.maximum(distance, 1e-12) ** 2
        weights /= weights.sum(axis=1, keepdims=True)
        return np.einsum("nk,nkj->nj", weights, self.data.velocity[index]) * self.flow_scale

    def velocity_at(self, position, time=0.0):
        # The export is steady; do not add the analytical model's sinusoid.
        return self.velocity_at_many([position])[0]
