"""
CFD-backed flow field for DynaMag.

Replaces the analytical PulsatileFlow / PoiseuilleFlow models with a
velocity field interpolated from a precomputed OpenFOAM solution.

Drop-in interface
-----------------
Every existing call site uses one of:

    network.velocity_at(pos, time)          # RealVesselNetwork
    segment.flow_velocity(pos, time)        # RealVesselSegment
    flow_model.velocity_at(pos, time)       # flow.py models

CFDFlowField exposes the same velocity_at(pos, time) signature, so it
can replace any of those without touching the controllers or integrator.

Two subclasses
--------------
SteadyCFDFlowField   -- loaded from a single .vtu / .vtk snapshot
                        (OpenFOAM simpleFoam output)
PulsatileCFDFlowField -- loaded from N phase snapshots
                        (OpenFOAM pimpleFoam output, one file per phase)
                        Interpolates in both space and time.

Offline build
-------------
Use build_cfd_interpolator.py to convert raw .vtu output into the
compact .npz format these classes load.  The .npz stores:
    points   -- (M, 3) float32  cell-centre coordinates [m]
    velocity -- (M, 3) float32  velocity vectors [m/s]  (steady)
             or (N, M, 3) float32                        (pulsatile)
    period   -- scalar float    cardiac period [s]       (pulsatile only)

Spatial lookup
--------------
Uses scipy.spatial.cKDTree for O(log M) nearest-neighbour queries.
For carrier positions inside the vessel lumen the nearest cell centre
is always a good approximation — cell sizes in the boundary-layer mesh
are 50–200 µm, comparable to or smaller than the 500 µm carrier.

For improved accuracy a linear barycentric interpolation from the k=4
nearest cells is used (inverse-distance weighting).

Usage
-----
    from cfd_flow_field import SteadyCFDFlowField, PulsatileCFDFlowField

    # Steady (simpleFoam)
    flow = SteadyCFDFlowField("abdominal_aorta_steady.npz")

    # Pulsatile (pimpleFoam, 20 phases)
    flow = PulsatileCFDFlowField("abdominal_aorta_pulsatile.npz")

    v = flow.velocity_at(position_m, time_s)

    # Swap into RealVesselNetwork (see patched_vessel_network.py)
    network = RealVesselNetworkCFD(pkl_file="...", cfd_flow=flow)
"""

from __future__ import annotations

import numpy as np
from pathlib import Path


def _load_npz(path: str | Path) -> dict:
    data = np.load(str(path), allow_pickle=False)
    return dict(data)


def _build_kdtree(points: np.ndarray):
    from scipy.spatial import cKDTree  # lazy import — not needed until first load
    return cKDTree(points)


class SteadyCFDFlowField:
    """
    Steady-state CFD velocity field (simpleFoam output).

    Parameters
    ----------
    npz_path : str or Path
        Path to the precomputed .npz archive produced by
        build_cfd_interpolator.py.  Must contain 'points' and 'velocity'.
    flow_scale : float
        Global multiplier applied to all velocities, matching the λ
        convention used in RealVesselNetwork.  Default 1.0.
    k_neighbours : int
        Number of nearest cell-centre neighbours used for IDW
        interpolation.  k=1 gives nearest-neighbour (fastest).
        k=4 gives smoother results near cell boundaries.  Default 4.
    extrapolate_zero : bool
        If True, positions outside the mesh bounding box return zero
        velocity instead of raising.  Default True (matches vessel
        network behaviour: points outside vessel wall → zero flow).
    """

    name = "cfd_steady"

    def __init__(
        self,
        npz_path: str | Path,
        flow_scale: float = 1.0,
        k_neighbours: int = 4,
        extrapolate_zero: bool = True,
    ):
        data = _load_npz(npz_path)
        self._points = data["points"].astype(np.float64)       # (M, 3)
        self._velocity = data["velocity"].astype(np.float64)   # (M, 3)
        self.flow_scale = float(flow_scale)
        self._k = int(k_neighbours)
        self._extrapolate_zero = extrapolate_zero
        self._tree = _build_kdtree(self._points)

        # Bounding box for quick out-of-domain check
        self._bbox_lo = self._points.min(axis=0)
        self._bbox_hi = self._points.max(axis=0)

        print(
            f"[CFD] SteadyCFDFlowField loaded: {len(self._points):,} cells  "
            f"flow_scale={flow_scale:.2f}  k={k_neighbours}"
        )

    def _in_bbox(self, pos: np.ndarray) -> bool:
        return bool(
            np.all(pos >= self._bbox_lo - 1e-4)
            and np.all(pos <= self._bbox_hi + 1e-4)
        )

    def velocity_at(self, position, time: float = 0.0) -> np.ndarray:
        """
        Return the CFD velocity vector at `position`.

        Parameters
        ----------
        position : array-like, shape (3,)
            Query point in metres (same coordinate frame as the vessel pkl).
        time : float
            Ignored for steady fields; kept for interface compatibility.

        Returns
        -------
        v : ndarray, shape (3,)  [m/s]
        """
        pos = np.asarray(position, dtype=np.float64)

        if self._extrapolate_zero and not self._in_bbox(pos):
            return np.zeros(3, dtype=np.float64)

        dists, idxs = self._tree.query(pos, k=self._k, workers=1)

        if self._k == 1:
            v = self._velocity[idxs]
        else:
            # Inverse-distance weighting; guard against exact hits (d=0)
            dists = np.maximum(dists, 1e-12)
            weights = 1.0 / dists
            weights /= weights.sum()
            v = (weights[:, None] * self._velocity[idxs]).sum(axis=0)

        return v * self.flow_scale

    def params_dict(self) -> dict:
        return {
            "flow_model": self.name,
            "n_cells": len(self._points),
            "flow_scale": self.flow_scale,
            "k_neighbours": self._k,
        }


class PulsatileCFDFlowField:
    """
    Pulsatile CFD velocity field (pimpleFoam phase-averaged output).

    Stores N velocity snapshots uniformly spaced over one cardiac cycle.
    At query time, interpolates spatially (IDW from k nearest cells) and
    temporally (linear between the two bounding phase snapshots).

    Parameters
    ----------
    npz_path : str or Path
        Path to the .npz archive.  Must contain:
            'points'   -- (M, 3) float32
            'velocity' -- (N, M, 3) float32
            'period'   -- scalar float  [seconds]; default 1.0 if absent
    flow_scale : float
        Global velocity multiplier.
    k_neighbours : int
        IDW neighbours for spatial interpolation.
    pulsation_amplitude : float or None
        If not None, the spatial CFD field is used for the spatial
        distribution but the temporal modulation is replaced by a simple
        sinusoidal envelope with this amplitude.  Useful when only a
        steady CFD solve is available but pulsatile time variation is
        still desired.  Set to None (default) to use the full phase
        sequence from the CFD solve.
    """

    name = "cfd_pulsatile"

    def __init__(
        self,
        npz_path: str | Path,
        flow_scale: float = 1.0,
        k_neighbours: int = 4,
        extrapolate_zero: bool = True,
        pulsation_amplitude: float | None = None,
    ):
        data = _load_npz(npz_path)
        self._points = data["points"].astype(np.float64)       # (M, 3)

        vel = data["velocity"].astype(np.float64)
        if vel.ndim == 2:
            # Single snapshot stored as steady — wrap to (1, M, 3)
            vel = vel[np.newaxis, ...]
        self._velocity = vel                                    # (N, M, 3)
        self._n_phases = self._velocity.shape[0]

        self._period = float(data["period"]) if "period" in data else 1.0
        self.flow_scale = float(flow_scale)
        self._k = int(k_neighbours)
        self._extrapolate_zero = extrapolate_zero
        self._pulsation_amplitude = pulsation_amplitude

        self._tree = _build_kdtree(self._points)
        self._bbox_lo = self._points.min(axis=0)
        self._bbox_hi = self._points.max(axis=0)

        print(
            f"[CFD] PulsatileCFDFlowField loaded: {len(self._points):,} cells  "
            f"{self._n_phases} phases  period={self._period:.3f}s  "
            f"flow_scale={flow_scale:.2f}"
        )

    def _in_bbox(self, pos: np.ndarray) -> bool:
        return bool(
            np.all(pos >= self._bbox_lo - 1e-4)
            and np.all(pos <= self._bbox_hi + 1e-4)
        )

    def _phase_weights(self, time: float):
        """
        Return (i0, i1, alpha) for linear temporal interpolation.
        v = (1 - alpha) * phase[i0] + alpha * phase[i1]
        """
        phase = (time % self._period) / self._period   # [0, 1)
        f = phase * self._n_phases
        i0 = int(f) % self._n_phases
        i1 = (i0 + 1) % self._n_phases
        alpha = f - int(f)
        return i0, i1, alpha

    def velocity_at(self, position, time: float = 0.0) -> np.ndarray:
        """
        Return the pulsatile CFD velocity vector at `position` and `time`.

        Parameters
        ----------
        position : array-like, shape (3,)
        time : float   [seconds]

        Returns
        -------
        v : ndarray, shape (3,)  [m/s]
        """
        pos = np.asarray(position, dtype=np.float64)

        if self._extrapolate_zero and not self._in_bbox(pos):
            return np.zeros(3, dtype=np.float64)

        dists, idxs = self._tree.query(pos, k=self._k, workers=1)

        if self._k == 1:
            idxs = np.array([idxs])
            dists = np.array([dists])

        dists = np.maximum(dists, 1e-12)
        weights = 1.0 / dists
        weights /= weights.sum()

        if self._pulsation_amplitude is not None:
            # Use phase 0 (mean field) with sinusoidal time modulation
            v_spatial = (weights[:, None] * self._velocity[0][idxs]).sum(axis=0)
            modulation = 1.0 + self._pulsation_amplitude * np.sin(
                2.0 * np.pi * time / self._period
            )
            v = v_spatial * modulation
        else:
            i0, i1, alpha = self._phase_weights(time)
            v0 = (weights[:, None] * self._velocity[i0][idxs]).sum(axis=0)
            v1 = (weights[:, None] * self._velocity[i1][idxs]).sum(axis=0)
            v = (1.0 - alpha) * v0 + alpha * v1

        return v * self.flow_scale

    def params_dict(self) -> dict:
        return {
            "flow_model": self.name,
            "n_cells": len(self._points),
            "n_phases": self._n_phases,
            "period_s": self._period,
            "flow_scale": self.flow_scale,
            "k_neighbours": self._k,
        }
