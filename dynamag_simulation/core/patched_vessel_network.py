"""
patched_vessel_network.py
=========================
Drop-in replacement for RealVesselNetwork that transparently routes all
velocity queries through a CFDFlowField when one is supplied, falling
back to the analytical Poiseuille model when none is provided.

The patch is non-invasive: the original real_vessel_network.py is
imported and subclassed — no editing of the original file required.

Usage
-----
    # Analytical (original behaviour, zero code changes to controllers)
    from dynamag_simulation.core.real_vessel_network import RealVesselNetwork
    network = RealVesselNetwork("abdominal_aorta.pkl", flow_scale=1.0)

    # CFD-backed (swap one import + add cfd_flow kwarg)
    from patched_vessel_network import RealVesselNetworkCFD
    from cfd_flow_field import PulsatileCFDFlowField

    cfd = PulsatileCFDFlowField("data/abdominal_aorta_pulsatile.npz", flow_scale=1.0)
    network = RealVesselNetworkCFD("abdominal_aorta.pkl", cfd_flow=cfd)

    # Both expose identical interfaces — controllers need no changes
    v = network.velocity_at(pos, time)

How the override works
----------------------
RealVesselNetworkCFD overrides only two methods:

    velocity_at(pos, time)
        If cfd_flow is set, delegates directly to cfd_flow.velocity_at().
        The CFD field already covers the full 3-D domain, so per-segment
        lookup is unnecessary.

    segment_flow_velocity(seg, pos, time)
        Called internally when only a segment's contribution is needed
        (e.g. by the route planner's authority estimation).  Delegates to
        CFD when available; falls back to the analytical segment model.

The analytical fallback is preserved for:
  - Segments not covered by the CFD mesh bounding box
  - Flow outside the vessel (r > R), where both models return zero
  - Runs without a cfd_flow argument (pure Poiseuille, unchanged)

Hybrid per-segment CFD
----------------------
If you want different flow models for different vessels (e.g. CFD for the
aortic root, analytical for pulmonary), use HybridVesselNetworkCFD.
Pass a dict mapping segment IDs to CFDFlowField objects.  Segments not
in the dict use the analytical model.

    hybrid = HybridVesselNetworkCFD(
        "combined.pkl",
        seg_cfd_map={
            0: SteadyCFDFlowField("aortic_root_steady.npz"),
            1: SteadyCFDFlowField("aortic_root_steady.npz"),
        }
    )
"""

from __future__ import annotations

import sys
import os
import numpy as np

# Allow running from the repo root or from scripts/ without install
_repo = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if _repo not in sys.path:
    sys.path.insert(0, _repo)

from dynamag_simulation.core.real_vessel_network import (
    RealVesselNetwork,
    RealVesselSegment,
)


class RealVesselNetworkCFD(RealVesselNetwork):
    """
    RealVesselNetwork with optional CFD-backed velocity queries.

    Parameters
    ----------
    pkl_file : str
        Path to vessel segment pickle (same as RealVesselNetwork).
    cfd_flow : SteadyCFDFlowField | PulsatileCFDFlowField | None
        Pre-loaded CFD flow field.  If None, falls back to the analytical
        Poiseuille model for all segments.
    use_analytical_outside_bbox : bool
        When the query point is outside the CFD mesh bounding box, fall
        back to the analytical model instead of returning zero.  Useful
        for vessel regions not covered by the CFD domain.  Default True.
    **kwargs
        Forwarded to RealVesselNetwork.__init__ (connection_tol, flow_scale, …).
    """

    def __init__(
        self,
        pkl_file: str,
        cfd_flow=None,
        use_analytical_outside_bbox: bool = True,
        **kwargs,
    ):
        super().__init__(pkl_file, **kwargs)
        self.cfd_flow = cfd_flow
        self._use_analytical_outside_bbox = use_analytical_outside_bbox

        if cfd_flow is not None:
            mode = cfd_flow.name
            print(f"[CFD] RealVesselNetworkCFD: flow model = {mode}")
        else:
            print("[CFD] RealVesselNetworkCFD: no CFD field supplied — analytical fallback")

    # ------------------------------------------------------------------
    # Core override: velocity_at
    # ------------------------------------------------------------------

    def velocity_at(self, pos, time: float = 0.0) -> np.ndarray:
        """
        Return velocity at `pos`, `time`.

        Routes to CFD field when available; falls back to analytical when:
          - cfd_flow is None
          - pos is outside the CFD bounding box (if use_analytical_outside_bbox)
        """
        if self.cfd_flow is None:
            return super().velocity_at(pos, time)

        pos = np.asarray(pos, dtype=np.float64)

        # Out-of-bbox fallback
        if self._use_analytical_outside_bbox and not self.cfd_flow._in_bbox(pos):
            return super().velocity_at(pos, time)

        return self.cfd_flow.velocity_at(pos, time)

    # ------------------------------------------------------------------
    # Segment-level override (used by route planner authority checks)
    # ------------------------------------------------------------------

    def segment_flow_velocity(
        self, seg: RealVesselSegment, pos, time: float = 0.0
    ) -> np.ndarray:
        """
        Flow velocity at `pos` attributed to `seg`.

        Used by ACBC/COMET authority estimation which calls this per-
        segment rather than going through velocity_at().
        Delegates to CFD when available; analytical otherwise.
        """
        if self.cfd_flow is None:
            return seg.flow_velocity(pos, time)

        pos = np.asarray(pos, dtype=np.float64)

        if self._use_analytical_outside_bbox and not self.cfd_flow._in_bbox(pos):
            return seg.flow_velocity(pos, time)

        return self.cfd_flow.velocity_at(pos, time)

    # ------------------------------------------------------------------
    # Reporting
    # ------------------------------------------------------------------

    def params_dict(self) -> dict:
        d = super().params_dict()
        if self.cfd_flow is not None:
            d.update(self.cfd_flow.params_dict())
        return d

    def segment_summary(self) -> str:
        base = super().segment_summary()
        if self.cfd_flow is not None:
            return base + f"  [CFD: {self.cfd_flow.name}  {self.cfd_flow._points.shape[0]:,} cells]"
        return base + "  [CFD: analytical fallback]"


class HybridVesselNetworkCFD(RealVesselNetworkCFD):
    """
    Per-segment CFD override.

    Useful when multiple vascular models are loaded together but only
    some have a CFD solution available.

    Parameters
    ----------
    seg_cfd_map : dict[int, CFDFlowField]
        Maps segment IDs to their corresponding CFD flow field.
        Segments not in the map use the analytical model.
    default_cfd_flow : CFDFlowField | None
        Applied to any segment not in seg_cfd_map.  If None, those
        segments use the analytical model.
    """

    def __init__(
        self,
        pkl_file: str,
        seg_cfd_map: dict | None = None,
        default_cfd_flow=None,
        **kwargs,
    ):
        # Pass default_cfd_flow as the global cfd_flow
        super().__init__(pkl_file, cfd_flow=default_cfd_flow, **kwargs)
        self._seg_cfd_map: dict = seg_cfd_map or {}

        n_mapped = len(self._seg_cfd_map)
        print(f"[CFD] HybridVesselNetworkCFD: {n_mapped} segments with custom CFD fields")

    def velocity_at(self, pos, time: float = 0.0) -> np.ndarray:
        pos = np.asarray(pos, dtype=np.float64)

        # Find which segment this position belongs to
        seg, _ = self.find_segment(pos)

        if seg is not None and seg.seg_id in self._seg_cfd_map:
            field = self._seg_cfd_map[seg.seg_id]
            if field._in_bbox(pos) or not self._use_analytical_outside_bbox:
                return field.velocity_at(pos, time)

        # Fall through to global CFD or analytical
        return super().velocity_at(pos, time)
