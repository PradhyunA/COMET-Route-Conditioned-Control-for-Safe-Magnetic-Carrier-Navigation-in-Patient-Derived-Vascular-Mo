"""Centreline navigation and existing wall model driven by a steady CFD field."""

import numpy as np

from .real_vessel_network import RealVesselNetwork


class RealVesselNetworkCFD(RealVesselNetwork):
    name = "real_vessel_network_steady_cfd"

    def __init__(self, pkl_file, cfd_flow, flow_scale=1.0, connection_tol=1e-6):
        if cfd_flow is None or cfd_flow.flow_scale != float(flow_scale):
            raise ValueError("A CFD field with the network's flow scale is required")
        super().__init__(pkl_file, connection_tol, flow_scale)
        self.cfd_flow = cfd_flow
        metadata = cfd_flow.metadata
        expected = metadata.get("vessel_sha256")
        if expected:
            import hashlib
            from pathlib import Path
            actual = hashlib.sha256(Path(self.source_file).read_bytes()).hexdigest()
            if actual != expected:
                raise ValueError("CFD field was imported against a different vessel geometry")
        mids = np.array([(seg.start + seg.end) * 0.5 for seg in self.segments])
        velocities = cfd_flow.velocity_at_many(mids)
        self._segment_cfd_velocity = dict(zip((s.seg_id for s in self.segments), velocities))

    def velocity_at(self, pos, time=0.0):
        segment, distance = self.find_segment(pos)
        if segment is None or distance > segment.radius:
            return np.zeros(3)
        return self.cfd_flow.velocity_at(pos, time)

    def _route_accessible_flow(self, seg, required):
        # Midpoint CFD vectors supply graph costs. No synthetic Poiseuille
        # profile is inferred from the segment's analytical v_mean parameter.
        return self._segment_cfd_velocity[seg.seg_id]

    def params_dict(self):
        return {**super().params_dict(), "cfd_file": self.cfd_flow.source_file,
                "flow_interpolation": "bounded 8-neighbour inverse-distance-squared"}

    def segment_summary(self):
        return (f"RealVesselNetworkCFD: {self.n_segments} segments, "
                f"steady CFD velocity multiplier={self.flow_scale:g}")
