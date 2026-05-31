"""
VesselNetwork: procedural bifurcating vascular tree with connected return path.

Topology
--------
A symmetric binary tree with n branching levels:

  inlet root
      -> bifurcating afferent tree
      -> terminal capillary connector vessels
      -> reconverging efferent tree
      -> outlet root

Main fixes in this version
--------------------------
1. Adds connector/capillary segments between afferent leaves and efferent leaves.
2. Uses physically reasonable child velocity scaling under Murray-style splitting.
3. Keeps efferent v_mean positive because efferent geometry is already reversed.
"""

import numpy as np
from dataclasses import dataclass, field
from typing import List, Optional, Tuple

from .constants import PI, EPS


# ------------------------------------------------------------------ #
#  Murray's law helpers                                                #
# ------------------------------------------------------------------ #

MURRAY_FACTOR = 2.0 ** (-1.0 / 3.0)   # r_child = r_parent * this

# If total volumetric flow splits equally into two child vessels:
# Q_child = Q_parent / 2
# A_child = pi * (r_parent * 2^(-1/3))^2 = A_parent * 2^(-2/3)
# v_child = Q_child / A_child = v_parent * 2^(-1/3)
SPEED_FACTOR = 2.0 ** (-1.0 / 3.0)


# ------------------------------------------------------------------ #
#  Segment dataclass                                                   #
# ------------------------------------------------------------------ #

@dataclass
class VesselSegment:
    """One cylindrical vessel segment."""
    seg_id: int
    start: np.ndarray
    end: np.ndarray
    radius: float
    v_mean: float
    level: int
    is_efferent: bool = False
    is_connector: bool = False

    _axis: np.ndarray = field(default_factory=lambda: np.zeros(3))
    _length: float = 0.0

    def __post_init__(self):
        diff = self.end - self.start
        self._length = float(np.linalg.norm(diff))

        if self._length > EPS:
            self._axis = diff / self._length
        else:
            self._axis = np.array([0.0, 1.0, 0.0])

    def closest_point_and_radial(
        self, pos: np.ndarray
    ) -> Tuple[np.ndarray, np.ndarray, float, float]:
        """
        Return:
            closest point on axis,
            radial vector,
            radial distance,
            axial t in [0, 1]
        """
        pos = np.asarray(pos, dtype=float)
        d = pos - self.start

        t = float(np.dot(d, self._axis)) / (self._length + EPS)
        t = float(np.clip(t, 0.0, 1.0))

        closest = self.start + t * self._length * self._axis
        radial_vec = pos - closest
        radial_dist = float(np.linalg.norm(radial_vec))

        return closest, radial_vec, radial_dist, t

    def contains(self, pos: np.ndarray, tol: float = 1e-9) -> bool:
        """True if pos is inside the cylindrical segment."""
        _, _, r, _ = self.closest_point_and_radial(pos)
        return r <= self.radius + tol

    def clip_to_interior(self, pos: np.ndarray, inset: float = 0.01) -> np.ndarray:
        """
        Project pos onto the segment axis, clamped slightly away from endpoints.
        """
        pos = np.asarray(pos, dtype=float)
        d = pos - self.start

        t = float(np.dot(d, self._axis)) / (self._length + 1e-30)
        t = float(np.clip(t, inset, 1.0 - inset))

        return self.start + t * self._length * self._axis

    def flow_velocity(self, pos: np.ndarray, time: float = 0.0) -> np.ndarray:
        """
        Poiseuille parabolic profile with mild pulsatile modulation.

        v(r) = v_max * (1 - (r/R)^2)
        v_max = 2 * abs(v_mean)
        """
        _, _, r, _ = self.closest_point_and_radial(pos)

        if r >= self.radius:
            return np.zeros(3)

        v_max = 2.0 * abs(self.v_mean)
        profile = 1.0 - (r / self.radius) ** 2
        pulsation = 1.0 + 0.25 * np.sin(2.0 * PI * 1.0 * time)

        speed = v_max * profile * pulsation

        direction = self._axis if self.v_mean >= 0 else -self._axis
        return speed * direction


# ------------------------------------------------------------------ #
#  Tree builder                                                        #
# ------------------------------------------------------------------ #

def _build_afferent_tree(
    root_start: np.ndarray,
    root_end: np.ndarray,
    root_radius: float,
    root_v_mean: float,
    n_levels: int,
    spread_x: float,
    spread_z: float,
    seg_length_scale: float,
) -> List[VesselSegment]:
    """
    Recursively build a symmetric bifurcating afferent tree.
    """
    segments: List[VesselSegment] = []
    next_id = [0]

    def _recurse(start, end, radius, v_mean, level):
        sid = next_id[0]
        next_id[0] += 1

        seg = VesselSegment(
            seg_id=sid,
            start=start.copy(),
            end=end.copy(),
            radius=radius,
            v_mean=v_mean,
            level=level,
            is_efferent=False,
            is_connector=False,
        )
        segments.append(seg)

        if level >= n_levels:
            return

        parent_axis = end - start
        parent_len = float(np.linalg.norm(parent_axis))

        if parent_len < EPS:
            return

        parent_axis /= parent_len

        child_len = parent_len * seg_length_scale
        child_radius = radius * MURRAY_FACTOR
        child_v_mean = v_mean * SPEED_FACTOR

        if level % 2 == 0:
            spread_dir = np.array([spread_x, 0.0, 0.0])
        else:
            spread_dir = np.array([0.0, 0.0, spread_z])

        for sign in (+1.0, -1.0):
            child_axis = parent_axis + sign * spread_dir * 0.5
            norm = np.linalg.norm(child_axis)

            if norm > EPS:
                child_axis = child_axis / norm

            child_end = end + child_axis * child_len

            _recurse(
                start=end,
                end=child_end,
                radius=child_radius,
                v_mean=child_v_mean,
                level=level + 1,
            )

    _recurse(root_start, root_end, root_radius, root_v_mean, level=0)
    return segments


# ------------------------------------------------------------------ #
#  VesselNetwork                                                       #
# ------------------------------------------------------------------ #

class VesselNetwork:
    """
    Bifurcating vascular tree with connected capillary return path.
    """

    name = "vessel_network"

    def __init__(
        self,
        root_radius: float = 0.006,
        root_v_mean: float = 0.05,
        n_levels: int = 4,
        trunk_length: float = 0.04,
        outlet_offset: float = 0.015,
        spread_x: float = 1.0,
        spread_z: float = 1.0,
        seg_length_scale: float = 0.75,
    ):
        self.root_radius = float(root_radius)
        self.root_v_mean = float(root_v_mean)
        self.n_levels = int(n_levels)
        self.trunk_length = float(trunk_length)
        self.outlet_offset = float(outlet_offset)

        # ---- afferent tree -----------------------------------------

        aff_start = np.array([0.0, -trunk_length / 2.0, 0.0])
        aff_end = np.array([0.0, trunk_length / 2.0, 0.0])

        self._afferent = _build_afferent_tree(
            root_start=aff_start,
            root_end=aff_end,
            root_radius=root_radius,
            root_v_mean=root_v_mean,
            n_levels=n_levels,
            spread_x=spread_x,
            spread_z=spread_z,
            seg_length_scale=seg_length_scale,
        )

        # ---- efferent return tree ----------------------------------
        # IMPORTANT:
        # Geometry is already reversed by swapping start/end.
        # Therefore v_mean stays positive.
        # Do NOT use -seg.v_mean here.

        self._efferent: List[VesselSegment] = []
        n_aff = len(self._afferent)

        for seg in self._afferent:
            eseg = VesselSegment(
                seg_id=seg.seg_id + n_aff,
                start=seg.end + np.array([outlet_offset, 0.0, 0.0]),
                end=seg.start + np.array([outlet_offset, 0.0, 0.0]),
                radius=seg.radius,
                v_mean=seg.v_mean,
                level=seg.level,
                is_efferent=True,
                is_connector=False,
            )
            self._efferent.append(eseg)

        # ---- capillary connector vessels ---------------------------
        # These bridge afferent leaves to matching efferent leaves.
        # Without this, the afferent and efferent trees are disconnected.

        self._connectors: List[VesselSegment] = []

        afferent_leaves = [s for s in self._afferent if s.level == self.n_levels]
        efferent_leaves = [s for s in self._efferent if s.level == self.n_levels]

        afferent_leaves = sorted(afferent_leaves, key=lambda s: s.seg_id)
        efferent_leaves = sorted(efferent_leaves, key=lambda s: s.seg_id)

        connector_start_id = len(self._afferent) + len(self._efferent)

        for i, (a_seg, e_seg) in enumerate(zip(afferent_leaves, efferent_leaves)):
            conn = VesselSegment(
                seg_id=connector_start_id + i,
                start=a_seg.end.copy(),
                end=e_seg.start.copy(),
                radius=a_seg.radius,
                v_mean=a_seg.v_mean,
                level=self.n_levels + 1,
                is_efferent=False,
                is_connector=True,
            )
            self._connectors.append(conn)

        self.segments: List[VesselSegment] = (
            self._afferent + self._connectors + self._efferent
        )
        self.n_segments = len(self.segments)

        self._build_spatial_index()

    # ------------------------------------------------------------------ #
    #  Spatial index                                                       #
    # ------------------------------------------------------------------ #

    def _build_spatial_index(self, grid_res: int = 20):
        """
        Divide the bounding box into a grid and map cells to nearby segments.
        """
        coords = np.array([
            pt
            for seg in self.segments
            for pt in (seg.start, seg.end)
        ])

        self._bbox_min = coords.min(axis=0) - 0.01
        self._bbox_max = coords.max(axis=0) + 0.01
        self._grid_res = int(grid_res)

        span = self._bbox_max - self._bbox_min
        span = np.where(span < EPS, EPS, span)
        self._cell_size = span / self._grid_res

        self._grid: dict = {}

        for seg in self.segments:
            lo = np.minimum(seg.start, seg.end) - seg.radius
            hi = np.maximum(seg.start, seg.end) + seg.radius

            i0, j0, k0 = self._world_to_cell(lo)
            i1, j1, k1 = self._world_to_cell(hi)

            for i in range(i0, i1 + 1):
                for j in range(j0, j1 + 1):
                    for k in range(k0, k1 + 1):
                        key = (i, j, k)
                        if key not in self._grid:
                            self._grid[key] = []
                        self._grid[key].append(seg)

    def _world_to_cell(self, pos: np.ndarray):
        pos = np.asarray(pos, dtype=float)

        denom = self._bbox_max - self._bbox_min
        denom = np.where(denom < EPS, EPS, denom)

        idx = (pos - self._bbox_min) / denom * self._grid_res
        i, j, k = np.clip(idx.astype(int), 0, self._grid_res - 1)

        return int(i), int(j), int(k)

    def _candidates(self, pos: np.ndarray) -> list:
        """
        Return candidate vessel segments near pos.
        """
        key = self._world_to_cell(pos)

        cands = self._grid.get(key, None)
        if cands:
            return cands

        i, j, k = key
        result = []
        seen = set()

        for di in (-1, 0, 1):
            for dj in (-1, 0, 1):
                for dk in (-1, 0, 1):
                    nkey = (i + di, j + dj, k + dk)
                    for seg in self._grid.get(nkey, []):
                        if seg.seg_id not in seen:
                            seen.add(seg.seg_id)
                            result.append(seg)

        return result if result else self.segments

    # ------------------------------------------------------------------ #
    #  Core spatial query                                                  #
    # ------------------------------------------------------------------ #

    def find_segment(
        self, pos: np.ndarray
    ) -> Tuple[Optional[VesselSegment], float]:
        """
        Return nearest segment and radial distance to its centerline.
        """
        pos = np.asarray(pos, dtype=float)

        best_seg = None
        best_dist = np.inf

        for seg in self._candidates(pos):
            _, _, r, _ = seg.closest_point_and_radial(pos)

            if r < best_dist:
                best_dist = r
                best_seg = seg

        if best_seg is None:
            for seg in self.segments:
                _, _, r, _ = seg.closest_point_and_radial(pos)

                if r < best_dist:
                    best_dist = r
                    best_seg = seg

        return best_seg, best_dist

    def in_any_vessel(self, pos: np.ndarray) -> bool:
        """
        True if pos is inside any vessel.
        """
        for seg in self._candidates(pos):
            if seg.contains(pos):
                return True
        return False

    # ------------------------------------------------------------------ #
    #  Flow interface                                                      #
    # ------------------------------------------------------------------ #

    def velocity_at(self, pos: np.ndarray, time: float = 0.0) -> np.ndarray:
        """
        Return blood flow velocity at position.
        """
        pos = np.asarray(pos, dtype=float)

        best_seg = None
        best_dist = np.inf

        for seg in self._candidates(pos):
            _, _, r, _ = seg.closest_point_and_radial(pos)

            if r < seg.radius and r < best_dist:
                best_dist = r
                best_seg = seg

        if best_seg is None:
            return np.zeros(3)

        return best_seg.flow_velocity(pos, time)

    def params_dict(self) -> dict:
        return {
            "flow_model": self.name,
            "n_levels": self.n_levels,
            "n_segments": self.n_segments,
            "n_afferent": len(self._afferent),
            "n_connectors": len(self._connectors),
            "n_efferent": len(self._efferent),
            "root_radius_m": self.root_radius,
            "root_v_mean_m_s": self.root_v_mean,
            "trunk_length_m": self.trunk_length,
            "outlet_offset_m": self.outlet_offset,
            "murray_factor": MURRAY_FACTOR,
            "speed_factor": SPEED_FACTOR,
        }

    # ------------------------------------------------------------------ #
    #  Workspace / boundary interface                                      #
    # ------------------------------------------------------------------ #

    @property
    def half(self) -> float:
        coords = []

        for seg in self.segments:
            coords.append(seg.start)
            coords.append(seg.end)

        arr = np.array(coords)
        return float(np.max(np.abs(arr))) * 1.1

    def clip(self, pos: np.ndarray) -> np.ndarray:
        """
        If pos is outside all vessels, project it back to the nearest vessel.
        """
        pos = np.asarray(pos, dtype=float)

        if self.in_any_vessel(pos):
            return pos.copy()

        seg, _ = self.find_segment(pos)

        if seg is None:
            return pos.copy()

        closest, radial_vec, radial_dist, _ = seg.closest_point_and_radial(pos)

        if radial_dist < EPS:
            perp = np.array([1.0, 0.0, 0.0])

            if abs(np.dot(perp, seg._axis)) > 0.9:
                perp = np.array([0.0, 1.0, 0.0])

            radial_hat = perp - np.dot(perp, seg._axis) * seg._axis
            radial_hat /= np.linalg.norm(radial_hat)
        else:
            radial_hat = radial_vec / radial_dist

        return closest + radial_hat * seg.radius * 0.99

    # ------------------------------------------------------------------ #
    #  Helpers for simulation setup                                        #
    # ------------------------------------------------------------------ #

    def inlet_start(self) -> np.ndarray:
        """
        Entry point of the afferent inlet segment.
        """
        return self._afferent[0].start.copy()

    def outlet_end(self) -> np.ndarray:
        """
        Physical exit point of the efferent root segment.
        """
        for seg in self._efferent:
            if seg.level == 0:
                return seg.end.copy()

        return self._efferent[0].end.copy()

    def random_point_in_vessel(self, rng=None) -> np.ndarray:
        """
        Sample a random point inside any vessel segment.
        """
        if rng is None:
            rng = np.random.default_rng()

        seg = rng.choice(self.segments)
        t = rng.uniform(0.0, 1.0)

        # sqrt gives uniform area distribution in disk.
        r = seg.radius * np.sqrt(rng.uniform(0.0, 1.0))
        phi = rng.uniform(0.0, 2.0 * PI)

        axis = seg._axis

        perp1 = np.array([1.0, 0.0, 0.0])
        if abs(np.dot(perp1, axis)) > 0.9:
            perp1 = np.array([0.0, 1.0, 0.0])

        perp1 = perp1 - np.dot(perp1, axis) * axis
        perp1 = perp1 / np.linalg.norm(perp1)

        perp2 = np.cross(axis, perp1)

        point_on_axis = seg.start + t * seg._length * axis

        return point_on_axis + r * (
            np.cos(phi) * perp1 + np.sin(phi) * perp2
        )

    def segment_summary(self) -> str:
        """
        Human-readable summary of all vessel segments.
        """
        lines = [
            (
                f"VesselNetwork: {self.n_segments} segments "
                f"({len(self._afferent)} afferent + "
                f"{len(self._connectors)} connectors + "
                f"{len(self._efferent)} efferent)"
            ),
            (
                f"  Levels: {self.n_levels}  "
                f"Root radius: {self.root_radius * 1000:.1f} mm  "
                f"Root v_mean: {self.root_v_mean * 100:.1f} cm/s"
            ),
            (
                f"  Murray radius factor: {MURRAY_FACTOR:.3f}  "
                f"Speed factor per branch: {SPEED_FACTOR:.3f}"
            ),
        ]

        for seg in self.segments:
            if seg.is_connector:
                tag = "conn"
            elif seg.is_efferent:
                tag = "eff"
            else:
                tag = "aff"

            lines.append(
                f"  [{tag} L{seg.level} id={seg.seg_id:3d}] "
                f"r={seg.radius * 1000:.2f} mm  "
                f"v={seg.v_mean * 100:.2f} cm/s  "
                f"len={seg._length * 100:.1f} cm"
            )

        return "\n".join(lines)

    # ------------------------------------------------------------------ #
    #  Graph pathfinding                                                   #
    # ------------------------------------------------------------------ #

    def _build_graph(self):
        """
        Build adjacency graph of segments connected at shared endpoints.
        """
        tol = 1e-6
        graph = {seg.seg_id: [] for seg in self.segments}

        endpoints = []

        for seg in self.segments:
            endpoints.append((seg.seg_id, "start", seg.start))
            endpoints.append((seg.seg_id, "end", seg.end))

        for i, (id_a, _, pt_a) in enumerate(endpoints):
            for id_b, _, pt_b in endpoints[i + 1:]:
                if id_a == id_b:
                    continue

                if np.linalg.norm(pt_a - pt_b) < tol:
                    junction = (pt_a + pt_b) / 2.0
                    graph[id_a].append((id_b, junction))
                    graph[id_b].append((id_a, junction))

        return graph

    def plan_waypoints(
        self,
        start_pos: np.ndarray,
        target_pos: np.ndarray,
    ) -> List[np.ndarray]:
        """
        Plan waypoints through the vessel segment graph using Dijkstra.
        """
        start_pos = np.asarray(start_pos, dtype=float)
        target_pos = np.asarray(target_pos, dtype=float)

        start_seg, _ = self.find_segment(start_pos)
        target_seg, _ = self.find_segment(target_pos)

        if start_seg is None or target_seg is None:
            return [target_pos]

        if start_seg.seg_id == target_seg.seg_id:
            return [target_pos]

        if not hasattr(self, "_graph"):
            self._graph = self._build_graph()

        graph = self._graph

        import heapq

        dist = {seg.seg_id: np.inf for seg in self.segments}
        prev = {seg.seg_id: None for seg in self.segments}
        junction = {seg.seg_id: None for seg in self.segments}

        dist[start_seg.seg_id] = 0.0
        heap = [(0.0, start_seg.seg_id)]

        seg_by_id = {seg.seg_id: seg for seg in self.segments}

        while heap:
            d, uid = heapq.heappop(heap)

            if d > dist[uid]:
                continue

            if uid == target_seg.seg_id:
                break

            for vid, jpt in graph.get(uid, []):
                seg_v = seg_by_id[vid]
                new_dist = d + seg_v._length

                if new_dist < dist[vid]:
                    dist[vid] = new_dist
                    prev[vid] = uid
                    junction[vid] = jpt
                    heapq.heappush(heap, (new_dist, vid))

        if dist[target_seg.seg_id] == np.inf:
            # This should be rare now because connectors exist.
            return [target_pos]

        path_ids = []
        cur = target_seg.seg_id

        while cur is not None:
            path_ids.append(cur)
            cur = prev[cur]

        path_ids.reverse()

        waypoints = []

        for sid in path_ids[1:]:
            jpt = junction[sid]

            if jpt is not None:
                seg = seg_by_id[sid]
                wp = seg.clip_to_interior(jpt)
                waypoints.append(wp)

        waypoints.append(target_pos)

        return waypoints

    def waypoint_navigator(
        self,
        start_pos: np.ndarray,
        target_pos: np.ndarray,
        arrival_tol: float = 0.003,
    ):
        """
        Returns a callable next_waypoint(current_pos).
        """
        waypoints = self.plan_waypoints(start_pos, target_pos)

        state = {
            "idx": 0,
            "waypoints": waypoints,
        }

        def next_waypoint(current_pos: np.ndarray) -> np.ndarray:
            wps = state["waypoints"]
            idx = state["idx"]

            if idx >= len(wps):
                return wps[-1]

            wp = wps[idx]

            if np.linalg.norm(np.asarray(current_pos) - wp) < arrival_tol:
                state["idx"] = min(idx + 1, len(wps) - 1)
                
            next_waypoint.current_idx = state["idx"]
            return state["waypoints"][state["idx"]]

        next_waypoint.waypoints = waypoints
        next_waypoint.total_hops = len(waypoints)

        return next_waypoint