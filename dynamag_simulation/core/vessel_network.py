"""
VesselNetwork: procedural 4-level bifurcating vascular tree.

Topology
--------
A symmetric binary tree with 4 branching levels:

  Level 0 (root):   1 inlet segment
  Level 1:          2 segments
  Level 2:          4 segments
  Level 3:          8 segments
  Level 4 (leaves): 16 terminal segments  <- capillary bed here
  ... mirror tree reconverges back ...
  Level 0 (root):   1 outlet segment (parallel to inlet)

Flow travels:
  inlet -> bifurcating afferent tree -> 16 terminals ->
  reconverging efferent tree -> outlet

Geometry
--------
The tree is laid out in the XZ plane, centered at the origin.
  - Inlet enters from -Y, runs along +Y
  - Each bifurcation spreads symmetrically in X and Z
  - The efferent (return) tree mirrors the afferent tree,
    offset slightly in X so inlet and outlet run side by side

Radii
-----
Murray's law (symmetric bifurcation):
    r_child = r_parent / 2^(1/3)  (~0.794 * r_parent)

Flow speed
----------
Conservation of mass: v * pi * r^2 = const
    v_child = v_parent * (r_parent / r_child)^2 = v_parent * 2^(2/3)

All units SI (meters, m/s).
"""

import numpy as np
from dataclasses import dataclass, field
from typing import List, Optional, Tuple

from .constants import PI, EPS


# ------------------------------------------------------------------ #
#  Murray's law helpers                                                #
# ------------------------------------------------------------------ #

MURRAY_FACTOR = 2.0 ** (-1.0 / 3.0)   # r_child = r_parent * this
SPEED_FACTOR  = 2.0 ** ( 2.0 / 3.0)   # v_child = v_parent * this


# ------------------------------------------------------------------ #
#  Segment dataclass                                                   #
# ------------------------------------------------------------------ #

@dataclass
class VesselSegment:
    """One cylindrical vessel segment."""
    seg_id:    int
    start:     np.ndarray   # 3D world coords (m)
    end:       np.ndarray   # 3D world coords (m)
    radius:    float        # vessel radius (m)
    v_mean:    float        # mean (centreline) flow speed (m/s), signed
    level:     int          # bifurcation depth (0 = root)
    is_efferent: bool = False   # True for return-flow segments

    # cached geometry (filled by VesselNetwork.__post_init__)
    _axis:   np.ndarray = field(default_factory=lambda: np.zeros(3))
    _length: float      = 0.0

    def __post_init__(self):
        diff = self.end - self.start
        self._length = float(np.linalg.norm(diff))
        if self._length > EPS:
            self._axis = diff / self._length
        else:
            self._axis = np.array([0.0, 1.0, 0.0])

    # -------------------------------------------------------------- #
    #  Geometry queries                                                #
    # -------------------------------------------------------------- #

    def closest_point_and_radial(
        self, pos: np.ndarray
    ) -> Tuple[np.ndarray, np.ndarray, float, float]:
        """
        Return (closest_point_on_axis, radial_vec, radial_dist, axial_t).

        axial_t is clamped to [0, 1] so the closest point is always
        within the segment (not extrapolated past the ends).
        """
        d = pos - self.start
        t = float(np.dot(d, self._axis)) / (self._length + EPS)
        t = float(np.clip(t, 0.0, 1.0))
        closest = self.start + t * self._length * self._axis
        radial_vec = pos - closest
        radial_dist = float(np.linalg.norm(radial_vec))
        return closest, radial_vec, radial_dist, t

    def contains(self, pos: np.ndarray, tol: float = 1e-9) -> bool:
        """True if pos is inside the cylinder (including end-caps)."""
        _, _, r, _ = self.closest_point_and_radial(pos)
        return r <= self.radius + tol

    def clip_to_interior(self, pos: np.ndarray, inset: float = 0.01) -> np.ndarray:
        """
        Project pos onto the segment axis, clamped to the interior.
        Used to place waypoints safely inside a segment near a junction.
        inset: fraction of segment length to move away from the end.
        """
        pos = np.asarray(pos, dtype=float)
        d = pos - self.start
        t = float(np.dot(d, self._axis)) / (self._length + 1e-30)
        t = float(np.clip(t, inset, 1.0 - inset))
        return self.start + t * self._length * self._axis

    def flow_velocity(self, pos: np.ndarray, time: float = 0.0) -> np.ndarray:
        """
        Poiseuille parabolic profile at pos, with pulsatile modulation.

        v(r) = v_max * (1 - (r/R)^2) * axis_hat
        v_max = 2 * v_mean  (parabolic profile)
        """
        _, radial_vec, r, _ = self.closest_point_and_radial(pos)
        if r >= self.radius:
            return np.zeros(3)
        v_max = 2.0 * abs(self.v_mean)
        profile = 1.0 - (r / self.radius) ** 2
        # mild pulsatile modulation (cardiac-like, ~1 Hz)
        pulsation = 1.0 + 0.25 * np.sin(2.0 * PI * 1.0 * time)
        speed = v_max * profile * pulsation
        direction = self._axis if self.v_mean >= 0 else -self._axis
        return speed * direction


# ------------------------------------------------------------------ #
#  Tree builder                                                        #
# ------------------------------------------------------------------ #

def _build_afferent_tree(
    root_start: np.ndarray,
    root_end:   np.ndarray,
    root_radius: float,
    root_v_mean: float,
    n_levels: int,
    spread_x: float,
    spread_z: float,
    seg_length_scale: float,
) -> List[VesselSegment]:
    """
    Recursively build a symmetric bifurcating tree.
    Returns list of VesselSegment (afferent, level 0..n_levels).
    """
    segments = []
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
        )
        segments.append(seg)

        if level >= n_levels:
            return  # leaf — no further bifurcation

        # child geometry: spread symmetrically around the parent endpoint
        parent_axis = (end - start)
        parent_len  = float(np.linalg.norm(parent_axis))
        if parent_len < EPS:
            return
        parent_axis /= parent_len

        child_len    = parent_len * seg_length_scale
        child_radius = radius * MURRAY_FACTOR
        child_v_mean = v_mean * SPEED_FACTOR

        # spread vector: rotate spread direction per level to avoid overlap
        if level % 2 == 0:
            spread_dir = np.array([spread_x, 0.0, 0.0])
        else:
            spread_dir = np.array([0.0, 0.0, spread_z])

        for sign in (+1.0, -1.0):
            # child axis = blend of parent axis and lateral spread
            child_axis = parent_axis + sign * spread_dir * 0.5
            child_axis_norm = np.linalg.norm(child_axis)
            if child_axis_norm > EPS:
                child_axis /= child_axis_norm
            child_end = end + child_axis * child_len
            _recurse(end, child_end, child_radius, child_v_mean, level + 1)

    _recurse(root_start, root_end, root_radius, root_v_mean, level=0)
    return segments


# ------------------------------------------------------------------ #
#  VesselNetwork                                                       #
# ------------------------------------------------------------------ #

class VesselNetwork:
    """
    4-level bifurcating vascular tree with a mirrored return path.

    Parameters
    ----------
    root_radius : float
        Radius of the inlet/outlet segment (m).  Default 0.006 m (6 mm).
    root_v_mean : float
        Mean flow speed in the inlet segment (m/s).  Default 0.05 m/s.
    n_levels : int
        Number of bifurcation levels (4 → 16 terminal branches).
    trunk_length : float
        Length of the inlet segment (m).
    outlet_offset : float
        Lateral separation between inlet and outlet trunks (m).
    """

    name = "vessel_network"

    def __init__(
        self,
        root_radius:    float = 0.006,
        root_v_mean:    float = 0.05,
        n_levels:       int   = 4,
        trunk_length:   float = 0.04,
        outlet_offset:  float = 0.015,
        spread_x:       float = 1.0,
        spread_z:       float = 1.0,
        seg_length_scale: float = 0.75,
    ):
        self.root_radius   = float(root_radius)
        self.root_v_mean   = float(root_v_mean)
        self.n_levels      = int(n_levels)
        self.trunk_length  = float(trunk_length)
        self.outlet_offset = float(outlet_offset)

        # ---- afferent (forward) tree --------------------------------
        aff_start = np.array([0.0, -trunk_length / 2, 0.0])
        aff_end   = np.array([0.0,  trunk_length / 2, 0.0])

        self._afferent = _build_afferent_tree(
            root_start       = aff_start,
            root_end         = aff_end,
            root_radius      = root_radius,
            root_v_mean      = root_v_mean,
            n_levels         = n_levels,
            spread_x         = spread_x,
            spread_z         = spread_z,
            seg_length_scale = seg_length_scale,
        )

        # ---- efferent (return) tree ---------------------------------
        # Mirror the afferent tree: swap start/end (reverse flow),
        # negate v_mean, offset in X by outlet_offset, mark efferent.
        self._efferent = []
        for seg in self._afferent:
            eseg = VesselSegment(
                seg_id     = seg.seg_id + len(self._afferent),
                start      = seg.end   + np.array([outlet_offset, 0.0, 0.0]),
                end        = seg.start + np.array([outlet_offset, 0.0, 0.0]),
                radius     = seg.radius,
                v_mean     = -seg.v_mean,   # reverse flow
                level      = seg.level,
                is_efferent= True,
            )
            self._efferent.append(eseg)

        self.segments: List[VesselSegment] = self._afferent + self._efferent
        self.n_segments = len(self.segments)

        # ---- spatial index: maps grid cell -> list of segment ids ----
        # Speeds up find_segment from O(N_segs) to O(1) average case.
        self._build_spatial_index()

    # -------------------------------------------------------------- #
    #  Spatial index                                                   #
    # -------------------------------------------------------------- #

    def _build_spatial_index(self, grid_res: int = 20):
        """
        Divide the bounding box into a grid and record which segments
        overlap each cell.  Lookup then only checks the few segments
        in the same cell as the query point.
        """
        # Compute bounding box of all segment endpoints
        coords = np.array([pt for seg in self.segments
                           for pt in (seg.start, seg.end)])
        self._bbox_min = coords.min(axis=0) - 0.01
        self._bbox_max = coords.max(axis=0) + 0.01
        self._grid_res = grid_res

        span = self._bbox_max - self._bbox_min
        self._cell_size = span / grid_res

        # For each segment, mark every grid cell its bounding box touches
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
        idx = ((np.asarray(pos) - self._bbox_min) /
               (self._bbox_max - self._bbox_min) * self._grid_res)
        i, j, k = np.clip(idx.astype(int), 0, self._grid_res - 1)
        return int(i), int(j), int(k)

    def _candidates(self, pos: np.ndarray) -> list:
        """Return candidate segments near pos via spatial index."""
        key = self._world_to_cell(pos)
        cands = self._grid.get(key, None)
        if cands:
            return cands
        # Fallback: expand search one cell in each direction
        i, j, k = key
        result = []
        seen = set()
        for di in (-1, 0, 1):
            for dj in (-1, 0, 1):
                for dk in (-1, 0, 1):
                    nkey = (i+di, j+dj, k+dk)
                    for seg in self._grid.get(nkey, []):
                        if seg.seg_id not in seen:
                            seen.add(seg.seg_id)
                            result.append(seg)
        return result if result else self.segments  # final fallback

    # -------------------------------------------------------------- #
    #  Core spatial query                                              #
    # -------------------------------------------------------------- #

    def find_segment(
        self, pos: np.ndarray
    ) -> Tuple[Optional[VesselSegment], float]:
        """
        Return (best_segment, radial_distance) for pos.
        Uses spatial index for O(1) average-case performance.
        """
        pos = np.asarray(pos, dtype=float)
        best_seg  = None
        best_dist = np.inf

        for seg in self._candidates(pos):
            _, _, r, _ = seg.closest_point_and_radial(pos)
            if r < best_dist:
                best_dist = r
                best_seg  = seg

        # Guarantee we always return something
        if best_seg is None:
            for seg in self.segments:
                _, _, r, _ = seg.closest_point_and_radial(pos)
                if r < best_dist:
                    best_dist = r
                    best_seg  = seg

        return best_seg, best_dist

    def in_any_vessel(self, pos: np.ndarray) -> bool:
        """True if pos is inside at least one vessel segment."""
        for seg in self._candidates(pos):
            if seg.contains(pos):
                return True
        return False

    # -------------------------------------------------------------- #
    #  Flow interface (drop-in replacement for flow.py models)         #
    # -------------------------------------------------------------- #

    def velocity_at(self, pos: np.ndarray, time: float = 0.0) -> np.ndarray:
        """
        Return blood flow velocity at pos using spatial index.
        """
        pos = np.asarray(pos, dtype=float)
        best_seg  = None
        best_dist = np.inf

        for seg in self._candidates(pos):
            _, _, r, _ = seg.closest_point_and_radial(pos)
            if r < seg.radius and r < best_dist:
                best_dist = r
                best_seg  = seg

        if best_seg is None:
            return np.zeros(3)

        return best_seg.flow_velocity(pos, time)

    def params_dict(self) -> dict:
        return {
            "flow_model":        self.name,
            "n_levels":          self.n_levels,
            "n_segments":        self.n_segments,
            "root_radius_m":     self.root_radius,
            "root_v_mean_m_s":   self.root_v_mean,
            "trunk_length_m":    self.trunk_length,
            "outlet_offset_m":   self.outlet_offset,
        }

    # -------------------------------------------------------------- #
    #  Workspace / boundary interface (replaces Workspace3D.clip)      #
    # -------------------------------------------------------------- #

    @property
    def half(self) -> float:
        """
        Bounding half-size for any code that still references
        workspace.half (e.g. plotting).  Computed from segment extent.
        """
        coords = []
        for seg in self.segments:
            coords.append(seg.start)
            coords.append(seg.end)
        arr = np.array(coords)
        return float(np.max(np.abs(arr))) * 1.1  # 10 % margin

    def clip(self, pos: np.ndarray) -> np.ndarray:
        """
        Vessel-aware position clamping.

        If pos is inside any vessel segment, return it unchanged.
        Otherwise find the nearest segment and project pos back to
        the vessel wall (closest point on axis + radius in the
        radial direction).
        """
        pos = np.asarray(pos, dtype=float)

        if self.in_any_vessel(pos):
            return pos.copy()

        seg, _ = self.find_segment(pos)
        if seg is None:
            return pos.copy()

        closest, radial_vec, radial_dist, _ = seg.closest_point_and_radial(pos)

        if radial_dist < EPS:
            # pos is exactly on the axis — push radially in arbitrary direction
            perp = np.array([1.0, 0.0, 0.0])
            if abs(np.dot(perp, seg._axis)) > 0.9:
                perp = np.array([0.0, 1.0, 0.0])
            radial_hat = perp - np.dot(perp, seg._axis) * seg._axis
            radial_hat /= np.linalg.norm(radial_hat)
        else:
            radial_hat = radial_vec / radial_dist

        # Place just inside the wall (1 % inset to avoid fp edge cases)
        return closest + radial_hat * seg.radius * 0.99

    # -------------------------------------------------------------- #
    #  Helpers for simulation setup                                    #
    # -------------------------------------------------------------- #

    def inlet_start(self) -> np.ndarray:
        """Entry point of the afferent inlet segment."""
        return self._afferent[0].start.copy()

    def outlet_end(self) -> np.ndarray:
        """Exit point of the efferent outlet segment."""
        # efferent root (level 0) has reversed direction,
        # so its 'end' is the physical exit
        for seg in self._efferent:
            if seg.level == 0:
                return seg.end.copy()
        return self._efferent[0].end.copy()

    def random_point_in_vessel(self, rng=None) -> np.ndarray:
        """Sample a uniformly random point inside any vessel segment."""
        if rng is None:
            rng = np.random.default_rng()
        seg = rng.choice(self.segments)
        t   = rng.uniform(0.0, 1.0)
        r   = rng.uniform(0.0, seg.radius)
        phi = rng.uniform(0.0, 2.0 * PI)
        axis = seg._axis
        # build two perpendicular vectors to axis
        perp1 = np.array([1.0, 0.0, 0.0])
        if abs(np.dot(perp1, axis)) > 0.9:
            perp1 = np.array([0.0, 1.0, 0.0])
        perp1 -= np.dot(perp1, axis) * axis
        perp1 /= np.linalg.norm(perp1)
        perp2  = np.cross(axis, perp1)
        point_on_axis = seg.start + t * seg._length * axis
        return point_on_axis + r * (np.cos(phi) * perp1 + np.sin(phi) * perp2)

    def segment_summary(self) -> str:
        """Human-readable summary of all segments."""
        lines = [
            f"VesselNetwork: {self.n_segments} segments "
            f"({len(self._afferent)} afferent + {len(self._efferent)} efferent)",
            f"  Levels: {self.n_levels}  "
            f"Root radius: {self.root_radius*1000:.1f} mm  "
            f"Root v_mean: {self.root_v_mean*100:.1f} cm/s",
        ]
        for seg in self.segments:
            tag = "eff" if seg.is_efferent else "aff"
            lines.append(
                f"  [{tag} L{seg.level} id={seg.seg_id:3d}] "
                f"r={seg.radius*1000:.2f} mm  "
                f"v={seg.v_mean*100:.2f} cm/s  "
                f"len={seg._length*100:.1f} cm"
            )
        return "\n".join(lines)

    # -------------------------------------------------------------- #
    #  Graph pathfinding                                               #
    # -------------------------------------------------------------- #

    def _build_graph(self):
        """
        Build an adjacency graph of segments connected at shared
        endpoints.  Two segments are connected if their endpoints are
        within a small tolerance of each other.

        Returns dict: seg_id -> list of (neighbour_seg_id, junction_point)
        """
        tol = 1e-6
        graph = {seg.seg_id: [] for seg in self.segments}

        endpoints = []  # (seg_id, which_end, point)
        for seg in self.segments:
            endpoints.append((seg.seg_id, 'start', seg.start))
            endpoints.append((seg.seg_id, 'end',   seg.end))

        # O(N^2) but only runs once at startup — 62 segs = 3844 pairs, fine
        for i, (id_a, end_a, pt_a) in enumerate(endpoints):
            for id_b, end_b, pt_b in endpoints[i+1:]:
                if id_a == id_b:
                    continue
                if np.linalg.norm(pt_a - pt_b) < tol:
                    # shared junction point
                    junction = (pt_a + pt_b) / 2.0
                    graph[id_a].append((id_b, junction))
                    graph[id_b].append((id_a, junction))

        return graph

    def plan_waypoints(
        self,
        start_pos:  np.ndarray,
        target_pos: np.ndarray,
    ) -> List[np.ndarray]:
        """
        Plan a sequence of 3D waypoints through the vessel graph
        from start_pos to target_pos.

        Uses Dijkstra on the segment graph (edge weight = segment length).
        Returns a list of positions [wp1, wp2, ..., target_pos] that
        the MPC should follow in order.

        If start and target are in the same segment, returns [target_pos].
        """
        start_pos  = np.asarray(start_pos,  dtype=float)
        target_pos = np.asarray(target_pos, dtype=float)

        # Find which segments contain start and target
        start_seg,  _ = self.find_segment(start_pos)
        target_seg, _ = self.find_segment(target_pos)

        if start_seg is None or target_seg is None:
            return [target_pos]

        if start_seg.seg_id == target_seg.seg_id:
            return [target_pos]

        # Build graph (cached after first call)
        if not hasattr(self, '_graph'):
            self._graph = self._build_graph()

        graph = self._graph

        # Dijkstra from start_seg to target_seg
        import heapq
        dist     = {seg.seg_id: np.inf for seg in self.segments}
        prev     = {seg.seg_id: None   for seg in self.segments}
        junction = {seg.seg_id: None   for seg in self.segments}

        dist[start_seg.seg_id] = 0.0
        heap = [(0.0, start_seg.seg_id)]

        seg_by_id = {seg.seg_id: seg for seg in self.segments}

        while heap:
            d, uid = heapq.heappop(heap)
            if d > dist[uid]:
                continue
            if uid == target_seg.seg_id:
                break
            for (vid, jpt) in graph.get(uid, []):
                seg_v   = seg_by_id[vid]
                new_dist = d + seg_v._length
                if new_dist < dist[vid]:
                    dist[vid]     = new_dist
                    prev[vid]     = uid
                    junction[vid] = jpt
                    heapq.heappush(heap, (new_dist, vid))

        # Reconstruct path of segment ids
        if dist[target_seg.seg_id] == np.inf:
            # No path found — return direct target
            return [target_pos]

        path_ids = []
        cur = target_seg.seg_id
        while cur is not None:
            path_ids.append(cur)
            cur = prev[cur]
        path_ids.reverse()  # start_seg -> ... -> target_seg

        # Convert segment path to 3D waypoints at junctions
        waypoints = []
        for i, sid in enumerate(path_ids[1:], start=1):
            jpt = junction[sid]
            if jpt is not None:
                # Place waypoint just inside the junction on the centreline
                seg = seg_by_id[sid]
                wp  = seg.clip_to_interior(jpt)
                waypoints.append(wp)

        waypoints.append(target_pos)
        return waypoints

    def waypoint_navigator(
        self,
        start_pos:  np.ndarray,
        target_pos: np.ndarray,
        arrival_tol: float = 0.003,
    ):
        """
        Returns a callable next_waypoint(current_pos) that advances
        through the planned waypoint list as each one is reached.

        Usage in simulation loop:
            nav = network.waypoint_navigator(start, target)
            ...
            current_target = nav(particle.pos)
        """
        waypoints = self.plan_waypoints(start_pos, target_pos)
        state = {'idx': 0, 'waypoints': waypoints}

        def next_waypoint(current_pos: np.ndarray) -> np.ndarray:
            wps  = state['waypoints']
            idx  = state['idx']
            if idx >= len(wps):
                return wps[-1]
            wp = wps[idx]
            if np.linalg.norm(np.asarray(current_pos) - wp) < arrival_tol:
                state['idx'] = min(idx + 1, len(wps) - 1)
            return state['waypoints'][state['idx']]

        next_waypoint.waypoints     = waypoints
        next_waypoint.total_hops    = len(waypoints)
        return next_waypoint
