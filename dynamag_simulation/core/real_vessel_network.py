"""
Real vessel network: graph, flow, waypoint navigation.

Navigator fix (v2):
    - Uses a true monotonic best-index tracker inside next_waypoint.
    - forward_search_points window searches from best_idx forward, not just
      from current_idx. This prevents navigator from lagging behind the
      particle when the particle skips a waypoint.
    - Arrival tolerance applied relative to current_idx, but best_idx can
      advance independently via closest-point search.
"""

import pickle
import heapq
import numpy as np
from dataclasses import dataclass
from typing import Optional, Tuple, List

from .constants import DEFAULT_TARGET_TOLERANCE, EPS, PI


@dataclass
class RealVesselSegment:
    seg_id: int
    start: np.ndarray
    end: np.ndarray
    radius: float
    v_mean: float = 0.02
    level: int = 0

    def __post_init__(self):
        self.start = np.asarray(self.start, dtype=float)
        self.end = np.asarray(self.end, dtype=float)
        diff = self.end - self.start
        self._length = float(np.linalg.norm(diff))
        self._axis = (
            diff / self._length
            if self._length > EPS
            else np.array([0.0, 1.0, 0.0])
        )

    def closest_point_and_radial(self, pos):
        pos = np.asarray(pos, dtype=float)
        d = pos - self.start
        t = float(np.dot(d, self._axis)) / (self._length + EPS)
        t = float(np.clip(t, 0.0, 1.0))
        closest = self.start + t * self._length * self._axis
        radial_vec = pos - closest
        radial_dist = float(np.linalg.norm(radial_vec))
        return closest, radial_vec, radial_dist, t

    def contains(self, pos, tol=1e-9):
        _, _, r, _ = self.closest_point_and_radial(pos)
        return r <= self.radius + tol

    def flow_velocity(self, pos, time=0.0):
        _, _, r, _ = self.closest_point_and_radial(pos)

        if r >= self.radius:
            return np.zeros(3)

        v_max = 2.0 * abs(self.v_mean)
        profile = 1.0 - (r / self.radius) ** 2
        pulsation = 1.0 + 0.15 * np.sin(2.0 * PI * time)

        return v_max * profile * pulsation * self._axis

    def clip_to_interior(self, pos, inset=0.01):
        pos = np.asarray(pos, dtype=float)
        d = pos - self.start
        t = float(np.dot(d, self._axis)) / (self._length + EPS)
        t = float(np.clip(t, inset, 1.0 - inset))
        return self.start + t * self._length * self._axis


class RealVesselNetwork:
    name = "real_vessel_network"

    def __init__(
        self,
        pkl_file="real_vessel_segments.pkl",
        connection_tol=1e-6,
        flow_scale=1.0,
    ):
        import os

        if not os.path.exists(pkl_file):
            repo_root = os.path.dirname(
                os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
            )

            candidates = [
                os.path.join("data", pkl_file),
                os.path.join(repo_root, "data", pkl_file),
            ]

            for candidate in candidates:
                if os.path.exists(candidate):
                    pkl_file = candidate
                    break

        with open(pkl_file, "rb") as f:
            raw_segments = pickle.load(f)

        self.source_file = pkl_file
        self.flow_scale = float(flow_scale)

        self.segments = []

        radii = []
        for s in raw_segments:
            radii.append(float(s["radius"]))

        r_ref = max(radii) if radii else 1.0
        v_mean_ref = 0.02

        for s in raw_segments:
            r = float(s["radius"])
            raw_v = float(s.get("v_mean", v_mean_ref))
            scaled_v = raw_v * (r / r_ref) * self.flow_scale

            self.segments.append(
                RealVesselSegment(
                    seg_id=int(s["seg_id"]),
                    start=np.asarray(s["start"], dtype=float),
                    end=np.asarray(s["end"], dtype=float),
                    radius=r,
                    v_mean=scaled_v,
                    level=int(s.get("level", 0)),
                )
            )

        self.n_segments = len(self.segments)
        self.connection_tol = float(connection_tol)

        self._build_spatial_index()
        self._graph = self._build_graph()

    @property
    def half(self):
        pts = []

        for s in self.segments:
            pts.append(s.start)
            pts.append(s.end)

        arr = np.array(pts)

        return float(np.max(np.abs(arr))) * 1.1

    def bounds(self, include_radius: bool = False):
        starts = np.array([s.start for s in self.segments], dtype=float)
        ends = np.array([s.end for s in self.segments], dtype=float)

        if not include_radius:
            pts = np.vstack([starts, ends])
            return pts.min(axis=0), pts.max(axis=0)

        radii = np.array([s.radius for s in self.segments], dtype=float)

        lo = np.minimum(starts, ends) - radii[:, None]
        hi = np.maximum(starts, ends) + radii[:, None]

        return lo.min(axis=0), hi.max(axis=0)

    def _build_spatial_index(self, grid_res=40):
        coords = np.array(
            [pt for seg in self.segments for pt in (seg.start, seg.end)]
        )

        self._bbox_min = coords.min(axis=0) - 0.01
        self._bbox_max = coords.max(axis=0) + 0.01
        self._grid_res = int(grid_res)
        self._grid = {}

        for seg in self.segments:
            lo = np.minimum(seg.start, seg.end) - seg.radius
            hi = np.maximum(seg.start, seg.end) + seg.radius

            i0, j0, k0 = self._world_to_cell(lo)
            i1, j1, k1 = self._world_to_cell(hi)

            for i in range(i0, i1 + 1):
                for j in range(j0, j1 + 1):
                    for k in range(k0, k1 + 1):
                        self._grid.setdefault((i, j, k), []).append(seg)

    def _world_to_cell(self, pos):
        pos = np.asarray(pos, dtype=float)

        denom = self._bbox_max - self._bbox_min
        denom = np.where(denom < EPS, EPS, denom)

        idx = (pos - self._bbox_min) / denom * self._grid_res
        i, j, k = np.clip(idx.astype(int), 0, self._grid_res - 1)

        return int(i), int(j), int(k)

    def _candidates(self, pos):
        key = self._world_to_cell(pos)
        cands = self._grid.get(key)

        if cands:
            return cands

        return self.segments

    def find_segment(self, pos) -> Tuple[Optional[RealVesselSegment], float]:
        best_seg = None
        best_dist = np.inf

        for seg in self._candidates(pos):
            _, _, r, _ = seg.closest_point_and_radial(pos)

            if r < best_dist:
                best_dist = r
                best_seg = seg

        return best_seg, float(best_dist)

    def in_any_vessel(self, pos):
        for seg in self._candidates(pos):
            if seg.contains(pos):
                return True

        return False

    def clip(self, pos):
        pos = np.asarray(pos, dtype=float)

        if self.in_any_vessel(pos):
            return pos.copy()

        seg, _ = self.find_segment(pos)

        if seg is None:
            return pos.copy()

        closest, radial_vec, radial_dist, _ = seg.closest_point_and_radial(pos)

        if radial_dist < EPS:
            radial_hat = np.array([1.0, 0.0, 0.0])
        else:
            radial_hat = radial_vec / radial_dist

        return closest + radial_hat * seg.radius * 0.98

    def velocity_at(self, pos, time=0.0):
        seg, dist = self.find_segment(pos)

        if seg is None or dist > seg.radius:
            return np.zeros(3)

        return seg.flow_velocity(pos, time)

    def _node_key(self, p):
        return tuple(np.round(np.asarray(p) / self.connection_tol).astype(int))

    def _build_graph(self):
        graph = {seg.seg_id: [] for seg in self.segments}
        endpoints = {}

        for seg in self.segments:
            for p in (seg.start, seg.end):
                key = self._node_key(p)
                endpoints.setdefault(key, []).append(seg.seg_id)

        seg_by_id = {s.seg_id: s for s in self.segments}

        for ids in endpoints.values():
            unique_ids = sorted(set(ids))

            for a in unique_ids:
                for b in unique_ids:
                    if a == b:
                        continue

                    graph[a].append((b, seg_by_id[b].start))

        return graph

    def inlet_start(self):
        all_starts = np.array([s.start for s in self.segments])
        idx = np.argmin(all_starts[:, 2])
        return all_starts[idx].copy()

    def outlet_end(self):
        all_ends = np.array([s.end for s in self.segments])
        idx = np.argmax(all_ends[:, 2])
        return all_ends[idx].copy()

    def random_point_in_vessel(self, rng=None):
        if rng is None:
            rng = np.random.default_rng()

        seg = rng.choice(self.segments)
        t = rng.uniform(0.05, 0.95)

        return seg.start + t * (seg.end - seg.start)

    def params_dict(self):
        return {
            "flow_model": self.name,
            "source_file": self.source_file,
            "n_segments": self.n_segments,
            "flow_scale": self.flow_scale,
        }

    def segment_summary(self):
        v_means = [seg.v_mean * 1000 for seg in self.segments]
        radii = [seg.radius * 1000 for seg in self.segments]

        return (
            f"RealVesselNetwork: {self.n_segments} segments, "
            f"flow_scale={self.flow_scale:.2f}, "
            f"v_mean={min(v_means):.1f}–{max(v_means):.1f} mm/s, "
            f"radius={min(radii):.2f}–{max(radii):.2f} mm"
        )

    def plan_waypoints(self, start_pos, target_pos) -> List[np.ndarray]:
        start_seg, _ = self.find_segment(start_pos)
        target_seg, _ = self.find_segment(target_pos)

        if start_seg is None or target_seg is None:
            return [np.asarray(target_pos, dtype=float)]

        if start_seg.seg_id == target_seg.seg_id:
            return [np.asarray(target_pos, dtype=float)]

        dist = {seg.seg_id: np.inf for seg in self.segments}
        prev = {seg.seg_id: None for seg in self.segments}

        dist[start_seg.seg_id] = 0.0
        heap = [(0.0, start_seg.seg_id)]

        seg_by_id = {seg.seg_id: seg for seg in self.segments}

        while heap:
            d, uid = heapq.heappop(heap)

            if d > dist[uid]:
                continue

            if uid == target_seg.seg_id:
                break

            for vid, _ in self._graph.get(uid, []):
                seg_v = seg_by_id[vid]
                nd = d + seg_v._length

                if nd < dist[vid]:
                    dist[vid] = nd
                    prev[vid] = uid
                    heapq.heappush(heap, (nd, vid))

        if dist[target_seg.seg_id] == np.inf:
            return [np.asarray(target_pos, dtype=float)]

        path_ids = []
        cur = target_seg.seg_id

        while cur is not None:
            path_ids.append(cur)
            cur = prev[cur]

        path_ids.reverse()

        waypoints = []

        for sid in path_ids[1:]:
            seg = seg_by_id[sid]
            waypoints.append(seg.start.copy())

        waypoints.append(np.asarray(target_pos, dtype=float))

        return waypoints

    def _sparsify_waypoints(self, points, min_spacing=0.0015):
        if points is None or len(points) == 0:
            return []

        if len(points) == 1:
            return [np.asarray(points[0], dtype=float)]

        sparse = [np.asarray(points[0], dtype=float)]
        last_kept = sparse[0]

        for p in points[1:]:
            p = np.asarray(p, dtype=float)

            if np.linalg.norm(p - last_kept) >= min_spacing:
                sparse.append(p)
                last_kept = p

        final_point = np.asarray(points[-1], dtype=float)

        if np.linalg.norm(final_point - sparse[-1]) > EPS:
            sparse.append(final_point)

        return sparse

    def waypoint_navigator(
        self,
        start_pos,
        target_pos,
        arrival_tol=DEFAULT_TARGET_TOLERANCE,
        waypoint_spacing=0.0015,
        lookahead=3,
        forward_search_points=50,
    ):
        """
        Build a waypoint navigator for MPC — v2.

        Key fix: the navigator maintains its own monotonic best_idx.
        Progress is advanced by searching forward_search_points ahead of
        best_idx, not just current_idx. This prevents the navigator from
        lagging when the particle physically passes a waypoint without
        triggering arrival_tol.

        lookahead: how many waypoints ahead of current_idx to return as target.
        forward_search_points: window for closest-point search from best_idx.
        """
        start_pos = np.asarray(start_pos, dtype=float)
        target_pos = np.asarray(target_pos, dtype=float)

        raw_waypoints = self.plan_waypoints(start_pos, target_pos)

        full_path = [
            start_pos,
            *[np.asarray(wp, dtype=float) for wp in raw_waypoints],
        ]

        sparse_path = self._sparsify_waypoints(
            full_path, min_spacing=waypoint_spacing,
        )

        if len(sparse_path) > 1:
            active_waypoints = sparse_path[1:]
        else:
            active_waypoints = [target_pos]

        # Navigator state: use a dict so it's mutable inside closure
        state = {
            "idx": 0,          # current waypoint index (arrival-tol based)
            "best_idx": 0,     # monotonic best (closest-point search based)
            "waypoints": active_waypoints,
            "raw_count": len(raw_waypoints),
        }

        def next_waypoint(current_pos):
            current_pos = np.asarray(current_pos, dtype=float)

            waypoints = state["waypoints"]
            idx = int(state["idx"])
            best_idx = int(state["best_idx"])
            n = len(waypoints)

            if n == 0:
                next_waypoint.current_idx = 0
                next_waypoint.lookahead_idx = 0
                return target_pos

            # ----------------------------------------------------------
            # Step 1: Update best_idx via forward closest-point search.
            # This is monotonic — best_idx never retreats.
            # Search from current best_idx forward in a window.
            # ----------------------------------------------------------
            search_start = best_idx
            search_end = min(n, search_start + max(1, int(forward_search_points)))

            window = np.asarray(waypoints[search_start:search_end], dtype=float)

            if len(window) > 0:
                dists = np.linalg.norm(window - current_pos, axis=1)
                local_best = int(np.argmin(dists))
                candidate_best = search_start + local_best

                if candidate_best > best_idx:
                    best_idx = candidate_best

            state["best_idx"] = best_idx

            # ----------------------------------------------------------
            # Step 2: Advance current idx by arrival tolerance.
            # Also advance to catch up with best_idx if it's ahead.
            # ----------------------------------------------------------
            idx = max(idx, best_idx)

            while idx < n - 1:
                dist_to_current_wp = float(
                    np.linalg.norm(current_pos - waypoints[idx])
                )

                if dist_to_current_wp < arrival_tol:
                    idx += 1
                else:
                    break

            state["idx"] = idx

            # ----------------------------------------------------------
            # Step 3: Choose lookahead target
            # ----------------------------------------------------------
            look_idx = min(idx + lookahead, n - 1)

            next_waypoint.current_idx = idx
            next_waypoint.best_idx = best_idx
            next_waypoint.lookahead_idx = look_idx
            next_waypoint.current_wp = waypoints[idx]
            next_waypoint.lookahead_wp = waypoints[look_idx]
            next_waypoint.current_wp_dist = float(
                np.linalg.norm(current_pos - waypoints[idx])
            )

            return waypoints[look_idx]

        next_waypoint.waypoints = active_waypoints
        next_waypoint.raw_waypoint_count = len(raw_waypoints)
        next_waypoint.total_hops = len(active_waypoints)
        next_waypoint.current_idx = 0
        next_waypoint.best_idx = 0
        next_waypoint.lookahead_idx = min(lookahead, len(active_waypoints) - 1)
        next_waypoint.current_wp = active_waypoints[0]
        next_waypoint.lookahead_wp = active_waypoints[next_waypoint.lookahead_idx]
        next_waypoint.current_wp_dist = float("inf")
        next_waypoint.waypoint_spacing = waypoint_spacing
        next_waypoint.lookahead = lookahead
        next_waypoint.forward_search_points = forward_search_points
        next_waypoint.arrival_tol = arrival_tol

        return next_waypoint
