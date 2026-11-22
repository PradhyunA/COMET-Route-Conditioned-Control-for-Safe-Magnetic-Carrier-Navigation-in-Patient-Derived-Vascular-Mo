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
from dataclasses import dataclass, field
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


@dataclass(frozen=True)
class RoutePlanningOptions:
    """Physical constraints and costs used by network route planning."""

    wall_clearance_m: float = 0.0
    particle_radius_m: float = 0.0
    minimum_surface_clearance_m: float = 0.0
    magnetic_authority_m_s: float = float("inf")
    minimum_authority_reserve_m_s: float = float("-inf")
    clearance_cost_weight: float = 1.0
    flow_cost_weight: float = 2.0
    authority_cost_weight: float = 3.0
    blocked_segment_ids: frozenset = field(default_factory=frozenset)

    @property
    def required_center_clearance_m(self):
        return max(
            float(self.wall_clearance_m),
            float(self.particle_radius_m) + float(self.minimum_surface_clearance_m),
        )


@dataclass
class RoutePlan:
    feasible: bool
    reason: str
    segment_ids: List[int]
    waypoints: List[np.ndarray]
    route_length_m: float = float("inf")
    min_radius_m: float = 0.0
    min_center_clearance_m: float = 0.0
    min_authority_reserve_m_s: float = float("-inf")


class RealVesselNetwork:
    name = "real_vessel_network"
    #: Shortcut out-and-back spurs in planned routes (see
    #: remove_route_backtracks). Off by default to keep published routes.
    remove_backtracks = False

    def __init__(
        self,
        pkl_file="vessels/abdominal_0033_H_ABAO_AAA/real_vessel_segments.pkl",
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
        self._seg_by_id = {seg.seg_id: seg for seg in self.segments}
        self._endpoint_segments = self._build_endpoint_segments()
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

    def lumen_clearance(self, pos):
        """Largest (radius - radial distance) over the segments near pos."""
        best = -np.inf
        for seg in self._candidates(pos):
            _, _, r, _ = seg.closest_point_and_radial(pos)
            best = max(best, seg.radius - r)
        return float(best)

    def _chord_is_clear(self, a, b, required_clearance, sample_spacing=5e-4):
        n = max(2, int(np.ceil(np.linalg.norm(b - a) / sample_spacing)) + 1)
        return all(
            self.lumen_clearance(a + t * (b - a)) >= required_clearance - EPS
            for t in np.linspace(0.0, 1.0, n)
        )

    def remove_route_backtracks(self, points, required_clearance, window=40):
        """
        Shortcut out-and-back spurs in a route polyline through the lumen.

        Overlapping centerlines that join only at some nodes make the
        segment graph route run past a junction and double back. At each
        reversal (consecutive edges more than 120 deg apart) the chord that
        removes the most route length while keeping `required_clearance`
        inside the lumen replaces the spur. Unfixable reversals are kept.
        """
        pts = [np.asarray(p, dtype=float) for p in points]
        search_from = 1
        while True:
            reversal = None
            for v in range(search_from, len(pts) - 1):
                e_in, e_out = pts[v] - pts[v - 1], pts[v + 1] - pts[v]
                denom = np.linalg.norm(e_in) * np.linalg.norm(e_out)
                if denom > EPS and float(e_in @ e_out) / denom < -0.5:
                    reversal = v
                    break
            if reversal is None:
                return pts

            arr = np.asarray(pts)
            cum = np.concatenate([[0.0], np.cumsum(np.linalg.norm(np.diff(arr, axis=0), axis=1))])
            lo, hi = max(0, reversal - window), min(len(pts) - 1, reversal + window)
            pairs = [
                (cum[j] - cum[i] - float(np.linalg.norm(arr[j] - arr[i])), i, j)
                for i in range(lo, reversal) for j in range(reversal + 1, hi + 1)
            ]
            chord = next(
                ((i, j) for _, i, j in sorted(pairs, reverse=True)
                 if self._chord_is_clear(arr[i], arr[j], required_clearance)),
                None,
            )
            if chord is None:
                search_from = reversal + 1
                continue
            i, j = chord
            pts = pts[:i + 1] + pts[j:]
            search_from = max(1, i)

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

    def _build_endpoint_segments(self):
        endpoints = {}
        for seg in self.segments:
            for point in (seg.start, seg.end):
                endpoints.setdefault(self._node_key(point), []).append(seg.seg_id)
        return {key: sorted(set(ids)) for key, ids in endpoints.items()}

    def shared_endpoint(self, first_id, second_id):
        first = self._seg_by_id[int(first_id)]
        second = self._seg_by_id[int(second_id)]
        second_keys = {
            self._node_key(second.start): second.start,
            self._node_key(second.end): second.end,
        }
        for point in (first.start, first.end):
            if self._node_key(point) in second_keys:
                return np.asarray(point, dtype=float).copy()
        return None

    def endpoint_segment_ids(self, point):
        return list(self._endpoint_segments.get(self._node_key(point), []))

    def _build_graph(self):
        graph = {seg.seg_id: [] for seg in self.segments}
        for ids in self._endpoint_segments.values():
            unique_ids = sorted(set(ids))

            for a in unique_ids:
                for b in unique_ids:
                    if a == b:
                        continue

                    graph[a].append((b, self._seg_by_id[b].start))

        return graph

    def inlet_start(self):
        all_starts = np.array([s.start for s in self.segments])
        idx = np.argmin(all_starts[:, 2])
        return all_starts[idx].copy()

    def outlet_end(self):
        all_ends = np.array([s.end for s in self.segments])
        idx = np.argmax(all_ends[:, 2])
        return all_ends[idx].copy()

    def random_point_in_vessel(self, rng=None, min_radius=0.0):
        if rng is None:
            rng = np.random.default_rng()

        candidates = [seg for seg in self.segments if seg.radius >= min_radius]
        if not candidates:
            raise ValueError("No vessel segment satisfies min_radius")
        if min_radius <= 0.0:
            seg = rng.choice(candidates)
        else:
            weights = np.asarray(
                [
                    seg._length * max(seg.radius - min_radius, EPS) ** 2
                    for seg in candidates
                ],
                dtype=float,
            )
            weights /= np.sum(weights)
            seg = candidates[int(rng.choice(len(candidates), p=weights))]
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

    def _route_accessible_flow(self, seg, required):
        flow_speed = 2.0 * abs(seg.v_mean)
        if required > 0.0:
            ride_fraction = np.clip(
                (seg.radius - required) / max(seg.radius, EPS), 0.0, 1.0,
            )
            flow_speed *= 1.0 - ride_fraction ** 2
        return flow_speed * seg._axis

    def _route_edge_metrics(self, seg, direction, options):
        required = options.required_center_clearance_m
        if seg.radius + EPS < required:
            return None

        accessible_flow = self._route_accessible_flow(seg, required)
        adverse_flow = max(0.0, -float(np.dot(accessible_flow, direction)))
        authority = float(options.magnetic_authority_m_s)
        reserve = authority - adverse_flow
        if reserve + EPS < options.minimum_authority_reserve_m_s:
            return None

        clearance_headroom = max(seg.radius - required, EPS)
        clearance_penalty = required / clearance_headroom if required > 0.0 else 0.0
        flow_penalty = adverse_flow / max(authority, EPS) if np.isfinite(authority) else 0.0
        authority_penalty = max(0.0, -reserve) / max(authority, EPS) if np.isfinite(authority) else 0.0
        multiplier = 1.0 + (
            options.clearance_cost_weight * clearance_penalty
            + options.flow_cost_weight * flow_penalty
            + options.authority_cost_weight * authority_penalty
        )
        return seg._length * multiplier, adverse_flow, reserve

    def plan_route(self, start_pos, target_pos, planning_options=None):
        options = planning_options or RoutePlanningOptions()
        start_seg, _ = self.find_segment(start_pos)
        target_seg, _ = self.find_segment(target_pos)

        if start_seg is None or target_seg is None:
            return RoutePlan(False, "endpoint_outside_network", [], [])

        blocked = set(options.blocked_segment_ids)
        if start_seg.seg_id in blocked or target_seg.seg_id in blocked:
            return RoutePlan(False, "endpoint_segment_blocked", [], [])

        required = options.required_center_clearance_m
        if start_seg.radius + EPS < required or target_seg.radius + EPS < required:
            return RoutePlan(False, "endpoint_clearance", [], [])

        if start_seg.seg_id == target_seg.seg_id:
            length = float(np.linalg.norm(np.asarray(target_pos) - np.asarray(start_pos)))
            direction = np.asarray(target_pos, dtype=float) - np.asarray(start_pos, dtype=float)
            direction /= max(float(np.linalg.norm(direction)), EPS)
            metrics = self._route_edge_metrics(start_seg, direction, options)
            if metrics is None:
                return RoutePlan(False, "endpoint_authority", [], [])
            return RoutePlan(
                True, "ok", [start_seg.seg_id], [np.asarray(target_pos, dtype=float)],
                length, start_seg.radius, start_seg.radius,
                metrics[2],
            )

        start_pos = np.asarray(start_pos, dtype=float)
        target_pos = np.asarray(target_pos, dtype=float)
        dist = {}
        prev = {}
        heap = []

        for exit_point in (start_seg.start, start_seg.end):
            direction = np.asarray(exit_point) - start_pos
            partial_length = float(np.linalg.norm(direction))
            direction /= max(partial_length, EPS)
            metrics = self._route_edge_metrics(start_seg, direction, options)
            if metrics is None:
                continue
            state = (start_seg.seg_id, self._node_key(exit_point))
            cost = metrics[0] * partial_length / max(start_seg._length, EPS)
            dist[state] = cost
            prev[state] = None
            heapq.heappush(heap, (cost, state))

        best_goal_cost = np.inf
        best_goal_state = None

        while heap:
            d, state = heapq.heappop(heap)

            if d > dist.get(state, np.inf) or d >= best_goal_cost:
                continue

            uid, endpoint_key = state
            current_seg = self._seg_by_id[uid]
            endpoint = (
                current_seg.start
                if self._node_key(current_seg.start) == endpoint_key
                else current_seg.end
            )

            if target_seg.seg_id in self._endpoint_segments.get(endpoint_key, []):
                direction = target_pos - endpoint
                partial_length = float(np.linalg.norm(direction))
                direction /= max(partial_length, EPS)
                metrics = self._route_edge_metrics(target_seg, direction, options)
                if metrics is not None:
                    goal_cost = d + (
                        metrics[0] * partial_length / max(target_seg._length, EPS)
                    )
                    if goal_cost < best_goal_cost:
                        best_goal_cost = goal_cost
                        best_goal_state = state

            for vid in self._endpoint_segments.get(endpoint_key, []):
                if vid == uid or vid == target_seg.seg_id or vid in blocked:
                    continue
                seg_v = self._seg_by_id[vid]
                if self._node_key(seg_v.start) == endpoint_key:
                    exit_point = seg_v.end
                else:
                    exit_point = seg_v.start
                direction = np.asarray(exit_point) - np.asarray(endpoint)
                direction /= max(float(np.linalg.norm(direction)), EPS)
                metrics = self._route_edge_metrics(seg_v, direction, options)
                if metrics is None:
                    continue
                nd = d + metrics[0]
                next_state = (vid, self._node_key(exit_point))

                if nd < dist.get(next_state, np.inf):
                    dist[next_state] = nd
                    prev[next_state] = state
                    heapq.heappush(heap, (nd, next_state))

        if best_goal_state is None:
            return RoutePlan(False, "no_feasible_route", [], [])

        path_states = []
        cur = best_goal_state

        while cur is not None:
            path_states.append(cur)
            cur = prev[cur]

        path_states.reverse()
        path_ids = [state[0] for state in path_states]
        if path_ids[-1] != target_seg.seg_id:
            path_ids.append(target_seg.seg_id)

        waypoints = []
        for sid, endpoint_key in path_states:
            seg = self._seg_by_id[sid]
            point = seg.start if self._node_key(seg.start) == endpoint_key else seg.end
            waypoints.append(np.asarray(point, dtype=float).copy())
        waypoints.append(target_pos)
        if self.remove_backtracks:
            waypoints = self.remove_route_backtracks(
                [start_pos, *waypoints], options.required_center_clearance_m,
            )[1:]

        radii = [self._seg_by_id[sid].radius for sid in path_ids]
        reserves = []
        route_length = 0.0
        route_points = [np.asarray(start_pos, dtype=float), *waypoints]
        for first, second in zip(route_points, route_points[1:]):
            route_length += float(np.linalg.norm(second - first))
        for index, sid in enumerate(path_ids):
            seg = self._seg_by_id[sid]
            if index + 1 < len(path_ids):
                exit_point = self.shared_endpoint(sid, path_ids[index + 1])
            else:
                exit_point = target_pos
            entry_point = start_pos if index == 0 else self.shared_endpoint(path_ids[index - 1], sid)
            direction = np.asarray(exit_point) - np.asarray(entry_point)
            direction /= max(float(np.linalg.norm(direction)), EPS)
            metrics = self._route_edge_metrics(seg, direction, options)
            reserves.append(metrics[2] if metrics is not None else float("-inf"))

        return RoutePlan(
            True, "ok", path_ids, waypoints, route_length,
            min(radii), min(radii), min(reserves),
        )

    def plan_waypoints(self, start_pos, target_pos, planning_options=None) -> List[np.ndarray]:
        plan = self.plan_route(start_pos, target_pos, planning_options)
        if plan.feasible:
            return plan.waypoints
        return [np.asarray(target_pos, dtype=float)]

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
        planning_options=None,
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

        options = planning_options or RoutePlanningOptions()
        route_plan = self.plan_route(start_pos, target_pos, options)
        raw_waypoints = (
            route_plan.waypoints
            if route_plan.feasible
            else [np.asarray(target_pos, dtype=float)]
        )

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

        def replan(current_pos, blocked_segment_ids=()):
            replanning_options = RoutePlanningOptions(
                wall_clearance_m=options.wall_clearance_m,
                particle_radius_m=options.particle_radius_m,
                minimum_surface_clearance_m=options.minimum_surface_clearance_m,
                magnetic_authority_m_s=options.magnetic_authority_m_s,
                minimum_authority_reserve_m_s=options.minimum_authority_reserve_m_s,
                clearance_cost_weight=options.clearance_cost_weight,
                flow_cost_weight=options.flow_cost_weight,
                authority_cost_weight=options.authority_cost_weight,
                blocked_segment_ids=frozenset(blocked_segment_ids),
            )
            plan = self.plan_route(current_pos, target_pos, replanning_options)
            if not plan.feasible:
                return plan
            replanned_path = self._sparsify_waypoints(
                [np.asarray(current_pos, dtype=float), *plan.waypoints],
                min_spacing=waypoint_spacing,
            )
            state["waypoints"] = replanned_path[1:] or [target_pos]
            state["idx"] = 0
            state["best_idx"] = 0
            state["raw_count"] = len(plan.waypoints)
            next_waypoint.waypoints = state["waypoints"]
            next_waypoint.route_plan = plan
            next_waypoint.route_segment_ids = list(plan.segment_ids)
            next_waypoint.total_hops = len(state["waypoints"])
            next_waypoint.current_idx = 0
            next_waypoint.best_idx = 0
            return plan

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
        next_waypoint.route_plan = route_plan
        next_waypoint.route_segment_ids = list(route_plan.segment_ids)
        next_waypoint.planning_options = options
        next_waypoint.replan = replan

        return next_waypoint
