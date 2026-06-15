"""
Authority-Conditioned Branch Commitment controller — v2.

Key fixes over v1:
    1. STALL_ESCAPE target uses best navigator progress index, not current_idx.
       Lookahead is small (12 waypoints) so the escape target stays near the
       particle's actual position in route space.

    2. Recovery target is route-conditioned. It finds the nearest waypoint on
       the planned path and steps a small amount downstream along the route
       tangent, preventing lateral/backward pull in curved vessels.

    3. State machine uses strict priority ordering with early-return semantics.
       Hard wall danger beats everything. STALL_ESCAPE is blocked when wall is
       unsafe. Recovery is bounded and cannot trap the controller.

    4. Navigator progress is tracked via a separate best-index counter inside
       the ACBC step, independent of the navigator's own current_idx.

States:
    FREE_NAVIGATION
    VIABILITY_ASSESSMENT
    BRANCH_COMMITMENT
    RECOVERY_MODE
    STALL_ESCAPE
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import Optional, List, Dict, Tuple

import numpy as np

from .constants import EPS
from .controllers import _simulate_one_step, continuous_mpc_controller


ACBC_FREE = "FREE_NAVIGATION"
ACBC_ASSESS = "VIABILITY_ASSESSMENT"
ACBC_COMMIT = "BRANCH_COMMITMENT"
ACBC_RECOVER = "RECOVERY_MODE"
ACBC_ESCAPE = "STALL_ESCAPE"

logger = logging.getLogger(__name__)


@dataclass
class BranchCandidate:
    branch_seg_id: int
    bifurcation_point: np.ndarray
    direction: np.ndarray
    target_point: np.ndarray

    flow_projection_m_s: float
    max_magnetic_projection_m_s: float
    authority_margin: float

    wall_safety_margin_m: float
    wall_safety_fraction: float

    route_alignment: float
    recoverability: float
    score: float
    capturable: bool


@dataclass
class ACBCStepInfo:
    state: str
    previous_state: str
    transition: bool

    control_target: np.ndarray
    nominal_target: np.ndarray

    near_bifurcation: bool
    bifurcation_distance_m: float
    bifurcation_point: Optional[np.ndarray]

    selected_branch_seg_id: Optional[int]
    selected_authority_margin: float
    selected_score: float
    target_branch_capturable: bool

    recovery_target: Optional[np.ndarray]
    num_candidates: int

    # Extended diagnostics
    stall_escape_target: Optional[np.ndarray]
    best_route_idx: int
    stagnant_steps: int
    wall_clearance_m: float
    wall_fraction: float


class ACBCController:
    """
    Authority-Conditioned Branch Commitment — v2.

    Route-conditioned hybrid supervisor over continuous MPC current allocator.
    """

    def __init__(
        self,
        network,
        magnet_array,
        workspace,
        max_current: float = 8.0,
        branch_detection_radius: Optional[float] = None,
        branch_lookahead: Optional[float] = None,
        route_corridor_radius: Optional[float] = None,
        authority_threshold: float = 1.02,
        commit_enter_authority: float = 1.05,
        commit_exit_authority: float = 0.80,
        recovery_threshold: float = 0.65,
        wall_clearance_threshold_m: float = 0.0007,
        min_wall_safety_fraction: float = 0.015,
        absolute_wall_clearance_threshold: float = 0.00045,
        surface_clearance_target_m: float = 0.0005,
        recovery_enter_wall_clearance: float = 0.00100,
        recovery_exit_wall_clearance: float = 0.00150,
        near_wall_edge_multiplier: float = 6.0,
        min_commitment_steps: int = 10,
        max_commitment_steps: int = 90,
        recovery_center_fraction: float = 0.55,
        recovery_route_fraction: float = 0.10,
        recovery_route_tangent_steps: int = 3,
        stall_window_steps: int = 15,
        stall_escape_lookahead: int = 12,
        stall_escape_release_hops: int = 3,
        stall_escape_max_steps: int = 120,
        stall_escape_min_wall_clearance: float = 0.00090,
        stall_escape_min_wall_fraction_multiplier: float = 1.00,
        recovery_escape_lockout_steps: int = 0,
        min_recovery_steps: int = 8,
        max_recovery_steps: int = 40,
        recovery_release_clearance: float = 0.00090,
        transition_cooldown_steps: int = 4,
        enable_wall_safety_current_filter: bool = True,
        enable_branch_commitment: bool = True,
        enable_authority_assessment: bool = True,
        enable_recovery: bool = True,
        enable_stall_escape: bool = True,
    ):
        self.network = network
        self.magnet_array = magnet_array
        self.workspace = workspace
        self.max_current = float(max_current)

        bounds_min, bounds_max = self.network.bounds(include_radius=True)
        span = bounds_max - bounds_min
        max_span = float(np.max(span))

        self.branch_detection_radius = (
            float(branch_detection_radius)
            if branch_detection_radius is not None
            else 0.08 * max_span
        )

        self.branch_lookahead = (
            float(branch_lookahead)
            if branch_lookahead is not None
            else 0.05 * max_span
        )

        self.route_corridor_radius = (
            float(route_corridor_radius)
            if route_corridor_radius is not None
            else max(0.005, 0.25 * self.branch_detection_radius)
        )

        self.authority_threshold = float(authority_threshold)
        self.commit_enter_authority = float(commit_enter_authority)
        self.commit_exit_authority = float(commit_exit_authority)
        self.recovery_threshold = float(recovery_threshold)
        self.wall_clearance_threshold_m = float(
            max(0.0, wall_clearance_threshold_m)
        )
        self.min_wall_safety_fraction = float(min_wall_safety_fraction)
        self.absolute_wall_clearance_threshold = float(
            absolute_wall_clearance_threshold
        )
        self.surface_clearance_target_m = float(
            max(0.0, surface_clearance_target_m)
        )
        self.recovery_enter_wall_clearance = float(recovery_enter_wall_clearance)
        self.recovery_exit_wall_clearance = float(recovery_exit_wall_clearance)
        self.near_wall_edge_multiplier = float(max(1.0, near_wall_edge_multiplier))

        self.recovery_center_fraction = float(
            np.clip(recovery_center_fraction, 0.05, 1.0)
        )
        self.recovery_route_fraction = float(
            np.clip(recovery_route_fraction, 0.0, 0.5)
        )
        self.recovery_route_tangent_steps = int(
            max(1, recovery_route_tangent_steps)
        )

        self.stall_window_steps = int(max(5, stall_window_steps))
        self.stall_escape_lookahead = int(max(2, stall_escape_lookahead))
        self.stall_escape_release_hops = int(max(1, stall_escape_release_hops))
        self.stall_escape_max_steps = int(max(10, stall_escape_max_steps))

        self.stall_escape_min_wall_clearance = float(
            max(0.0, stall_escape_min_wall_clearance)
        )
        self.stall_escape_min_wall_fraction_multiplier = float(
            max(1.0, stall_escape_min_wall_fraction_multiplier)
        )

        self.recovery_escape_lockout_steps = int(
            max(0, recovery_escape_lockout_steps)
        )
        self.last_recovery_step: Optional[int] = None

        self.min_recovery_steps = int(max(0, min_recovery_steps))
        self.max_recovery_steps = int(max(10, max_recovery_steps))
        self.recovery_release_clearance = float(recovery_release_clearance)
        self.transition_cooldown_steps = int(max(0, transition_cooldown_steps))
        self.enable_wall_safety_current_filter = bool(
            enable_wall_safety_current_filter
        )
        self.enable_branch_commitment = bool(enable_branch_commitment)
        self.enable_authority_assessment = bool(enable_authority_assessment)
        self.enable_recovery = bool(enable_recovery)
        self.enable_stall_escape = bool(enable_stall_escape)

        # Route progress tracking (internal, monotonic)
        # best_route_idx tracks the highest waypoint index the particle has
        # been closest to. This is separate from the navigator's current_idx
        # so ACBC can detect stalls independently.
        self.best_route_idx = -1
        self.stagnant_steps = 0

        self.escape_start_step: Optional[int] = None
        self.escape_start_route_idx: Optional[int] = None

        self.state = ACBC_FREE
        self.prev_state = ACBC_FREE

        self.selected_branch: Optional[BranchCandidate] = None
        self.selected_branch_id: Optional[int] = None
        self.last_bifurcation_point: Optional[np.ndarray] = None

        self.commit_start_step: Optional[int] = None
        self.commit_target_idx: Optional[int] = None
        self.min_commitment_steps = int(max(0, min_commitment_steps))
        self.max_commitment_steps = int(max_commitment_steps)
        self.commit_target_reach_radius = max(
            0.004,
            0.35 * self.branch_lookahead,
        )
        self.last_transition_step: Optional[int] = None

        self.recovery_count = 0
        self.commitment_count = 0
        self.transition_count = 0
        self.stall_escape_count = 0
        self.step_count = 0

        self._bifurcation_nodes = self._build_bifurcation_nodes()

    # ------------------------------------------------------------------
    # Bifurcation / route utilities
    # ------------------------------------------------------------------

    def _build_bifurcation_nodes(self) -> List[Tuple[np.ndarray, List[int]]]:
        endpoint_map: Dict[Tuple[int, int, int], List[int]] = {}

        for seg in self.network.segments:
            for p in (seg.start, seg.end):
                key = self.network._node_key(p)
                endpoint_map.setdefault(key, []).append(seg.seg_id)

        seg_by_id = {s.seg_id: s for s in self.network.segments}
        nodes = []

        for ids in endpoint_map.values():
            unique_ids = sorted(set(ids))

            if len(unique_ids) >= 3:
                pts = []

                for sid in unique_ids:
                    seg = seg_by_id[sid]
                    pts.append(seg.start)
                    pts.append(seg.end)

                node_point = np.mean(np.asarray(pts), axis=0)
                nodes.append((node_point, unique_ids))

        return nodes

    def _nearest_route_index(self, pos, navigator) -> int:
        """Find the closest waypoint index to pos (no monotonicity enforced)."""
        waypoints = getattr(navigator, "waypoints", [])

        if waypoints is None or len(waypoints) == 0:
            return 0

        pts = np.asarray(waypoints, dtype=float)
        dists = np.linalg.norm(pts - np.asarray(pos, dtype=float), axis=1)

        return int(np.argmin(dists))

    def _navigator_route_index(self, pos, navigator) -> int:
        """Get navigator's current waypoint index."""
        idx = getattr(navigator, "current_idx", None)

        if idx is not None:
            try:
                return int(idx)
            except Exception:
                pass

        return self._nearest_route_index(pos, navigator)

    def _update_best_route_idx(self, pos, navigator) -> int:
        """
        Update best_route_idx by finding the closest waypoint in a forward
        search window from best_route_idx. This is monotonic — it never retreats.

        Returns the updated best_route_idx.
        """
        waypoints = getattr(navigator, "waypoints", [])

        if waypoints is None or len(waypoints) == 0:
            return self.best_route_idx

        pos = np.asarray(pos, dtype=float)
        n = len(waypoints)

        # Search from current best index forward in a window.
        # Also include a small backward window in case navigator is ahead.
        search_start = max(0, self.best_route_idx)
        search_end = min(n, search_start + 50)

        pts = np.asarray(waypoints[search_start:search_end], dtype=float)

        if len(pts) == 0:
            return self.best_route_idx

        dists = np.linalg.norm(pts - pos, axis=1)
        local_best = int(np.argmin(dists))
        candidate = search_start + local_best

        # Monotonic: only advance
        if candidate > self.best_route_idx:
            self.best_route_idx = candidate
            self.stagnant_steps = 0
        else:
            self.stagnant_steps += 1

        return self.best_route_idx

    def _update_stall_status(self, pos, navigator) -> bool:
        """
        Returns True if route progress has stalled.

        Uses best_route_idx (monotonic) rather than navigator's current_idx.
        """
        self._update_best_route_idx(pos, navigator)

        total_hops = int(getattr(navigator, "total_hops", 0) or 0)
        has_route_ahead = True

        if total_hops > 0:
            has_route_ahead = self.best_route_idx < total_hops - 3

        return bool(
            self.stagnant_steps >= self.stall_window_steps
            and has_route_ahead
        )

    def _stall_escape_target(self, particle_pos, navigator):
        """
        Choose a stall escape target that is just ahead of where the particle
        actually is in route space (based on best_route_idx, not current_idx).

        Uses a small lookahead so the particle moves to a nearby reachable
        point rather than jumping 90 waypoints ahead.
        """
        waypoints = getattr(navigator, "waypoints", None)

        if waypoints is None or len(waypoints) == 0:
            return np.asarray(particle_pos, dtype=float)

        # Base on best_route_idx (monotonic), not current_idx (can lag)
        base_idx = max(0, self.best_route_idx)
        target_idx = min(
            len(waypoints) - 1,
            base_idx + self.stall_escape_lookahead,
        )

        return self.workspace.clip(np.asarray(waypoints[target_idx], dtype=float))

    def _route_conditioned_recovery_target(self, particle_pos, navigator):
        """
        Recovery target that pulls toward vessel centerline while staying
        downstream along the planned route.

        Steps:
        1. Find nearest vessel centerline point (recentering component).
        2. Find nearest waypoint on planned route.
        3. Compute downstream route tangent from waypoints.
        4. Target = partial_step_to_center + downstream_tangent_step.

        This prevents recovery from pulling the particle backward or sideways
        in curved vessel geometries.
        """
        particle_pos = np.asarray(particle_pos, dtype=float)

        # --- Recentering component ---
        seg, _ = self.network.find_segment(particle_pos)

        if seg is None:
            center_target = particle_pos.copy()
        else:
            closest, _, _, _ = seg.closest_point_and_radial(particle_pos)
            to_center = closest - particle_pos
            dist_to_center = float(np.linalg.norm(to_center))

            if dist_to_center < EPS:
                center_target = closest.copy()
            else:
                center_target = (
                    particle_pos + self.recovery_center_fraction * to_center
                )

        # --- Route-conditioned downstream component ---
        waypoints = getattr(navigator, "waypoints", None)

        if waypoints is None or len(waypoints) == 0:
            return self.workspace.clip(center_target)

        # Find nearest waypoint (near best_route_idx)
        base_idx = max(0, self.best_route_idx)
        search_start = max(0, base_idx - 2)
        search_end = min(len(waypoints), base_idx + 10)

        pts = np.asarray(waypoints[search_start:search_end], dtype=float)
        dists = np.linalg.norm(pts - particle_pos, axis=1)
        local_best = int(np.argmin(dists))
        near_idx = search_start + local_best

        # Get downstream tangent from waypoints
        tangent_end_idx = min(
            len(waypoints) - 1,
            near_idx + self.recovery_route_tangent_steps,
        )

        if tangent_end_idx > near_idx:
            wp_near = np.asarray(waypoints[near_idx], dtype=float)
            wp_far = np.asarray(waypoints[tangent_end_idx], dtype=float)
            route_vec = wp_far - wp_near
            route_norm = float(np.linalg.norm(route_vec))

            if route_norm > EPS:
                route_hat = route_vec / route_norm

                downstream_step = min(
                    self.recovery_route_fraction * self.branch_lookahead,
                    0.30 * (seg._length if seg is not None else 0.01),
                )

                target = center_target + downstream_step * route_hat
            else:
                target = center_target
        else:
            target = center_target

        return self.workspace.clip(target)

    def _escape_release_ready(self, route_idx: int) -> bool:
        if self.escape_start_step is None or self.escape_start_route_idx is None:
            return False

        escape_age = self.step_count - self.escape_start_step
        progress_hops = route_idx - self.escape_start_route_idx

        return bool(
            progress_hops >= self.stall_escape_release_hops
            or escape_age >= self.stall_escape_max_steps
        )

    def _recent_recovery_lockout_active(self) -> bool:
        if self.last_recovery_step is None:
            return False

        return bool(
            self.step_count - self.last_recovery_step
            < self.recovery_escape_lockout_steps
        )

    def _point_segment_distance(self, p, a, b):
        p = np.asarray(p, dtype=float)
        a = np.asarray(a, dtype=float)
        b = np.asarray(b, dtype=float)

        ab = b - a
        denom = np.dot(ab, ab)

        if denom < EPS:
            return float(np.linalg.norm(p - a)), a, 0.0

        t = float(np.dot(p - a, ab) / denom)
        t = float(np.clip(t, 0.0, 1.0))

        closest = a + t * ab
        dist = float(np.linalg.norm(p - closest))

        return dist, closest, t

    def _route_relevant_bifurcation(self, pos, navigator):
        pos = np.asarray(pos, dtype=float)

        waypoints = getattr(navigator, "waypoints", [])

        if waypoints is None or len(waypoints) == 0:
            return None, [], np.inf

        route_pts = [pos]
        idx0 = self._nearest_route_index(pos, navigator)

        max_scan = min(len(waypoints), idx0 + 40)

        for wp in waypoints[idx0:max_scan]:
            route_pts.append(np.asarray(wp, dtype=float))

        if len(route_pts) < 2:
            return None, [], np.inf

        best_node = None
        best_ids = []
        best_along = np.inf

        cumulative = 0.0

        for i in range(len(route_pts) - 1):
            a = route_pts[i]
            b = route_pts[i + 1]
            seg_len = float(np.linalg.norm(b - a))

            if seg_len < EPS:
                continue

            for node_point, node_ids in self._bifurcation_nodes:
                lateral_dist, _, t = self._point_segment_distance(node_point, a, b)
                along = cumulative + t * seg_len

                if along < -EPS:
                    continue

                if along > self.branch_detection_radius:
                    continue

                if lateral_dist > self.route_corridor_radius:
                    continue

                if along < best_along:
                    best_along = along
                    best_node = node_point
                    best_ids = node_ids

            cumulative += seg_len

            if cumulative > self.branch_detection_radius:
                break

        if best_node is None:
            return None, [], np.inf

        return np.asarray(best_node, dtype=float), best_ids, float(best_along)

    def _branch_direction_from_bifurcation(self, seg, bif_point) -> np.ndarray:
        bif_point = np.asarray(bif_point, dtype=float)

        ds = np.linalg.norm(seg.start - bif_point)
        de = np.linalg.norm(seg.end - bif_point)

        far_point = seg.end if de > ds else seg.start
        vec = far_point - bif_point
        norm = np.linalg.norm(vec)

        if norm < EPS:
            return seg._axis.copy()

        return vec / norm

    def _candidate_branches(self, bif_point, seg_ids, particle_pos=None):
        seg_by_id = {s.seg_id: s for s in self.network.segments}
        candidates = []
        fallback = []

        for sid in seg_ids:
            seg = seg_by_id[sid]
            direction = self._branch_direction_from_bifurcation(seg, bif_point)
            fallback.append((seg, direction))

            if particle_pos is not None:
                back_vec = np.asarray(particle_pos, dtype=float) - np.asarray(
                    bif_point, dtype=float,
                )
                back_norm = np.linalg.norm(back_vec)

                if back_norm > EPS:
                    back_hat = back_vec / back_norm

                    if float(np.dot(direction, back_hat)) > 0.35:
                        continue

            candidates.append((seg, direction))

        if len(candidates) == 0:
            return fallback

        return candidates

    # ------------------------------------------------------------------
    # ACBC metrics
    # ------------------------------------------------------------------

    def _max_magnetic_projection_speed(self, particle, pos, direction) -> float:
        pos = np.asarray(pos, dtype=float)
        direction = np.asarray(direction, dtype=float)
        direction = direction / (np.linalg.norm(direction) + EPS)

        n = self.magnet_array.n_magnets
        best_proj = 0.0

        test_currents = []

        for i in range(n):
            c = np.zeros(n)
            c[i] = self.max_current
            test_currents.append(c)

        all_on = np.ones(n) * self.max_current
        test_currents.append(all_on)

        previous_currents = self.magnet_array.currents.copy()

        for currents in test_currents:
            self.magnet_array.set_currents(currents)

            B = self.magnet_array.field_at(pos)
            grad = self.magnet_array.grad_B_squared(pos)
            F = particle.magnetic_force(B, grad)

            fmag = np.linalg.norm(F)

            if fmag < EPS:
                continue

            gamma_eff, _, _ = particle._effective_drag(fmag)
            vmag = F / gamma_eff
            proj = float(np.dot(vmag, direction))

            if proj > best_proj:
                best_proj = proj

        self.magnet_array.set_currents(previous_currents)

        return max(0.0, best_proj)

    def _wall_safety_for_segment(self, seg, particle):
        margin = float(seg.radius - particle.radius)
        fraction = margin / (seg.radius + EPS)
        return margin, fraction

    def _wall_safety_at_position(self, particle_pos, particle):
        seg, _ = self.network.find_segment(particle_pos)

        if seg is None:
            return -np.inf, -np.inf, -np.inf

        _, _, radial_dist, _ = seg.closest_point_and_radial(particle_pos)

        clearance = seg.radius - radial_dist
        margin = clearance - particle.radius
        fraction = margin / (seg.radius + EPS)

        return float(clearance), float(margin), float(fraction)

    def _required_wall_clearance(self, particle) -> float:
        """Common center-to-wall clearance criterion used by every controller."""
        return self.wall_clearance_threshold_m

    def _hard_wall_unsafe(self, wall_clearance, wall_fraction, particle) -> bool:
        return bool(wall_clearance < self._required_wall_clearance(particle))

    def _near_wall_safety_band(self, wall_clearance, particle) -> bool:
        safety_band = max(
            self.recovery_exit_wall_clearance,
            self._required_wall_clearance(particle),
        )
        return bool(wall_clearance < safety_band)

    def _edge_weight_with_near_wall_boost(
        self,
        base_w_edge: float,
        current_w_edge: float,
        wall_clearance: float,
        particle,
    ) -> float:
        if self._near_wall_safety_band(wall_clearance, particle):
            return float(
                max(current_w_edge, self.near_wall_edge_multiplier * base_w_edge)
            )

        return float(current_w_edge)

    def _filter_wall_unsafe_currents(
        self,
        particle,
        particle_pos,
        currents,
        dt: float,
        flow,
        sim_time: float,
    ):
        """
        Last-resort ACBC safety filter.

        The low-level MPC wall term is a soft cost. Near tight bends it can still
        accept a high-progress current vector whose one-step prediction skims
        inside the carrier clearance band. This filter keeps the selected current
        direction but backs off its magnitude until the one-step prediction
        preserves the required center-to-wall clearance.
        """
        currents = np.asarray(currents, dtype=float)
        required = self._required_wall_clearance(particle)

        def predict(candidate):
            next_pos, force, _ = _simulate_one_step(
                particle=particle,
                particle_pos=particle_pos,
                currents=candidate,
                magnet_array=self.magnet_array,
                workspace=self.workspace,
                dt=dt,
                flow=flow,
                sim_time=sim_time,
            )
            clearance, _, fraction = self._wall_safety_at_position(
                next_pos,
                particle,
            )
            safe = (
                clearance >= required
                and fraction >= self.min_wall_safety_fraction
            )
            return safe, clearance, force

        safe, _, force = predict(currents)
        if safe:
            return currents, force, False

        zero = np.zeros_like(currents)
        zero_safe, _, zero_force = predict(zero)
        best_currents = zero if zero_safe else currents
        best_force = zero_force if zero_safe else force

        lo = 0.0 if zero_safe else None
        hi = 1.0

        if lo is not None:
            for _ in range(10):
                alpha = 0.5 * (lo + hi)
                cand = alpha * currents
                cand_safe, _, cand_force = predict(cand)
                if cand_safe:
                    lo = alpha
                    best_currents = cand
                    best_force = cand_force
                else:
                    hi = alpha

        logger.debug(
            "ACBC wall safety current filter active: required_clearance=%.6g",
            required,
        )
        return best_currents, best_force, True

    def _recoverability_score(self, particle_pos, particle) -> float:
        seg, _ = self.network.find_segment(particle_pos)

        if seg is None:
            return 0.0

        closest, _, radial_dist, _ = seg.closest_point_and_radial(particle_pos)

        center_dir = closest - particle_pos
        center_norm = np.linalg.norm(center_dir)

        if center_norm < EPS:
            return 1.0

        center_dir = center_dir / center_norm

        v_recenter = self._max_magnetic_projection_speed(
            particle, particle_pos, center_dir,
        )

        clearance = seg.radius - radial_dist
        danger = max(0.0, particle.radius - clearance)

        scale = 1e-6 + max(particle.radius, 0.05 * seg.radius)
        recoverability = (v_recenter + 1e-12) / (danger / scale + 1e-12)

        return float(np.clip(recoverability, 0.0, 10.0))

    def _evaluate_branch(
        self,
        particle,
        particle_pos,
        nominal_target,
        bif_point,
        seg,
        direction,
        flow,
        sim_time,
    ) -> BranchCandidate:
        direction = np.asarray(direction, dtype=float)
        direction = direction / (np.linalg.norm(direction) + EPS)

        target_point = bif_point + direction * min(self.branch_lookahead, seg._length)
        target_point = self.workspace.clip(target_point)

        eval_point = bif_point + direction * min(
            0.25 * self.branch_lookahead, seg._length,
        )
        eval_point = self.workspace.clip(eval_point)

        flow_v = (
            flow.velocity_at(eval_point, sim_time)
            if flow is not None
            else np.zeros(3)
        )

        flow_projection = float(np.dot(flow_v, direction))

        max_mag_projection = self._max_magnetic_projection_speed(
            particle, eval_point, direction,
        )

        opposing_flow = max(0.0, -flow_projection)

        authority_margin = (
            max_mag_projection + max(0.0, flow_projection) + EPS
        ) / (opposing_flow + 1e-6)

        wall_margin, wall_fraction = self._wall_safety_for_segment(seg, particle)

        route_vec = np.asarray(nominal_target, dtype=float) - np.asarray(
            bif_point, dtype=float,
        )
        route_norm = np.linalg.norm(route_vec)

        if route_norm < EPS:
            route_alignment = 0.0
        else:
            route_alignment = float(np.dot(direction, route_vec / route_norm))

        recoverability = self._recoverability_score(particle_pos, particle)

        authority_ok = (
            authority_margin > self.commit_enter_authority
            if self.enable_authority_assessment
            else True
        )

        capturable = (
            authority_ok
            and wall_margin >= self.surface_clearance_target_m
            and wall_fraction >= self.min_wall_safety_fraction
            and recoverability >= self.recovery_threshold
            and route_alignment > -0.25
        )

        score = (
            3.0 * route_alignment
            + 2.0 * np.tanh(authority_margin)
            + 1.5 * np.tanh(recoverability)
            + 1.0 * np.clip(wall_fraction, -1.0, 1.0)
            + 0.25 * flow_projection
        )

        if not capturable:
            score -= 2.0

        return BranchCandidate(
            branch_seg_id=seg.seg_id,
            bifurcation_point=np.asarray(bif_point, dtype=float),
            direction=direction,
            target_point=target_point,
            flow_projection_m_s=flow_projection,
            max_magnetic_projection_m_s=max_mag_projection,
            authority_margin=float(authority_margin),
            wall_safety_margin_m=float(wall_margin),
            wall_safety_fraction=float(wall_fraction),
            route_alignment=float(route_alignment),
            recoverability=float(recoverability),
            score=float(score),
            capturable=bool(capturable),
        )

    # ------------------------------------------------------------------
    # Hybrid state machine
    # ------------------------------------------------------------------

    def _state_age(self) -> int:
        """Return elapsed steps in the current dwell-tracked state."""
        if self.state == ACBC_COMMIT and self.commit_start_step is not None:
            return self.step_count - self.commit_start_step

        if self.state == ACBC_RECOVER and self.last_recovery_step is not None:
            return self.step_count - self.last_recovery_step

        return 0

    def _cooldown_active(self) -> bool:
        if self.last_transition_step is None:
            return False

        return bool(
            self.step_count - self.last_transition_step
            < self.transition_cooldown_steps
        )

    def _dwell_blocks_exit(self, new_state: str) -> bool:
        """
        Prevent rapid state oscillation by requiring a short minimum residence
        time in commitment and recovery. Hard safety transitions pass
        hard_override=True and bypass this check.
        """
        if new_state == self.state:
            return False

        age = self._state_age()

        if self.state == ACBC_COMMIT and new_state != ACBC_COMMIT:
            if age < self.min_commitment_steps:
                logger.debug(
                    "ACBC transition blocked: commitment dwell time active "
                    "(%s/%s)",
                    age,
                    self.min_commitment_steps,
                )
                return True

        if self.state == ACBC_RECOVER and new_state != ACBC_RECOVER:
            if age < self.min_recovery_steps:
                logger.debug(
                    "ACBC transition blocked: recovery dwell time active "
                    "(%s/%s)",
                    age,
                    self.min_recovery_steps,
                )
                return True

        return False

    def _acquire_branch_lock(self, selected: BranchCandidate, navigator):
        self.selected_branch = selected
        self.selected_branch_id = selected.branch_seg_id
        self.commit_start_step = self.step_count
        self.commit_target_idx = self._nearest_route_index(
            selected.target_point, navigator,
        )
        logger.debug(
            "ACBC branch lock acquired: branch=%s target_idx=%s step=%s",
            self.selected_branch_id,
            self.commit_target_idx,
            self.step_count,
        )

    def _release_branch_lock(self, reason: str):
        if self.selected_branch is not None or self.selected_branch_id is not None:
            logger.debug(
                "ACBC branch lock released: branch=%s reason=%s step=%s",
                self.selected_branch_id,
                reason,
                self.step_count,
            )

        self.selected_branch = None
        self.selected_branch_id = None
        self.commit_start_step = None
        self.commit_target_idx = None

    def _enter_recovery_for_wall_safety(self, reason: str) -> bool:
        self._release_branch_lock(reason)
        self.recovery_count += 1
        return self._transition(
            ACBC_RECOVER,
            hard_override=True,
            reason=reason,
        )

    def _refresh_locked_branch(
        self,
        particle,
        particle_pos,
        nominal_target,
        flow,
        sim_time,
    ) -> Optional[BranchCandidate]:
        """
        Re-evaluate only the currently locked branch after dwell expires.
        This preserves branch commitment while still allowing release if
        authority collapses below the hysteretic exit threshold.
        """
        if self.selected_branch is None:
            return None

        seg_by_id = {s.seg_id: s for s in self.network.segments}
        seg = seg_by_id.get(self.selected_branch.branch_seg_id)

        if seg is None:
            return self.selected_branch

        refreshed = self._evaluate_branch(
            particle=particle,
            particle_pos=particle_pos,
            nominal_target=nominal_target,
            bif_point=self.selected_branch.bifurcation_point,
            seg=seg,
            direction=self.selected_branch.direction,
            flow=flow,
            sim_time=sim_time,
        )
        self.selected_branch = refreshed
        self.selected_branch_id = refreshed.branch_seg_id
        return refreshed

    def _transition(
        self,
        new_state: str,
        *,
        hard_override: bool = False,
        reason: str = "",
    ):
        if new_state != self.state:
            old_state = self.state

            if not hard_override:
                if self._cooldown_active():
                    age = self.step_count - self.last_transition_step
                    logger.debug(
                        "ACBC transition blocked: cooldown active "
                        "(%s/%s) %s -> %s",
                        age,
                        self.transition_cooldown_steps,
                        old_state,
                        new_state,
                    )
                    self.prev_state = self.state
                    return False

                if self._dwell_blocks_exit(new_state):
                    self.prev_state = self.state
                    return False

            self.prev_state = self.state
            self.state = new_state
            self.transition_count += 1
            self.last_transition_step = self.step_count

            if new_state == ACBC_COMMIT:
                self.commit_start_step = self.step_count
                logger.debug(
                    "ACBC entering BRANCH_COMMITMENT at step=%s reason=%s",
                    self.step_count,
                    reason,
                )

            if new_state == ACBC_RECOVER:
                self.last_recovery_step = self.step_count
                logger.debug(
                    "ACBC entering RECOVERY_MODE at step=%s reason=%s",
                    self.step_count,
                    reason,
                )

            if new_state == ACBC_ESCAPE:
                self.escape_start_step = self.step_count
                self.escape_start_route_idx = self.best_route_idx

            if old_state == ACBC_COMMIT and new_state != ACBC_COMMIT:
                logger.debug(
                    "ACBC exiting BRANCH_COMMITMENT at step=%s reason=%s",
                    self.step_count,
                    reason,
                )
                self._release_branch_lock(reason or f"transition_to_{new_state}")

            if old_state == ACBC_RECOVER and new_state != ACBC_RECOVER:
                logger.debug(
                    "ACBC exiting RECOVERY_MODE at step=%s reason=%s",
                    self.step_count,
                    reason,
                )

            if old_state == ACBC_ESCAPE and new_state != ACBC_ESCAPE:
                self.escape_start_step = None
                self.escape_start_route_idx = None

            return True

        self.prev_state = self.state
        return False

    def step(
        self,
        particle,
        final_target,
        navigator,
        dt: float,
        flow,
        sim_time: float,
        initial_guess=None,
        w_progress: float = 120.0,
        w_align: float = 8.0,
        w_dist: float = 1.0,
        w_curr: float = 1e-5,
        w_smooth: float = 1e-4,
        w_edge: float = 50.0,
        w_overshoot: float = 2.0,
        w_route: float = 0.0,
        route_tangent: Optional[np.ndarray] = None,
    ):
        self.step_count += 1
        step_initial_state = self.state

        particle_pos = particle.pos.copy()
        nominal_target = navigator(particle_pos)
        route_idx = self._navigator_route_index(particle_pos, navigator)

        # Update best route index (monotonic) and stall status
        stalled = self._update_stall_status(particle_pos, navigator)

        bif_point, bif_ids, bif_dist = self._route_relevant_bifurcation(
            particle_pos, navigator,
        )

        near_bifurcation = bif_point is not None

        wall_clearance, wall_margin, wall_fraction = self._wall_safety_at_position(
            particle_pos, particle,
        )

        # --- Wall danger classification ---
        wall_unsafe = self._hard_wall_unsafe(
            wall_clearance,
            wall_fraction,
            particle,
        )

        safe_for_escape = wall_clearance >= self.stall_escape_min_wall_clearance

        near_wall_warning = not safe_for_escape

        recovery_lockout_active = self._recent_recovery_lockout_active()

        stall_escape_allowed = bool(
            stalled
            and safe_for_escape
            and not recovery_lockout_active
        )

        candidates: List[BranchCandidate] = []
        selected: Optional[BranchCandidate] = None
        recovery_target = None
        escape_target = None
        transitioned = False

        # ==============================================================
        # PRIORITY 1: Hard wall danger → RECOVERY_MODE
        # This overrides everything except existing recovery.
        # ==============================================================
        if self.enable_recovery and wall_unsafe and self.state != ACBC_RECOVER:
            transitioned = (
                self._enter_recovery_for_wall_safety("hard_wall_safety")
                or transitioned
            )

        # ==============================================================
        # PRIORITY 2: Stall escape (only when safe AND not recovering)
        # ==============================================================
        elif (
            self.enable_stall_escape
            and stall_escape_allowed
            and self.state not in (ACBC_ESCAPE, ACBC_RECOVER)
        ):
            self._release_branch_lock("stall_escape")
            self.stall_escape_count += 1
            transitioned = self._transition(
                ACBC_ESCAPE,
                reason="stall_escape",
            ) or transitioned

        # ==============================================================
        # Stalled but near wall → recovery (not escape)
        # ==============================================================
        if (
            stalled
            and near_wall_warning
            and not safe_for_escape
            and self.enable_recovery
            and self.state not in (ACBC_RECOVER, ACBC_ESCAPE)
        ):
            self._release_branch_lock("stalled_near_wall")
            self.recovery_count += 1
            transitioned = self._transition(
                ACBC_RECOVER,
                reason="stalled_near_wall",
            ) or transitioned

        # ==============================================================
        # State-specific logic
        # ==============================================================

        if self.state == ACBC_FREE:
            if near_bifurcation:
                transitioned = self._transition(
                    ACBC_ASSESS,
                    reason="near_bifurcation",
                ) or transitioned
            else:
                transitioned = self._transition(ACBC_FREE) or transitioned

        if self.state == ACBC_ASSESS:
            if not near_bifurcation:
                self._release_branch_lock("left_bifurcation_region")
                transitioned = self._transition(
                    ACBC_FREE,
                    reason="left_bifurcation_region",
                ) or transitioned
            else:
                branches = self._candidate_branches(
                    bif_point, bif_ids, particle_pos=particle_pos,
                )

                for seg, direction in branches:
                    cand = self._evaluate_branch(
                        particle=particle,
                        particle_pos=particle_pos,
                        nominal_target=nominal_target,
                        bif_point=bif_point,
                        seg=seg,
                        direction=direction,
                        flow=flow,
                        sim_time=sim_time,
                    )
                    candidates.append(cand)

                if candidates:
                    candidates.sort(key=lambda c: c.score, reverse=True)
                    selected = candidates[0]
                    self.last_bifurcation_point = bif_point

                    if selected.capturable and self.enable_branch_commitment:
                        committed = self._transition(
                            ACBC_COMMIT,
                            reason=(
                                "authority_margin="
                                f"{selected.authority_margin:.3f}"
                            ),
                        )
                        transitioned = committed or transitioned
                        if committed:
                            self.commitment_count += 1
                            self._acquire_branch_lock(selected, navigator)
                        else:
                            self.selected_branch = selected
                    else:
                        if self.enable_recovery and wall_unsafe:
                            transitioned = (
                                self._enter_recovery_for_wall_safety(
                                    "assessment_wall_safety",
                                )
                                or transitioned
                            )
                        else:
                            self._release_branch_lock("branch_not_capturable")
                            transitioned = self._transition(
                                ACBC_FREE,
                                reason="branch_not_capturable",
                            ) or transitioned
                else:
                    self._release_branch_lock("no_branch_candidates")
                    transitioned = self._transition(
                        ACBC_FREE,
                        reason="no_branch_candidates",
                    ) or transitioned

        elif self.state == ACBC_COMMIT:
            selected = self.selected_branch

            if selected is None:
                transitioned = self._transition(
                    ACBC_FREE,
                    reason="missing_branch_lock",
                ) or transitioned
            else:
                dist_from_bif = float(
                    np.linalg.norm(particle_pos - selected.bifurcation_point)
                )

                dist_to_commit_target = float(
                    np.linalg.norm(particle_pos - selected.target_point)
                )

                if self.commit_start_step is None:
                    self.commit_start_step = self.step_count

                commit_age = self.step_count - self.commit_start_step

                if commit_age >= self.min_commitment_steps:
                    selected = self._refresh_locked_branch(
                        particle=particle,
                        particle_pos=particle_pos,
                        nominal_target=nominal_target,
                        flow=flow,
                        sim_time=sim_time,
                    )

                authority_lost = bool(
                    selected is not None
                    and self.enable_authority_assessment
                    and selected.authority_margin < self.commit_exit_authority
                )

                commitment_complete = (
                    dist_to_commit_target < self.commit_target_reach_radius
                    or dist_from_bif > 0.75 * self.branch_detection_radius
                    or commit_age >= self.max_commitment_steps
                )

                if authority_lost:
                    transitioned = self._transition(
                        ACBC_FREE,
                        reason=(
                            "authority_below_exit_threshold="
                            f"{selected.authority_margin:.3f}"
                        ),
                    ) or transitioned
                elif commitment_complete:
                    transitioned = self._transition(
                        ACBC_FREE,
                        reason="commitment_complete",
                    ) or transitioned

        elif self.state == ACBC_ESCAPE:
            if self.enable_recovery and wall_unsafe:
                # Hard wall danger overrides escape
                transitioned = (
                    self._enter_recovery_for_wall_safety("escape_wall_safety")
                    or transitioned
                )
            elif self._escape_release_ready(self.best_route_idx):
                # Use best_route_idx for release check, not navigator's route_idx
                transitioned = self._transition(
                    ACBC_FREE,
                    reason="escape_release_ready",
                ) or transitioned

        elif self.state == ACBC_RECOVER:
            recovery_target = self._route_conditioned_recovery_target(
                particle_pos, navigator,
            )

            wall_clearance, wall_margin, wall_fraction = self._wall_safety_at_position(
                particle_pos, particle,
            )

            wall_safe_again = wall_clearance > self.recovery_exit_wall_clearance

            if self.last_recovery_step is None:
                recovery_age = 0
            else:
                recovery_age = self.step_count - self.last_recovery_step

            recovery_timed_out_safely = bool(
                recovery_age >= self.max_recovery_steps
                and wall_clearance >= self.recovery_enter_wall_clearance
            )

            can_leave_recovery = bool(wall_safe_again or recovery_timed_out_safely)

            if can_leave_recovery:
                if near_bifurcation:
                    transitioned = self._transition(
                        ACBC_ASSESS,
                        reason="recovery_clear_near_bifurcation",
                    ) or transitioned
                else:
                    transitioned = self._transition(
                        ACBC_FREE,
                        reason="recovery_clear",
                    ) or transitioned

        if self.state == ACBC_ASSESS and len(candidates) == 0:
            self._release_branch_lock("assessment_no_candidates")
            transitioned = self._transition(
                ACBC_FREE,
                reason="assessment_no_candidates",
            ) or transitioned

        # ==============================================================
        # Choose control target
        # ==============================================================

        if self.state == ACBC_FREE:
            control_target = nominal_target

        elif self.state == ACBC_ASSESS:
            if self.selected_branch is not None:
                control_target = self.selected_branch.target_point
            else:
                control_target = nominal_target

        elif self.state == ACBC_COMMIT:
            if self.selected_branch is not None:
                control_target = self.selected_branch.target_point
            else:
                control_target = nominal_target

        elif self.state == ACBC_ESCAPE:
            escape_target = self._stall_escape_target(particle_pos, navigator)
            control_target = escape_target

        elif self.state == ACBC_RECOVER:
            if recovery_target is None:
                recovery_target = self._route_conditioned_recovery_target(
                    particle_pos, navigator,
                )
            control_target = recovery_target

        else:
            control_target = nominal_target

        # ==============================================================
        # Low-level current allocation weights
        # ==============================================================

        if self.state == ACBC_ESCAPE:
            w_progress_eff = 4.00 * w_progress
            w_align_eff = 2.00 * w_align
            w_dist_eff = 1.50 * w_dist
            w_curr_eff = 0.10 * w_curr
            w_smooth_eff = 0.10 * w_smooth
            w_edge_eff = 0.60 * w_edge
            w_overshoot_eff = 0.50 * w_overshoot
            w_route_eff = 2.0 * w_route

        elif self.state == ACBC_RECOVER:
            w_progress_eff = 0.40 * w_progress
            w_align_eff = 0.60 * w_align
            w_dist_eff = 1.50 * w_dist
            w_curr_eff = w_curr
            w_smooth_eff = w_smooth
            w_edge_eff = 2.50 * w_edge
            w_overshoot_eff = w_overshoot
            w_route_eff = 0.5 * w_route

        else:
            w_progress_eff = w_progress
            w_align_eff = w_align
            w_dist_eff = w_dist
            w_curr_eff = w_curr
            w_smooth_eff = w_smooth
            w_edge_eff = w_edge
            w_overshoot_eff = w_overshoot
            w_route_eff = w_route

        w_edge_eff = self._edge_weight_with_near_wall_boost(
            base_w_edge=w_edge,
            current_w_edge=w_edge_eff,
            wall_clearance=wall_clearance,
            particle=particle,
        )

        # Build route tangent for MPC if not provided externally
        if route_tangent is None:
            waypoints = getattr(navigator, "waypoints", None)
            if waypoints is not None and len(waypoints) > 1:
                tidx = min(self.best_route_idx, len(waypoints) - 1)
                tidx_next = min(tidx + 3, len(waypoints) - 1)
                if tidx_next > tidx:
                    rt = (
                        np.asarray(waypoints[tidx_next], dtype=float)
                        - np.asarray(waypoints[tidx], dtype=float)
                    )
                    rtn = np.linalg.norm(rt)
                    route_tangent = rt / rtn if rtn > EPS else None

        best_currents, best_force, best_cost, converged = continuous_mpc_controller(
            particle=particle,
            particle_pos=particle_pos,
            target_pos=control_target,
            magnet_array=self.magnet_array,
            workspace=self.workspace,
            dt=dt,
            max_current=self.max_current,
            initial_guess=initial_guess,
            w_progress=w_progress_eff,
            w_align=w_align_eff,
            w_dist=w_dist_eff,
            w_curr=w_curr_eff,
            w_smooth=w_smooth_eff,
            w_edge=w_edge_eff,
            w_overshoot=w_overshoot_eff,
            w_route=w_route_eff,
            route_tangent=route_tangent,
            wall_buffer_m=self._required_wall_clearance(particle),
            flow=flow,
            sim_time=sim_time,
        )

        if self.enable_wall_safety_current_filter:
            best_currents, best_force, _ = self._filter_wall_unsafe_currents(
                particle=particle,
                particle_pos=particle_pos,
                currents=best_currents,
                dt=dt,
                flow=flow,
                sim_time=sim_time,
            )

        selected_branch_id = None
        selected_margin = 0.0
        selected_score = 0.0
        target_capturable = False

        if self.selected_branch is not None:
            selected_branch_id = self.selected_branch.branch_seg_id
            selected_margin = self.selected_branch.authority_margin
            selected_score = self.selected_branch.score
            target_capturable = self.selected_branch.capturable

        info = ACBCStepInfo(
            state=self.state,
            previous_state=step_initial_state if transitioned else self.state,
            transition=transitioned,
            control_target=np.asarray(control_target, dtype=float),
            nominal_target=np.asarray(nominal_target, dtype=float),
            near_bifurcation=bool(near_bifurcation),
            bifurcation_distance_m=float(bif_dist),
            bifurcation_point=None
            if bif_point is None
            else np.asarray(bif_point, dtype=float),
            selected_branch_seg_id=selected_branch_id,
            selected_authority_margin=float(selected_margin),
            selected_score=float(selected_score),
            target_branch_capturable=bool(target_capturable),
            recovery_target=None
            if recovery_target is None
            else np.asarray(recovery_target, dtype=float),
            num_candidates=len(candidates),
            stall_escape_target=None
            if escape_target is None
            else np.asarray(escape_target, dtype=float),
            best_route_idx=self.best_route_idx,
            stagnant_steps=self.stagnant_steps,
            wall_clearance_m=float(wall_clearance),
            wall_fraction=float(wall_fraction),
        )

        return best_currents, best_force, best_cost, converged, info

    def summary_dict(self):
        return {
            "acbc_steps": self.step_count,
            "acbc_transitions": self.transition_count,
            "acbc_commitments": self.commitment_count,
            "acbc_recoveries": self.recovery_count,
            "acbc_stall_escapes": self.stall_escape_count,
            "acbc_branch_detection_radius_m": self.branch_detection_radius,
            "acbc_branch_lookahead_m": self.branch_lookahead,
            "acbc_route_corridor_radius_m": self.route_corridor_radius,
            "acbc_authority_threshold": self.authority_threshold,
            "acbc_commit_enter_authority": self.commit_enter_authority,
            "acbc_commit_exit_authority": self.commit_exit_authority,
            "acbc_recovery_threshold": self.recovery_threshold,
            "acbc_wall_clearance_threshold_m": self.wall_clearance_threshold_m,
            "acbc_min_wall_safety_fraction": self.min_wall_safety_fraction,
            "acbc_absolute_wall_clearance_threshold_m": (
                self.absolute_wall_clearance_threshold
            ),
            "acbc_surface_clearance_target_m": self.surface_clearance_target_m,
            "acbc_recovery_enter_wall_clearance_m": (
                self.recovery_enter_wall_clearance
            ),
            "acbc_recovery_exit_wall_clearance_m": (
                self.recovery_exit_wall_clearance
            ),
            "acbc_near_wall_edge_multiplier": self.near_wall_edge_multiplier,
            "acbc_min_commitment_steps": self.min_commitment_steps,
            "acbc_max_commitment_steps": self.max_commitment_steps,
            "acbc_commit_target_reach_radius_m": self.commit_target_reach_radius,
            "acbc_recovery_center_fraction": self.recovery_center_fraction,
            "acbc_recovery_route_fraction": self.recovery_route_fraction,
            "acbc_stall_window_steps": self.stall_window_steps,
            "acbc_stall_escape_lookahead": self.stall_escape_lookahead,
            "acbc_stall_escape_release_hops": self.stall_escape_release_hops,
            "acbc_stall_escape_max_steps": self.stall_escape_max_steps,
            "acbc_stall_escape_min_wall_clearance_m": (
                self.stall_escape_min_wall_clearance
            ),
            "acbc_stall_escape_min_wall_fraction_multiplier": (
                self.stall_escape_min_wall_fraction_multiplier
            ),
            "acbc_recovery_escape_lockout_steps": (
                self.recovery_escape_lockout_steps
            ),
            "acbc_min_recovery_steps": self.min_recovery_steps,
            "acbc_max_recovery_steps": self.max_recovery_steps,
            "acbc_recovery_release_clearance_m": self.recovery_release_clearance,
            "acbc_transition_cooldown_steps": self.transition_cooldown_steps,
            "acbc_enable_wall_safety_current_filter": (
                self.enable_wall_safety_current_filter
            ),
            "acbc_enable_branch_commitment": self.enable_branch_commitment,
            "acbc_enable_authority_assessment": self.enable_authority_assessment,
            "acbc_enable_recovery": self.enable_recovery,
            "acbc_enable_stall_escape": self.enable_stall_escape,
        }
