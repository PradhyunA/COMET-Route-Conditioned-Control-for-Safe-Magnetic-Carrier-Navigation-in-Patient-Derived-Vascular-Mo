"""
Authority-Conditioned Branch Commitment controller.

ACBC is a hybrid supervisory controller for magnetic carrier navigation
through bifurcating vascular graphs.

Main correction:
    ACBC is route-conditioned.

It does NOT activate near the nearest bifurcation anywhere in the vessel tree.
It only activates when a bifurcation lies ahead on the planned route.

Recovery correction:
    Recovery is softened.

Instead of snapping directly to the vessel centerline during RECOVERY_MODE,
ACBC targets a partial step toward the centerline. This reduces aggressive
sideways recovery forces, lowers velocity spikes, and preserves wall-safety
behavior.

Stall correction:
    Stall escape is added.

If the route waypoint index does not improve for many consecutive control
steps while the particle is safely away from the vessel wall, ACBC temporarily
targets a farther downstream waypoint and boosts forward-progress weighting.

Latest correction:
    Stall escape has stronger wall-safety gating and recovery lockout.

Trial 1 needs STALL_ESCAPE at the high-flow wp≈184 plateau, where wall
clearance is several millimeters. Trial 2 showed that STALL_ESCAPE should NOT
activate near the wall, and should not immediately re-enter after recovery.
Therefore STALL_ESCAPE now requires:
    - wall clearance >= 2.5 mm by default
    - sufficient wall safety fraction
    - no recent RECOVERY_MODE lockout

States:
    FREE_NAVIGATION
    VIABILITY_ASSESSMENT
    BRANCH_COMMITMENT
    RECOVERY_MODE
    STALL_ESCAPE
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional, List, Dict, Tuple

import numpy as np

from .constants import EPS
from .controllers import continuous_mpc_controller


ACBC_FREE = "FREE_NAVIGATION"
ACBC_ASSESS = "VIABILITY_ASSESSMENT"
ACBC_COMMIT = "BRANCH_COMMITMENT"
ACBC_RECOVER = "RECOVERY_MODE"
ACBC_ESCAPE = "STALL_ESCAPE"


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


class ACBCController:
    """
    Authority-Conditioned Branch Commitment.

    ACBC is a route-conditioned hybrid supervisor. It uses the existing
    continuous MPC controller only as a low-level current allocator.

    The novelty is the supervisory decision:
        free navigation vs. assess viability vs. commit vs. recover vs. escape.
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
        authority_threshold: float = 1.05,
        recovery_threshold: float = 0.85,
        min_wall_safety_fraction: float = 0.05,
        absolute_wall_clearance_threshold: float = 0.00075,
        max_commitment_steps: int = 40,
        recovery_center_fraction: float = 0.45,
        recovery_route_fraction: float = 0.12,
        stall_window_steps: int = 45,
        stall_escape_lookahead: int = 24,
        stall_escape_release_hops: int = 5,
        stall_escape_max_steps: int = 90,
        stall_escape_min_wall_clearance: float = 0.0025,
        stall_escape_min_wall_fraction_multiplier: float = 2.0,
        recovery_escape_lockout_steps: int = 80,
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
        self.recovery_threshold = float(recovery_threshold)
        self.min_wall_safety_fraction = float(min_wall_safety_fraction)

        self.absolute_wall_clearance_threshold = float(
            absolute_wall_clearance_threshold
        )

        self.recovery_center_fraction = float(
            np.clip(recovery_center_fraction, 0.05, 1.0)
        )
        self.recovery_route_fraction = float(
            np.clip(recovery_route_fraction, 0.0, 0.5)
        )

        # Stall escape configuration.
        self.stall_window_steps = int(max(5, stall_window_steps))
        self.stall_escape_lookahead = int(max(4, stall_escape_lookahead))
        self.stall_escape_release_hops = int(max(1, stall_escape_release_hops))
        self.stall_escape_max_steps = int(max(10, stall_escape_max_steps))

        # Extra wall-safety gate for STALL_ESCAPE.
        #
        # This is intentionally much stricter than the global wall-unsafe
        # trigger. STALL_ESCAPE is aggressive; it should only happen when the
        # particle is clearly away from the wall.
        self.stall_escape_min_wall_clearance = float(
            max(0.0, stall_escape_min_wall_clearance)
        )
        self.stall_escape_min_wall_fraction_multiplier = float(
            max(1.0, stall_escape_min_wall_fraction_multiplier)
        )

        # Recovery lockout:
        # after RECOVERY_MODE is entered, do not allow STALL_ESCAPE for this
        # many controller steps. This prevents:
        #     RECOVERY -> ESCAPE -> wall scrape -> RECOVERY -> ESCAPE
        self.recovery_escape_lockout_steps = int(
            max(0, recovery_escape_lockout_steps)
        )
        self.last_recovery_step: Optional[int] = None

        self.best_route_idx = -1
        self.stagnant_steps = 0
        self.escape_start_step: Optional[int] = None
        self.escape_start_route_idx: Optional[int] = None

        self.state = ACBC_FREE
        self.prev_state = ACBC_FREE

        self.selected_branch: Optional[BranchCandidate] = None
        self.last_bifurcation_point: Optional[np.ndarray] = None

        self.commit_start_step: Optional[int] = None
        self.max_commitment_steps = int(max_commitment_steps)
        self.commit_target_reach_radius = max(
            0.004,
            0.35 * self.branch_lookahead,
        )

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
        waypoints = getattr(navigator, "waypoints", [])

        if waypoints is None or len(waypoints) == 0:
            return 0

        pts = np.asarray(waypoints, dtype=float)
        dists = np.linalg.norm(pts - np.asarray(pos, dtype=float), axis=1)

        return int(np.argmin(dists))

    def _navigator_route_index(self, pos, navigator) -> int:
        """
        Return navigator current_idx when available. Fall back to nearest route
        waypoint index.
        """
        idx = getattr(navigator, "current_idx", None)

        if idx is not None:
            try:
                return int(idx)
            except Exception:
                pass

        return self._nearest_route_index(pos, navigator)

    def _update_stall_status(self, route_idx: int, navigator) -> bool:
        """
        Detect when route progress has plateaued.

        A stall means:
            - waypoint index has not improved for stall_window_steps
            - there are still downstream route waypoints ahead
        """
        total_hops = int(getattr(navigator, "total_hops", 0) or 0)

        if route_idx > self.best_route_idx:
            self.best_route_idx = int(route_idx)
            self.stagnant_steps = 0
        else:
            self.stagnant_steps += 1

        has_route_ahead = True

        if total_hops > 0:
            has_route_ahead = route_idx < total_hops - 3

        return bool(
            self.stagnant_steps >= self.stall_window_steps
            and has_route_ahead
        )

    def _stall_escape_target(self, particle_pos, navigator):
        """
        Pick a farther downstream waypoint on the same planned route.

        This does not change the route or invent a new path. It only tells the
        low-level allocator to stop obsessing over a local waypoint it cannot
        cleanly reach.
        """
        waypoints = getattr(navigator, "waypoints", None)

        if waypoints is None or len(waypoints) == 0:
            return np.asarray(particle_pos, dtype=float)

        current_idx = self._navigator_route_index(particle_pos, navigator)
        target_idx = min(
            len(waypoints) - 1,
            current_idx + self.stall_escape_lookahead,
        )

        return self.workspace.clip(np.asarray(waypoints[target_idx], dtype=float))

    def _escape_release_ready(self, route_idx: int) -> bool:
        """
        Leave STALL_ESCAPE after meaningful downstream progress, or after a
        maximum number of escape steps so the state does not become permanent.
        """
        if self.escape_start_step is None or self.escape_start_route_idx is None:
            return False

        escape_age = self.step_count - self.escape_start_step
        progress_hops = route_idx - self.escape_start_route_idx

        return bool(
            progress_hops >= self.stall_escape_release_hops
            or escape_age >= self.stall_escape_max_steps
        )

    def _recent_recovery_lockout_active(self) -> bool:
        """
        Returns True if STALL_ESCAPE should be blocked because recovery was
        entered recently.
        """
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
        """
        Find the next bifurcation that lies on the planned route ahead.

        This prevents ACBC from triggering on irrelevant nearby branch nodes.
        """
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
                    bif_point,
                    dtype=float,
                )
                back_norm = np.linalg.norm(back_vec)

                if back_norm > EPS:
                    back_hat = back_vec / back_norm

                    # Exclude the branch that points mostly back toward where
                    # the particle came from.
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
        """
        Return local wall safety at the particle's current position.

        clearance:
            center of particle to vessel wall distance before subtracting
            particle radius.

        margin:
            remaining clearance between particle surface and vessel wall.

        fraction:
            margin normalized by vessel radius.
        """
        seg, _ = self.network.find_segment(particle_pos)

        if seg is None:
            return -np.inf, -np.inf, -np.inf

        _, _, radial_dist, _ = seg.closest_point_and_radial(particle_pos)

        clearance = seg.radius - radial_dist
        margin = clearance - particle.radius
        fraction = margin / (seg.radius + EPS)

        return float(clearance), float(margin), float(fraction)

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
            particle,
            particle_pos,
            center_dir,
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
            0.25 * self.branch_lookahead,
            seg._length,
        )
        eval_point = self.workspace.clip(eval_point)

        flow_v = (
            flow.velocity_at(eval_point, sim_time)
            if flow is not None
            else np.zeros(3)
        )

        flow_projection = float(np.dot(flow_v, direction))

        max_mag_projection = self._max_magnetic_projection_speed(
            particle,
            eval_point,
            direction,
        )

        opposing_flow = max(0.0, -flow_projection)

        authority_margin = (
            max_mag_projection + max(0.0, flow_projection) + EPS
        ) / (opposing_flow + 1e-6)

        wall_margin, wall_fraction = self._wall_safety_for_segment(seg, particle)

        route_vec = np.asarray(nominal_target, dtype=float) - np.asarray(
            bif_point,
            dtype=float,
        )
        route_norm = np.linalg.norm(route_vec)

        if route_norm < EPS:
            route_alignment = 0.0
        else:
            route_alignment = float(np.dot(direction, route_vec / route_norm))

        recoverability = self._recoverability_score(particle_pos, particle)

        capturable = (
            authority_margin >= self.authority_threshold
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

    def _recovery_target(self, particle_pos, nominal_target=None):
        """
        Recovery target gently recenters the particle.

        Previous behavior:
            If the particle was near the wall, ACBC targeted the exact
            centerline point. That created large sideways corrections and
            velocity spikes in RECOVERY_MODE.

        Current behavior:
            Target only a partial step toward the centerline, optionally with
            a small downstream route nudge once partial recentering is defined.
        """
        particle_pos = np.asarray(particle_pos, dtype=float)

        seg, _ = self.network.find_segment(particle_pos)

        if seg is None:
            return particle_pos.copy()

        closest, _, radial_dist, _ = seg.closest_point_and_radial(particle_pos)

        to_center = closest - particle_pos
        dist_to_center = float(np.linalg.norm(to_center))

        if dist_to_center < EPS:
            base_target = closest.copy()
        else:
            base_target = particle_pos + self.recovery_center_fraction * to_center

        if nominal_target is None:
            return self.workspace.clip(base_target)

        route_vec = np.asarray(nominal_target, dtype=float) - base_target
        route_norm = float(np.linalg.norm(route_vec))

        if route_norm < EPS:
            return self.workspace.clip(base_target)

        route_hat = route_vec / route_norm

        downstream_step = min(
            self.recovery_route_fraction * self.branch_lookahead,
            0.20 * seg._length,
        )

        target = base_target + downstream_step * route_hat

        return self.workspace.clip(target)

    # ------------------------------------------------------------------
    # Hybrid state machine
    # ------------------------------------------------------------------

    def _transition(self, new_state: str):
        if new_state != self.state:
            old_state = self.state

            self.prev_state = self.state
            self.state = new_state
            self.transition_count += 1

            if new_state == ACBC_COMMIT:
                self.commit_start_step = self.step_count

            if new_state == ACBC_RECOVER:
                self.last_recovery_step = self.step_count

            if new_state == ACBC_ESCAPE:
                self.escape_start_step = self.step_count
                self.escape_start_route_idx = self.best_route_idx

            if old_state == ACBC_COMMIT and new_state != ACBC_COMMIT:
                self.commit_start_step = None

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
    ):
        self.step_count += 1

        particle_pos = particle.pos.copy()
        nominal_target = navigator(particle_pos)
        route_idx = self._navigator_route_index(particle_pos, navigator)

        bif_point, bif_ids, bif_dist = self._route_relevant_bifurcation(
            particle_pos,
            navigator,
        )

        near_bifurcation = bif_point is not None

        wall_clearance, wall_margin, wall_fraction = self._wall_safety_at_position(
            particle_pos,
            particle,
        )

        # Hard wall-unsafe condition.
        # This is the emergency trigger: the particle is already too close to
        # the wall and should immediately prioritize recentering.
        wall_unsafe = (
            wall_clearance < self.absolute_wall_clearance_threshold
            or wall_fraction < self.min_wall_safety_fraction
        )

        stalled = self._update_stall_status(route_idx, navigator)

        # Soft near-wall warning.
        # This is stricter than wall_unsafe and is used for hysteresis:
        # if the route is stalled while the particle is not safely centered,
        # prefer RECOVERY_MODE over aggressive STALL_ESCAPE.
        near_wall_warning = (
            wall_clearance < self.stall_escape_min_wall_clearance
            or wall_fraction
            < self.stall_escape_min_wall_fraction_multiplier
            * self.min_wall_safety_fraction
        )

        # Important safety gate:
        # STALL_ESCAPE should only activate when the particle is safely away
        # from the wall. Trial 1 had several millimeters of clearance during
        # the useful escape. Trial 2 hit escape near ~0.1--1.7 mm clearance,
        # which should instead force recovery/recentering.
        safe_for_escape = (
            wall_clearance >= self.stall_escape_min_wall_clearance
            and wall_fraction
            >= self.stall_escape_min_wall_fraction_multiplier
            * self.min_wall_safety_fraction
        )

        recovery_lockout_active = self._recent_recovery_lockout_active()

        stall_escape_allowed = bool(
            stalled
            and safe_for_escape
            and not recovery_lockout_active
        )

        candidates: List[BranchCandidate] = []
        selected: Optional[BranchCandidate] = None
        recovery_target = None
        transitioned = False

        # --------------------------------------------------------------
        # Global wall-safety guard
        # --------------------------------------------------------------

        if (
            wall_unsafe
            or (stalled and near_wall_warning)
        ) and self.state != ACBC_RECOVER:
            self.selected_branch = None
            self.recovery_count += 1
            transitioned = self._transition(ACBC_RECOVER) or transitioned

        # --------------------------------------------------------------
        # Route-progress stall guard
        # --------------------------------------------------------------

        if stall_escape_allowed and self.state != ACBC_ESCAPE:
            self.selected_branch = None
            self.stall_escape_count += 1
            transitioned = self._transition(ACBC_ESCAPE) or transitioned

        # --------------------------------------------------------------
        # State transitions
        # --------------------------------------------------------------

        if self.state == ACBC_FREE:
            if near_bifurcation:
                transitioned = self._transition(ACBC_ASSESS) or transitioned
            else:
                transitioned = self._transition(ACBC_FREE) or transitioned

        if self.state == ACBC_ASSESS:
            if not near_bifurcation:
                self.selected_branch = None
                transitioned = self._transition(ACBC_FREE) or transitioned
            else:
                branches = self._candidate_branches(
                    bif_point,
                    bif_ids,
                    particle_pos=particle_pos,
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
                    self.selected_branch = selected
                    self.last_bifurcation_point = bif_point

                    if selected.capturable:
                        self.commitment_count += 1
                        transitioned = self._transition(ACBC_COMMIT) or transitioned
                    else:
                        # Only recover if the selected branch is truly unsafe.
                        # Otherwise, stay in free navigation instead of trapping.
                        if selected.authority_margin < self.recovery_threshold:
                            self.recovery_count += 1
                            transitioned = self._transition(ACBC_RECOVER) or transitioned
                        else:
                            self.selected_branch = None
                            transitioned = self._transition(ACBC_FREE) or transitioned
                else:
                    self.selected_branch = None
                    transitioned = self._transition(ACBC_FREE) or transitioned

        elif self.state == ACBC_COMMIT:
            selected = self.selected_branch

            if selected is None:
                transitioned = self._transition(ACBC_FREE) or transitioned
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

                commitment_complete = (
                    dist_to_commit_target < self.commit_target_reach_radius
                    or dist_from_bif > 0.75 * self.branch_detection_radius
                    or commit_age >= self.max_commitment_steps
                )

                if commitment_complete:
                    self.selected_branch = None
                    transitioned = self._transition(ACBC_FREE) or transitioned

        elif self.state == ACBC_ESCAPE:
            # If clearance degrades while escaping, or if recovery was recent,
            # immediately return to recovery.
            if not safe_for_escape or recovery_lockout_active:
                self.recovery_count += 1
                transitioned = self._transition(ACBC_RECOVER) or transitioned
            elif self._escape_release_ready(route_idx):
                transitioned = self._transition(ACBC_FREE) or transitioned

        elif self.state == ACBC_RECOVER:
            recovery_target = self._recovery_target(
                particle_pos,
                nominal_target=nominal_target,
            )

            recoverability = self._recoverability_score(particle_pos, particle)

            wall_clearance, wall_margin, wall_fraction = self._wall_safety_at_position(
                particle_pos,
                particle,
            )

            # Recovery release must be LESS strict than STALL_ESCAPE entry.
            # Otherwise Trial 1 can get stuck around ~2.26-2.28 mm clearance
            # because STALL_ESCAPE requires 2.5 mm by default.
            #
            # This creates three useful zones:
            #   < 0.75 mm  -> forced RECOVERY
            #   0.75-1.8 mm -> remain in RECOVERY
            #   1.8-2.5 mm -> FREE/COMMIT allowed, STALL_ESCAPE blocked
            #   > 2.5 mm -> STALL_ESCAPE allowed if stalled
            recovery_release_clearance = 0.0018  # 1.8 mm
            recovery_release_fraction = 1.25 * self.min_wall_safety_fraction

            wall_safe_again = (
                wall_clearance >= recovery_release_clearance
                and wall_fraction >= recovery_release_fraction
            )

            can_leave_recovery = bool(
                recoverability >= self.recovery_threshold
                and wall_safe_again
            )

            if can_leave_recovery:
                if near_bifurcation:
                    transitioned = self._transition(ACBC_ASSESS) or transitioned
                else:
                    transitioned = self._transition(ACBC_FREE) or transitioned

        # --------------------------------------------------------------
        # Final guard against impossible ACBC state
        # --------------------------------------------------------------

        if self.state == ACBC_ASSESS and len(candidates) == 0:
            self.selected_branch = None
            transitioned = self._transition(ACBC_FREE) or transitioned

        # --------------------------------------------------------------
        # Choose control target
        # --------------------------------------------------------------

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
            control_target = self._stall_escape_target(particle_pos, navigator)

        elif self.state == ACBC_RECOVER:
            recovery_target = self._recovery_target(
                particle_pos,
                nominal_target=nominal_target,
            )
            control_target = recovery_target

        else:
            control_target = nominal_target

        # --------------------------------------------------------------
        # Low-level current allocation
        # --------------------------------------------------------------

        if self.state == ACBC_ESCAPE:
            w_progress_eff = 1.85 * w_progress
            w_align_eff = 1.35 * w_align
            w_dist_eff = 1.25 * w_dist
            w_curr_eff = 0.50 * w_curr
            w_smooth_eff = 0.50 * w_smooth
            w_edge_eff = 1.10 * w_edge
            w_overshoot_eff = w_overshoot
        elif self.state == ACBC_RECOVER:
            # During near-wall recovery, reduce forward aggression and increase
            # edge cost. This should help Trial 2 stop scraping the wall.
            w_progress_eff = 0.55 * w_progress
            w_align_eff = 0.75 * w_align
            w_dist_eff = 1.35 * w_dist
            w_curr_eff = w_curr
            w_smooth_eff = w_smooth
            w_edge_eff = 2.00 * w_edge
            w_overshoot_eff = w_overshoot
        else:
            w_progress_eff = w_progress
            w_align_eff = w_align
            w_dist_eff = w_dist
            w_curr_eff = w_curr
            w_smooth_eff = w_smooth
            w_edge_eff = w_edge
            w_overshoot_eff = w_overshoot

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
            previous_state=self.prev_state,
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
            "acbc_recovery_threshold": self.recovery_threshold,
            "acbc_min_wall_safety_fraction": self.min_wall_safety_fraction,
            "acbc_absolute_wall_clearance_threshold_m": (
                self.absolute_wall_clearance_threshold
            ),
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
        }