"""
Authority-Conditioned Branch Commitment controller.

ACBC is a hybrid supervisory controller for magnetic nanoparticle navigation
through bifurcating vascular graphs.

Core idea:
    Do not blindly track waypoints near bifurcations.
    Instead, detect upcoming branch points, estimate whether the target branch
    is magnetically capturable under local flow and wall constraints, and
    explicitly switch between:

        FREE_NAVIGATION
        VIABILITY_ASSESSMENT
        BRANCH_COMMITMENT
        RECOVERY_MODE

The low-level current optimizer is still the existing continuous MPC allocator.
ACBC decides what local target should be given to that allocator.
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
    Hybrid supervisor for branch-aware magnetic navigation.

    Parameters
    ----------
    network:
        RealVesselNetwork-like object.
    magnet_array:
        ElectromagnetArray3D instance.
    workspace:
        WorkspaceVessel wrapper.
    max_current:
        Maximum coil current.
    branch_detection_radius:
        Distance upstream of a bifurcation where ACBC becomes active.
        If None, chosen from vessel span.
    branch_lookahead:
        How far into each branch to evaluate candidate commitment.
        If None, chosen from vessel span.
    authority_threshold:
        Minimum magnetic authority margin required for branch commitment.
    recovery_threshold:
        Margin below which ACBC enters recovery mode.
    """

    def __init__(
        self,
        network,
        magnet_array,
        workspace,
        max_current: float = 8.0,
        branch_detection_radius: Optional[float] = None,
        branch_lookahead: Optional[float] = None,
        authority_threshold: float = 1.05,
        recovery_threshold: float = 0.85,
        min_wall_safety_fraction: float = 0.05,
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

        self.authority_threshold = float(authority_threshold)
        self.recovery_threshold = float(recovery_threshold)
        self.min_wall_safety_fraction = float(min_wall_safety_fraction)

        self.state = ACBC_FREE
        self.prev_state = ACBC_FREE

        self.selected_branch: Optional[BranchCandidate] = None
        self.last_bifurcation_point: Optional[np.ndarray] = None

        self.recovery_count = 0
        self.commitment_count = 0
        self.transition_count = 0
        self.step_count = 0

        self._bifurcation_nodes = self._build_bifurcation_nodes()

    # ------------------------------------------------------------------
    # Graph / bifurcation utilities
    # ------------------------------------------------------------------

    def _build_bifurcation_nodes(self) -> List[Tuple[np.ndarray, List[int]]]:
        """
        Reconstruct bifurcation nodes from shared segment endpoints.

        A bifurcation node is any endpoint shared by three or more segments.
        """
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

    def _nearest_bifurcation(self, pos) -> Tuple[Optional[np.ndarray], List[int], float]:
        pos = np.asarray(pos, dtype=float)

        best_point = None
        best_ids: List[int] = []
        best_dist = np.inf

        for point, ids in self._bifurcation_nodes:
            d = float(np.linalg.norm(pos - point))
            if d < best_dist:
                best_dist = d
                best_point = point
                best_ids = ids

        return best_point, best_ids, best_dist

    def _branch_direction_from_bifurcation(self, seg, bif_point) -> np.ndarray:
        """
        Direction pointing away from bifurcation into the candidate branch.
        """
        bif_point = np.asarray(bif_point, dtype=float)

        ds = np.linalg.norm(seg.start - bif_point)
        de = np.linalg.norm(seg.end - bif_point)

        far_point = seg.end if de > ds else seg.start
        vec = far_point - bif_point
        norm = np.linalg.norm(vec)

        if norm < EPS:
            return seg._axis.copy()

        return vec / norm

    def _candidate_branches(self, bif_point, seg_ids) -> List[Tuple[object, np.ndarray]]:
        seg_by_id = {s.seg_id: s for s in self.network.segments}
        out = []

        for sid in seg_ids:
            seg = seg_by_id[sid]
            direction = self._branch_direction_from_bifurcation(seg, bif_point)
            out.append((seg, direction))

        return out

    # ------------------------------------------------------------------
    # Viability / authority metrics
    # ------------------------------------------------------------------

    def _max_magnetic_projection_speed(self, particle, pos, direction) -> float:
        """
        Estimate maximum achievable magnetic drift speed along `direction`.

        This is deliberately not full MPC. It is an ACBC-specific authority
        probe: sample each actuator at max current and ask how much drift it
        can create along the candidate branch direction.
        """
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

        return max(0.0, best_proj)

    def _wall_safety_for_segment(self, seg, particle) -> Tuple[float, float]:
        """
        Wall safety based on vessel radius minus particle radius.

        Returns absolute margin in meters and normalized margin.
        """
        margin = float(seg.radius - particle.radius)
        fraction = margin / (seg.radius + EPS)
        return margin, fraction

    def _recoverability_score(self, particle_pos, particle) -> float:
        """
        Estimate whether the particle can still recenter away from the wall.

        1. Find current segment.
        2. Compute direction from particle to local centerline.
        3. Estimate magnetic drift available in that direction.
        4. Normalize into [0, ~inf).
        """
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
        final_target,
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

        eval_point = bif_point + direction * min(0.25 * self.branch_lookahead, seg._length)
        eval_point = self.workspace.clip(eval_point)

        flow_v = flow.velocity_at(eval_point, sim_time) if flow is not None else np.zeros(3)
        flow_projection = float(np.dot(flow_v, direction))

        max_mag_projection = self._max_magnetic_projection_speed(
            particle,
            eval_point,
            direction,
        )

        # If flow helps, it contributes positively. If flow opposes, it is a burden.
        opposing_flow = max(0.0, -flow_projection)
        authority_margin = (
            max_mag_projection + max(0.0, flow_projection) + EPS
        ) / (opposing_flow + 1e-6)

        wall_margin, wall_fraction = self._wall_safety_for_segment(seg, particle)

        route_vec = np.asarray(nominal_target, dtype=float) - np.asarray(bif_point, dtype=float)
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
        )

        score = (
            2.5 * route_alignment
            + 2.0 * np.tanh(authority_margin)
            + 1.5 * np.tanh(recoverability)
            + 1.0 * np.clip(wall_fraction, -1.0, 1.0)
            + 0.5 * flow_projection
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

    def _recovery_target(self, particle_pos):
        """
        Recovery target: local centerline projection.

        This is intentionally a discrete ACBC recovery mode, not just a wall
        penalty. It asks the low-level controller to recentralize before branch
        commitment.
        """
        seg, _ = self.network.find_segment(particle_pos)

        if seg is None:
            return np.asarray(particle_pos, dtype=float)

        closest, _, _, _ = seg.closest_point_and_radial(particle_pos)
        return closest

    # ------------------------------------------------------------------
    # Hybrid state machine
    # ------------------------------------------------------------------

    def _transition(self, new_state: str):
        if new_state != self.state:
            self.prev_state = self.state
            self.state = new_state
            self.transition_count += 1
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
        """
        Run one ACBC decision step.

        Returns
        -------
        best_currents, best_force, best_cost, converged, info
        """
        self.step_count += 1

        particle_pos = particle.pos.copy()
        nominal_target = navigator(particle_pos)

        bif_point, bif_ids, bif_dist = self._nearest_bifurcation(particle_pos)
        near_bifurcation = (
            bif_point is not None
            and bif_dist <= self.branch_detection_radius
        )

        candidates: List[BranchCandidate] = []
        selected: Optional[BranchCandidate] = None
        recovery_target = None

        transitioned = False

        # --------------------------------------------------------------
        # State transition logic
        # --------------------------------------------------------------

        if self.state == ACBC_FREE:
            if near_bifurcation:
                transitioned = self._transition(ACBC_ASSESS)
            else:
                transitioned = self._transition(ACBC_FREE)

        if self.state == ACBC_ASSESS:
            if not near_bifurcation:
                self.selected_branch = None
                transitioned = self._transition(ACBC_FREE) or transitioned
            else:
                branches = self._candidate_branches(bif_point, bif_ids)

                for seg, direction in branches:
                    cand = self._evaluate_branch(
                        particle=particle,
                        particle_pos=particle_pos,
                        final_target=final_target,
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
                        self.recovery_count += 1
                        transitioned = self._transition(ACBC_RECOVER) or transitioned
                else:
                    transitioned = self._transition(ACBC_FREE) or transitioned

        elif self.state == ACBC_COMMIT:
            selected = self.selected_branch

            if selected is None:
                transitioned = self._transition(ACBC_FREE) or transitioned
            else:
                dist_from_bif = np.linalg.norm(particle_pos - selected.bifurcation_point)

                # Once sufficiently downstream of the bifurcation, return to nominal navigation.
                if dist_from_bif > 1.25 * self.branch_detection_radius:
                    self.selected_branch = None
                    transitioned = self._transition(ACBC_FREE) or transitioned

        elif self.state == ACBC_RECOVER:
            recovery_target = self._recovery_target(particle_pos)

            recoverability = self._recoverability_score(particle_pos, particle)

            if recoverability >= self.recovery_threshold and not near_bifurcation:
                transitioned = self._transition(ACBC_FREE) or transitioned
            elif recoverability >= self.recovery_threshold and near_bifurcation:
                transitioned = self._transition(ACBC_ASSESS) or transitioned

        # --------------------------------------------------------------
        # Select control target based on hybrid state
        # --------------------------------------------------------------

        if self.state == ACBC_FREE:
            control_target = nominal_target

        elif self.state == ACBC_ASSESS:
            # If assessment has a selected viable candidate, bias toward it.
            if self.selected_branch is not None:
                control_target = self.selected_branch.target_point
            else:
                control_target = nominal_target

        elif self.state == ACBC_COMMIT:
            if self.selected_branch is not None:
                control_target = self.selected_branch.target_point
            else:
                control_target = nominal_target

        elif self.state == ACBC_RECOVER:
            recovery_target = self._recovery_target(particle_pos)
            control_target = recovery_target

        else:
            control_target = nominal_target

        # --------------------------------------------------------------
        # Low-level current allocation
        # --------------------------------------------------------------

        best_currents, best_force, best_cost, converged = continuous_mpc_controller(
            particle=particle,
            particle_pos=particle_pos,
            target_pos=control_target,
            magnet_array=self.magnet_array,
            workspace=self.workspace,
            dt=dt,
            max_current=self.max_current,
            initial_guess=initial_guess,
            w_progress=w_progress,
            w_align=w_align,
            w_dist=w_dist,
            w_curr=w_curr,
            w_smooth=w_smooth,
            w_edge=w_edge,
            w_overshoot=w_overshoot,
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
            bifurcation_point=None if bif_point is None else np.asarray(bif_point, dtype=float),
            selected_branch_seg_id=selected_branch_id,
            selected_authority_margin=float(selected_margin),
            selected_score=float(selected_score),
            target_branch_capturable=bool(target_capturable),
            recovery_target=None if recovery_target is None else np.asarray(recovery_target, dtype=float),
            num_candidates=len(candidates),
        )

        return best_currents, best_force, best_cost, converged, info

    def summary_dict(self):
        return {
            "acbc_steps": self.step_count,
            "acbc_transitions": self.transition_count,
            "acbc_commitments": self.commitment_count,
            "acbc_recoveries": self.recovery_count,
            "acbc_branch_detection_radius_m": self.branch_detection_radius,
            "acbc_branch_lookahead_m": self.branch_lookahead,
            "acbc_authority_threshold": self.authority_threshold,
            "acbc_recovery_threshold": self.recovery_threshold,
        }