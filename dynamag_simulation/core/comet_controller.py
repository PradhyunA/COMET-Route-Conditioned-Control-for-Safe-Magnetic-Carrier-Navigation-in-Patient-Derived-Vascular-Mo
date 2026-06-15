"""
COMET: Commitment-Oriented Measure-Transport Control.

COMET is a post-MPC controller for magnetic carrier navigation through
branching vascular networks. Instead of optimizing a short-horizon,
time-indexed control sequence (MPC) or supervising MPC with a discrete
state machine (ACBC), COMET reasons about *branch-conditioned occupancy
flow*: it maintains a soft commitment measure ``q`` over candidate
downstream branches, transports that measure toward branches that maximize
expected downstream success, and shapes the carrier state into the chosen
branch's capture basin with an allocation layer that solves for coil
currents directly against a local, exact field surrogate.

Architecture (one control step)
-------------------------------
1.  Route/arc state      : project the particle onto the planned centerline
                           polyline; track monotonic arc progress + stalls.
2.  Bifurcation horizon  : next route-relevant bifurcation node ahead.
3.  Branch hypotheses    : for each candidate branch at that node compute
                           downstream value (graph distance to target),
                           authority reserve r, capture probability p, wall
                           risk, and control effort.
4.  Measure transport    : score branches, soften with an entropic
                           (temperature-scheduled) softmax, blend with the
                           previous measure (mirror-descent update), apply
                           commitment hysteresis + dwell to prevent chatter.
5.  Capture-basin control: build a desired velocity that follows the
                           centerline corridor, pre-positions laterally into
                           the q-weighted capture corridor, compensates flow
                           and repels walls.
6.  Current allocation   : minimize ||v(I) - v_des||^2 + effort + smoothness
                           + predicted wall penalty over the coil currents,
                           using a local field surrogate that is exact w.r.t.
                           the simulator's dipole + soft-core model. A final
                           safety backoff guarantees the one-step prediction
                           respects the wall clearance threshold.

The controller only uses information available at run time: current particle
state, the vessel graph/geometry, the planned route, the flow model, and the
magnet array limits. It never reads future ground-truth outcomes.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple

import numpy as np

from .constants import EPS

logger = logging.getLogger(__name__)


COMET_CRUISE = "COMET_CRUISE"        # no bifurcation in horizon
COMET_APPROACH = "COMET_APPROACH"    # soft intent transport upstream of node
COMET_COMMIT = "COMET_COMMIT"        # hardened commitment near/through node
COMET_RESCUE = "COMET_RESCUE"        # high-authority capture-basin recovery
COMET_BOOST = "COMET_BOOST"          # stall recovery boost
COMET_TERMINAL = "COMET_TERMINAL_CAPTURE"
COMET_TERMINAL_PRECISION = "COMET_TERMINAL_PRECISION"


# ----------------------------------------------------------------------
# Configuration
# ----------------------------------------------------------------------

@dataclass
class COMETConfig:
    """Hyperparameters for COMET. Distances in meters, speeds in m/s."""

    # --- intent / measure transport -----------------------------------
    #: Softmax temperature far from a bifurcation (keeps q high-entropy).
    entropy_temperature_far: float = 2.5
    #: Softmax temperature at the bifurcation (q close to argmax).
    entropy_temperature_near: float = 0.30
    #: Arc distance at which commitment hardens (mode -> COMMIT).
    commitment_distance_threshold: float = 0.010
    #: Arc window in which bifurcations are considered "in horizon".
    bifurcation_window: float = 0.045
    #: Reserve (m) below which the controller must hard-commit early.
    hard_commit_reserve_threshold: float = 0.0
    #: Extend the commitment horizon by local flow speed times this reaction
    #: time, capped by ``max_flow_commitment_extension``.
    flow_commitment_reaction_time: float = 0.080
    max_flow_commitment_extension: float = 0.012
    #: Minimum steps to hold a hard commitment before reconsidering.
    min_commitment_dwell_steps: int = 15
    #: Additive score penalty for branches other than the incumbent.
    branch_switch_penalty: float = 1.25
    #: Mirror-descent retention of the previous measure q (0 = memoryless).
    intent_smoothing: float = 0.45
    #: Capture probability of the incumbent below which an emergency
    #: branch switch is allowed even during dwell.
    emergency_capture_floor: float = 0.02

    # --- branch score weights ------------------------------------------
    target_value_weight: float = 3.0
    capture_probability_weight: float = 1.0
    authority_reserve_weight: float = 1.0
    downstream_value_weight: float = 2.5
    control_effort_weight: float = 0.4
    wall_risk_weight: float = 1.5
    #: Length scale (m) for downstream-value exponential decay.
    downstream_value_scale: float = 0.025

    # --- capture probability model -------------------------------------
    #: Logistic gain on normalized reserve.
    capture_reserve_gain: float = 3.0
    #: Logistic gain on velocity/branch alignment.
    capture_alignment_gain: float = 1.0
    #: Extra reserve demand when flow preferentially carries the particle
    #: toward a competing branch.
    wrong_branch_flow_penalty_gain: float = 0.35
    #: Additional loss of confidence for negative reserve close to a node.
    capture_urgency_gain: float = 0.75
    #: Offset (m) past the node where the capture point sits.
    capture_point_offset: float = 0.004
    #: Floor on the along-route approach speed used for time-to-bifurcation.
    min_approach_speed: float = 0.004

    # --- capture-basin shaping control ----------------------------------
    #: Requested corridor-following speed (the allocator delivers the best
    #: achievable velocity toward it within actuator limits).
    cruise_speed: float = 0.180
    #: Gain converting progress need into tracking speed (dimensionless).
    progress_gain: float = 1.0
    #: Centerline pull gain [1/s].
    lateral_capture_gain: float = 8.0
    #: Cap on the centering/capture lateral speed.
    max_lateral_speed: float = 0.040
    #: Multiplier on the adverse (lateral) flow component to cancel.
    flow_compensation_gain: float = 1.0
    #: First-order smoothing of the desired velocity (anti-oscillation).
    damping_gain: float = 0.30
    #: Arc lookahead for the tracking aim point.
    lookahead_distance: float = 0.0080
    #: Radius used as the reference for geometry-scaled distances.
    geometry_reference_radius: float = 0.0030
    geometry_scale_min: float = 0.30
    geometry_scale_max: float = 2.00
    #: Max fraction of the aim distance covered in one step (overshoot guard).
    max_step_fraction: float = 0.70
    #: Adverse-flow wall-riding: when the route opposes the centerline flow
    #: by more than this fraction of magnetic authority, shift the corridor
    #: toward the slow near-wall boundary layer (Poiseuille profile).
    wall_ride_flow_fraction: float = 0.45
    #: Center-to-wall clearance maintained while wall-riding.
    wall_ride_clearance: float = 0.0018
    #: Within this distance of the final target (with line-of-sight inside
    #: the vessel), aim straight at the target and drop corridor shaping.
    terminal_capture_distance: float = 0.012
    terminal_capture_exit_distance: float = 0.020
    #: Radius for the local high-precision magnetic servo.
    terminal_precision_distance: float = 0.0030
    #: Required positive closing-speed reserve before direct terminal capture
    #: may replace route following. Infeasible adverse-flow approaches retain
    #: COMET's boundary-layer wall-riding strategy.
    terminal_direct_authority_margin: float = -0.002
    terminal_precision_kp: float = 55.0
    terminal_precision_kv: float = 0.0
    terminal_precision_max_speed: float = 0.120
    terminal_effort_multiplier: float = 0.01
    terminal_smoothness_multiplier: float = 0.20
    terminal_hold_radius: float = 0.00020
    terminal_cem_iterations: int = 5
    terminal_cem_population: int = 48
    terminal_cem_elites: int = 10
    #: Wall repulsion activates below threshold + this margin.
    wall_repulsion_margin: float = 0.0004
    #: Wall repulsion speed at zero clearance.
    wall_repulsion_speed: float = 0.025

    # --- current allocation ----------------------------------------------
    #: Per-coil current limit; None -> use magnet_array.max_current.
    max_control_norm: Optional[float] = None
    #: Quadratic effort weight (energy regularizer), normalized units.
    current_effort_weight: float = 0.05
    #: Current smoothness weight (chatter regularizer), normalized units.
    current_smoothness_weight: float = 0.01
    #: Weight of the predicted wall-clearance violation penalty.
    wall_penalty_weight: float = 40.0
    #: CEM iterations / population / elite count.
    cem_iterations: int = 3
    cem_population: int = 32
    cem_elites: int = 8
    #: Initial CEM standard deviation as a fraction of the current limit.
    cem_initial_sigma_fraction: float = 0.45
    #: Finite-difference step for the local field surrogate (matches
    #: ElectromagnetArray3D.grad_B_squared default).
    fd_step: float = 1e-4

    # --- stall handling ---------------------------------------------------
    #: Steps without arc progress before BOOST engages.
    stall_window_steps: int = 18
    #: Minimum arc progress (m) that counts as progress.
    stall_progress_epsilon: float = 0.0003
    #: Speed multiplier in BOOST mode.
    boost_speed_multiplier: float = 1.6
    #: Effort weight multiplier in BOOST mode (cheap current).
    boost_effort_multiplier: float = 0.05
    #: BOOST is a bounded attempt, followed by route backtracking/replanning.
    boost_max_steps: int = 45
    recovery_backtrack_steps: int = 60
    recovery_backtrack_radii: float = 4.0
    max_route_replans: int = 3

    # --- emergency branch rescue ----------------------------------------
    rescue_capture_floor: float = 0.30
    rescue_reserve_threshold: float = -0.0010
    rescue_min_flow_scale: float = 4.0
    rescue_target_horizon: float = 0.025
    flow_branch_adaptation_min_scale: float = 4.0
    flow_branch_adaptation_route_horizon: float = 0.120
    terminal_feasibility_min_flow_scale: float = 4.0
    rescue_exit_capture: float = 0.60
    rescue_exit_reserve: float = 0.0010
    rescue_min_dwell_steps: int = 10
    rescue_distance_multiplier: float = 1.5
    rescue_lateral_gain_multiplier: float = 1.8
    rescue_max_lateral_speed: float = 0.070
    rescue_effort_multiplier: float = 0.03

    # --- safety -----------------------------------------------------------
    #: Required center-to-wall clearance (benchmark criterion).
    wall_clearance_threshold: float = 0.0007
    #: Deprecated compatibility field. Hard feasibility uses the single
    #: center-to-wall threshold above; soft wall repulsion has its own margin.
    safety_filter_margin: float = 0.0

    #: Verbose per-step debug logging.
    debug: bool = False


# ----------------------------------------------------------------------
# Step diagnostics
# ----------------------------------------------------------------------

@dataclass
class BranchHypothesis:
    seg_id: int
    direction: np.ndarray
    capture_point: np.ndarray
    branch_radius: float
    downstream_distance: float
    downstream_value: float
    is_target_branch: bool
    required_lateral: float
    reachable_lateral: float
    reserve: float
    capture_probability: float
    wall_risk: float
    effort: float
    score: float
    time_to_bifurcation: float = np.inf
    wrong_branch_flow_penalty: float = 0.0
    q: float = 0.0


@dataclass
class COMETStepInfo:
    mode: str
    previous_mode: str
    transition: bool
    q: Dict[int, float]
    selected_branch_seg_id: Optional[int]
    selected_reserve: float
    selected_capture_probability: float
    dist_to_bifurcation: float
    temperature: float
    entropy: float
    aim_point: np.ndarray
    control_norm: float
    commit_reason: str
    stagnant_steps: int
    wall_clearance_m: float
    route_arc_m: float
    num_candidates: int
    safety_filter_active: bool
    terminal_mode_active: bool
    terminal_stage: str
    target_distance_m: float
    predicted_flow_at_target: np.ndarray
    terminal_correction: np.ndarray
    current_saturation: bool
    final_hold_steps: int
    overshoot_count: int
    direct_terminal_feasible: bool


# ----------------------------------------------------------------------
# Local field surrogate
# ----------------------------------------------------------------------

class _LocalFieldModel:
    """
    Exact local surrogate of ElectromagnetArray3D around one position.

    The array's field is a superposition of per-coil dipole fields (linear
    in the commanded currents) followed by a scalar soft-saturation core
    map applied to the *total* field. Precomputing the per-coil unit fields
    at the 7 finite-difference stencil points makes force/velocity
    evaluation for arbitrary current vectors a couple of matrix products,
    while reproducing field_at()/grad_B_squared() to floating-point
    accuracy.
    """

    _REQUIRED_ATTRS = (
        "positions", "orientations", "moment_per_amp",
        "_a_sq", "_mu_over_4pi",
        "core_saturation_tesla", "core_relative_permeability",
    )

    def __init__(self, magnet_array, pos: np.ndarray, h: float):
        for attr in self._REQUIRED_ATTRS:
            if not hasattr(magnet_array, attr):
                raise AttributeError(
                    "COMET local field surrogate requires "
                    f"ElectromagnetArray3D-compatible attribute '{attr}'."
                )

        pos = np.asarray(pos, dtype=float)
        self.h = float(h)
        self.b_sat = float(magnet_array.core_saturation_tesla)
        self.mu_r = float(magnet_array.core_relative_permeability)

        offsets = np.zeros((7, 3), dtype=float)
        offsets[1, 0] = +h
        offsets[2, 0] = -h
        offsets[3, 1] = +h
        offsets[4, 1] = -h
        offsets[5, 2] = +h
        offsets[6, 2] = -h
        pts = pos[None, :] + offsets                      # (7, 3)

        positions = np.asarray(magnet_array.positions, dtype=float)
        orientations = np.asarray(magnet_array.orientations, dtype=float)
        m_unit = float(magnet_array.moment_per_amp) * orientations  # (n, 3)

        r_vec = pts[:, None, :] - positions[None, :, :]    # (7, n, 3)
        r_sq = np.einsum("kni,kni->kn", r_vec, r_vec)      # (7, n)
        r = np.sqrt(np.maximum(r_sq, 1e-20))
        r_hat = r_vec / r[:, :, None]

        m_dot_rhat = np.einsum("ni,kni->kn", m_unit, r_hat)
        denom = (r_sq + float(magnet_array._a_sq)) ** 1.5

        # Per-amp field of each coil at each stencil point: (7, n, 3)
        self.unit_fields = (
            float(magnet_array._mu_over_4pi)
            * (3.0 * m_dot_rhat[:, :, None] * r_hat - m_unit[None, :, :])
            / denom[:, :, None]
        )

    def _core_magnitude(self, air_mag: np.ndarray) -> np.ndarray:
        return self.b_sat * np.tanh(
            self.mu_r * air_mag / (self.b_sat + EPS)
        )

    def field_and_grad(
        self, currents: np.ndarray,
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        Effective B at the center and grad|B|^2 for a batch of currents.

        currents: (s, n) -> returns B (s, 3), grad (s, 3).
        """
        currents = np.atleast_2d(np.asarray(currents, dtype=float))
        # (s, 7, 3)
        b_air = np.einsum("knj,sn->skj", self.unit_fields, currents)
        air_mag = np.linalg.norm(b_air, axis=2)            # (s, 7)
        eff_mag = self._core_magnitude(air_mag)            # (s, 7)
        b_sq = eff_mag ** 2

        grad = np.stack(
            [
                (b_sq[:, 1] - b_sq[:, 2]) / (2.0 * self.h),
                (b_sq[:, 3] - b_sq[:, 4]) / (2.0 * self.h),
                (b_sq[:, 5] - b_sq[:, 6]) / (2.0 * self.h),
            ],
            axis=1,
        )

        center_air = b_air[:, 0, :]
        center_air_mag = np.maximum(air_mag[:, 0], EPS)
        b_center = center_air * (eff_mag[:, 0] / center_air_mag)[:, None]

        return b_center, grad


def _magnetic_velocity_batch(
    particle, b_center: np.ndarray, grad: np.ndarray,
) -> np.ndarray:
    """
    Vectorized force -> overdamped magnetic drift velocity, replicating
    MagneticParticle3D.magnetic_force + _effective_drag (Schiller-Naumann).
    """
    b_mag = np.linalg.norm(b_center, axis=1)
    linear = b_mag < particle.B_sat

    scale = np.where(
        linear,
        particle.force_prefactor_linear,
        particle.force_prefactor_sat / (2.0 * np.maximum(b_mag, EPS)),
    )
    force = scale[:, None] * grad                            # (s, 3)
    f_mag = np.linalg.norm(force, axis=1)

    gamma_s = particle.drag_stokes
    v = f_mag / gamma_s
    re = particle.rho * v * particle.diameter / particle.eta

    gamma_eff = np.full_like(v, gamma_s)
    active = re >= 0.01

    if np.any(active):
        re_a = re[active]
        f_a = f_mag[active]
        for _ in range(15):
            corr = 1.0 + 0.15 * re_a ** 0.687
            g = gamma_s * corr
            v_a = f_a / g
            re_new = particle.rho * v_a * particle.diameter / particle.eta
            if np.max(np.abs(re_new - re_a) / (re_a + 1e-30)) < 1e-4:
                re_a = re_new
                break
            re_a = re_new
        gamma_eff[active] = gamma_s * (1.0 + 0.15 * re_a ** 0.687)

    safe = f_mag > EPS
    velocity = np.zeros_like(force)
    velocity[safe] = force[safe] / gamma_eff[safe, None]

    return velocity, force


# ----------------------------------------------------------------------
# Controller
# ----------------------------------------------------------------------

class COMETController:
    """
    Commitment-Oriented Measure-Transport controller.

    Exposes the same call style as ACBCController.step(): it receives the
    particle, final target, waypoint navigator, time step, flow model and
    simulation time, and returns (currents, force, cost, converged, info).
    """

    def __init__(
        self,
        network,
        magnet_array,
        workspace,
        max_current: float = 16.0,
        config: Optional[COMETConfig] = None,
        wall_clearance_threshold_m: Optional[float] = None,
    ):
        self.network = network
        self.magnet_array = magnet_array
        self.workspace = workspace
        self.cfg = config or COMETConfig()

        if wall_clearance_threshold_m is not None:
            self.cfg.wall_clearance_threshold = float(wall_clearance_threshold_m)

        self.max_current = float(
            self.cfg.max_control_norm
            if self.cfg.max_control_norm is not None
            else max_current
        )

        self._seg_by_id = {s.seg_id: s for s in network.segments}
        self._bifurcation_nodes = self._build_bifurcation_nodes()
        self._downstream_cache: Dict[Tuple[int, ...], Dict[int, float]] = {}

        # Route polyline cache (per navigator instance).
        self._route_key: Optional[int] = None
        self._route_pts: Optional[np.ndarray] = None
        self._route_cum: Optional[np.ndarray] = None
        self._route_nodes: List[Tuple[float, np.ndarray, List[int]]] = []
        self._route_vertex_idx = 0

        # Intent state.
        self.q: Dict[int, float] = {}
        self._active_node_key: Optional[Tuple[float, float, float]] = None
        self.mode = COMET_CRUISE
        self.incumbent_seg_id: Optional[int] = None
        self.commit_step: Optional[int] = None
        self.commit_reason = ""
        self.transition_count = 0
        self.hard_commit_count = 0
        self.emergency_switch_count = 0

        # Progress / stall state.
        self._arc_now = 0.0
        self._arc_best = 0.0
        self.stagnant_steps = 0
        self._boost_until = 0
        self._recovery_until = 0
        self._recovery_target_arc = 0.0
        self._route_replans = 0

        # Control memory.
        self._v_des_prev = np.zeros(3, dtype=float)
        self._prev_currents = np.zeros(magnet_array.n_magnets, dtype=float)
        self._ride_dir: Optional[np.ndarray] = None

        # Terminal intercept state.
        self._terminal_best = np.inf
        self._terminal_stuck = 0
        self._dash_until = 0
        self._terminal_hold_steps = 0
        self._terminal_overshoot_count = 0
        self._terminal_prev_distance = np.inf
        self._mode_enter_step = 0
        self._flow_branch_adaptation_route_enabled = False

        self.step_count = 0
        self._rng = np.random.default_rng(7)
        self._i_lim = np.full(
            magnet_array.n_magnets, self.max_current, dtype=float,
        )

    # ------------------------------------------------------------------
    # Hardware-aware current limits
    # ------------------------------------------------------------------

    def _effective_current_limits(self) -> np.ndarray:
        """
        Per-coil current limit actually enforceable by the hardware model
        (peak limit, thermal derating, voltage limit, power limit).

        Planning against the effective limit keeps the local surrogate
        consistent with the currents the magnet array will realize.
        """
        m = self.magnet_array
        n = m.n_magnets
        limit = np.full(n, self.max_current, dtype=float)

        if not getattr(m, "enforce_hardware_limits", False):
            return limit

        try:
            derating = m._thermal_current_derating()
            resistances = m._coil_resistances_now()
        except AttributeError:
            return limit

        limit = np.minimum(limit, m.max_current * derating)
        limit = np.minimum(
            limit, m.supply_voltage_limit / np.maximum(resistances, EPS),
        )
        limit = np.minimum(
            limit, np.sqrt(m.max_coil_power_w / np.maximum(resistances, EPS)),
        )
        return np.maximum(limit, 0.0)

    # ------------------------------------------------------------------
    # Graph utilities
    # ------------------------------------------------------------------

    def _build_bifurcation_nodes(self):
        endpoint_map: Dict[Tuple[int, int, int], List[int]] = {}
        point_map: Dict[Tuple[int, int, int], np.ndarray] = {}

        for seg in self.network.segments:
            for p in (seg.start, seg.end):
                key = self.network._node_key(p)
                endpoint_map.setdefault(key, []).append(seg.seg_id)
                point_map[key] = np.asarray(p, dtype=float)

        self._endpoint_map = {
            key: sorted(set(ids)) for key, ids in endpoint_map.items()
        }

        nodes = []
        for key, ids in self._endpoint_map.items():
            if len(ids) >= 3:
                nodes.append((point_map[key], ids))

        return nodes

    def _walk_branch(
        self,
        node_point: np.ndarray,
        first_seg,
        probe_length: float = 0.004,
    ) -> Tuple[Optional[np.ndarray], Optional[np.ndarray], int, float]:
        """
        Walk downstream from a node through (possibly micro-tessellated)
        segments until ``probe_length`` away, a dead end, or another
        branching. Real vessel files often tessellate the centerline into
        sub-millimeter segments, so a single segment's far endpoint is a
        poor estimate of the branch direction.

        Returns (unit_direction, reached_point, last_seg_id, min_radius)
        or (None, None, first_seg_id, radius) when degenerate.
        """
        node_point = np.asarray(node_point, dtype=float)
        point = node_point
        seg = first_seg
        visited = {seg.seg_id}
        reached = None
        min_radius = float(seg.radius)

        for _ in range(64):
            ds = np.linalg.norm(seg.start - point)
            de = np.linalg.norm(seg.end - point)
            far = seg.end if de > ds else seg.start
            reached = np.asarray(far, dtype=float)
            min_radius = min(min_radius, float(seg.radius))

            if np.linalg.norm(reached - node_point) >= probe_length:
                break

            key = self.network._node_key(far)
            next_ids = [
                sid for sid in self._endpoint_map.get(key, [])
                if sid not in visited
            ]
            if len(next_ids) != 1:
                break  # dead end or nested branching: stop the probe here
            seg = self._seg_by_id[next_ids[0]]
            visited.add(seg.seg_id)
            point = reached

        if reached is None:
            return None, None, first_seg.seg_id, min_radius

        direction = reached - node_point
        norm = float(np.linalg.norm(direction))
        if norm < 1e-7:
            return None, None, first_seg.seg_id, min_radius

        return direction / norm, reached, seg.seg_id, min_radius

    def _downstream_distances(self, final_target) -> Dict[int, float]:
        """Graph distance (m) from every segment to the target segment."""
        target_seg, _ = self.network.find_segment(np.asarray(final_target))
        if target_seg is None:
            return {}

        cache_key = (target_seg.seg_id,)
        if cache_key in self._downstream_cache:
            return self._downstream_cache[cache_key]

        import heapq

        dist = {sid: np.inf for sid in self._seg_by_id}
        dist[target_seg.seg_id] = 0.0
        heap = [(0.0, target_seg.seg_id)]

        while heap:
            d, uid = heapq.heappop(heap)
            if d > dist[uid]:
                continue
            for vid, _ in self.network._graph.get(uid, []):
                nd = d + self._seg_by_id[vid]._length
                if nd < dist[vid]:
                    dist[vid] = nd
                    heapq.heappush(heap, (nd, vid))

        self._downstream_cache[cache_key] = dist
        return dist

    # ------------------------------------------------------------------
    # Route polyline utilities
    # ------------------------------------------------------------------

    def _ensure_route(self, navigator, start_pos: np.ndarray, force: bool = False):
        key = id(navigator)
        if key == self._route_key and not force:
            return

        new_navigator = key != self._route_key

        waypoints = getattr(navigator, "waypoints", None)
        if waypoints is None or len(waypoints) == 0:
            pts = np.asarray([start_pos], dtype=float)
        else:
            pts = np.asarray(
                [start_pos] + [np.asarray(w, dtype=float) for w in waypoints],
                dtype=float,
            )

        seg_lens = np.linalg.norm(np.diff(pts, axis=0), axis=1)
        cum = np.concatenate([[0.0], np.cumsum(seg_lens)])

        self._route_key = key
        self._route_pts = pts
        self._route_cum = cum
        self._flow_branch_adaptation_route_enabled = True
        self._route_vertex_idx = 0
        self._arc_now = 0.0
        self._arc_best = 0.0
        self.stagnant_steps = 0
        self.q = {}
        self._active_node_key = None
        self.incumbent_seg_id = None
        self.mode = COMET_CRUISE
        self._v_des_prev = np.zeros(3, dtype=float)
        self._ride_dir = None
        self._terminal_best = np.inf
        self._terminal_stuck = 0
        self._dash_until = 0
        self._terminal_hold_steps = 0
        self._terminal_overshoot_count = 0
        self._terminal_prev_distance = np.inf
        self._mode_enter_step = self.step_count
        self._boost_until = 0
        self._recovery_until = 0
        self._recovery_target_arc = 0.0
        if new_navigator:
            self._route_replans = 0

        # Build route junctions from consecutive selected-route segments.
        # Spatial proximity is insufficient in dense/tessellated networks:
        # nearby unrelated branches otherwise collapse into one mega-node.
        self._route_nodes = []
        route_ids = list(getattr(navigator, "route_segment_ids", []))
        if len(pts) >= 2 and len(route_ids) >= 2:
            for first_id, second_id in zip(route_ids, route_ids[1:]):
                node_point = self.network.shared_endpoint(first_id, second_id)
                if node_point is None:
                    continue
                seg_ids = self.network.endpoint_segment_ids(node_point)
                if len(seg_ids) < 3:
                    continue
                arc, _ = self._project_point_on_route(node_point)
                self._route_nodes.append((arc, node_point, seg_ids))
            self._route_nodes.sort(key=lambda item: item[0])

    def _local_radius(self, pos: np.ndarray) -> float:
        seg, _ = self.network.find_segment(pos)
        return float(seg.radius) if seg is not None else self.cfg.geometry_reference_radius

    def _geometry_scale(self, pos: np.ndarray) -> float:
        return float(np.clip(
            self._local_radius(pos) / max(self.cfg.geometry_reference_radius, EPS),
            self.cfg.geometry_scale_min,
            self.cfg.geometry_scale_max,
        ))

    def _project_point_on_route(self, point) -> Tuple[float, float]:
        pts = self._route_pts
        cum = self._route_cum
        point = np.asarray(point, dtype=float)

        if len(pts) < 2:
            return 0.0, float(np.linalg.norm(point - pts[0]))

        a = pts[:-1]
        b = pts[1:]
        ab = b - a
        ab_len_sq = np.einsum("ij,ij->i", ab, ab)
        ab_len_sq = np.where(ab_len_sq < EPS, EPS, ab_len_sq)
        t = np.clip(
            np.einsum("ij,ij->i", point[None, :] - a, ab) / ab_len_sq,
            0.0, 1.0,
        )
        closest = a + t[:, None] * ab
        d = np.linalg.norm(point[None, :] - closest, axis=1)
        i = int(np.argmin(d))
        arc = float(cum[i] + t[i] * np.sqrt(ab_len_sq[i]))
        return arc, float(d[i])

    def _route_state(self, pos: np.ndarray):
        """
        Project pos onto the route polyline near the tracked vertex.

        Returns (arc, closest_point, tangent, perp_vector).
        """
        pts = self._route_pts
        cum = self._route_cum
        n = len(pts)

        if n == 1:
            tangent = np.array([0.0, 0.0, 1.0])
            return 0.0, pts[0].copy(), tangent, pts[0] - pos

        i0 = max(0, self._route_vertex_idx - 8)
        i1 = min(n - 1, self._route_vertex_idx + 40)

        a = pts[i0:i1]
        b = pts[i0 + 1:i1 + 1]
        ab = b - a
        ab_len = np.linalg.norm(ab, axis=1)
        ab_len = np.where(ab_len < EPS, EPS, ab_len)
        t = np.clip(
            np.einsum("ij,ij->i", pos[None, :] - a, ab) / ab_len ** 2,
            0.0, 1.0,
        )
        closest = a + t[:, None] * ab
        d = np.linalg.norm(pos[None, :] - closest, axis=1)
        j = int(np.argmin(d))

        seg_idx = i0 + j
        arc = float(cum[seg_idx] + t[j] * ab_len[j])
        closest_pt = closest[j]
        tangent = ab[j] / ab_len[j]

        # Advance the window anchor monotonically (small backtrack allowed
        # through the i0 offset above).
        if seg_idx > self._route_vertex_idx:
            self._route_vertex_idx = seg_idx

        perp = closest_pt - pos
        perp = perp - np.dot(perp, tangent) * tangent

        return arc, closest_pt, tangent, perp

    def _route_point_at_arc(self, arc: float) -> np.ndarray:
        pts = self._route_pts
        cum = self._route_cum

        if len(pts) == 1:
            return pts[0].copy()

        arc = float(np.clip(arc, 0.0, cum[-1]))
        i = int(np.searchsorted(cum, arc, side="right") - 1)
        i = min(max(i, 0), len(pts) - 2)
        seg_len = max(cum[i + 1] - cum[i], EPS)
        t = (arc - cum[i]) / seg_len
        return pts[i] + t * (pts[i + 1] - pts[i])

    def _route_tangent_at_arc(self, arc: float) -> np.ndarray:
        pts = self._route_pts
        cum = self._route_cum

        if len(pts) < 2:
            return np.array([0.0, 0.0, 1.0])

        arc = float(np.clip(arc, 0.0, cum[-1] - EPS))
        i = int(np.searchsorted(cum, arc, side="right") - 1)
        i = min(max(i, 0), len(pts) - 2)
        v = pts[i + 1] - pts[i]
        n = np.linalg.norm(v)
        return v / n if n > EPS else np.array([0.0, 0.0, 1.0])

    def _next_bifurcation(self, arc_now: float, window: Optional[float] = None):
        """First route bifurcation ahead of arc_now within the window."""
        window = self.cfg.bifurcation_window if window is None else float(window)
        for arc_node, point, seg_ids in self._route_nodes:
            if arc_node < arc_now - 0.002:
                continue
            if arc_node - arc_now > window:
                return None
            return arc_node, point, seg_ids
        return None

    # ------------------------------------------------------------------
    # Wall clearance helpers
    # ------------------------------------------------------------------

    def _line_of_sight(self, pos: np.ndarray, target: np.ndarray) -> bool:
        """True if the straight segment pos->target stays inside the lumen."""
        margin = self.cfg.wall_clearance_threshold
        for frac in np.linspace(0.0, 1.0, 9):
            p = pos + frac * (target - pos)
            seg, radial = self.network.find_segment(p)
            if seg is None or radial > seg.radius - margin:
                return False
        return True

    def _local_segments(self, pos: np.ndarray, extra_ids=(), probes=()):
        """Candidate segments around ``pos`` and optional probe points.

        Probe points matter for the clearance prediction: a one-step move
        can cross into a neighbouring grid cell whose thin branches are
        invisible from the current cell.
        """
        segs = list(self.network._candidates(pos))
        seen = {s.seg_id for s in segs}
        for p in probes:
            for seg in self.network._candidates(p):
                if seg.seg_id not in seen:
                    segs.append(seg)
                    seen.add(seg.seg_id)
        for sid in extra_ids:
            if sid not in seen:
                segs.append(self._seg_by_id[sid])
                seen.add(sid)
        return segs

    @staticmethod
    def _clearance_batch(segs, positions: np.ndarray) -> np.ndarray:
        """Center-to-wall clearance for a batch of positions.

        Mirrors ``RealVesselNetwork.find_segment`` semantics (the benchmark
        clearance metric): the clearance is taken w.r.t. the segment whose
        *axis* is closest, not the segment giving the most generous
        clearance. This makes the safety filter avoid corner-cutting past
        thin neighbouring branches.
        """
        positions = np.atleast_2d(positions)
        best_radial = np.full(len(positions), np.inf)
        clearance = np.full(len(positions), -np.inf)

        for seg in segs:
            d = positions - seg.start[None, :]
            t = np.clip(
                d @ seg._axis / (seg._length + EPS), 0.0, 1.0,
            )
            closest = seg.start[None, :] + t[:, None] * (
                seg._length * seg._axis[None, :]
            )
            radial = np.linalg.norm(positions - closest, axis=1)
            mask = radial < best_radial
            best_radial[mask] = radial[mask]
            clearance[mask] = seg.radius - radial[mask]

        return clearance

    # ------------------------------------------------------------------
    # Authority / branch evaluation
    # ------------------------------------------------------------------

    def _authority_speed_along(
        self, model: _LocalFieldModel, particle, direction: np.ndarray,
    ) -> float:
        """Max achievable magnetic drift speed along a unit direction."""
        n = self.magnet_array.n_magnets
        i_lim = self._i_lim
        patterns = np.zeros((2 * n + 2, n), dtype=float)
        for i in range(n):
            patterns[2 * i, i] = i_lim[i]
            patterns[2 * i + 1, i] = -i_lim[i]
        patterns[-2, :] = i_lim
        patterns[-1, :] = -i_lim

        b, grad = model.field_and_grad(patterns)
        velocity, _ = _magnetic_velocity_batch(particle, b, grad)
        proj = velocity @ direction
        return float(max(0.0, np.max(proj)))

    def _evaluate_branches(
        self,
        particle,
        pos: np.ndarray,
        model: _LocalFieldModel,
        node_point: np.ndarray,
        seg_ids: List[int],
        arc_to_node: float,
        approach_tangent: np.ndarray,
        v_flow: np.ndarray,
        v_mag_prev: np.ndarray,
        downstream: Dict[int, float],
        flow_adaptation_active: bool = False,
        capture_point_offset: Optional[float] = None,
    ) -> List[BranchHypothesis]:
        cfg = self.cfg
        node_point = np.asarray(node_point, dtype=float)
        high_flow = flow_adaptation_active
        capture_offset = (
            cfg.capture_point_offset
            if capture_point_offset is None else float(capture_point_offset)
        )

        probes = []
        for sid in seg_ids:
            seg = self._seg_by_id[sid]
            direction, reached, last_sid, min_radius = self._walk_branch(
                node_point, seg, probe_length=max(capture_offset, 0.001),
            )
            if direction is None:
                continue

            # Drop the incoming branch (points back along the approach).
            if float(np.dot(direction, approach_tangent)) < -0.5:
                continue

            probes.append((seg, direction, reached, last_sid, min_radius))

        # Merge near-duplicate directions produced by micro-segment stubs.
        candidates = []
        for probe in probes:
            seg, direction, reached, last_sid, min_radius = probe
            duplicate = None
            for i, kept in enumerate(candidates):
                if float(np.dot(direction, kept[1])) > 0.95:
                    duplicate = i
                    break
            if duplicate is None:
                candidates.append(probe)
            else:
                kept = candidates[duplicate]
                d_new = min(
                    downstream.get(seg.seg_id, np.inf),
                    downstream.get(last_sid, np.inf),
                )
                d_old = min(
                    downstream.get(kept[0].seg_id, np.inf),
                    downstream.get(kept[3], np.inf),
                )
                if d_new < d_old:
                    candidates[duplicate] = probe

        if not candidates:
            return []

        # Approach kinematics.
        v_along = float(np.dot(v_flow + v_mag_prev, approach_tangent))
        v_along = max(v_along, cfg.min_approach_speed)
        tau = max(arc_to_node, 0.0) / v_along

        # Flow-carried arrival prediction.
        p_arrival = pos + v_flow * tau
        flow_branch_alignment = [
            float(np.dot(v_flow, direction))
            for _, direction, _, _, _ in candidates
        ]
        best_flow_alignment = max(flow_branch_alignment)

        finite = [
            min(downstream.get(seg.seg_id, np.inf), downstream.get(last_sid, np.inf))
            for seg, _, _, last_sid, _ in candidates
        ]
        d_min = min(finite) if finite else np.inf

        hypotheses = []
        for candidate_idx, (
            (seg, direction, reached, last_sid, min_radius), d_target,
        ) in enumerate(zip(
            candidates, finite,
        )):
            capture_point = node_point + direction * min(
                capture_offset,
                float(np.linalg.norm(reached - node_point)),
            )

            err = capture_point - p_arrival
            err_lat = err - np.dot(err, approach_tangent) * approach_tangent
            required_geometric = float(np.linalg.norm(err_lat))

            # Constant local-flow prediction already appears in ``err_lat``.
            # Add a smaller uncertainty reserve when the stream aligns more
            # strongly with a competing branch; this prevents optimistic
            # capture estimates near fast bifurcations without double-counting
            # the full drift.
            wrong_flow_advantage = max(
                0.0,
                best_flow_alignment - flow_branch_alignment[candidate_idx],
            )
            wrong_flow_penalty = (
                cfg.wrong_branch_flow_penalty_gain
                * wrong_flow_advantage
                * tau
                if high_flow else 0.0
            )
            required = required_geometric + wrong_flow_penalty

            if required_geometric > EPS:
                lat_dir = err_lat / required_geometric
                v_lat_max = self._authority_speed_along(
                    model, particle, lat_dir,
                )
            else:
                v_lat_max = self._authority_speed_along(
                    model, particle, direction,
                )

            reachable = v_lat_max * (tau + 2.0 * 0.02)
            reserve = reachable - required

            v_total = v_flow + v_mag_prev
            v_norm = np.linalg.norm(v_total)
            alignment = (
                float(np.dot(v_total / v_norm, direction))
                if v_norm > EPS else 0.0
            )

            seg_scale = max(min_radius, 1e-4)
            z = (
                cfg.capture_reserve_gain * np.tanh(reserve / seg_scale)
                + cfg.capture_alignment_gain * alignment
                - (cfg.capture_urgency_gain if high_flow else 0.0)
                * np.clip(1.0 - tau / 0.25, 0.0, 1.0)
                * max(0.0, -reserve / seg_scale)
            )
            p_capture = float(1.0 / (1.0 + np.exp(-z)))

            min_safe_radius = cfg.wall_clearance_threshold
            wall_risk = float(
                max(0.0, (min_safe_radius - min_radius) / min_safe_radius)
            )

            effort = float(required / (reachable + 1e-6))

            if np.isfinite(d_target) and np.isfinite(d_min):
                downstream_value = float(
                    np.exp(-(d_target - d_min) / cfg.downstream_value_scale)
                )
            else:
                downstream_value = 0.0

            is_target = bool(np.isfinite(d_target) and d_target <= d_min + EPS)

            score = (
                cfg.target_value_weight * (1.0 if is_target else 0.0)
                + cfg.capture_probability_weight * float(np.log(p_capture + 1e-6))
                + cfg.authority_reserve_weight * float(np.tanh(reserve / seg_scale))
                + cfg.downstream_value_weight * downstream_value
                - cfg.control_effort_weight * effort
                - cfg.wall_risk_weight * wall_risk
            )
            if not np.isfinite(d_target):
                score -= 6.0

            hypotheses.append(BranchHypothesis(
                seg_id=seg.seg_id,
                direction=direction,
                capture_point=capture_point,
                branch_radius=float(min_radius),
                downstream_distance=float(d_target),
                downstream_value=downstream_value,
                is_target_branch=is_target,
                required_lateral=required,
                reachable_lateral=reachable,
                reserve=float(reserve),
                capture_probability=p_capture,
                wall_risk=wall_risk,
                effort=effort,
                score=float(score),
                time_to_bifurcation=float(tau),
                wrong_branch_flow_penalty=float(wrong_flow_penalty),
            ))

        return hypotheses

    # ------------------------------------------------------------------
    # Measure transport / commitment
    # ------------------------------------------------------------------

    @staticmethod
    def _harden(q: Dict[int, float], leader_sid: int, floor: float = 0.02):
        """Concentrate the measure on the committed branch (in place)."""
        others = [sid for sid in q if sid != leader_sid]
        if not others:
            q[leader_sid] = 1.0
            return
        for sid in others:
            q[sid] = floor / len(others)
        q[leader_sid] = 1.0 - floor

    def _update_measure(
        self,
        hypotheses: List[BranchHypothesis],
        node_key,
        arc_to_node: float,
        commitment_distance: float,
    ) -> Tuple[Dict[int, float], float, str]:
        cfg = self.cfg

        if node_key != self._active_node_key:
            # New bifurcation: reset measure and commitment memory.
            self._active_node_key = node_key
            self.q = {}
            self.incumbent_seg_id = None
            self.commit_step = None
            self.commit_reason = ""

        leader_prev = self.incumbent_seg_id
        scores = np.array([h.score for h in hypotheses], dtype=float)

        # Anti-chatter: incumbent advantage.
        if leader_prev is not None:
            for i, h in enumerate(hypotheses):
                if h.seg_id != leader_prev:
                    scores[i] -= cfg.branch_switch_penalty

        # Temperature schedule: geometric interpolation in proximity.
        prox = float(np.clip(
            1.0 - arc_to_node / max(cfg.bifurcation_window, EPS), 0.0, 1.0,
        ))
        temperature = float(
            cfg.entropy_temperature_far
            * (cfg.entropy_temperature_near / cfg.entropy_temperature_far)
            ** prox
        )

        reason = ""
        lead_idx = int(np.argmax(scores))
        lead = hypotheses[lead_idx]

        hard_commit = False
        if arc_to_node < commitment_distance:
            hard_commit = True
            reason = f"distance<{commitment_distance * 1000:.1f}mm"
        elif lead.reserve < cfg.hard_commit_reserve_threshold:
            hard_commit = True
            reason = f"reserve={lead.reserve * 1000:.1f}mm"

        if hard_commit:
            temperature = min(temperature, cfg.entropy_temperature_near)

        logits = scores / max(temperature, 1e-3)
        logits -= np.max(logits)
        q_raw = np.exp(logits)
        q_raw /= np.sum(q_raw)

        # Mirror-descent (entropic transport) blend with the previous measure.
        rho = cfg.intent_smoothing if not hard_commit else 0.15
        q_new = {}
        for i, h in enumerate(hypotheses):
            q_prev = self.q.get(h.seg_id, 1.0 / len(hypotheses))
            q_new[h.seg_id] = (q_prev ** rho) * (q_raw[i] ** (1.0 - rho))

        total = sum(q_new.values())
        for sid in q_new:
            q_new[sid] /= max(total, EPS)

        # Commitment with dwell + hysteresis.
        dwell_active = (
            self.commit_step is not None
            and self.step_count - self.commit_step
            < cfg.min_commitment_dwell_steps
        )

        if dwell_active and self.incumbent_seg_id is not None:
            incumbent = next(
                (h for h in hypotheses if h.seg_id == self.incumbent_seg_id),
                None,
            )
            emergency = (
                incumbent is not None
                and incumbent.capture_probability < cfg.emergency_capture_floor
                and any(
                    h.reserve > 0 and np.isfinite(h.downstream_distance)
                    for h in hypotheses if h.seg_id != incumbent.seg_id
                )
            )
            if not emergency and incumbent is not None:
                # Hold the committed branch.
                self._harden(q_new, incumbent.seg_id)
                self.q = q_new
                return q_new, temperature, self.commit_reason
            if emergency:
                self.emergency_switch_count += 1
                reason = "emergency_override"
                logger.debug(
                    "COMET emergency branch override at step=%s", self.step_count,
                )

        if hard_commit:
            if self.incumbent_seg_id != lead.seg_id:
                self.incumbent_seg_id = lead.seg_id
                self.commit_step = self.step_count
                self.commit_reason = reason
                self.hard_commit_count += 1
                logger.debug(
                    "COMET hard commit: branch=%s reason=%s step=%s",
                    lead.seg_id, reason, self.step_count,
                )
            self._harden(q_new, lead.seg_id)
        else:
            # Soft leadership tracking (upstream).
            self.incumbent_seg_id = max(q_new, key=q_new.get)

        self.q = q_new
        return q_new, temperature, reason or self.commit_reason

    # ------------------------------------------------------------------
    # Current allocation
    # ------------------------------------------------------------------

    def _allocation_seeds(
        self, pos: np.ndarray, v_des_dir: np.ndarray, initial_guess,
    ) -> np.ndarray:
        n = self.magnet_array.n_magnets
        i_lim = self._i_lim
        seeds = [np.zeros(n, dtype=float), self._prev_currents.copy()]

        if initial_guess is not None:
            seeds.append(np.asarray(initial_guess, dtype=float))

        pull = self.magnet_array.positions - pos[None, :]
        pull /= np.linalg.norm(pull, axis=1)[:, None] + EPS
        proj = pull @ v_des_dir

        max_abs = float(np.max(np.abs(proj)))
        if max_abs > EPS:
            signed = i_lim * proj / max_abs
            seeds.extend([
                signed, 0.5 * signed, -signed, -0.5 * signed,
            ])

        pos_proj = np.clip(proj, 0.0, None)
        if float(np.max(pos_proj)) > EPS:
            rect = i_lim * pos_proj / float(np.max(pos_proj))
            seeds.extend([rect, 0.5 * rect, -rect])

        best = int(np.argmax(np.abs(proj)))
        single = np.zeros(n)
        single[best] = i_lim[best] * np.sign(proj[best] or 1.0)
        seeds.extend([single, -single])

        return np.clip(np.asarray(seeds, dtype=float), -i_lim, i_lim)

    def _allocate_currents(
        self,
        particle,
        pos: np.ndarray,
        model: _LocalFieldModel,
        v_des_mag: np.ndarray,
        v_flow: np.ndarray,
        dt: float,
        segs,
        effort_weight: float,
        initial_guess=None,
        flow=None,
        sim_time: float = 0.0,
        smoothness_weight: Optional[float] = None,
        cem_iterations: Optional[int] = None,
        cem_population: Optional[int] = None,
        cem_elites: Optional[int] = None,
        terminal_target: Optional[np.ndarray] = None,
    ) -> Tuple[np.ndarray, float, bool]:
        cfg = self.cfg
        n = self.magnet_array.n_magnets
        i_lim = self._i_lim
        i_max = max(float(np.mean(i_lim)), EPS)
        smoothness_weight = (
            cfg.current_smoothness_weight
            if smoothness_weight is None else float(smoothness_weight)
        )
        cem_iterations = (
            cfg.cem_iterations if cem_iterations is None else int(cem_iterations)
        )
        cem_population = (
            cfg.cem_population if cem_population is None else int(cem_population)
        )
        cem_elites = cfg.cem_elites if cem_elites is None else int(cem_elites)

        v_norm_sq = max(float(np.dot(v_des_mag, v_des_mag)), 1e-8)
        des_dir = (
            v_des_mag / np.sqrt(v_norm_sq)
            if v_norm_sq > 1e-12 else np.array([0.0, 0.0, 1.0])
        )
        required_clear = cfg.wall_clearance_threshold

        # Only segments whose axis can plausibly become the closest one
        # within the prediction horizon matter for the clearance terms.
        near_segs = [
            seg for seg in segs
            if seg.closest_point_and_radial(pos)[2] - seg.radius < 0.015
        ]
        if near_segs:
            segs = near_segs

        # Precompute segment geometry once; the clearance terms dominate
        # the allocation cost when evaluated per segment in Python.
        seg_start = np.array([s.start for s in segs], dtype=float)
        seg_axis = np.array([s._axis for s in segs], dtype=float)
        seg_len = np.array([s._length for s in segs], dtype=float)
        seg_rad = np.array([s.radius for s in segs], dtype=float)
        seg_scaled_axis = seg_len[:, None] * seg_axis

        def clearance_fn(positions: np.ndarray) -> np.ndarray:
            """Closest-axis clearance (find_segment semantics), vectorized."""
            d = positions[:, None, :] - seg_start[None, :, :]
            t = np.einsum("nsd,sd->ns", d, seg_axis) / (seg_len[None, :] + EPS)
            np.clip(t, 0.0, 1.0, out=t)
            closest = seg_start[None, :, :] + t[..., None] * seg_scaled_axis[None, :, :]
            radial = np.linalg.norm(positions[:, None, :] - closest, axis=2)
            idx = np.argmin(radial, axis=1)
            rows = np.arange(len(positions))
            return seg_rad[idx] - radial[rows, idx]

        def cost_batch(currents: np.ndarray) -> np.ndarray:
            b, grad = model.field_and_grad(currents)
            v_mag, _ = _magnetic_velocity_batch(particle, b, grad)

            track = np.sum((v_mag - v_des_mag[None, :]) ** 2, axis=1) / v_norm_sq
            if terminal_target is not None:
                p_half = pos[None, :] + (
                    v_mag + v_flow[None, :]
                ) * (0.5 * dt)
                # The precision servo is deliberately quasi-static. Reuse
                # the measured local drift inside CEM; querying the network
                # for every candidate made terminal-only cases hundreds of
                # milliseconds slower without adding useful horizon fidelity.
                p_end = p_half + (
                    v_mag + v_flow[None, :]
                ) * (0.5 * dt)
                target_scale_sq = max(
                    float(np.sum((terminal_target - pos) ** 2)), 1e-8,
                )
                endpoint = np.sum(
                    (p_end - terminal_target[None, :]) ** 2, axis=1,
                ) / target_scale_sq
                track = 0.20 * track + 4.0 * endpoint
            effort = effort_weight * np.sum(currents ** 2, axis=1) / (n * i_max ** 2)
            smooth = smoothness_weight * np.sum(
                (currents - self._prev_currents[None, :]) ** 2, axis=1,
            ) / (n * i_max ** 2)

            # Wall penalty over a short constant-velocity horizon: narrow
            # clearance slivers at junctions (thin side-branch capture
            # regions) are invisible to a one-step prediction until the
            # particle is already inside them.
            v_tot = v_mag + v_flow[None, :]
            wall = np.zeros(len(currents))
            for k, discount in ((1, 1.0), (2, 0.6), (3, 0.35)):
                p_k = pos[None, :] + v_tot * (k * dt)
                clearance = clearance_fn(p_k)
                violation = np.maximum(0.0, required_clear - clearance)
                wall += discount * (violation / required_clear) ** 2
            wall *= cfg.wall_penalty_weight

            return track + effort + smooth + wall

        seeds = self._allocation_seeds(pos, des_dir, initial_guess)
        seed_costs = cost_batch(seeds)

        best_idx = int(np.argmin(seed_costs))
        best_currents = seeds[best_idx].copy()
        best_cost = float(seed_costs[best_idx])

        # Cross-entropy refinement around the best seed.
        mean = best_currents.copy()
        sigma = np.full(n, cfg.cem_initial_sigma_fraction * i_max)

        for _ in range(cem_iterations):
            samples = mean[None, :] + sigma[None, :] * self._rng.standard_normal(
                (cem_population, n),
            )
            samples = np.clip(samples, -i_lim, i_lim)
            samples[0] = mean  # keep the running mean in the pool
            costs = cost_batch(samples)

            order = np.argsort(costs)
            elites = samples[order[:cem_elites]]
            mean = np.mean(elites, axis=0)
            sigma = np.std(elites, axis=0) + 0.02 * i_max

            if costs[order[0]] < best_cost:
                best_cost = float(costs[order[0]])
                best_currents = samples[order[0]].copy()

        # Predictive wall-safety backoff. The prediction uses two half
        # steps with a mid-step flow refresh: junction flow can rotate
        # substantially within one control step at high flow scales, so a
        # single-endpoint prediction misses mid-step wall grazes.
        def predicted_clearance(v_mag_batch: np.ndarray) -> np.ndarray:
            p_half = pos[None, :] + (v_mag_batch + v_flow[None, :]) * (0.5 * dt)
            if flow is not None:
                v_half = np.array([
                    flow.velocity_at(p, sim_time + 0.5 * dt) for p in p_half
                ], dtype=float)
            else:
                v_half = np.zeros_like(p_half)
            p_full = p_half + (v_mag_batch + v_half) * (0.5 * dt)
            c_half = clearance_fn(p_half)
            c_full = clearance_fn(p_full)
            return np.minimum(c_half, c_full)

        safety_active = False
        b, grad = model.field_and_grad(best_currents[None, :])
        v_mag, _ = _magnetic_velocity_batch(particle, b, grad)
        clearance = float(predicted_clearance(v_mag)[0])

        if clearance < cfg.wall_clearance_threshold:
            safety_active = True
            alphas = np.linspace(0.0, 1.0, 9)
            scaled = alphas[:, None] * best_currents[None, :]
            b, grad = model.field_and_grad(scaled)
            v_a, _ = _magnetic_velocity_batch(particle, b, grad)
            c_a = predicted_clearance(v_a)

            safe = c_a >= cfg.wall_clearance_threshold
            if np.any(safe):
                # Largest backoff factor that restores predicted clearance.
                alpha = float(alphas[np.where(safe)[0].max()])
                best_currents = alpha * best_currents
            else:
                # Flow-dominated danger: no scaling is safe, so keep the
                # candidate that maximizes predicted clearance (the cost
                # already pushes away from the wall).
                alpha = float(alphas[int(np.argmax(c_a))])
                best_currents = alpha * best_currents

        return best_currents, best_cost, safety_active

    # ------------------------------------------------------------------
    # Main step
    # ------------------------------------------------------------------

    def step(
        self,
        particle,
        final_target,
        navigator,
        dt: float,
        flow,
        sim_time: float,
        initial_guess=None,
    ):
        cfg = self.cfg
        self.step_count += 1
        prev_mode = self.mode

        pos = np.asarray(particle.pos, dtype=float).copy()
        final_target = np.asarray(final_target, dtype=float)

        # Keep the navigator's internal progress state in sync (its return
        # value is not used as the control target; COMET builds its own).
        navigator(pos)
        self._ensure_route(navigator, pos)

        arc_now, route_pt, tangent, perp = self._route_state(pos)
        self._arc_now = arc_now
        geometry_scale = self._geometry_scale(pos)
        bifurcation_window = cfg.bifurcation_window * geometry_scale
        commitment_distance = cfg.commitment_distance_threshold * geometry_scale
        lookahead_distance = cfg.lookahead_distance * geometry_scale
        capture_point_offset = cfg.capture_point_offset * geometry_scale
        terminal_capture_distance = cfg.terminal_capture_distance * geometry_scale

        # --- stall accounting -----------------------------------------
        if arc_now > self._arc_best + cfg.stall_progress_epsilon:
            self._arc_best = arc_now
            self.stagnant_steps = 0
        else:
            self.stagnant_steps += 1

        dist_to_final = float(np.linalg.norm(final_target - pos))
        stalled = (
            self.stagnant_steps >= cfg.stall_window_steps
            and dist_to_final > 0.002
        )

        # Bounded recovery sequence: boost, backtrack, then replan around the
        # next route segment. Progress immediately cancels the sequence.
        if not stalled:
            self._boost_until = 0
            self._recovery_until = 0
        elif self._recovery_until and self.step_count >= self._recovery_until:
            route_ids = list(getattr(navigator, "route_segment_ids", []))
            current_seg, _ = self.network.find_segment(pos)
            blocked = []
            if current_seg is not None and current_seg.seg_id in route_ids:
                index = route_ids.index(current_seg.seg_id)
                if index + 1 < len(route_ids) - 1:
                    blocked = [route_ids[index + 1]]
            if (
                hasattr(navigator, "replan")
                and self._route_replans < cfg.max_route_replans
            ):
                plan = navigator.replan(pos, blocked_segment_ids=blocked)
                if plan.feasible:
                    self._route_replans += 1
                    self._ensure_route(navigator, pos, force=True)
                    arc_now, route_pt, tangent, perp = self._route_state(pos)
                    self._arc_now = arc_now
            self._boost_until = 0
            self._recovery_until = 0
            self.stagnant_steps = 0
            stalled = False
        elif self._boost_until and self.step_count >= self._boost_until:
            self._boost_until = 0
            self._recovery_until = self.step_count + cfg.recovery_backtrack_steps
            self._recovery_target_arc = max(
                0.0,
                arc_now - cfg.recovery_backtrack_radii * self._local_radius(pos),
            )
        elif self._boost_until == 0 and self._recovery_until == 0:
            self._boost_until = self.step_count + cfg.boost_max_steps

        boost_active = self.step_count < self._boost_until
        recovery_active = self.step_count < self._recovery_until

        # Terminal capture: the target is close and directly reachable
        # inside the lumen, so route/branch shaping only gets in the way
        # (e.g. a route that folds back along the vessel axis).
        terminal_radius = geometry_scale * (
            cfg.terminal_capture_exit_distance
            if prev_mode in (COMET_TERMINAL, COMET_TERMINAL_PRECISION)
            else cfg.terminal_capture_distance
        )
        terminal_was_active = prev_mode in (
            COMET_TERMINAL, COMET_TERMINAL_PRECISION,
        )
        terminal_hysteresis_active = (
            terminal_was_active
        )
        terminal_line_of_sight = self._line_of_sight(pos, final_target)
        remaining_route_arc = max(float(self._route_cum[-1] - arc_now), 0.0)
        route_terminal_candidate = remaining_route_arc < terminal_capture_distance
        direct_terminal_candidate = (
            dist_to_final < terminal_radius
            and (
                terminal_was_active
                or terminal_line_of_sight
            )
        )
        terminal_candidate = direct_terminal_candidate or route_terminal_candidate

        # --- flow / local field model ----------------------------------
        v_flow = (
            np.asarray(flow.velocity_at(pos, sim_time), dtype=float)
            if flow is not None else np.zeros(3)
        )
        model = _LocalFieldModel(self.magnet_array, pos, cfg.fd_step)
        self._i_lim = self._effective_current_limits()

        v_mag_prev = np.asarray(particle.vel, dtype=float)

        # Geometric line of sight is not sufficient under adverse flow. Keep
        # route following and boundary-layer wall riding until direct target
        # closure has positive magnetic-authority reserve. Very close states
        # always enter the precision servo so it can cancel local drift.
        direct_terminal_feasible = False
        terminal_closing_reserve = -np.inf
        target_authority = 0.0
        target_direction = np.zeros(3)
        if dist_to_final > EPS:
            target_direction = (final_target - pos) / dist_to_final
        if direct_terminal_candidate and dist_to_final > EPS:
            target_authority = self._authority_speed_along(
                model, particle, target_direction,
            )
            terminal_closing_reserve = (
                target_authority + float(np.dot(v_flow, target_direction))
            )
            high_flow_terminal_gate = (
                float(getattr(self.network, "flow_scale", 1.0))
                >= cfg.terminal_feasibility_min_flow_scale
            )
            direct_terminal_feasible = (
                not high_flow_terminal_gate
                or terminal_hysteresis_active
                or
                terminal_closing_reserve
                >= (
                    -0.004
                    if prev_mode in (COMET_TERMINAL, COMET_TERMINAL_PRECISION)
                    else cfg.terminal_direct_authority_margin
                )
                or dist_to_final <= cfg.terminal_precision_distance
            )
        terminal = route_terminal_candidate or (
            direct_terminal_candidate and direct_terminal_feasible
        )
        direct_terminal = terminal and terminal_line_of_sight
        terminal_precision = (
            direct_terminal
            and dist_to_final <= cfg.terminal_precision_distance
        )
        terminal_stage = (
            "precision" if terminal_precision else "capture" if terminal else ""
        )
        v_flow_target = (
            np.asarray(flow.velocity_at(final_target, sim_time), dtype=float)
            if flow is not None and terminal_candidate else np.zeros(3)
        )

        # --- branch hypotheses + measure transport ----------------------
        nxt = self._next_bifurcation(arc_now, bifurcation_window)
        hypotheses: List[BranchHypothesis] = []
        q: Dict[int, float] = {}
        temperature = cfg.entropy_temperature_far
        commit_reason = ""
        dist_to_bif = np.inf
        selected: Optional[BranchHypothesis] = None
        high_flow = (
            float(getattr(self.network, "flow_scale", 1.0))
            >= cfg.flow_branch_adaptation_min_scale
        )
        dynamic_commitment_distance = commitment_distance
        if high_flow:
            dynamic_commitment_distance += min(
                cfg.max_flow_commitment_extension,
                cfg.flow_commitment_reaction_time * float(np.linalg.norm(v_flow)),
            )

        downstream = self._downstream_distances(final_target)

        if nxt is not None:
            arc_node, node_point, seg_ids = nxt
            dist_to_bif = max(arc_node - arc_now, 0.0)
            approach_tangent = self._route_tangent_at_arc(
                max(arc_node - 0.002, 0.0),
            )

            hypotheses = self._evaluate_branches(
                particle=particle,
                pos=pos,
                model=model,
                node_point=node_point,
                seg_ids=seg_ids,
                arc_to_node=dist_to_bif,
                approach_tangent=approach_tangent,
                v_flow=v_flow,
                v_mag_prev=v_mag_prev,
                downstream=downstream,
                flow_adaptation_active=high_flow,
                capture_point_offset=capture_point_offset,
            )

            if hypotheses:
                node_key = self.network._node_key(node_point)
                q, temperature, commit_reason = self._update_measure(
                    hypotheses,
                    node_key,
                    dist_to_bif,
                    dynamic_commitment_distance,
                )
                for h in hypotheses:
                    h.q = q.get(h.seg_id, 0.0)
                selected = max(hypotheses, key=lambda h: h.q)

                if self.commit_step is not None or (
                    dist_to_bif < dynamic_commitment_distance
                ) or (
                    selected.reserve < cfg.hard_commit_reserve_threshold
                ):
                    self.mode = COMET_COMMIT
                else:
                    self.mode = COMET_APPROACH
            else:
                self.mode = COMET_CRUISE
        else:
            # Left the bifurcation horizon: release commitment memory.
            self.mode = COMET_CRUISE
            self._active_node_key = None
            self.incumbent_seg_id = None
            self.commit_step = None

        rescue_entry = bool(
            selected is not None
            and float(getattr(self.network, "flow_scale", 1.0))
            >= cfg.rescue_min_flow_scale
            and dist_to_final < cfg.rescue_target_horizon
            and dist_to_bif
            < cfg.rescue_distance_multiplier * dynamic_commitment_distance
            and (
                selected.reserve < cfg.rescue_reserve_threshold
                or selected.capture_probability < cfg.rescue_capture_floor
            )
        )
        rescue_hold = bool(
            prev_mode == COMET_RESCUE
            and selected is not None
            and float(getattr(self.network, "flow_scale", 1.0))
            >= cfg.rescue_min_flow_scale
            and dist_to_final < cfg.rescue_target_horizon
            and dist_to_bif
            < cfg.rescue_distance_multiplier * dynamic_commitment_distance
            and (
                self.step_count - self._mode_enter_step
                < cfg.rescue_min_dwell_steps
                or selected.reserve < cfg.rescue_exit_reserve
                or selected.capture_probability < cfg.rescue_exit_capture
            )
        )
        rescue = rescue_entry or rescue_hold
        if rescue:
            self.mode = COMET_RESCUE

        if boost_active:
            self.mode = COMET_BOOST
        elif recovery_active:
            self.mode = COMET_RESCUE
        if terminal:
            self.mode = (
                COMET_TERMINAL_PRECISION
                if terminal_precision else COMET_TERMINAL
            )

        # --- capture-basin shaping: desired velocity ---------------------
        prox = float(np.clip(
            1.0 - dist_to_bif / max(bifurcation_window, EPS), 0.0, 1.0,
        )) if np.isfinite(dist_to_bif) else 0.0

        lookahead = lookahead_distance
        effort_weight = cfg.current_effort_weight
        boost_factor = 1.0

        if self.mode == COMET_BOOST:
            lookahead *= 2.0
            boost_factor = cfg.boost_speed_multiplier
            effort_weight *= cfg.boost_effort_multiplier
        elif self.mode == COMET_RESCUE:
            effort_weight *= cfg.rescue_effort_multiplier
        elif terminal_precision:
            effort_weight *= cfg.terminal_effort_multiplier
        elif terminal and stalled:
            # Stalled in the endgame (e.g. fighting peak pulsatile flow):
            # relax the effort penalty so the allocator uses full authority.
            effort_weight *= cfg.boost_effort_multiplier

        aim_arc = self._recovery_target_arc if recovery_active else arc_now + lookahead
        aim_route = self._route_point_at_arc(aim_arc)
        aim_tangent = self._route_tangent_at_arc(aim_arc)

        # Capture-basin shaping: shift the (always-forward) route aim point
        # laterally toward the q-weighted capture corridor. With a soft q the
        # weighted capture point sits between branch entrances, preserving
        # optionality; as q hardens it converges to the selected branch's
        # entry corridor. The offset is purely lateral so it can never stall
        # forward progress, and it is capped by the local safe clearance.
        if recovery_active and not terminal:
            aim = aim_route
        elif terminal:
            aim = (
                final_target.copy()
                if terminal_line_of_sight
                else self._route_point_at_arc(arc_now + lookahead)
            )

            if (
                terminal_precision
                and np.isfinite(self._terminal_prev_distance)
                and dist_to_final > self._terminal_prev_distance + 2e-5
            ):
                self._terminal_overshoot_count += 1
            self._terminal_prev_distance = dist_to_final
            if dist_to_final <= cfg.terminal_hold_radius:
                self._terminal_hold_steps += 1
            else:
                self._terminal_hold_steps = 0

            # Track terminal progress. If the direct pursuit has stalled
            # (the particle hovers at the standoff where authority equals
            # the adverse stream), run a timed intercept dash: aim upstream
            # of the target by the drift expected during the crossing so
            # the stream sweeps the particle through the tolerance sphere.
            if dist_to_final < self._terminal_best - 1e-5:
                self._terminal_best = dist_to_final
                self._terminal_stuck = 0
            else:
                self._terminal_stuck += 1

            if (
                self.step_count >= self._dash_until
                and self._terminal_stuck > 25
            ):
                self._dash_until = self.step_count + 30
                self._terminal_stuck = 0

            if (
                not terminal_precision
                and terminal_line_of_sight
                and self.step_count < self._dash_until
                and flow is not None
            ):
                v_flow_tgt = v_flow_target
                to_tgt = final_target - pos
                tgt_dist = float(np.linalg.norm(to_tgt))
                if tgt_dist > EPS and float(np.linalg.norm(v_flow_tgt)) > 0.01:
                    closing = max(
                        self._authority_speed_along(
                            model, particle, to_tgt / tgt_dist,
                        ),
                        0.03,
                    )
                    lead = -v_flow_tgt * (tgt_dist / closing)
                    lead_norm = float(np.linalg.norm(lead))
                    if lead_norm > 0.010:
                        lead *= 0.010 / lead_norm
                    aim = final_target + lead
        elif hypotheses and selected is not None:
            if rescue:
                aim_capture = selected.capture_point.copy()
            else:
                aim_capture = np.zeros(3)
                for h in hypotheses:
                    aim_capture += h.q * h.capture_point

            offset = aim_capture - aim_route
            offset_lat = offset - np.dot(offset, aim_tangent) * aim_tangent

            seg_aim, _ = self.network.find_segment(aim_route)
            if seg_aim is not None:
                max_offset = max(
                    seg_aim.radius - cfg.wall_clearance_threshold, 0.0,
                )
            else:
                max_offset = 0.002
            off_norm = float(np.linalg.norm(offset_lat))
            if off_norm > max_offset and off_norm > EPS:
                offset_lat *= max_offset / off_norm

            capture_blend = 1.0 if rescue else prox
            aim = aim_route + capture_blend * offset_lat
        else:
            aim = aim_route

        # --- adverse-flow wall riding ----------------------------------
        # Poiseuille flow vanishes at the vessel wall. When the route runs
        # against a centerline flow faster than the magnets can fight, the
        # corridor target is shifted toward the slow near-wall boundary
        # layer (while preserving the clearance criterion) instead of
        # fighting the fastest part of the profile on the axis.
        seg_here, _ = self.network.find_segment(pos)
        ride_offset = np.zeros(3)
        if direct_terminal:
            self._ride_dir = None
        elif flow is not None and seg_here is not None:
            v_flow_center = np.asarray(
                flow.velocity_at(route_pt, sim_time), dtype=float,
            )
            adverse = max(0.0, -float(np.dot(v_flow_center, tangent)))
            if adverse > EPS:
                v_cap = self._authority_speed_along(model, particle, tangent)
                thresh = cfg.wall_ride_flow_fraction * max(v_cap, 1e-4)
                lam = float(np.clip((adverse - thresh) / max(thresh, 1e-4), 0.0, 1.0))
                if lam > 0.0:
                    closest_here, radial_vec_here, radial_here, _ = (
                        seg_here.closest_point_and_radial(pos)
                    )
                    if radial_here > 1e-6:
                        ride_dir = radial_vec_here / radial_here
                    elif self._ride_dir is not None:
                        ride_dir = self._ride_dir
                    else:
                        ride_dir = np.cross(tangent, np.array([0.0, 0.0, 1.0]))
                        if np.linalg.norm(ride_dir) < 1e-6:
                            ride_dir = np.cross(tangent, np.array([0.0, 1.0, 0.0]))
                        ride_dir /= np.linalg.norm(ride_dir) + EPS
                    # Keep the riding side stable (hysteresis on direction).
                    if self._ride_dir is not None and float(
                        np.dot(ride_dir, self._ride_dir)
                    ) < 0.0:
                        ride_dir = self._ride_dir
                    ride_dir = ride_dir - np.dot(ride_dir, tangent) * tangent
                    norm = float(np.linalg.norm(ride_dir))
                    if norm > 1e-6:
                        ride_dir /= norm
                        self._ride_dir = ride_dir
                        # Cap the ride radius by the narrowest vessel coming
                        # up on the route (don't get pinched by a taper).
                        r_ride = seg_here.radius
                        for d_arc in (0.5 * lookahead, lookahead):
                            p_ahead = self._route_point_at_arc(
                                arc_now + d_arc,
                            )
                            s_a, _ = self.network.find_segment(p_ahead)
                            if s_a is not None:
                                r_ride = min(r_ride, s_a.radius)
                        ride_clearance = max(
                            cfg.wall_clearance_threshold,
                            min(
                                0.75 * r_ride,
                                cfg.wall_ride_clearance * geometry_scale,
                            ),
                        )
                        r_off = lam * max(r_ride - ride_clearance, 0.0)
                        ride_offset = r_off * ride_dir
                else:
                    self._ride_dir = None
            else:
                self._ride_dir = None

        aim = aim + ride_offset

        to_aim = aim - pos
        aim_dist = float(np.linalg.norm(to_aim))
        aim_dir = to_aim / aim_dist if aim_dist > EPS else tangent

        if (
            self.mode == COMET_BOOST
            and not terminal
            and high_flow
        ):
            boost_dir = 0.65 * tangent + 0.35 * aim_dir
            boost_norm = float(np.linalg.norm(boost_dir))
            if boost_norm > EPS:
                aim_dir = boost_dir / boost_norm

        # Tracking speed: request the cruise cap and let the allocator
        # deliver its best achievable velocity along the aim direction
        # (the canonical-pattern authority probe underestimates what the
        # full multi-coil optimization can produce, so it is not used to
        # throttle progress). Overshoot is guarded relative to aim distance.
        speed = cfg.cruise_speed * cfg.progress_gain * boost_factor
        if direct_terminal and not terminal_precision:
            # Keep the request modest so the allocator spends its authority
            # on flow compensation first (a near-feasible v_des tracks the
            # target line instead of being sheared downstream).
            speed = 0.5 * cfg.cruise_speed
        speed = min(speed, cfg.max_step_fraction * max(aim_dist, EPS) / dt)

        # Slow down approaching the final target (anti-overshoot).
        if dist_to_final < 4.0 * speed * dt:
            speed = min(speed, 0.55 * dist_to_final / dt + 1e-4)

        if terminal_precision:
            terminal_correction = (
                cfg.terminal_precision_kp * (final_target - pos)
                - cfg.terminal_precision_kv
                * np.asarray(particle.realized_vel, dtype=float)
            )
            correction_norm = float(np.linalg.norm(terminal_correction))
            if correction_norm > cfg.terminal_precision_max_speed:
                terminal_correction *= (
                    cfg.terminal_precision_max_speed / correction_norm
                )
            if self._terminal_hold_steps > 0:
                terminal_correction = (
                    -cfg.terminal_precision_kv
                    * np.asarray(particle.realized_vel, dtype=float)
                )
            v_track = terminal_correction
        else:
            v_track = speed * aim_dir
            terminal_correction = v_track.copy() if terminal else np.zeros(3)

        # Corridor pull: toward the centerline normally, toward the offset
        # boundary-layer corridor while wall-riding, and off entirely in
        # terminal capture (the aim point IS the target).
        if direct_terminal:
            v_center = np.zeros(3)
        else:
            corridor_error = perp + ride_offset
            corridor_error = (
                corridor_error - np.dot(corridor_error, tangent) * tangent
            )
            lateral_gain = cfg.lateral_capture_gain
            max_lateral_speed = cfg.max_lateral_speed
            if rescue:
                lateral_gain *= cfg.rescue_lateral_gain_multiplier
                max_lateral_speed = cfg.rescue_max_lateral_speed
            v_center = lateral_gain * corridor_error
            c_norm = float(np.linalg.norm(v_center))
            if c_norm > max_lateral_speed:
                v_center *= max_lateral_speed / c_norm

        # Wall repulsion.
        v_wall = np.zeros(3)
        wall_clearance = np.nan
        if seg_here is not None:
            closest, radial_vec, radial_dist, _ = (
                seg_here.closest_point_and_radial(pos)
            )
            wall_clearance = float(seg_here.radius - radial_dist)
            repel_band = cfg.wall_clearance_threshold + cfg.wall_repulsion_margin
            if wall_clearance < repel_band and radial_dist > EPS:
                inward = (closest - pos) / radial_dist
                deficit = (repel_band - wall_clearance) / repel_band
                v_wall = cfg.wall_repulsion_speed * deficit * inward

        v_des_total = v_track + v_center + v_wall

        # First-order damping of the desired velocity (anti-oscillation).
        damping_gain = 0.0 if terminal_precision else cfg.damping_gain
        v_des_total = (
            damping_gain * self._v_des_prev
            + (1.0 - damping_gain) * v_des_total
        )
        self._v_des_prev = v_des_total.copy()

        # Flow compensation: the allocator matches the *magnetic* component.
        flow_estimate = v_flow
        if terminal_precision:
            flow_estimate = 0.65 * v_flow + 0.35 * v_flow_target
        v_des_mag = (
            v_des_total - cfg.flow_compensation_gain * flow_estimate
        )

        # --- allocation ---------------------------------------------------
        extra_ids = [h.seg_id for h in hypotheses]
        if seg_here is not None:
            extra_ids.append(seg_here.seg_id)
        segs = self._local_segments(
            pos,
            extra_ids,
            probes=(pos + 0.003 * aim_dir, pos + 0.006 * aim_dir),
        )

        currents, cost, safety_active = self._allocate_currents(
            particle=particle,
            pos=pos,
            model=model,
            v_des_mag=v_des_mag,
            v_flow=v_flow,
            dt=dt,
            segs=segs,
            effort_weight=effort_weight,
            initial_guess=initial_guess,
            flow=flow,
            sim_time=sim_time,
            smoothness_weight=(
                cfg.current_smoothness_weight
                * cfg.terminal_smoothness_multiplier
                if terminal_precision else None
            ),
            cem_iterations=(
                cfg.terminal_cem_iterations if terminal_precision else None
            ),
            cem_population=(
                cfg.terminal_cem_population if terminal_precision else None
            ),
            cem_elites=(
                cfg.terminal_cem_elites if terminal_precision else None
            ),
            terminal_target=final_target if terminal_precision else None,
        )
        self._prev_currents = currents.copy()
        current_saturation = bool(np.any(
            np.abs(currents) >= 0.98 * np.maximum(self._i_lim, EPS)
        ))

        # Exact force from the true magnet model for the simulator.
        previous_currents = self.magnet_array.currents.copy()
        self.magnet_array.set_currents(currents)
        B = self.magnet_array.field_at(pos)
        grad = self.magnet_array.grad_B_squared(pos)
        force = particle.magnetic_force(B, grad)
        self.magnet_array.set_currents(previous_currents)

        transition = self.mode != prev_mode
        if transition:
            self.transition_count += 1
            self._mode_enter_step = self.step_count

        if cfg.debug:
            q_str = {sid: round(val, 3) for sid, val in q.items()}
            logger.debug(
                "COMET step=%d mode=%s arc=%.4f bif=%.4f q=%s sel=%s "
                "reserve=%.4f |I|=%.2f reason=%s",
                self.step_count, self.mode, arc_now, dist_to_bif, q_str,
                selected.seg_id if selected else None,
                selected.reserve if selected else np.nan,
                float(np.max(np.abs(currents))), commit_reason,
            )

        entropy = 0.0
        if q:
            p = np.array(list(q.values()), dtype=float)
            p = p[p > 0]
            entropy = float(-np.sum(p * np.log(p)))

        info = COMETStepInfo(
            mode=self.mode,
            previous_mode=prev_mode,
            transition=transition,
            q=dict(q),
            selected_branch_seg_id=selected.seg_id if selected else None,
            selected_reserve=float(selected.reserve) if selected else np.nan,
            selected_capture_probability=(
                float(selected.capture_probability) if selected else np.nan
            ),
            dist_to_bifurcation=float(dist_to_bif),
            temperature=float(temperature),
            entropy=entropy,
            aim_point=np.asarray(aim, dtype=float),
            control_norm=float(np.linalg.norm(currents)),
            commit_reason=commit_reason,
            stagnant_steps=int(self.stagnant_steps),
            wall_clearance_m=float(wall_clearance),
            route_arc_m=float(arc_now),
            num_candidates=len(hypotheses),
            safety_filter_active=bool(safety_active),
            terminal_mode_active=bool(terminal),
            terminal_stage=terminal_stage,
            target_distance_m=float(dist_to_final),
            predicted_flow_at_target=v_flow_target.copy(),
            terminal_correction=np.asarray(
                terminal_correction, dtype=float,
            ).copy(),
            current_saturation=current_saturation,
            final_hold_steps=int(self._terminal_hold_steps),
            overshoot_count=int(self._terminal_overshoot_count),
            direct_terminal_feasible=bool(direct_terminal_feasible),
        )

        return currents, force, float(cost), True, info

    # ------------------------------------------------------------------

    def summary_dict(self) -> dict:
        cfg = self.cfg
        return {
            "comet_steps": self.step_count,
            "comet_transitions": self.transition_count,
            "comet_hard_commits": self.hard_commit_count,
            "comet_emergency_switches": self.emergency_switch_count,
            "comet_entropy_temperature_far": cfg.entropy_temperature_far,
            "comet_entropy_temperature_near": cfg.entropy_temperature_near,
            "comet_commitment_distance_threshold_m": (
                cfg.commitment_distance_threshold
            ),
            "comet_flow_commitment_reaction_time_s": (
                cfg.flow_commitment_reaction_time
            ),
            "comet_max_flow_commitment_extension_m": (
                cfg.max_flow_commitment_extension
            ),
            "comet_hard_commit_reserve_threshold_m": (
                cfg.hard_commit_reserve_threshold
            ),
            "comet_min_commitment_dwell_steps": cfg.min_commitment_dwell_steps,
            "comet_branch_switch_penalty": cfg.branch_switch_penalty,
            "comet_target_value_weight": cfg.target_value_weight,
            "comet_capture_probability_weight": cfg.capture_probability_weight,
            "comet_authority_reserve_weight": cfg.authority_reserve_weight,
            "comet_downstream_value_weight": cfg.downstream_value_weight,
            "comet_control_effort_weight": cfg.control_effort_weight,
            "comet_wall_risk_weight": cfg.wall_risk_weight,
            "comet_flow_compensation_gain": cfg.flow_compensation_gain,
            "comet_lateral_capture_gain": cfg.lateral_capture_gain,
            "comet_progress_gain": cfg.progress_gain,
            "comet_damping_gain": cfg.damping_gain,
            "comet_cruise_speed_m_s": cfg.cruise_speed,
            "comet_terminal_capture_distance_m": cfg.terminal_capture_distance,
            "comet_terminal_precision_distance_m": (
                cfg.terminal_precision_distance
            ),
            "comet_terminal_precision_kp": cfg.terminal_precision_kp,
            "comet_terminal_precision_kv": cfg.terminal_precision_kv,
            "comet_terminal_overshoots": self._terminal_overshoot_count,
            "comet_max_control_norm_A": self.max_current,
            "comet_wall_clearance_threshold_m": cfg.wall_clearance_threshold,
        }
