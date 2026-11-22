"""
Receding-horizon path-tracking MPC for magnetic carrier steering.

A textbook nonlinear MPC baseline with no task-specific heuristics. Each
control step solves

    min_{I_0..I_{N-1}}  sum_{k=1}^{N} q_k ||p_k - r_k||^2
                        + sum_{k=0}^{N-1} r ||I_k / I_max||^2 + s ||(I_k - I_{k-1}) / I_max||^2
                        + rho * sum_{k=1}^{N} max(0, c_req - c(p_k))^2

    s.t.  p_{k+1} = p_k + dt * (u(p_k) + (F_mag(p_k, I_k) + F_g) / gamma(|F|))
          |I_k| <= I_max

with position errors in mm, q_k = q for k < N and q_N = q_terminal. The
dynamics are the overdamped carrier model the simulator integrates (dipole
coil fields with core saturation, saturated-magnetisation force,
Schiller-Naumann drag, gravity, blood flow). The wall constraint is soft
(quadratic exact-penalty), the standard treatment of state constraints
that may be infeasible under disturbances.

The reference r_k follows the shared planned route at a constant speed from
the carrier's projection onto it, clipped at the target (path-following
MPC). Inputs use move blocking: the horizon is split into `control_blocks`
equal blocks of constant current. The problem is solved with bounded
L-BFGS-B, warm-started from the shifted previous solution plus single-coil and random seeds (zero current is a
stationary point because the force is quadratic in the currents). For speed the
coil field at each horizon step is evaluated at the nominal (warm-start)
predicted position, a real-time-iteration style linearisation of the
field map; the clearance is evaluated at the candidate positions.
"""

from dataclasses import dataclass

import numpy as np
from scipy.optimize import minimize

from .comet_controller import _LocalFieldModel, _magnetic_velocity_batch
from .constants import EPS

_MM = 1e-3


@dataclass
class TrackingMPCConfig:
    horizon_steps: int = 10
    control_blocks: int = 2
    reference_speed_m_s: float = 0.020
    q_tracking: float = 1.0
    q_terminal: float = 5.0
    r_current: float = 1e-3
    s_rate: float = 1e-2
    rho_wall: float = 1e3
    maxiter: int = 30
    fd_step_fraction: float = 1e-3
    random_starts: int = 16
    optimized_starts: int = 2


class TrackingMPC:
    """Path-tracking NMPC; command() matches the CEM-MPC controller API."""

    def __init__(self, network, route_points, wall_clearance_m, current_limit,
                 config=None, rng_seed=0):
        self.cfg = config or TrackingMPCConfig()
        self._rng = np.random.default_rng(rng_seed)
        self.network = network
        self.wall_clearance_m = float(wall_clearance_m)
        self.current_limit = float(current_limit)
        self.set_route(route_points)
        self._plan = None

    def set_route(self, route_points):
        pts = np.asarray(route_points, dtype=float)
        keep = np.concatenate([[True], np.linalg.norm(np.diff(pts, axis=0), axis=1) > EPS])
        self._pts = pts[keep]
        self._cum = np.concatenate([[0.0], np.cumsum(np.linalg.norm(np.diff(self._pts, axis=0), axis=1))])
        self._arc = 0.0
        self._plan = None

    # ------------------------------------------------------------------
    # Route reference
    # ------------------------------------------------------------------

    def _project(self, pos):
        """Arc length of the closest route point, searched forward of the last one."""
        a, b = self._pts[:-1], self._pts[1:]
        ab = b - a
        ab_len_sq = np.maximum(np.einsum("ij,ij->i", ab, ab), EPS)
        t = np.clip(np.einsum("ij,ij->i", pos[None, :] - a, ab) / ab_len_sq, 0.0, 1.0)
        d = np.linalg.norm(pos[None, :] - (a + t[:, None] * ab), axis=1)
        arcs = self._cum[:-1] + t * np.sqrt(ab_len_sq)
        window = arcs >= self._arc - 0.005
        j = int(np.argmin(np.where(window, d, np.inf)))
        self._arc = max(self._arc, float(arcs[j]))
        return self._arc

    def _point_at(self, arc):
        arc = float(np.clip(arc, 0.0, self._cum[-1]))
        i = int(np.clip(np.searchsorted(self._cum, arc, side="right") - 1, 0, len(self._pts) - 2))
        t = (arc - self._cum[i]) / max(self._cum[i + 1] - self._cum[i], EPS)
        return self._pts[i] + t * (self._pts[i + 1] - self._pts[i])

    def reference(self, pos, dt):
        arc0 = self._project(pos)
        n = self.cfg.horizon_steps
        step = self.cfg.reference_speed_m_s * dt
        return np.array([self._point_at(arc0 + (k + 1) * step) for k in range(n)])

    # ------------------------------------------------------------------
    # Prediction and cost
    # ------------------------------------------------------------------

    def _expand(self, z):
        """(s, blocks * n) decision batch -> (s, N, n) current sequence."""
        n_blocks = self.cfg.control_blocks
        s = z.shape[0]
        blocks = z.reshape(s, n_blocks, -1)
        idx = np.minimum(np.arange(self.cfg.horizon_steps) * n_blocks // self.cfg.horizon_steps, n_blocks - 1)
        return blocks[:, idx, :]

    def _cost_batch(self, z, ctx):
        cfg = self.cfg
        currents = self._expand(z) * self.current_limit
        s = z.shape[0]
        p = np.repeat(ctx["pos"][None, :], s, axis=0)
        cost = np.zeros(s)
        prev = ctx["prev"]
        for k in range(cfg.horizon_steps):
            b, grad = ctx["models"][k].field_and_grad(currents[:, k, :])
            v_mag, _ = _magnetic_velocity_batch(ctx["particle"], b, grad)
            p = p + ctx["dt"] * (v_mag + ctx["flow"][k][None, :])
            q = cfg.q_terminal if k == cfg.horizon_steps - 1 else cfg.q_tracking
            err = (p - ctx["ref"][k][None, :]) / _MM
            cost += q * np.einsum("ij,ij->i", err, err)
            axis_pt, axis_dir, radius = ctx["walls"][k]
            rel = p - axis_pt[None, :]
            radial = np.linalg.norm(rel - np.outer(rel @ axis_dir, axis_dir), axis=1)
            violation = np.maximum(0.0, ctx["c_req"] - (radius - radial)) / _MM
            cost += cfg.rho_wall * violation ** 2
            u = currents[:, k, :] / self.current_limit
            du = (currents[:, k, :] - prev) / self.current_limit
            cost += cfg.r_current * np.einsum("ij,ij->i", u, u) + cfg.s_rate * np.einsum("ij,ij->i", du, du)
            prev = currents[:, k, :]
        return cost

    def _nominal_positions(self, particle, pos, magnet_array, flow, sim_time, dt):
        """Roll the warm-start plan through the true field model."""
        positions = []
        p = pos.copy()
        plan = self._plan
        for k in range(self.cfg.horizon_steps):
            positions.append(p.copy())
            if plan is None:
                continue
            model = _LocalFieldModel(magnet_array, p, h=1e-4)
            b, grad = model.field_and_grad(plan[k][None, :] * self.current_limit)
            v_mag, _ = _magnetic_velocity_batch(particle, b, grad)
            u = np.asarray(flow.velocity_at(p, sim_time + k * dt), dtype=float) if flow is not None else 0.0
            p = p + dt * (v_mag[0] + u)
        return positions

    def _wall_frame(self, point):
        seg, _ = self.network.find_segment(point)
        if seg is None:
            return np.asarray(point, dtype=float), np.array([0.0, 0.0, 1.0]), np.inf
        return np.asarray(seg.start, dtype=float), np.asarray(seg._axis, dtype=float), float(seg.radius)

    # ------------------------------------------------------------------
    # Control
    # ------------------------------------------------------------------

    def command(self, particle, particle_pos, magnet_array, dt, flow=None, sim_time=0.0,
                previous_currents=None, **_unused):
        cfg = self.cfg
        n = magnet_array.n_magnets
        pos = np.asarray(particle_pos, dtype=float)
        ref = self.reference(pos, dt)

        nominal = self._nominal_positions(particle, pos, magnet_array, flow, sim_time, dt)
        flow_pred = [
            np.asarray(flow.velocity_at(p, sim_time + k * dt), dtype=float) if flow is not None else np.zeros(3)
            for k, p in enumerate(nominal)
        ]
        prev = (np.zeros(n) if previous_currents is None
                else np.asarray(previous_currents, dtype=float))
        ctx = {
            "pos": pos,
            "ref": ref,
            "dt": dt,
            "particle": particle,
            "flow": flow_pred,
            "models": [_LocalFieldModel(magnet_array, p, h=1e-4) for p in nominal],
            "walls": [self._wall_frame(r) for r in ref],
            "c_req": self.wall_clearance_m,
            "prev": prev[None, :],
        }

        n_vars = cfg.control_blocks * n
        if self._plan is None:
            z0 = np.tile(prev / self.current_limit, cfg.control_blocks)
        else:
            shifted = np.vstack([self._plan[1:], self._plan[-1:]])
            idx = np.arange(cfg.control_blocks) * cfg.horizon_steps // cfg.control_blocks
            z0 = shifted[idx].reshape(-1)
        z0 = np.clip(z0, -1.0, 1.0)
        h = cfg.fd_step_fraction

        def fun_and_grad(z):
            batch = np.vstack([z[None, :], z[None, :] + h * np.eye(n_vars)])
            costs = self._cost_batch(batch, ctx)
            return float(costs[0]), (costs[1:] - costs[0]) / h

        # The force is quadratic in the currents, so zero current is a
        # stationary point; seed the solver from the warm start plus
        # single-coil probes and random draws held over the horizon.
        probes = np.vstack([np.eye(n), -np.eye(n)])
        draws = self._rng.uniform(-1.0, 1.0, (cfg.random_starts, n))
        seeds = np.vstack([z0[None, :], np.tile(np.vstack([probes, draws]), cfg.control_blocks)])
        seed_costs = self._cost_batch(seeds, ctx)
        order = np.argsort(seed_costs)
        best_z, best_cost = seeds[order[0]], float(seed_costs[order[0]])
        converged = False
        for start in seeds[order[:cfg.optimized_starts]]:
            result = minimize(fun_and_grad, start, jac=True, method="L-BFGS-B",
                              bounds=[(-1.0, 1.0)] * n_vars,
                              options={"maxiter": cfg.maxiter})
            if result.fun < best_cost:
                best_z, best_cost = np.clip(result.x, -1.0, 1.0), float(result.fun)
            converged = converged or bool(result.success)

        self._plan = self._expand(best_z[None, :])[0]
        currents = self._plan[0] * self.current_limit

        magnet_array.set_currents(currents)
        force = particle.magnetic_force(
            magnet_array.field_at(pos), magnet_array.grad_B_squared(pos),
        )
        return currents, force, best_cost, converged
