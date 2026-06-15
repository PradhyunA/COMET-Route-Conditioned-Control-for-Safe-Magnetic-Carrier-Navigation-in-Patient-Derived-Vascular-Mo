"""Classical PID baseline controller for magnetic vessel navigation."""

from __future__ import annotations

import time
from dataclasses import dataclass
from typing import Optional

import numpy as np

try:
    from scipy.optimize import minimize
except Exception:  # pragma: no cover - exercised in dependency-light envs
    minimize = None

from .constants import EPS


@dataclass
class PIDConfig:
    """Hyperparameters for the route-aware classical PID baseline."""

    kp: float = 18.0
    ki: float = 1.5
    kd: float = 0.20
    flow_compensation_gain: float = 1.0
    wall_repulsion_gain: float = 0.040
    wall_clearance_threshold: float = 0.0007
    waypoint_lookahead_distance: float = 0.002
    waypoint_reached_radius: float = 0.0015
    max_integral_norm: float = 0.004
    derivative_filter_alpha: float = 0.75
    max_control_norm: float = 0.120
    max_current: Optional[float] = None
    current_effort_weight: float = 1e-4
    current_smoothness_weight: float = 1e-4
    wall_penalty_weight: float = 20.0
    optimizer_maxiter: int = 40
    debug: bool = False


@dataclass
class PIDStepInfo:
    target: np.ndarray
    waypoint_index: int
    waypoint_distance_m: float
    position_error_norm_m: float
    p_norm: float
    i_norm: float
    d_norm: float
    flow_compensation_norm: float
    wall_repulsion_norm: float
    desired_velocity_norm: float
    current_norm: float
    max_abs_current: float
    saturated: bool
    integral_norm: float
    wall_clearance_m: float
    controller_step_time_s: float


class PIDController:
    """
    Path-aware classical PID controller.

    The high-level controller asks for a magnetic velocity from PID feedback on
    the local route waypoint. A bounded one-step allocator converts that desired
    magnetic velocity to bidirectional coil currents under the same current
    limits used by the other controllers.
    """

    def __init__(self, network, magnet_array, workspace, config: PIDConfig | None = None):
        self.network = network
        self.magnet_array = magnet_array
        self.workspace = workspace
        self.cfg = config if config is not None else PIDConfig()

        self.integral_error = np.zeros(3, dtype=float)
        self.filtered_derivative = np.zeros(3, dtype=float)
        self.last_waypoint = None
        self.last_waypoint_index = -1

    def reset(self):
        self.integral_error[:] = 0.0
        self.filtered_derivative[:] = 0.0
        self.last_waypoint = None
        self.last_waypoint_index = -1

    def _current_limit(self) -> float:
        if self.cfg.max_current is not None:
            return float(self.cfg.max_current)
        return float(getattr(self.magnet_array, "max_current", 8.0))

    @staticmethod
    def _clip_norm(v, max_norm):
        v = np.asarray(v, dtype=float)
        n = float(np.linalg.norm(v))
        if max_norm is not None and max_norm > 0.0 and n > max_norm:
            return v * (float(max_norm) / (n + EPS)), True
        return v, False

    def _magnetic_velocity_and_force(self, particle, pos, currents):
        self.magnet_array.set_currents(np.asarray(currents, dtype=float))
        B = self.magnet_array.field_at(pos)
        grad = self.magnet_array.grad_B_squared(pos)
        force = particle.magnetic_force(B, grad)
        force_norm = float(np.linalg.norm(force))

        if force_norm <= EPS:
            return np.zeros(3, dtype=float), force

        if hasattr(particle, "_effective_drag"):
            gamma_eff, _, _ = particle._effective_drag(force_norm)
        else:
            gamma_eff = float(getattr(particle, "drag", 1.0))

        return force / max(gamma_eff, EPS), force

    def _wall_geometry(self, pos):
        if self.network is None or not hasattr(self.network, "find_segment"):
            return np.nan, np.zeros(3, dtype=float)

        seg, _ = self.network.find_segment(pos)
        if seg is None:
            return np.nan, np.zeros(3, dtype=float)

        closest, radial_vec, radial_dist, _ = seg.closest_point_and_radial(pos)
        clearance = float(seg.radius - radial_dist)

        if radial_dist > EPS:
            inward = -radial_vec / radial_dist
        else:
            inward = np.asarray(closest, dtype=float) - np.asarray(pos, dtype=float)
            inward, _ = self._clip_norm(inward, 1.0)

        return clearance, inward

    def _wall_repulsion(self, pos):
        clearance, inward = self._wall_geometry(pos)
        threshold = max(float(self.cfg.wall_clearance_threshold), EPS)

        if not np.isfinite(clearance) or clearance >= threshold:
            return np.zeros(3, dtype=float), clearance

        activation = float(np.clip((threshold - clearance) / threshold, 0.0, 1.0))
        return self.cfg.wall_repulsion_gain * activation * activation * inward, clearance

    def _predict_wall_penalty(self, pos, total_velocity, dt):
        if self.network is None or not hasattr(self.workspace, "clip"):
            return 0.0

        predicted = np.asarray(pos, dtype=float) + np.asarray(total_velocity, dtype=float) * float(dt)
        predicted = self.workspace.clip(predicted)
        clearance, _ = self._wall_geometry(predicted)

        if not np.isfinite(clearance):
            return 0.0

        threshold = max(float(self.cfg.wall_clearance_threshold), 0.0)
        violation = max(0.0, threshold - clearance)
        return violation * violation

    def _heuristic_starts(self, pos, desired_velocity, initial_guess):
        n = int(self.magnet_array.n_magnets)
        i_lim = self._current_limit()
        zero = np.zeros(n, dtype=float)
        starts = [zero]

        if initial_guess is not None:
            starts.insert(0, np.clip(np.asarray(initial_guess, dtype=float), -i_lim, i_lim))

        desired_norm = float(np.linalg.norm(desired_velocity))
        if desired_norm > EPS and hasattr(self.magnet_array, "positions"):
            desired_hat = desired_velocity / desired_norm
            pull_dirs = np.asarray(self.magnet_array.positions, dtype=float) - np.asarray(pos, dtype=float)
            pull_norms = np.linalg.norm(pull_dirs, axis=1)
            pull_dirs = pull_dirs / (pull_norms[:, None] + EPS)
            proj = pull_dirs @ desired_hat
            max_abs = float(np.max(np.abs(proj))) if proj.size else 0.0

            if max_abs > EPS:
                signed = i_lim * proj / max_abs
                starts.extend([0.5 * signed, signed, -0.5 * signed])

            if proj.size:
                single = zero.copy()
                single[int(np.argmax(np.abs(proj)))] = i_lim * np.sign(proj[int(np.argmax(np.abs(proj)))])
                starts.append(single)

        direct = np.zeros(n, dtype=float)
        direct[: min(3, n)] = np.asarray(desired_velocity, dtype=float)[: min(3, n)]
        if np.linalg.norm(direct) > EPS:
            starts.extend([
                np.clip(direct, -i_lim, i_lim),
                np.clip(0.5 * direct, -i_lim, i_lim),
            ])

        unique = []
        for s in starts:
            s = np.nan_to_num(np.asarray(s, dtype=float), nan=0.0, posinf=i_lim, neginf=-i_lim)
            s = np.clip(s, -i_lim, i_lim)
            if not any(np.allclose(s, u) for u in unique):
                unique.append(s)
        return unique

    def _allocate_currents(self, particle, pos, desired_velocity, flow_velocity, dt, initial_guess):
        i_lim = self._current_limit()
        bounds = [(-i_lim, i_lim)] * int(self.magnet_array.n_magnets)
        prev = (
            np.zeros(self.magnet_array.n_magnets, dtype=float)
            if initial_guess is None
            else np.clip(np.asarray(initial_guess, dtype=float), -i_lim, i_lim)
        )

        scale = max(float(self.cfg.max_control_norm), EPS)

        def objective(currents):
            currents = np.asarray(currents, dtype=float)
            v_mag, _ = self._magnetic_velocity_and_force(particle, pos, currents)
            err = (v_mag - desired_velocity) / scale
            effort = np.sum((currents / max(i_lim, EPS)) ** 2)
            smooth = np.sum(((currents - prev) / max(i_lim, EPS)) ** 2)
            wall = self._predict_wall_penalty(pos, v_mag + flow_velocity, dt)
            return float(
                np.dot(err, err)
                + self.cfg.current_effort_weight * effort
                + self.cfg.current_smoothness_weight * smooth
                + self.cfg.wall_penalty_weight * wall
            )

        starts = self._heuristic_starts(pos, desired_velocity, initial_guess)
        best = starts[0].copy()
        best_cost = objective(best)
        converged = False

        if minimize is None:
            for start in starts:
                for scale in (1.0, 0.5, 0.25):
                    cand = np.clip(scale * start, -i_lim, i_lim)
                    cand_cost = objective(cand)
                    if cand_cost < best_cost:
                        best = cand.copy()
                        best_cost = cand_cost
            converged = True
        else:
            for start in starts:
                result = minimize(
                    objective,
                    start,
                    method="SLSQP",
                    bounds=bounds,
                    options={
                        "maxiter": int(self.cfg.optimizer_maxiter),
                        "ftol": 1e-6,
                        "disp": False,
                    },
                )
                cand = np.clip(result.x if hasattr(result, "x") else start, -i_lim, i_lim)
                cand_cost = objective(cand)

                if result.success:
                    converged = True

                if cand_cost < best_cost:
                    best = cand.copy()
                    best_cost = cand_cost

        best = np.nan_to_num(best, nan=0.0, posinf=i_lim, neginf=-i_lim)
        best = np.clip(best, -i_lim, i_lim)
        _, force = self._magnetic_velocity_and_force(particle, pos, best)
        saturated = bool(np.any(np.isclose(np.abs(best), i_lim, rtol=1e-3, atol=1e-6)))
        return best, force, float(best_cost), converged, saturated

    def _waypoint(self, particle_pos, final_target, navigator):
        if navigator is None:
            return np.asarray(final_target, dtype=float), -1, float(np.linalg.norm(final_target - particle_pos))

        waypoint = np.asarray(navigator(particle_pos), dtype=float)
        idx = int(getattr(navigator, "lookahead_idx", getattr(navigator, "current_idx", -1)))
        dist = float(getattr(navigator, "current_wp_dist", np.linalg.norm(waypoint - particle_pos)))
        return waypoint, idx, dist

    def step(
        self,
        particle,
        final_target,
        navigator,
        dt: float,
        flow=None,
        sim_time: float = 0.0,
        initial_guess=None,
    ):
        t0 = time.perf_counter()
        cfg = self.cfg

        pos = np.asarray(particle.pos, dtype=float)
        waypoint, waypoint_idx, waypoint_dist = self._waypoint(pos, final_target, navigator)

        if (
            self.last_waypoint is None
            or waypoint_idx != self.last_waypoint_index
            or np.linalg.norm(waypoint - self.last_waypoint) > cfg.waypoint_reached_radius
        ):
            self.integral_error[:] = 0.0

        self.last_waypoint = waypoint.copy()
        self.last_waypoint_index = waypoint_idx

        error = waypoint - pos
        prev_integral = self.integral_error.copy()
        self.integral_error += error * float(dt)
        self.integral_error, _ = self._clip_norm(
            self.integral_error,
            cfg.max_integral_norm,
        )

        measured_velocity = np.asarray(
            getattr(particle, "realized_vel", getattr(particle, "vel", np.zeros(3))),
            dtype=float,
        )
        if not np.all(np.isfinite(measured_velocity)):
            measured_velocity = np.zeros(3, dtype=float)

        derivative_raw = -measured_velocity
        alpha = float(np.clip(cfg.derivative_filter_alpha, 0.0, 1.0))
        self.filtered_derivative = (
            alpha * self.filtered_derivative + (1.0 - alpha) * derivative_raw
        )

        flow_velocity = (
            np.asarray(flow.velocity_at(pos, sim_time), dtype=float)
            if flow is not None and hasattr(flow, "velocity_at")
            else np.zeros(3, dtype=float)
        )
        if not np.all(np.isfinite(flow_velocity)):
            flow_velocity = np.zeros(3, dtype=float)

        wall_term, wall_clearance = self._wall_repulsion(pos)
        p_term = cfg.kp * error
        i_term = cfg.ki * self.integral_error
        d_term = cfg.kd * self.filtered_derivative
        flow_term = -cfg.flow_compensation_gain * flow_velocity

        desired_velocity = p_term + i_term + d_term + flow_term + wall_term
        desired_velocity = np.nan_to_num(desired_velocity, nan=0.0, posinf=0.0, neginf=0.0)
        desired_velocity, clipped_desired = self._clip_norm(
            desired_velocity,
            cfg.max_control_norm,
        )

        currents, force, cost, converged, saturated_current = self._allocate_currents(
            particle=particle,
            pos=pos,
            desired_velocity=desired_velocity,
            flow_velocity=flow_velocity,
            dt=dt,
            initial_guess=initial_guess,
        )

        saturated = bool(clipped_desired or saturated_current)
        if saturated:
            self.integral_error = prev_integral

        elapsed = time.perf_counter() - t0
        info = PIDStepInfo(
            target=waypoint.copy(),
            waypoint_index=waypoint_idx,
            waypoint_distance_m=waypoint_dist,
            position_error_norm_m=float(np.linalg.norm(error)),
            p_norm=float(np.linalg.norm(p_term)),
            i_norm=float(np.linalg.norm(i_term)),
            d_norm=float(np.linalg.norm(d_term)),
            flow_compensation_norm=float(np.linalg.norm(flow_term)),
            wall_repulsion_norm=float(np.linalg.norm(wall_term)),
            desired_velocity_norm=float(np.linalg.norm(desired_velocity)),
            current_norm=float(np.linalg.norm(currents)),
            max_abs_current=float(np.max(np.abs(currents))) if len(currents) else 0.0,
            saturated=saturated,
            integral_norm=float(np.linalg.norm(self.integral_error)),
            wall_clearance_m=float(wall_clearance),
            controller_step_time_s=float(elapsed),
        )

        if cfg.debug:
            print(
                "PID "
                f"wp={info.waypoint_index} "
                f"err={info.position_error_norm_m * 1000:.2f}mm "
                f"|v_des|={info.desired_velocity_norm * 1000:.1f}mm/s "
                f"|I|={info.max_abs_current:.2f}A "
                f"sat={info.saturated}",
                flush=True,
            )

        return currents, force, cost, converged, info
