"""SMC-DOB-inspired route-following controller.

This is an adapted published-style baseline based on Bhattacharjee et al.,
"Robust Autonomous Control of a Magnetic Millirobot in In Vitro Cardiac
Flow" (arXiv:2604.01523).  It is deliberately labelled *inspired*: the paper
uses a planar permanent-magnet millirobot, a 2-D cardiac phantom, camera
localization, and a different electromagnetic system.  This implementation
preserves the paper's controller structure while adapting it to this
project's overdamped 3-D particle dynamics and existing bounded SLSQP current
allocator.

The adapted velocity-domain equations are

    e       = x_d - x
    e_dot   = v_d - v_realized
    s       = gamma * LPF(e_dot) + lambda * e
    d_hat   = LPF(v_realized - v_mag,prev - v_flow,prev)
    v_eq    = v_d + k_v * LPF(e_dot) + k_p * e
    v_sw    = k_sw * tanh(s / phi)
    v_mag,d = v_eq + v_sw - g_f * v_flow - g_d * d_hat + v_wall

The requested magnetic velocity is mapped to coil currents by the same
non-CEM, bidirectional SLSQP allocator used by the route-aware PID baseline.
No COMET branch probability, downstream value, terminal precision, or CEM
allocation is used.
"""

from __future__ import annotations

import time
from dataclasses import dataclass

import numpy as np

from .constants import EPS
from .pid_controller import PIDConfig, PIDController


@dataclass
class SMCDOBInspiredConfig(PIDConfig):
    """Gains for the adapted three-dimensional SMC-DOB baseline."""

    # Paper-reported sliding-surface structure (gamma=2, lambda=1).
    sliding_gamma: float = 2.0
    sliding_lambda: float = 1.0

    # Velocity-domain equivalent-control gains for the overdamped model.
    equivalent_velocity_gain: float = 0.55
    equivalent_position_gain: float = 10.0

    # Smooth switching term.  Units are velocity-domain equivalents of the
    # paper's force-domain K4 and boundary layer phi.
    switching_gain: float = 0.045
    boundary_layer: float = 0.020

    # DOB-like filtered model residual.  The moderate-flow paper setting uses
    # eta=0.12 s; retain that time scale here.
    disturbance_time_constant: float = 0.12
    disturbance_gain: float = 1.0
    max_disturbance_norm: float = 0.080

    # Fixed-route reference.  The allocator will return the best achievable
    # velocity under the shared current limits.
    route_speed: float = 0.120
    terminal_distance: float = 0.006
    terminal_position_gain: float = 35.0
    max_step_fraction: float = 0.70

    # Shared physical/allocator settings.
    flow_compensation_gain: float = 1.0
    wall_repulsion_gain: float = 0.040
    max_control_norm: float = 0.180
    current_effort_weight: float = 1e-4
    current_smoothness_weight: float = 1e-4
    wall_penalty_weight: float = 20.0
    optimizer_maxiter: int = 40


@dataclass
class SMCDOBStepInfo:
    """Per-step diagnostics for the adapted SMC-DOB controller."""

    target: np.ndarray
    waypoint_index: int
    waypoint_distance_m: float
    position_error_norm_m: float
    velocity_error_norm_m_s: float
    filtered_velocity_error_norm_m_s: float
    sliding_surface_norm: float
    equivalent_velocity_norm_m_s: float
    switching_velocity_norm_m_s: float
    disturbance_residual_norm_m_s: float
    disturbance_estimate_norm_m_s: float
    flow_compensation_norm_m_s: float
    wall_repulsion_norm_m_s: float
    reference_velocity_norm_m_s: float
    desired_velocity_norm: float
    current_norm: float
    max_abs_current: float
    saturated: bool
    wall_clearance_m: float
    controller_step_time_s: float

    # Compatibility aliases used by the shared route-controller simulator.
    @property
    def p_norm(self):
        return self.equivalent_velocity_norm_m_s

    @property
    def i_norm(self):
        return self.disturbance_estimate_norm_m_s

    @property
    def d_norm(self):
        return self.switching_velocity_norm_m_s

    @property
    def flow_compensation_norm(self):
        return self.flow_compensation_norm_m_s

    @property
    def wall_repulsion_norm(self):
        return self.wall_repulsion_norm_m_s

    @property
    def integral_norm(self):
        return self.disturbance_estimate_norm_m_s


class SMCDOBInspiredController(PIDController):
    """Three-dimensional SMC-DOB-inspired fixed-route controller."""

    def __init__(self, network, magnet_array, workspace, config=None):
        cfg = config if config is not None else SMCDOBInspiredConfig()
        if not isinstance(cfg, SMCDOBInspiredConfig):
            raise TypeError("config must be an SMCDOBInspiredConfig")
        super().__init__(network, magnet_array, workspace, cfg)
        self.cfg: SMCDOBInspiredConfig = cfg
        self.filtered_velocity_error = np.zeros(3, dtype=float)
        self.filtered_disturbance_residual = np.zeros(3, dtype=float)
        self.previous_magnetic_velocity = np.zeros(3, dtype=float)
        self.previous_flow_velocity = np.zeros(3, dtype=float)
        self.has_previous_command = False

    def reset(self):
        super().reset()
        self.filtered_velocity_error[:] = 0.0
        self.filtered_disturbance_residual[:] = 0.0
        self.previous_magnetic_velocity[:] = 0.0
        self.previous_flow_velocity[:] = 0.0
        self.has_previous_command = False

    @staticmethod
    def _finite_vector(value):
        value = np.asarray(value, dtype=float)
        if value.shape != (3,) or not np.all(np.isfinite(value)):
            return np.zeros(3, dtype=float)
        return value

    def _route_tangent(self, waypoint, particle_pos, final_target, navigator):
        fallback = np.asarray(waypoint, dtype=float) - np.asarray(
            particle_pos, dtype=float,
        )
        if np.linalg.norm(fallback) <= EPS:
            fallback = np.asarray(final_target, dtype=float) - np.asarray(
                particle_pos, dtype=float,
            )

        waypoints = getattr(navigator, "waypoints", None) if navigator is not None else None
        if waypoints is not None and len(waypoints) > 1:
            idx = int(getattr(navigator, "current_idx", 0))
            idx = int(np.clip(idx, 0, len(waypoints) - 1))
            next_idx = min(idx + 3, len(waypoints) - 1)
            if next_idx > idx:
                tangent = np.asarray(waypoints[next_idx], dtype=float) - np.asarray(
                    waypoints[idx], dtype=float,
                )
                if np.linalg.norm(tangent) > EPS:
                    fallback = tangent

        norm = float(np.linalg.norm(fallback))
        if norm <= EPS:
            return np.zeros(3, dtype=float)
        return fallback / norm

    def _reference_velocity(
        self,
        pos,
        waypoint,
        final_target,
        tangent,
        dt,
    ):
        cfg = self.cfg
        final_error = np.asarray(final_target, dtype=float) - pos
        final_dist = float(np.linalg.norm(final_error))
        if final_dist <= cfg.terminal_distance:
            reference = cfg.terminal_position_gain * final_error
            reference, _ = self._clip_norm(reference, cfg.route_speed)
            return reference

        aim_error = np.asarray(waypoint, dtype=float) - pos
        aim_dist = float(np.linalg.norm(aim_error))
        speed = min(
            cfg.route_speed,
            cfg.max_step_fraction * max(aim_dist, EPS) / max(float(dt), EPS),
        )
        return speed * tangent

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
        dt = max(float(dt), EPS)

        pos = np.asarray(particle.pos, dtype=float)
        waypoint, waypoint_idx, waypoint_dist = self._waypoint(
            pos, final_target, navigator,
        )
        self.last_waypoint = waypoint.copy()
        self.last_waypoint_index = waypoint_idx

        tangent = self._route_tangent(
            waypoint=waypoint,
            particle_pos=pos,
            final_target=final_target,
            navigator=navigator,
        )
        reference_velocity = self._reference_velocity(
            pos=pos,
            waypoint=waypoint,
            final_target=final_target,
            tangent=tangent,
            dt=dt,
        )

        measured_velocity = self._finite_vector(
            getattr(particle, "realized_vel", getattr(particle, "vel", np.zeros(3)))
        )
        flow_velocity = (
            self._finite_vector(flow.velocity_at(pos, sim_time))
            if flow is not None and hasattr(flow, "velocity_at")
            else np.zeros(3, dtype=float)
        )

        error = np.asarray(waypoint, dtype=float) - pos
        velocity_error = reference_velocity - measured_velocity

        eta = max(float(cfg.disturbance_time_constant), dt)
        filter_alpha = float(np.clip(1.0 - np.exp(-dt / eta), 0.0, 1.0))
        self.filtered_velocity_error += filter_alpha * (
            velocity_error - self.filtered_velocity_error
        )

        if self.has_previous_command:
            residual = (
                measured_velocity
                - self.previous_magnetic_velocity
                - self.previous_flow_velocity
            )
        else:
            residual = np.zeros(3, dtype=float)
        residual = self._finite_vector(residual)
        self.filtered_disturbance_residual += filter_alpha * (
            residual - self.filtered_disturbance_residual
        )
        self.filtered_disturbance_residual, _ = self._clip_norm(
            self.filtered_disturbance_residual,
            cfg.max_disturbance_norm,
        )

        sliding_surface = (
            cfg.sliding_gamma * self.filtered_velocity_error
            + cfg.sliding_lambda * error
        )
        equivalent_velocity = (
            reference_velocity
            + cfg.equivalent_velocity_gain * self.filtered_velocity_error
            + cfg.equivalent_position_gain * error
        )
        switching_velocity = cfg.switching_gain * np.tanh(
            sliding_surface / max(float(cfg.boundary_layer), EPS)
        )
        disturbance_compensation = (
            cfg.flow_compensation_gain * flow_velocity
            + cfg.disturbance_gain * self.filtered_disturbance_residual
        )
        wall_velocity, wall_clearance = self._wall_repulsion(pos)

        desired_velocity = (
            equivalent_velocity
            + switching_velocity
            - disturbance_compensation
            + wall_velocity
        )
        desired_velocity = np.nan_to_num(
            desired_velocity, nan=0.0, posinf=0.0, neginf=0.0,
        )
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
        magnetic_velocity, _ = self._magnetic_velocity_and_force(
            particle, pos, currents,
        )
        self.previous_magnetic_velocity = self._finite_vector(magnetic_velocity)
        self.previous_flow_velocity = flow_velocity.copy()
        self.has_previous_command = True

        elapsed = time.perf_counter() - t0
        saturated = bool(clipped_desired or saturated_current)
        info = SMCDOBStepInfo(
            target=waypoint.copy(),
            waypoint_index=waypoint_idx,
            waypoint_distance_m=waypoint_dist,
            position_error_norm_m=float(np.linalg.norm(error)),
            velocity_error_norm_m_s=float(np.linalg.norm(velocity_error)),
            filtered_velocity_error_norm_m_s=float(
                np.linalg.norm(self.filtered_velocity_error)
            ),
            sliding_surface_norm=float(np.linalg.norm(sliding_surface)),
            equivalent_velocity_norm_m_s=float(np.linalg.norm(equivalent_velocity)),
            switching_velocity_norm_m_s=float(np.linalg.norm(switching_velocity)),
            disturbance_residual_norm_m_s=float(np.linalg.norm(residual)),
            disturbance_estimate_norm_m_s=float(
                np.linalg.norm(self.filtered_disturbance_residual)
            ),
            flow_compensation_norm_m_s=float(
                np.linalg.norm(cfg.flow_compensation_gain * flow_velocity)
            ),
            wall_repulsion_norm_m_s=float(np.linalg.norm(wall_velocity)),
            reference_velocity_norm_m_s=float(np.linalg.norm(reference_velocity)),
            desired_velocity_norm=float(np.linalg.norm(desired_velocity)),
            current_norm=float(np.linalg.norm(currents)),
            max_abs_current=float(np.max(np.abs(currents))) if len(currents) else 0.0,
            saturated=saturated,
            wall_clearance_m=float(wall_clearance),
            controller_step_time_s=float(elapsed),
        )

        if cfg.debug:
            print(
                "SMC-DOB-inspired "
                f"wp={info.waypoint_index} "
                f"err={info.position_error_norm_m * 1000:.2f}mm "
                f"|s|={info.sliding_surface_norm:.3g} "
                f"|d_hat|={info.disturbance_estimate_norm_m_s * 1000:.2f}mm/s "
                f"|v_des|={info.desired_velocity_norm * 1000:.1f}mm/s "
                f"|I|max={info.max_abs_current:.2f}A",
                flush=True,
            )

        return currents, force, cost, converged, info
