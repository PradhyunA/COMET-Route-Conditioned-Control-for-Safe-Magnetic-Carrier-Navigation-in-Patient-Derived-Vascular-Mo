import numpy as np

from dynamag_simulation.core.smc_dob_controller import (
    SMCDOBInspiredConfig,
    SMCDOBInspiredController,
)


class DummyParticle:
    def __init__(self, pos):
        self.pos = np.asarray(pos, dtype=float)
        self.vel = np.zeros(3, dtype=float)
        self.realized_vel = np.zeros(3, dtype=float)
        self.drag = 1.0

    def magnetic_force(self, _field, grad_b_squared):
        return np.asarray(grad_b_squared, dtype=float)

    def _effective_drag(self, _force_norm):
        return 1.0, None, None


class DummyMagnetArray:
    n_magnets = 3

    def __init__(self, max_current=1.0):
        self.max_current = float(max_current)
        self.currents = np.zeros(3, dtype=float)
        self.positions = np.eye(3, dtype=float)

    def set_currents(self, currents):
        self.currents = np.clip(
            np.asarray(currents, dtype=float),
            -self.max_current,
            self.max_current,
        )

    def field_at(self, _pos):
        return self.currents.copy()

    def grad_B_squared(self, _pos):
        return self.currents.copy()


class DummySegment:
    radius = 0.002

    def closest_point_and_radial(self, pos):
        pos = np.asarray(pos, dtype=float)
        closest = np.array([np.clip(pos[0], 0.0, 1.0), 0.0, 0.0])
        radial = pos - closest
        return closest, radial, float(np.linalg.norm(radial)), closest[0]


class DummyNetwork:
    def __init__(self):
        self.segment = DummySegment()

    def find_segment(self, pos):
        _, _, radial, _ = self.segment.closest_point_and_radial(pos)
        return self.segment, radial


class DummyWorkspace:
    def clip(self, pos):
        return np.asarray(pos, dtype=float)


class DummyFlow:
    def __init__(self, velocity):
        self.velocity = np.asarray(velocity, dtype=float)

    def velocity_at(self, _pos, _time=0.0):
        return self.velocity.copy()


class DummyNavigator:
    def __init__(self):
        self.waypoints = [
            np.array([0.002, 0.0, 0.0]),
            np.array([0.004, 0.0, 0.0]),
            np.array([0.006, 0.0, 0.0]),
        ]
        self.current_idx = 0
        self.lookahead_idx = 0
        self.current_wp_dist = np.inf

    def __call__(self, pos):
        pos = np.asarray(pos, dtype=float)
        self.current_wp_dist = float(
            np.linalg.norm(pos - self.waypoints[self.current_idx])
        )
        return self.waypoints[self.current_idx]


def make_controller(max_current=0.5, **overrides):
    params = {
        "max_current": max_current,
        "max_control_norm": 0.2,
        "optimizer_maxiter": 20,
    }
    params.update(overrides)
    magnets = DummyMagnetArray(max_current=max_current)
    return SMCDOBInspiredController(
        DummyNetwork(),
        magnets,
        DummyWorkspace(),
        SMCDOBInspiredConfig(**params),
    )


def test_smc_dob_command_is_finite_and_current_bounded():
    controller = make_controller(max_current=0.5)
    particle = DummyParticle([0.0, 0.0, 0.0])

    currents, force, cost, _, info = controller.step(
        particle=particle,
        final_target=np.array([0.006, 0.0, 0.0]),
        navigator=DummyNavigator(),
        dt=0.02,
        flow=DummyFlow([0.01, 0.0, 0.0]),
    )

    assert currents.shape == (3,)
    assert np.all(np.isfinite(currents))
    assert np.max(np.abs(currents)) <= 0.5 + 1e-9
    assert np.all(np.isfinite(force))
    assert np.isfinite(cost)
    assert info.sliding_surface_norm > 0.0
    assert info.flow_compensation_norm_m_s > 0.0


def test_dob_filtered_residual_updates_from_realized_velocity_mismatch():
    controller = make_controller(
        max_current=1.0,
        disturbance_time_constant=0.04,
        max_disturbance_norm=1.0,
    )
    particle = DummyParticle([0.0, 0.0, 0.0])
    navigator = DummyNavigator()
    flow = DummyFlow([0.01, 0.0, 0.0])

    controller.step(
        particle=particle,
        final_target=np.array([0.006, 0.0, 0.0]),
        navigator=navigator,
        dt=0.02,
        flow=flow,
    )
    particle.realized_vel = (
        controller.previous_magnetic_velocity
        + controller.previous_flow_velocity
        + np.array([0.03, -0.01, 0.0])
    )

    _, _, _, _, info = controller.step(
        particle=particle,
        final_target=np.array([0.006, 0.0, 0.0]),
        navigator=navigator,
        dt=0.02,
        flow=flow,
        initial_guess=np.zeros(3),
    )

    assert info.disturbance_residual_norm_m_s > 0.0
    assert info.disturbance_estimate_norm_m_s > 0.0


def test_smc_dob_uses_smooth_boundary_layer_switching():
    controller = make_controller(boundary_layer=0.02, switching_gain=0.04)
    particle = DummyParticle([0.0, 0.0, 0.0])

    _, _, _, _, info = controller.step(
        particle=particle,
        final_target=np.array([0.006, 0.0, 0.0]),
        navigator=DummyNavigator(),
        dt=0.02,
        flow=DummyFlow([0.0, 0.0, 0.0]),
    )

    assert 0.0 < info.switching_velocity_norm_m_s
    assert info.switching_velocity_norm_m_s <= np.sqrt(3.0) * 0.04 + 1e-12
