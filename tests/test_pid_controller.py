import numpy as np

from dynamag_simulation.core.pid_controller import PIDConfig, PIDController


class DummyParticle:
    def __init__(self, pos):
        self.pos = np.asarray(pos, dtype=float)
        self.vel = np.zeros(3, dtype=float)
        self.realized_vel = np.zeros(3, dtype=float)
        self.drag = 1.0

    def magnetic_force(self, B, grad_Bsq):
        return np.asarray(grad_Bsq, dtype=float)

    def _effective_drag(self, force_norm):
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

    def field_at(self, pos):
        return self.currents.copy()

    def grad_B_squared(self, pos):
        return self.currents.copy()


class DummySegment:
    radius = 0.002

    def __init__(self):
        self.start = np.zeros(3, dtype=float)
        self.end = np.array([1.0, 0.0, 0.0], dtype=float)
        self._axis = np.array([1.0, 0.0, 0.0], dtype=float)
        self._length = 1.0

    def closest_point_and_radial(self, pos):
        pos = np.asarray(pos, dtype=float)
        t = float(np.clip(pos[0], 0.0, 1.0))
        closest = np.array([t, 0.0, 0.0], dtype=float)
        radial = pos - closest
        return closest, radial, float(np.linalg.norm(radial)), t


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

    def velocity_at(self, pos, time=0.0):
        return self.velocity.copy()


class DummyNavigator:
    def __init__(self):
        self.waypoints = [
            np.array([0.001, 0.0, 0.0]),
            np.array([0.004, 0.0, 0.0]),
        ]
        self.total_hops = len(self.waypoints)
        self.current_idx = 0
        self.lookahead_idx = 0
        self.current_wp_dist = np.inf

    def __call__(self, pos):
        pos = np.asarray(pos, dtype=float)
        if np.linalg.norm(pos - self.waypoints[self.current_idx]) < 0.0002:
            self.current_idx = min(self.current_idx + 1, len(self.waypoints) - 1)
        self.lookahead_idx = self.current_idx
        self.current_wp_dist = float(np.linalg.norm(pos - self.waypoints[self.current_idx]))
        return self.waypoints[self.current_idx]


def make_controller(max_current=1.0, **overrides):
    params = {
        "kp": 10.0,
        "ki": 2.0,
        "kd": 0.1,
        "max_current": max_current,
        "max_control_norm": 0.2,
        "optimizer_maxiter": 20,
    }
    params.update(overrides)
    cfg = PIDConfig(**params)
    magnets = DummyMagnetArray(max_current=max_current)
    return PIDController(DummyNetwork(), magnets, DummyWorkspace(), cfg)


def test_pid_produces_bounded_finite_command():
    controller = make_controller(max_current=0.5)
    particle = DummyParticle([0.0, 0.0, 0.0])
    currents, force, cost, converged, info = controller.step(
        particle=particle,
        final_target=np.array([0.004, 0.0, 0.0]),
        navigator=DummyNavigator(),
        dt=0.02,
        flow=DummyFlow([0.0, 0.0, 0.0]),
    )

    assert currents.shape == (3,)
    assert np.all(np.isfinite(currents))
    assert np.all(np.abs(currents) <= 0.5 + 1e-9)
    assert np.all(np.isfinite(force))
    assert np.isfinite(cost)
    assert np.isfinite(info.current_norm)


def test_integral_term_remains_bounded():
    controller = make_controller(
        max_current=10.0,
        max_integral_norm=0.0005,
        max_control_norm=10.0,
    )
    particle = DummyParticle([0.0, 0.0, 0.0])
    navigator = DummyNavigator()

    for _ in range(50):
        controller.step(
            particle=particle,
            final_target=np.array([0.004, 0.0, 0.0]),
            navigator=navigator,
            dt=0.02,
            flow=DummyFlow([0.0, 0.0, 0.0]),
        )

    assert np.linalg.norm(controller.integral_error) <= 0.0005 + 1e-12


def test_waypoint_advances_when_particle_gets_close():
    controller = make_controller()
    navigator = DummyNavigator()
    particle = DummyParticle([0.001, 0.0, 0.0])

    controller.step(
        particle=particle,
        final_target=np.array([0.004, 0.0, 0.0]),
        navigator=navigator,
        dt=0.02,
        flow=DummyFlow([0.0, 0.0, 0.0]),
    )

    assert navigator.current_idx == 1
    assert controller.last_waypoint_index == 1


def test_flow_compensation_changes_command():
    no_flow_controller = make_controller(flow_compensation_gain=1.0)
    flow_controller = make_controller(flow_compensation_gain=1.0)
    particle = DummyParticle([0.0, 0.0, 0.0])
    target = np.array([0.004, 0.0, 0.0])

    currents_no_flow, *_ = no_flow_controller.step(
        particle=particle,
        final_target=target,
        navigator=DummyNavigator(),
        dt=0.02,
        flow=DummyFlow([0.0, 0.0, 0.0]),
    )
    currents_flow, _, _, _, info_flow = flow_controller.step(
        particle=particle,
        final_target=target,
        navigator=DummyNavigator(),
        dt=0.02,
        flow=DummyFlow([0.02, 0.0, 0.0]),
    )

    assert info_flow.flow_compensation_norm > 0.0
    assert not np.allclose(currents_no_flow, currents_flow)


def test_controller_respects_current_limits_under_saturation():
    controller = make_controller(max_current=0.05, max_control_norm=1.0)
    particle = DummyParticle([0.0, 0.0, 0.0])

    currents, _, _, _, info = controller.step(
        particle=particle,
        final_target=np.array([0.5, 0.0, 0.0]),
        navigator=None,
        dt=0.02,
        flow=DummyFlow([0.0, 0.0, 0.0]),
    )

    assert np.max(np.abs(currents)) <= 0.05 + 1e-9
    assert info.saturated
