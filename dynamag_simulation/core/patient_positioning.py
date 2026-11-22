"""
Patient positioning: pick the table orientation that minimises the
magnetic force a planned route needs.

The patient lies on a table that can roll about the body's long axis
(supine, lateral decubitus, prone and the obliques between them) and tilt
head-up or head-down (Trendelenburg). Each orientation fixes the direction
of gravity in the anatomy frame. On route edge i the magnets must supply

    F_req,i = gamma(|s_i|) * s_i - F_g,    s_i = v_progress * d_i - u_i

where d_i is the edge's travel direction, u_i the blood velocity at the
edge midpoint, gamma the Schiller-Naumann drag of the carrier and F_g its
net weight. The chosen orientation minimises max_i |F_req,i|, with ties
broken by the mean.

The VMR models used here are in scanner coordinates, so the body's long
axis is +z.
"""

from dataclasses import dataclass

import numpy as np

from .constants import STANDARD_GRAVITY_M_S2

DEFAULT_LONG_AXIS = (0.0, 0.0, 1.0)
DEFAULT_ROLL_STEP_DEG = 45.0
DEFAULT_MAX_TILT_DEG = 20.0
DEFAULT_PROGRESS_SPEED_M_S = 0.010


@dataclass(frozen=True)
class PositioningChoice:
    gravity_m_s2: tuple
    roll_deg: float
    tilt_deg: float
    peak_required_force_n: float
    mean_required_force_n: float
    worst_orientation_peak_force_n: float
    net_weight_n: float

    def as_dict(self):
        return {
            "gravity_m_s2": list(self.gravity_m_s2),
            "table_roll_deg": self.roll_deg,
            "table_tilt_deg": self.tilt_deg,
            "peak_required_force_n": self.peak_required_force_n,
            "mean_required_force_n": self.mean_required_force_n,
            "worst_orientation_peak_force_n": self.worst_orientation_peak_force_n,
            "net_weight_n": self.net_weight_n,
        }


def table_orientations(long_axis=DEFAULT_LONG_AXIS, roll_step_deg=DEFAULT_ROLL_STEP_DEG,
                       max_tilt_deg=DEFAULT_MAX_TILT_DEG):
    """Yield (roll_deg, tilt_deg, unit gravity direction) for each table pose."""
    axis = np.asarray(long_axis, dtype=float)
    axis = axis / np.linalg.norm(axis)
    helper = np.array([1.0, 0.0, 0.0]) if abs(axis[0]) < 0.9 else np.array([0.0, 1.0, 0.0])
    e1 = np.cross(axis, helper)
    e1 /= np.linalg.norm(e1)
    e2 = np.cross(axis, e1)

    tilts = [0.0] if max_tilt_deg <= 0 else np.linspace(-max_tilt_deg, max_tilt_deg, 5)
    for roll in np.arange(0.0, 360.0, roll_step_deg):
        r = np.radians(roll)
        lateral = np.cos(r) * e1 + np.sin(r) * e2
        for tilt in tilts:
            t = np.radians(tilt)
            yield float(roll), float(tilt), np.cos(t) * lateral + np.sin(t) * axis


def _drag_coefficient(particle, slip_speed):
    re = particle.rho * slip_speed * particle.diameter / particle.eta
    if re < 0.01:
        return particle.drag_stokes
    return particle.drag_stokes * (1.0 + 0.15 * re ** 0.687)


def route_drag_demand(network, waypoints, particle, progress_speed_m_s=DEFAULT_PROGRESS_SPEED_M_S):
    """Drag force (N) needed on each route edge to move at progress speed, shape (n, 3)."""
    points = np.asarray(waypoints, dtype=float)
    demands = []
    for a, b in zip(points[:-1], points[1:]):
        edge = b - a
        length = float(np.linalg.norm(edge))
        if length < 1e-9:
            continue
        flow = np.asarray(network.velocity_at(0.5 * (a + b)), dtype=float)
        slip = progress_speed_m_s * edge / length - flow
        demands.append(_drag_coefficient(particle, float(np.linalg.norm(slip))) * slip)
    return np.asarray(demands, dtype=float).reshape(-1, 3)


def choose_patient_orientation(network, waypoints, particle, net_weight_n,
                               long_axis=DEFAULT_LONG_AXIS,
                               roll_step_deg=DEFAULT_ROLL_STEP_DEG,
                               max_tilt_deg=DEFAULT_MAX_TILT_DEG,
                               progress_speed_m_s=DEFAULT_PROGRESS_SPEED_M_S):
    """Return the PositioningChoice minimising the route's peak required force."""
    demand = route_drag_demand(network, waypoints, particle, progress_speed_m_s)
    if len(demand) == 0:
        raise ValueError("Route needs at least two distinct waypoints")

    best = None
    worst_peak = 0.0
    for roll, tilt, g_dir in table_orientations(long_axis, roll_step_deg, max_tilt_deg):
        required = np.linalg.norm(demand - net_weight_n * g_dir, axis=1)
        score = (float(required.max()), float(required.mean()))
        worst_peak = max(worst_peak, score[0])
        if best is None or score < best[0]:
            best = (score, roll, tilt, g_dir)

    (peak, mean), roll, tilt, g_dir = best
    return PositioningChoice(
        gravity_m_s2=tuple(float(v) for v in STANDARD_GRAVITY_M_S2 * g_dir),
        roll_deg=roll,
        tilt_deg=tilt,
        peak_required_force_n=peak,
        mean_required_force_n=mean,
        worst_orientation_peak_force_n=worst_peak,
        net_weight_n=float(net_weight_n),
    )
