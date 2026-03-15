"""
Background flow field models for vascular phantom simulation.

In the overdamped regime, the particle velocity is:
    v_particle = v_flow(x, t) + F_magnetic / gamma

Flow models:
    NoFlow          -- static fluid (default, backward compatible)
    UniformFlow     -- constant velocity vector everywhere
    PoiseuilleFlow  -- parabolic profile in a cylindrical channel
    PulsatileFlow   -- Poiseuille modulated by a cardiac waveform
"""

import numpy as np
from .constants import PI


class NoFlow:
    """Static fluid -- no background flow velocity."""
    name = "no_flow"

    def velocity_at(self, position, time=0.0):
        return np.zeros(3, dtype=float)

    def params_dict(self):
        return {"flow_model": self.name}


class UniformFlow:
    """
    Spatially uniform, time-constant flow.

    Suitable when the channel cross-section is large compared to the
    particle and the region of interest is far from walls.

    Parameters
    ----------
    velocity : array-like, shape (3,)
        Flow velocity vector in m/s.
    """
    name = "uniform"

    def __init__(self, velocity):
        self.velocity = np.asarray(velocity, dtype=float)

    def velocity_at(self, position, time=0.0):
        return self.velocity.copy()

    def params_dict(self):
        return {
            "flow_model": self.name,
            "flow_vx": self.velocity[0],
            "flow_vy": self.velocity[1],
            "flow_vz": self.velocity[2],
            "flow_speed_m_s": float(np.linalg.norm(self.velocity)),
        }


class PoiseuilleFlow:
    """
    Steady Poiseuille (parabolic) flow in a cylindrical channel.

    v(r) = v_max * (1 - (r/R)^2) * axis_hat

    where r is the radial distance from the channel centerline.

    Parameters
    ----------
    center : array-like, shape (3,)
        A point on the channel centerline.
    axis : array-like, shape (3,)
        Unit vector along the channel direction.
    radius : float
        Channel radius in meters.
    v_max : float
        Centerline (peak) velocity in m/s.
    """
    name = "poiseuille"

    def __init__(self, center, axis, radius, v_max):
        self.center = np.asarray(center, dtype=float)
        self.axis = np.asarray(axis, dtype=float)
        self.axis = self.axis / np.linalg.norm(self.axis)
        self.radius = float(radius)
        self.v_max = float(v_max)

    def velocity_at(self, position, time=0.0):
        pos = np.asarray(position, dtype=float)
        # Project position onto plane perpendicular to axis through center
        d = pos - self.center
        axial = np.dot(d, self.axis)
        radial_vec = d - axial * self.axis
        r = np.linalg.norm(radial_vec)

        if r >= self.radius:
            return np.zeros(3, dtype=float)

        profile = 1.0 - (r / self.radius) ** 2
        return self.v_max * profile * self.axis

    def mean_velocity(self):
        """Cross-section averaged velocity = v_max / 2."""
        return self.v_max / 2.0

    def params_dict(self):
        return {
            "flow_model": self.name,
            "flow_channel_radius_m": self.radius,
            "flow_v_max_m_s": self.v_max,
            "flow_v_mean_m_s": self.mean_velocity(),
            "flow_axis": self.axis.tolist(),
        }


class PulsatileFlow:
    """
    Pulsatile Poiseuille flow: cardiac-cycle modulation.

    v(r, t) = v_max * (1 - (r/R)^2) * [1 + amplitude * sin(2*pi*f*t)] * axis_hat

    Parameters
    ----------
    center, axis, radius, v_max : same as PoiseuilleFlow
    heart_rate_bpm : float
        Heart rate in beats per minute (default 60 = 1 Hz).
    pulsation_amplitude : float
        Fractional amplitude of pulsation (default 0.3 = +/- 30%).
    """
    name = "pulsatile"

    def __init__(self, center, axis, radius, v_max,
                 heart_rate_bpm=60.0, pulsation_amplitude=0.3):
        self.center = np.asarray(center, dtype=float)
        self.axis = np.asarray(axis, dtype=float)
        self.axis = self.axis / np.linalg.norm(self.axis)
        self.radius = float(radius)
        self.v_max = float(v_max)
        self.freq = float(heart_rate_bpm) / 60.0
        self.amplitude = float(pulsation_amplitude)

    def velocity_at(self, position, time=0.0):
        pos = np.asarray(position, dtype=float)
        d = pos - self.center
        axial = np.dot(d, self.axis)
        radial_vec = d - axial * self.axis
        r = np.linalg.norm(radial_vec)

        if r >= self.radius:
            return np.zeros(3, dtype=float)

        profile = 1.0 - (r / self.radius) ** 2
        pulsation = 1.0 + self.amplitude * np.sin(2.0 * PI * self.freq * time)
        return self.v_max * profile * pulsation * self.axis

    def params_dict(self):
        return {
            "flow_model": self.name,
            "flow_channel_radius_m": self.radius,
            "flow_v_max_m_s": self.v_max,
            "flow_heart_rate_bpm": self.freq * 60.0,
            "flow_pulsation_amplitude": self.amplitude,
            "flow_axis": self.axis.tolist(),
        }
