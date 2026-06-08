"""
Electromagnet array: 9-magnet triangular-prism regularized dipole model.

Body-scale scanner version.

Layout:
    Front triangle (z = -d)
        A
       / \
      B---C

    Back triangle (z = +d)
        D
       / \
      E---F

    Mid-edge magnets:
        G between A-D
        H between B-E
        I between C-F

All magnets point toward the configured array center.

Each solenoid is modelled as a regularized magnetic dipole.

Magnetic moment:
    m = N I pi R^2

Regularized dipole field:
    B(r) = (mu_0 / 4pi)
           [3(m·r_hat)r_hat - m]
           / (r^2 + a^2)^(3/2)

This file intentionally models a fixed body-scale steering device, not a
vessel-bounds-scaled device.

Important:
    This is a simulation authority model.
    It does NOT solve coil resistance, heating, power electronics, duty cycle,
    voltage limits, insulation, cooling, or patient safety constraints.

Reference:
Jackson, Classical Electrodynamics, ch. 5
Furlani, Permanent Magnet and Electromechanical Devices (2001)
"""

from __future__ import annotations

import numpy as np

from .constants import MU_0, PI, EPS


class ElectromagnetArray3D:
    """
    9 electromagnets arranged in a triangular prism.

    Default body-scale scanner geometry:
        placement_radius = 0.300 m   # coil centers 30 cm from device center
        coil_radius      = 0.080 m   # 16 cm coil diameter
        coil_length      = 0.280 m   # 28 cm core/coil length
        core_radius      = 0.040 m   # 8 cm core diameter
        turns            = 20000
        max_current      = 60 A peak/pulsed simulation limit

    Core model:
        core_relative_permeability = 90
        core_saturation_tesla      = 6.0

    Why these values:
        - 35 / 8.0 was too weak and barely progressed.
        - 120 / 2.2 caused repeated flat-gradient saturation.
        - 90 / 6.0 is a compromise: high authority without the low 2.2 T cap.

    Notes:
        - This is a simulation authority model.
        - AWG, resistance, heating, voltage, and thermal limits are not solved here.
        - The steel-core model is an effective gain with soft saturation.
        - 60 A should be interpreted as a pulsed peak simulation limit,
          not continuous DC.
    """

    def __init__(
        self,
        coil_radius: float = 0.080,
        coil_length: float = 0.280,
        turns: float = 20000.0,
        max_current: float = 60.0,
        placement_radius: float = 0.300,
        center=None,
        core_radius: float = 0.040,
        core_relative_permeability: float = 90.0,
        core_saturation_tesla: float = 6.0,
    ):
        self.placement_radius = float(placement_radius)
        self.center = (
            np.zeros(3, dtype=float)
            if center is None
            else np.asarray(center, dtype=float)
        )

        self.coil_radius = float(coil_radius)
        self.coil_length = float(coil_length)
        self.turns = float(turns)
        self.max_current = float(max_current)

        self.core_radius = float(core_radius)
        self.core_relative_permeability = float(core_relative_permeability)
        self.core_saturation_tesla = float(core_saturation_tesla)

        self._mu_over_4pi = MU_0 / (4.0 * PI)

        # Regularization length.
        # Use a nonzero effective source size so the dipole field does not blow up
        # near a coil. This also roughly represents finite coil/core size.
        #
        # For a large body-scale scanner, the core radius dominates the
        # regularization. This prevents unrealistic near-singular fields while
        # preserving useful gradients in the bore.
        reg_len = max(0.5 * self.coil_radius, self.core_radius, 1e-4)
        self._a_sq = reg_len ** 2

        self._local_positions = self._build_local_positions()
        self.positions = self._local_positions + self.center
        self.n_magnets = len(self.positions)

        self.orientations = self._build_orientations(self._local_positions)

        self.currents = np.zeros(self.n_magnets, dtype=float)

        # Magnetic moment per ampere.
        # m = N I A, where A = pi R^2.
        self.moment_per_amp = self.turns * PI * self.coil_radius ** 2

    def _build_local_positions(self):
        d = self.placement_radius

        front = np.array(
            [
                [0.0, 2.0 * d / np.sqrt(3.0), -d],
                [-d, -d / np.sqrt(3.0), -d],
                [d, -d / np.sqrt(3.0), -d],
            ],
            dtype=float,
        )

        back = np.array(
            [
                [0.0, 2.0 * d / np.sqrt(3.0), d],
                [-d, -d / np.sqrt(3.0), d],
                [d, -d / np.sqrt(3.0), d],
            ],
            dtype=float,
        )

        mid = np.array(
            [
                0.5 * (front[0] + back[0]),
                0.5 * (front[1] + back[1]),
                0.5 * (front[2] + back[2]),
            ],
            dtype=float,
        )

        return np.vstack((front, back, mid))

    def _build_orientations(self, local_positions):
        orientations = np.zeros_like(local_positions)

        for i, pos in enumerate(local_positions):
            norm = np.linalg.norm(pos)

            if norm < 1e-12:
                orientations[i] = np.array([0.0, 0.0, 1.0])
            else:
                # Point all magnets toward array center.
                orientations[i] = -pos / norm

        return orientations

    def recenter(self, center):
        """
        Move the entire magnetic steering frame to a new center.

        This is useful for simulating a moving scanner/gantry frame.

        It does not change coil geometry, orientations, turns, or currents.
        """
        self.center = np.asarray(center, dtype=float)
        self.positions = self._local_positions + self.center

    def set_currents(self, currents):
        currents = np.asarray(currents, dtype=float)

        if currents.shape != (self.n_magnets,):
            raise ValueError(
                f"expected {self.n_magnets} currents, got shape {currents.shape}"
            )

        self.currents = np.clip(
            currents,
            -self.max_current,
            self.max_current,
        )

    def _apply_core_model(self, B):
        """
        Effective soft-saturating steel-core approximation.

        A naive B *= mu_r followed by hard clipping can flatten gradients too
        aggressively. This uses smooth tanh saturation:

            B_out = B_hat * B_sat * tanh(mu_eff * |B_air| / B_sat)

        This preserves a usable gradient before saturation while preventing
        unrealistic multi-tesla fields.
        """
        B = np.asarray(B, dtype=float)
        B_mag = float(np.linalg.norm(B))

        if B_mag < EPS:
            return B

        direction = B / B_mag

        effective_mag = self.core_saturation_tesla * np.tanh(
            self.core_relative_permeability * B_mag
            / (self.core_saturation_tesla + EPS)
        )

        return direction * effective_mag

    def field_at(self, point):
        """
        Net B-field in Tesla at point.
        """
        point = np.asarray(point, dtype=float)

        B_total_air = np.zeros(3, dtype=float)

        for i in range(self.n_magnets):
            current = self.currents[i]

            if abs(current) < EPS:
                continue

            m_mag = self.moment_per_amp * current
            m_vec = m_mag * self.orientations[i]

            r_vec = point - self.positions[i]
            r_sq = float(np.dot(r_vec, r_vec))

            r = np.sqrt(r_sq)

            if r < 1e-10:
                continue

            r_hat = r_vec / r
            m_dot_rhat = float(np.dot(m_vec, r_hat))

            denom = (r_sq + self._a_sq) ** 1.5

            B_i = self._mu_over_4pi * (
                3.0 * m_dot_rhat * r_hat - m_vec
            ) / denom

            B_total_air += B_i

        return self._apply_core_model(B_total_air)

    def grad_B_squared(self, point, h: float = 1e-4):
        """
        Gradient of |B|^2 using central differences.

        h = 1e-4 m gives a stable 0.1 mm finite-difference step for vessel-scale
        steering. Smaller values can become noisy with saturation.
        """
        point = np.asarray(point, dtype=float)

        def B_sq(p):
            B = self.field_at(p)
            return float(np.dot(B, B))

        grad = np.zeros(3, dtype=float)

        for axis in range(3):
            dp = np.zeros(3, dtype=float)
            dp[axis] = h

            grad[axis] = (
                B_sq(point + dp) - B_sq(point - dp)
            ) / (2.0 * h)

        return grad

    def params_dict(self):
        return {
            "n_magnets": self.n_magnets,
            "placement_radius_m": self.placement_radius,
            "coil_radius_m": self.coil_radius,
            "coil_diameter_m": 2.0 * self.coil_radius,
            "coil_length_m": self.coil_length,
            "turns": self.turns,
            "max_current_A": self.max_current,
            "current_mode": "peak_pulsed_simulation_limit",
            "core_radius_m": self.core_radius,
            "core_diameter_m": 2.0 * self.core_radius,
            "core_relative_permeability_effective": (
                self.core_relative_permeability
            ),
            "core_saturation_tesla": self.core_saturation_tesla,
            "moment_per_amp_A_m2": self.moment_per_amp,
            "thermal_model_included": False,
            "awg_model_included": False,
            "body_scale_scanner_model": True,
        }