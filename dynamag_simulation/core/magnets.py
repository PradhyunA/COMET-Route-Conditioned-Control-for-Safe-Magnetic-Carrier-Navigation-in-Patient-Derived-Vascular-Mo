"""
Electromagnet array: 9-magnet triangular-prism regularized dipole model.

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

Reference:
Jackson, Classical Electrodynamics, ch. 5
Furlani, Permanent Magnet and Electromechanical Devices (2001)
"""

import numpy as np

from .constants import MU_0, PI, EPS


class ElectromagnetArray3D:
    """
    9 electromagnets arranged in a triangular prism.

    Magnetic moment:
        m = N I pi R^2

    Regularized dipole field:
        B(r) = (mu_0 / 4pi)
               [3(m·r_hat)r_hat - m]
               / (r^2 + a^2)^(3/2)

    A simple 1018 steel-core approximation is included:
        B_core = mu_r * B_air

    with saturation capped at ~1.8 T.
    """

    def __init__(
        self,
        coil_radius: float = 0.03,
        coil_length: float = 0.12,
        turns: float = 3000.0,
        max_current: float = 16.0,
        placement_radius: float = 0.02,
        center=None,
    ):
        self.placement_radius = float(placement_radius)
        self.center = (
            np.zeros(3, dtype=float)
            if center is None
            else np.asarray(center, dtype=float)
        )

        d = self.placement_radius

        front = np.array([
            [0.0,      2.0 * d / np.sqrt(3.0), -d],
            [-d,      -d / np.sqrt(3.0),      -d],
            [ d,      -d / np.sqrt(3.0),      -d],
        ])

        back = np.array([
            [0.0,      2.0 * d / np.sqrt(3.0),  d],
            [-d,      -d / np.sqrt(3.0),       d],
            [ d,      -d / np.sqrt(3.0),       d],
        ])

        mid = np.array([
            0.5 * (front[0] + back[0]),
            0.5 * (front[1] + back[1]),
            0.5 * (front[2] + back[2]),
        ])

        local_positions = np.vstack((front, back, mid))
        self.positions = local_positions + self.center

        self.n_magnets = len(self.positions)

        # Point all magnets toward the array center.
        self.orientations = np.zeros_like(local_positions)

        for i, pos in enumerate(local_positions):
            norm = np.linalg.norm(pos)

            if norm < 1e-12:
                self.orientations[i] = np.array([0.0, 0.0, 1.0])
            else:
                self.orientations[i] = -pos / norm

        self.coil_radius = float(coil_radius)
        self.coil_length = float(coil_length)
        self.turns = float(turns)
        self.max_current = float(max_current)

        self.currents = np.zeros(self.n_magnets, dtype=float)

        self.moment_per_amp = (
            self.turns *
            PI *
            self.coil_radius ** 2
        )

        self._mu_over_4pi = MU_0 / (4.0 * PI)

        # Regularization length
        self._a_sq = self.coil_radius ** 2

        # --------------------------------------------------
        # 1018 Steel Core Approximation
        # --------------------------------------------------

        self.core_relative_permeability = 200.0
        self.core_saturation_tesla = 1.8

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
        Simple 1018 steel core approximation.

        Amplify field by mu_r then clamp near saturation.
        """

        B_mag = np.linalg.norm(B)

        if B_mag < EPS:
            return B

        amplified = B * self.core_relative_permeability

        amp_mag = np.linalg.norm(amplified)

        if amp_mag > self.core_saturation_tesla:
            amplified *= (
                self.core_saturation_tesla /
                amp_mag
            )

        return amplified

    def field_at(self, point):
        """
        Net B-field (Tesla) at point.
        """

        point = np.asarray(point, dtype=float)

        B_total = np.zeros(3, dtype=float)

        for i in range(self.n_magnets):

            current = self.currents[i]

            m_mag = self.moment_per_amp * current

            m_vec = m_mag * self.orientations[i]

            r_vec = point - self.positions[i]

            r_sq = np.dot(r_vec, r_vec)

            r = np.sqrt(r_sq)

            if r < 1e-10:
                continue

            r_hat = r_vec / r

            m_dot_rhat = np.dot(m_vec, r_hat)

            denom = (r_sq + self._a_sq) ** 1.5

            B_i = self._mu_over_4pi * (
                3.0 * m_dot_rhat * r_hat
                - m_vec
            ) / denom

            B_total += B_i

        B_total = self._apply_core_model(B_total)

        return B_total

    def grad_B_squared(self, point, h: float = 5e-5):
        """
        Gradient of |B|^2 using central differences.
        """

        point = np.asarray(point, dtype=float)

        def B_sq(p):
            B = self.field_at(p)
            return np.dot(B, B)

        grad = np.zeros(3)

        for axis in range(3):
            dp = np.zeros(3)
            dp[axis] = h

            grad[axis] = (
                B_sq(point + dp)
                - B_sq(point - dp)
            ) / (2.0 * h)

        return grad
