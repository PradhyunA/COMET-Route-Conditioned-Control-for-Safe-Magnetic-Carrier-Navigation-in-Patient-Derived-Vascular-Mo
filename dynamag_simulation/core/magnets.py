"""
Electromagnet array: 8-corner regularized dipole model.

Each solenoid is modelled as a regularized magnetic dipole. The
regularization replaces |r|^3 with (|r|^2 + a^2)^(3/2) where a is
a softening length set to the coil radius. This prevents the
unphysical field divergence of a point dipole at close range and
better approximates the finite-extent solenoid field.

Reference: Jackson, Classical Electrodynamics, ch. 5 (dipole
regularization); Furlani, Permanent Magnet and Electromechanical
Devices (2001), ch. 4 (finite-coil corrections).
"""

import itertools
import numpy as np

from .constants import MU_0, PI, HALF, EPS


class ElectromagnetArray3D:
    """
    8 corner electromagnets modelled as regularized magnetic dipoles.

    Magnetic moment per coil:
        m_vec = N * I * pi * R^2 * orientation_hat   (A*m^2)

    Regularized dipole field:
        B(r) = (mu_0 / 4pi) * [3(m . r_hat) r_hat - m] / (|r|^2 + a^2)^(3/2)

    where a = coil_radius prevents the 1/r^3 singularity.
    At |r| >> a this reduces to the standard dipole formula.
    """

    def __init__(
        self,
        coil_radius: float = 0.03,
        coil_length: float = 0.12,
        turns: float = 1500,
        max_current: float = 8.0,
        placement_radius: float = None,
    ):
        """
        Parameters
        ----------
        placement_radius : float or None
            Half-side of the cube on whose corners the 8 coils sit (m).
            Defaults to HALF (76.2 mm).  Reducing this brings the magnets
            closer to the vessel network and dramatically increases the
            field gradient (scales as ~1/r^4 for gradient force).
        """
        pr = HALF if placement_radius is None else float(placement_radius)
        self.placement_radius = pr
        signs = list(itertools.product((-1, 1), repeat=3))
        self.positions = np.array(
            [[s[0] * pr, s[1] * pr, s[2] * pr] for s in signs],
            dtype=float,
        )
        inv_sqrt3 = 1.0 / np.sqrt(3.0)
        self.orientations = np.array(
            [[-s[0] * inv_sqrt3, -s[1] * inv_sqrt3, -s[2] * inv_sqrt3]
             for s in signs],
            dtype=float,
        )

        self.n_magnets = 8
        self.coil_radius = float(coil_radius)
        self.coil_length = float(coil_length)
        self.turns = float(turns)
        self.max_current = float(max_current)
        self.currents = np.zeros(self.n_magnets, dtype=float)

        self.moment_per_amp = self.turns * PI * self.coil_radius ** 2
        self._mu_over_4pi = MU_0 / (4.0 * PI)

        # Softening length: prevents singularity at r -> 0.
        # Set to coil_radius since the dipole model is invalid for r < R anyway.
        self._a_sq = self.coil_radius ** 2

    def set_currents(self, currents):
        self.currents = np.clip(
            np.asarray(currents, dtype=float),
            -self.max_current, self.max_current,
        )

    def field_at(self, point):
        """Net B-field (Tesla) at *point* from all 8 regularized dipoles."""
        point = np.asarray(point, dtype=float)
        B_total = np.zeros(3, dtype=float)

        for i in range(self.n_magnets):
            m_mag = self.moment_per_amp * self.currents[i]
            m_vec = m_mag * self.orientations[i]

            r_vec = point - self.positions[i]
            r_sq = np.dot(r_vec, r_vec)
            r = np.sqrt(r_sq)
            if r < 1e-10:
                continue
            r_hat = r_vec / r

            m_dot_rhat = np.dot(m_vec, r_hat)

            # Regularized denominator: (r^2 + a^2)^(3/2)
            denom = (r_sq + self._a_sq) ** 1.5

            B_i = self._mu_over_4pi * (
                3.0 * m_dot_rhat * r_hat - m_vec
            ) / denom

            B_total += B_i

        return B_total

    def grad_B_squared(self, point, h: float = 5e-5):
        """Gradient of |B|^2 via central finite differences (T^2/m)."""
        point = np.asarray(point, dtype=float)

        def B_sq(p):
            B = self.field_at(p)
            return np.dot(B, B)

        grad = np.zeros(3, dtype=float)
        for axis in range(3):
            dp = np.zeros(3)
            dp[axis] = h
            grad[axis] = (B_sq(point + dp) - B_sq(point - dp)) / (2.0 * h)

        return grad
