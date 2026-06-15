"""
Superparamagnetic particle model with nonlinear drag correction.

Force model
-----------
Linear (unsaturated):   F = (V * chi_eff) / (2 mu_0) * grad(|B|^2)
Saturated:              F = V * M_s * grad(|B|)   = V * M_s / (2|B|) * grad(|B|^2)

Drag model
----------
Schiller-Naumann correlation (valid for Re up to ~800):
    C_D = (24 / Re) * (1 + 0.15 * Re^0.687)

This reduces to Stokes drag at Re << 1 and provides physically
correct increased drag at higher Re, preventing unphysical velocities
near field singularities (e.g., close to dipole sources).

At each step, the effective drag is solved iteratively since Re depends
on velocity which depends on drag.
"""

import numpy as np

from .constants import MU_0, K_B, PI, EPS
from .profiles import PhysicalProfile, MICROSPHERE_500UM


class MagneticParticle3D:
    """Superparamagnetic microsphere in viscous fluid."""

    def __init__(
        self,
        position,
        profile: PhysicalProfile = MICROSPHERE_500UM,
        rng_seed: int = 42,
    ):
        self.pos = np.asarray(position, dtype=float).copy()
        self.vel = np.zeros(3, dtype=float)
        self.realized_vel = np.zeros(3, dtype=float)

        self.profile = profile
        self.radius = profile.diameter / 2.0
        self.volume = (4.0 / 3.0) * PI * self.radius ** 3
        self.magnetic_volume = self.volume * profile.magnetic_loading_fraction

        self.chi_eff = profile.chi_eff
        self.M_sat = profile.M_sat
        self.B_sat = MU_0 * self.M_sat / self.chi_eff

        self.force_prefactor_linear = (
            self.magnetic_volume * self.chi_eff / (2.0 * MU_0)
        )
        self.force_prefactor_sat = self.magnetic_volume * self.M_sat

        self.eta = profile.fluid_viscosity
        self.rho = profile.fluid_density
        self.diameter = profile.diameter
        self.drag_stokes = (
            6.0
            * PI
            * self.eta
            * self.radius
            * profile.hydrodynamic_drag_multiplier
        )

        # Keep .drag as Stokes value for backward compat & diffusion calc
        self.drag = self.drag_stokes

        self.temperature = profile.temperature
        self.D = K_B * self.temperature / self.drag_stokes
        self.enable_brownian = profile.enable_brownian
        self.rng = np.random.default_rng(rng_seed)

    def _effective_drag(self, force_mag):
        """
        Iterative Schiller-Naumann drag for a given force magnitude.

        Solves: v = F / gamma_eff(Re(v)) via fixed-point iteration.
        Returns (gamma_eff, velocity_mag, Re).
        """
        gamma_s = self.drag_stokes
        v = force_mag / gamma_s  # Stokes estimate
        re = self.rho * v * self.diameter / self.eta

        if re < 0.01:
            return gamma_s, v, re

        for _ in range(15):
            correction = 1.0 + 0.15 * re ** 0.687
            gamma_eff = gamma_s * correction
            v_new = force_mag / gamma_eff
            re_new = self.rho * v_new * self.diameter / self.eta
            if abs(re_new - re) < 1e-4 * (re + 1e-30):
                return gamma_eff, v_new, re_new
            re = re_new
            v = v_new

        return gamma_eff, v, re

    def magnetic_force(self, B_at_pos, grad_B_sq):
        """Magnetic gradient force in Newtons."""
        B_mag = np.linalg.norm(B_at_pos)

        if B_mag < self.B_sat:
            F = self.force_prefactor_linear * grad_B_sq
        else:
            F = self.force_prefactor_sat * grad_B_sq / (2.0 * B_mag)

        return F

    def update(self, force, dt, flow_velocity=None):
        """
        Overdamped step with background flow, nonlinear drag, and Brownian noise.

        The particle velocity in the overdamped regime is:
            v_particle = v_flow + F_magnetic / gamma_eff

        The drag and Reynolds number are computed from the *magnetic*
        velocity component only (velocity relative to the fluid), since
        the particle is entrained by the flow with no relative slip at
        the flow velocity.

        After this call:
          - self.vel          = magnetic component F / gamma_eff
          - self.realized_vel = (pos_new - pos_old) / dt (total)
          - self.drag         = effective drag used this step
        """
        pos_old = self.pos.copy()

        if flow_velocity is None:
            v_flow = np.zeros(3)
        else:
            v_flow = np.asarray(flow_velocity, dtype=float)

        force_mag = np.linalg.norm(force)
        if force_mag < EPS:
            self.vel = np.zeros(3)
            gamma_eff = self.drag_stokes
        else:
            gamma_eff, _, _ = self._effective_drag(force_mag)
            self.vel = force / gamma_eff

        self.drag = gamma_eff

        # Total deterministic step: flow convection + magnetic drift
        det_step = (v_flow + self.vel) * dt

        if self.enable_brownian:
            sigma = np.sqrt(2.0 * self.D * dt)
            brownian_step = sigma * self.rng.standard_normal(3)
        else:
            brownian_step = np.zeros(3)

        self.pos = pos_old + det_step + brownian_step
        self.realized_vel = (self.pos - pos_old) / dt
