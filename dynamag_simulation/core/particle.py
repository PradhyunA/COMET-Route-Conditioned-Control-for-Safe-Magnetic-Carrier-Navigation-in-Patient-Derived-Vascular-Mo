"""
Superparamagnetic particle model with full inertial dynamics.

Equation of motion
------------------
    m * dv/dt = F_mag + F_drag + F_brownian

    F_drag = -gamma_eff(Re) * (v - v_flow)

    Re = rho_fluid * |v - v_flow| * d / eta

Drag model
----------
Schiller-Naumann correlation (valid Re up to ~800):
    C_D = (24 / Re) * (1 + 0.15 * Re^0.687)

which gives:
    gamma_eff = gamma_stokes * (1 + 0.15 * Re^0.687)

This reduces to Stokes drag at Re << 1.

Integration scheme
------------------
Semi-implicit first-order method per substep:

    gamma_n = gamma_eff( Re(v_n - v_flow) )          [explicit Re evaluation]

    v_{n+1} = (m * v_n + dt * (F_mag + gamma_n * v_flow + F_brown)) /
              (m + dt * gamma_n)

    x_{n+1} = x_n + dt * v_{n+1}

This is unconditionally stable in the drag term (implicit) while the
magnetic force and flow remain explicit. At Re << 1 and m -> 0 it
recovers the overdamped limit v = (F_mag + gamma * v_flow) / gamma
to machine precision.

Particle mass
-------------
Composite density from magnetic loading fraction:
    rho_particle = rho_iron_oxide * f_mag + rho_matrix * (1 - f_mag)
    rho_iron_oxide ~ 5000 kg/m^3  (magnetite Fe3O4)
    rho_matrix     ~ 1200 kg/m^3  (polymer/lipid carrier)

No particle_density field is needed in PhysicalProfile; it is derived.
"""

import numpy as np

from .constants import MU_0, K_B, PI, EPS
from .profiles import PhysicalProfile, MICROSPHERE_500UM

# Constituent densities for composite particle mass estimate
_RHO_IRON_OXIDE = 5000.0   # kg/m^3  (magnetite)
_RHO_MATRIX     = 1200.0   # kg/m^3  (polymer/lipid carrier)


class MagneticParticle3D:
    """Superparamagnetic microsphere in viscous fluid with full inertia."""

    def __init__(
        self,
        position,
        profile: PhysicalProfile = MICROSPHERE_500UM,
        rng_seed: int = 42,
    ):
        self.pos = np.asarray(position, dtype=float).copy()
        self.vel = np.zeros(3, dtype=float)          # true velocity [m/s]
        self.realized_vel = np.zeros(3, dtype=float)

        self.profile = profile
        self.radius = profile.diameter / 2.0
        self.volume = (4.0 / 3.0) * PI * self.radius ** 3
        self.magnetic_volume = self.volume * profile.magnetic_loading_fraction

        # Magnetic parameters
        self.chi_eff = profile.chi_eff
        self.M_sat   = profile.M_sat
        self.B_sat   = MU_0 * self.M_sat / self.chi_eff

        self.force_prefactor_linear = (
            self.magnetic_volume * self.chi_eff / (2.0 * MU_0)
        )
        self.force_prefactor_sat = self.magnetic_volume * self.M_sat

        # Fluid properties
        self.eta = profile.fluid_viscosity
        self.rho = profile.fluid_density
        self.diameter = profile.diameter

        # Stokes drag coefficient (includes geometry multiplier)
        self.drag_stokes = (
            6.0
            * PI
            * self.eta
            * self.radius
            * profile.hydrodynamic_drag_multiplier
        )
        # self.drag: effective drag used in the last step (backward compat)
        self.drag = self.drag_stokes

        # Particle mass via composite density
        rho_particle = (
            _RHO_IRON_OXIDE * profile.magnetic_loading_fraction
            + _RHO_MATRIX   * (1.0 - profile.magnetic_loading_fraction)
        )
        self.mass = self.volume * rho_particle

        # Momentum relaxation time (Stokes estimate, useful for diagnostics)
        self.tau_p = self.mass / self.drag_stokes

        # Brownian diffusion (Einstein relation, Stokes drag)
        self.temperature     = profile.temperature
        self.D               = K_B * self.temperature / self.drag_stokes
        self.enable_brownian = profile.enable_brownian
        self.rng = np.random.default_rng(rng_seed)

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _gamma_eff(self, v_rel_mag: float) -> float:
        """
        Schiller-Naumann effective drag coefficient.

        Parameters
        ----------
        v_rel_mag : float
            Magnitude of velocity relative to fluid [m/s].

        Returns
        -------
        gamma_eff : float  [N*s/m]
        """
        Re = self.rho * v_rel_mag * self.diameter / self.eta
        if Re < 1e-3:
            return self.drag_stokes
        correction = 1.0 + 0.15 * Re ** 0.687
        return self.drag_stokes * correction

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def magnetic_force(self, B_at_pos, grad_B_sq):
        """
        Magnetic gradient force [N].

        Parameters
        ----------
        B_at_pos  : array (3,)  Field vector at particle position.
        grad_B_sq : array (3,)  Gradient of |B|^2.
        """
        B_mag = np.linalg.norm(B_at_pos)
        if B_mag < self.B_sat:
            return self.force_prefactor_linear * grad_B_sq
        else:
            return self.force_prefactor_sat * grad_B_sq / (2.0 * B_mag + EPS)

    def update(self, force, dt, flow_velocity=None):
        """
        Advance particle by one substep using a semi-implicit inertial scheme.

        Scheme (unconditionally stable in drag term):

            gamma_n   = gamma_eff( |v_n - v_flow| )
            v_{n+1}   = (m*v_n + dt*(F_mag + gamma_n*v_flow + F_brown/dt)) /
                        (m + dt*gamma_n)
            x_{n+1}   = x_n + dt * v_{n+1}

        In the overdamped limit (m -> 0, tau_p << dt) this recovers:
            v = (F_mag + gamma*v_flow) / gamma  [= F/gamma + v_flow]

        Parameters
        ----------
        force        : array (3,)  Magnetic force [N].
        dt           : float       Substep duration [s].
        flow_velocity: array (3,) or None  Background fluid velocity [m/s].

        Side effects
        ------------
        self.vel          <- v_{n+1}  (true particle velocity)
        self.drag         <- gamma_eff used this step
        self.realized_vel <- displacement / dt  (same as vel for single substep)
        self.pos          <- updated position
        """
        pos_old = self.pos.copy()
        v_flow  = (
            np.zeros(3) if flow_velocity is None
            else np.asarray(flow_velocity, dtype=float)
        )

        # Evaluate drag at current relative velocity
        v_rel     = self.vel - v_flow
        v_rel_mag = np.linalg.norm(v_rel)
        gamma_n   = self._gamma_eff(v_rel_mag)
        self.drag = gamma_n

        # Brownian impulse: F_brown drawn as force-equivalent over dt
        if self.enable_brownian:
            # sigma_v from fluctuation-dissipation: var(delta_x) = 2*D*dt
            # Force-equivalent: F_b = m * delta_v / dt, but we add directly
            # as a position kick: delta_x ~ N(0, sqrt(2*D*dt))
            # Keep as a position-level correction applied after velocity update
            sigma_x   = np.sqrt(2.0 * self.D * dt)
            brown_pos = sigma_x * self.rng.standard_normal(3)
        else:
            brown_pos = np.zeros(3)

        # Semi-implicit velocity update
        # v_{n+1} = (m*v_n + dt*(F + gamma*v_flow)) / (m + dt*gamma)
        denom    = self.mass + dt * gamma_n
        self.vel = (self.mass * self.vel + dt * (force + gamma_n * v_flow)) / denom

        # Position update
        self.pos          = pos_old + dt * self.vel + brown_pos
        self.realized_vel = (self.pos - pos_old) / dt

    def reynolds_number(self) -> float:
        """Current Reynolds number based on last computed velocity."""
        v_rel_mag = np.linalg.norm(self.vel)   # approximate; exact only after update
        return self.rho * v_rel_mag * self.diameter / (self.eta + EPS)
