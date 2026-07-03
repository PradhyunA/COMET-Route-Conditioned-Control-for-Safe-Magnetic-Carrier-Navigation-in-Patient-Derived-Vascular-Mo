"""
Adaptive substepping integrator for inertial magnetic particle dynamics.

The particle now carries true momentum (m*dv/dt != 0), so displacement
estimation for substep sizing uses the actual particle velocity rather
than the overdamped approximation (F/gamma).

Substep sizing criterion
------------------------
Maximum displacement per substep is bounded by max_substep_disp.
Estimated total displacement is:

    est = |v_particle + v_flow| * dt

where v_particle is the current true velocity (not F/drag).  This is
conservative: it uses velocity at the start of the interval.  For
rapidly accelerating particles the controller's dt is typically short
enough that this does not over-subdivide.

Flow support
------------
The flow object must implement:
    velocity_at(position: np.ndarray, time: float) -> np.ndarray (3,)

NoFlow() from flow.py satisfies this interface.
"""

import numpy as np
from .flow import NoFlow

_NO_FLOW = NoFlow()


def adaptive_step(
    particle,
    force,
    dt: float,
    workspace,
    max_substep_disp: float = 5e-3,
    flow=None,
    sim_time: float = 0.0,
):
    """
    Advance *particle* by one control interval of duration *dt*.

    Parameters
    ----------
    particle : MagneticParticle3D
        Particle with inertial state (pos, vel, mass, drag_stokes, …).
    force : np.ndarray, shape (3,)
        Magnetic force, held constant across all substeps.
    dt : float
        Control interval duration [s].
    workspace : Workspace3D
        Provides workspace.clip(pos) to enforce domain boundaries.
    max_substep_disp : float
        Maximum allowed displacement per substep [m].
        Default 5 mm — coarser than the overdamped version is fine
        because the semi-implicit scheme is unconditionally stable.
    flow : flow field object or None
        Must implement ``velocity_at(position, time) -> ndarray (3,)``.
        Defaults to NoFlow (zero everywhere).
    sim_time : float
        Simulation time at the start of this control interval [s].

    Returns
    -------
    realized_displacement : np.ndarray, shape (3,)
        Net displacement over the full interval.
    n_substeps : int
        Number of substeps taken.

    Notes
    -----
    Because the semi-implicit drag update is unconditionally stable,
    substep subdivision is only needed to keep the *spatial* resolution
    fine enough (so that force/flow evaluated at particle position stays
    accurate).  It is NOT required for numerical stability.

    The particle's velocity state (particle.vel) is preserved across
    substeps and carries momentum from one control interval to the next,
    which is the key difference from the overdamped integrator.
    """
    if flow is None:
        flow = _NO_FLOW

    # ------------------------------------------------------------------
    # Determine number of substeps from current velocity estimate
    # ------------------------------------------------------------------
    v_flow_est  = flow.velocity_at(particle.pos, sim_time)

    # Total speed estimate: particle velocity + flow convection
    # For the very first step particle.vel == 0; fall back to F/drag to
    # avoid zero-substep estimate on startup.
    v_particle_mag = np.linalg.norm(particle.vel)
    if v_particle_mag < 1e-12:
        # Startup: use overdamped estimate as a conservative seed
        v_particle_mag = np.linalg.norm(force) / (particle.drag_stokes + 1e-30)

    v_total_mag = v_particle_mag + np.linalg.norm(v_flow_est)
    total_est   = v_total_mag * dt

    n_subs  = max(1, int(np.ceil(total_est / max_substep_disp)))
    sub_dt  = dt / n_subs

    # ------------------------------------------------------------------
    # Substep loop
    # ------------------------------------------------------------------
    pos_start = particle.pos.copy()

    for k in range(n_subs):
        sub_time  = sim_time + k * sub_dt
        v_flow_k  = flow.velocity_at(particle.pos, sub_time)
        particle.update(force, sub_dt, flow_velocity=v_flow_k)
        particle.pos = workspace.clip(particle.pos)

    # ------------------------------------------------------------------
    # Aggregate realized displacement
    # ------------------------------------------------------------------
    realized_disp         = particle.pos - pos_start
    particle.realized_vel = realized_disp / dt

    return realized_disp, n_subs
