"""
Adaptive substepping integrator for overdamped magnetic particle dynamics.

Supports background flow fields: the flow velocity at the particle
position is evaluated at each sub-step and added to the magnetic drift.
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
    force : np.ndarray, shape (3,)
        Magnetic force (held constant across sub-steps).
    dt : float
        Control interval duration.
    workspace : Workspace3D
    max_substep_disp : float
        Maximum displacement per sub-step before subdivision.
    flow : flow field object (optional)
        Must implement ``velocity_at(position, time)``.
    sim_time : float
        Current simulation time (for time-dependent flows).

    Returns
    -------
    realized_displacement : np.ndarray, shape (3,)
    n_substeps : int
    """
    if flow is None:
        flow = _NO_FLOW

    # Estimate total displacement (magnetic + flow) to decide sub-stepping
    v_flow = flow.velocity_at(particle.pos, sim_time)
    body_force = getattr(particle, "body_force", 0.0)
    mag_disp = np.linalg.norm(((force + body_force) / particle.drag) * dt)
    flow_disp = np.linalg.norm(v_flow * dt)
    total_est = mag_disp + flow_disp

    n_subs = max(1, int(np.ceil(total_est / max_substep_disp)))
    sub_dt = dt / n_subs

    pos_start = particle.pos.copy()

    for k in range(n_subs):
        sub_time = sim_time + k * sub_dt
        v_flow_k = flow.velocity_at(particle.pos, sub_time)
        particle.update(force, sub_dt, flow_velocity=v_flow_k)
        particle.pos = workspace.clip(particle.pos)

    realized_disp = particle.pos - pos_start
    particle.realized_vel = realized_disp / dt

    return realized_disp, n_subs
