"""
MPC controller for 3D magnetic particle steering under flow.

Single-horizon (1-step lookahead) optimisation solved by SLSQP with
multiple heuristic initial guesses to avoid poor local minima.

The controller's prediction accounts for background flow: it predicts
where the particle will be after one step of flow convection + magnetic
drift, and optimises currents to steer against/with the flow.
"""

import numpy as np
from scipy.optimize import minimize

from .constants import EPS
from .flow import NoFlow

_NO_FLOW = NoFlow()


def _simulate_one_step(particle, particle_pos, currents,
                       magnet_array, workspace, dt,
                       flow=None, sim_time=0.0):
    """Deterministic 1-step prediction including flow convection."""
    particle_pos = np.asarray(particle_pos, dtype=float)
    magnet_array.set_currents(currents)

    B = magnet_array.field_at(particle_pos)
    grad_Bsq = magnet_array.grad_B_squared(particle_pos)
    force = particle.magnetic_force(B, grad_Bsq)

    force_mag = np.linalg.norm(force)
    if force_mag > EPS:
        gamma_eff, _, _ = particle._effective_drag(force_mag)
        v_mag = force / gamma_eff
    else:
        v_mag = np.zeros(3)

    if flow is None:
        v_flow = np.zeros(3)
    else:
        v_flow = flow.velocity_at(particle_pos, sim_time)

    step_vec = (v_flow + v_mag) * dt
    next_pos = workspace.clip(particle_pos + step_vec)

    return next_pos, force, step_vec


def _edge_penalty(pos, workspace, edge_buffer=0.006):
    margins = workspace.half - np.abs(pos)
    return max(0.0, edge_buffer - float(np.min(margins)))


def _overshoot_penalty(step_vec, current_distance, fraction=0.65):
    return max(0.0, np.linalg.norm(step_vec) - fraction * current_distance)


def _heuristic_starts(target_pos, particle_pos, magnet_array, max_current):
    """Cheap nonzero initial guesses to seed the optimizer."""
    desired = np.asarray(target_pos, dtype=float) - np.asarray(particle_pos, dtype=float)
    dist = np.linalg.norm(desired)
    n = magnet_array.n_magnets

    if dist < EPS:
        return [np.zeros(n, dtype=float)]

    desired_hat = desired / dist
    proj = magnet_array.orientations @ desired_hat
    sign_seed = np.sign(proj)
    sign_seed[sign_seed == 0.0] = 1.0

    full_seed = max_current * sign_seed
    half_seed = 0.5 * max_current * sign_seed
    neg_full = -full_seed

    best_idx = int(np.argmax(np.abs(proj)))
    single = np.zeros(n, dtype=float)
    single[best_idx] = max_current * np.sign(
        proj[best_idx] if proj[best_idx] != 0 else 1.0
    )

    return [
        np.zeros(n, dtype=float),
        full_seed.astype(float),
        half_seed.astype(float),
        neg_full.astype(float),
        single.astype(float),
    ]


def _mpc_cost(currents, particle, particle_pos, target_pos,
              magnet_array, workspace, dt,
              w_progress, w_align, w_dist, w_curr,
              w_smooth, prev_currents, w_edge, w_overshoot,
              flow, sim_time):
    """Scalar cost for a candidate current vector (lower is better)."""
    particle_pos = np.asarray(particle_pos, dtype=float)
    target_pos = np.asarray(target_pos, dtype=float)
    currents = np.asarray(currents, dtype=float)

    desired = target_pos - particle_pos
    current_distance = np.linalg.norm(desired)
    if current_distance < EPS:
        return 0.0
    desired_hat = desired / current_distance

    next_pos, force, step_vec = _simulate_one_step(
        particle, particle_pos, currents, magnet_array, workspace, dt,
        flow=flow, sim_time=sim_time,
    )

    next_distance = np.linalg.norm(target_pos - next_pos)
    progress = current_distance - next_distance
    step_along = np.dot(step_vec, desired_hat)

    force_norm = np.linalg.norm(force)
    alignment = np.dot(desired_hat, force / (force_norm + EPS))

    edge_cost = _edge_penalty(next_pos, workspace)
    overshoot_cost = _overshoot_penalty(step_vec, current_distance)

    smoothness_cost = 0.0
    if prev_currents is not None:
        smoothness_cost = np.sum((currents - prev_currents) ** 2)

    return float(
        -w_progress * progress
        - w_align * step_along
        - 0.5 * alignment
        + w_dist * (next_distance ** 2)
        + w_curr * np.sum(currents ** 2)
        + w_smooth * smoothness_cost
        + w_edge * edge_cost
        + w_overshoot * overshoot_cost
    )


def continuous_mpc_controller(
    particle,
    particle_pos,
    target_pos,
    magnet_array,
    workspace,
    dt: float,
    max_current: float = 8.0,
    initial_guess=None,
    w_progress: float = 80.0,
    w_align: float = 8.0,
    w_dist: float = 2.0,
    w_curr: float = 1e-5,
    w_smooth: float = 0.5,
    w_edge: float = 20.0,
    w_overshoot: float = 2.0,
    flow=None,
    sim_time: float = 0.0,
):
    """
    1-step MPC: find currents that minimise a weighted cost.

    The prediction includes background flow so the controller
    compensates for convection when planning magnetic actuation.

    Returns (best_currents, best_force, best_cost, optimizer_converged).
    """
    particle_pos = np.asarray(particle_pos, dtype=float)
    target_pos = np.asarray(target_pos, dtype=float)

    if flow is None:
        flow = _NO_FLOW

    if np.linalg.norm(target_pos - particle_pos) < EPS:
        z = np.zeros(magnet_array.n_magnets, dtype=float)
        return z, np.zeros(3, dtype=float), 0.0, True

    bounds = [(-max_current, max_current)] * magnet_array.n_magnets

    starts = _heuristic_starts(target_pos, particle_pos, magnet_array, max_current)
    if initial_guess is not None:
        ig = np.clip(np.asarray(initial_guess, dtype=float),
                     -max_current, max_current)
        starts.insert(0, ig)

    best_currents = np.zeros(magnet_array.n_magnets, dtype=float)
    best_cost = np.inf
    any_converged = False

    for x0 in starts:
        result = minimize(
            fun=_mpc_cost,
            x0=x0,
            args=(particle, particle_pos, target_pos, magnet_array,
                  workspace, dt, w_progress, w_align, w_dist, w_curr,
                  w_smooth, initial_guess, w_edge, w_overshoot,
                  flow, sim_time),
            method="SLSQP",
            bounds=bounds,
            options={"maxiter": 80, "ftol": 1e-7, "disp": False},
        )

        if result.success:
            any_converged = True

        cand = np.clip(result.x, -max_current, max_current)
        cand_cost = _mpc_cost(
            cand, particle, particle_pos, target_pos, magnet_array,
            workspace, dt, w_progress, w_align, w_dist, w_curr,
            w_smooth, initial_guess, w_edge, w_overshoot,
            flow, sim_time,
        )

        if cand_cost < best_cost:
            best_cost = cand_cost
            best_currents = cand.copy()

    magnet_array.set_currents(best_currents)
    B = magnet_array.field_at(particle_pos)
    grad_Bsq = magnet_array.grad_B_squared(particle_pos)
    best_force = particle.magnetic_force(B, grad_Bsq)

    return best_currents, best_force, float(best_cost), any_converged
