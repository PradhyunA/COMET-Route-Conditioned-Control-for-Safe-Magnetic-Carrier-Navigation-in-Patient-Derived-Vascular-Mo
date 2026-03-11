import numpy as np
from scipy.optimize import minimize

from simulation_3d_realistic import MAX_STEP, EPS


def simulate_one_step_3d(
    particle,
    particle_pos,
    currents,
    magnet_array,
    workspace,
    dt,
    max_step=MAX_STEP,
):
    """
    Predict one deterministic step forward for a given 8-current vector.

    Returns
    -------
    next_pos : np.ndarray shape (3,)
    force    : np.ndarray shape (3,)
    raw_step : np.ndarray shape (3,)   # before limiting
    step_vec : np.ndarray shape (3,)   # after limiting
    """
    particle_pos = np.asarray(particle_pos, dtype=float)
    currents = np.asarray(currents, dtype=float)

    magnet_array.set_currents(currents)

    B = magnet_array.field_at(particle_pos)
    grad_Bsq = magnet_array.grad_B_squared(particle_pos)
    force = particle.magnetic_force(B, grad_Bsq)

    raw_step = (force / particle.drag) * dt
    step_vec = raw_step.copy()

    step_mag = np.linalg.norm(step_vec)
    if step_mag > max_step:
        step_vec *= max_step / (step_mag + EPS)

    next_pos = workspace.clip(particle_pos + step_vec)
    return next_pos, force, raw_step, step_vec


def edge_penalty_3d(pos, workspace, edge_buffer=0.006):
    margins = workspace.half - np.abs(pos)
    min_margin = np.min(margins)
    return max(0.0, edge_buffer - min_margin)


def overshoot_penalty_3d(raw_step, current_distance, overshoot_fraction=0.65):
    step_mag = np.linalg.norm(raw_step)
    return max(0.0, step_mag - overshoot_fraction * current_distance)


def heuristic_initial_guesses_3d(target_pos, particle_pos, magnet_array, max_current):
    """
    few cheap, nonzero seeds so the optimizer does not get stuck at zero current.
    """
    desired = np.asarray(target_pos, dtype=float) - np.asarray(particle_pos, dtype=float)
    dist = np.linalg.norm(desired)
    if dist < EPS:
        return [np.zeros(magnet_array.n_magnets, dtype=float)]

    desired_hat = desired / dist

    # Use inward coil orientations as a directional heuristic.
    proj = magnet_array.orientations @ desired_hat
    sign_seed = np.sign(proj)
    sign_seed[sign_seed == 0.0] = 1.0

    full_seed = max_current * sign_seed
    half_seed = 0.5 * max_current * sign_seed
    neg_full_seed = -full_seed

    # Strongest aligned single coil pulse
    best_idx = int(np.argmax(np.abs(proj)))
    single_seed = np.zeros(magnet_array.n_magnets, dtype=float)
    single_seed[best_idx] = max_current * np.sign(proj[best_idx] if proj[best_idx] != 0 else 1.0)

    return [
        np.zeros(magnet_array.n_magnets, dtype=float),
        full_seed.astype(float),
        half_seed.astype(float),
        neg_full_seed.astype(float),
        single_seed.astype(float),
    ]


def mpc_cost_3d(
    currents,
    particle,
    particle_pos,
    target_pos,
    magnet_array,
    workspace,
    dt,
    max_step=MAX_STEP,
    w_progress=80.0,
    w_align=8.0,
    w_dist=2.0,
    w_curr=1e-5,
    w_edge=20.0,
    w_overshoot=2.0,
):
    """
    Lower is better.
    """
    particle_pos = np.asarray(particle_pos, dtype=float)
    target_pos = np.asarray(target_pos, dtype=float)
    currents = np.asarray(currents, dtype=float)

    desired = target_pos - particle_pos
    current_distance = np.linalg.norm(desired)

    if current_distance < EPS:
        return 0.0

    desired_hat = desired / current_distance

    next_pos, force, raw_step, step_vec = simulate_one_step_3d(
        particle=particle,
        particle_pos=particle_pos,
        currents=currents,
        magnet_array=magnet_array,
        workspace=workspace,
        dt=dt,
        max_step=max_step,
    )

    next_distance = np.linalg.norm(target_pos - next_pos)
    progress = current_distance - next_distance
    step_along = np.dot(step_vec, desired_hat)

    force_norm = np.linalg.norm(force)
    alignment = np.dot(desired_hat, force / (force_norm + EPS))

    edge_cost = edge_penalty_3d(next_pos, workspace)
    overshoot_cost = overshoot_penalty_3d(raw_step, current_distance)

    cost = (
        -w_progress * progress
        -w_align * step_along
        -0.5 * alignment
        + w_dist * (next_distance ** 2)
        + w_curr * np.sum(currents ** 2)
        + w_edge * edge_cost
        + w_overshoot * overshoot_cost
    )

    return float(cost)


def continuous_mpc_controller_3d(
    particle,
    particle_pos,
    target_pos,
    magnet_array,
    workspace,
    dt,
    max_current=8.0,
    max_step=MAX_STEP,
    initial_guess=None,
    w_progress=80.0,
    w_align=8.0,
    w_dist=2.0,
    w_curr=1e-5,
    w_edge=20.0,
    w_overshoot=2.0,
):
    """
    Continuous 1-step MPC controller for the realistic 3D system.

    Returns
    -------
    best_currents : np.ndarray shape (8,)
    best_force    : np.ndarray shape (3,)
    best_cost     : float
    """
    particle_pos = np.asarray(particle_pos, dtype=float)
    target_pos = np.asarray(target_pos, dtype=float)

    if np.linalg.norm(target_pos - particle_pos) < EPS:
        zero = np.zeros(magnet_array.n_magnets, dtype=float)
        return zero, np.zeros(3, dtype=float), 0.0

    bounds = [(-max_current, max_current)] * magnet_array.n_magnets

    candidate_starts = heuristic_initial_guesses_3d(
        target_pos=target_pos,
        particle_pos=particle_pos,
        magnet_array=magnet_array,
        max_current=max_current,
    )

    if initial_guess is not None:
        ig = np.clip(np.asarray(initial_guess, dtype=float), -max_current, max_current)
        candidate_starts.insert(0, ig)

    best_currents = np.zeros(magnet_array.n_magnets, dtype=float)
    best_cost = np.inf

    for x0 in candidate_starts:
        result = minimize(
            fun=mpc_cost_3d,
            x0=x0,
            args=(
                particle,
                particle_pos,
                target_pos,
                magnet_array,
                workspace,
                dt,
                max_step,
                w_progress,
                w_align,
                w_dist,
                w_curr,
                w_edge,
                w_overshoot,
            ),
            method="SLSQP",
            bounds=bounds,
            options={
                "maxiter": 80,
                "ftol": 1e-7,
                "disp": False,
            },
        )

        cand_x = np.clip(result.x, -max_current, max_current)

        cand_cost = mpc_cost_3d(
            cand_x,
            particle,
            particle_pos,
            target_pos,
            magnet_array,
            workspace,
            dt,
            max_step=max_step,
            w_progress=w_progress,
            w_align=w_align,
            w_dist=w_dist,
            w_curr=w_curr,
            w_edge=w_edge,
            w_overshoot=w_overshoot,
        )

        if cand_cost < best_cost:
            best_cost = cand_cost
            best_currents = cand_x.copy()

    magnet_array.set_currents(best_currents)
    B = magnet_array.field_at(particle_pos)
    grad_Bsq = magnet_array.grad_B_squared(particle_pos)
    best_force = particle.magnetic_force(B, grad_Bsq)

    return best_currents, best_force, float(best_cost)