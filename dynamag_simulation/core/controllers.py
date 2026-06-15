"""
MPC controller for 3D magnetic particle steering in a vessel network.

Changes from v1:
  1. Bidirectional current allocation:
        I_i ∈ [-max_current, max_current]
     This matches the electromagnet model, where current sign reverses coil
     polarity.
  2. Wall penalty uses vessel radial distance instead of cubic edge distance.
  3. Flow-aware 1-step prediction includes background blood-flow convection.
  4. Optional route-tangent alignment term w_route * step_along_route.
     This keeps the particle moving along the vessel axis during MPC optimization,
     preventing wall-hugging when Euclidean target direction diverges from route.
  5. Fast-mode evaluates all heuristic starts and optimizes the best starts.
"""

import numpy as np
from scipy.optimize import minimize

from .constants import EPS
from .flow import NoFlow

_NO_FLOW = NoFlow()


def _simulate_one_step(
    particle,
    particle_pos,
    currents,
    magnet_array,
    workspace,
    dt,
    flow=None,
    sim_time=0.0,
):
    """Deterministic 1-step prediction including flow convection."""
    particle_pos = np.asarray(particle_pos, dtype=float)
    currents = np.asarray(currents, dtype=float)

    magnet_array.set_currents(currents)

    B = magnet_array.field_at(particle_pos)
    grad_Bsq = magnet_array.grad_B_squared(particle_pos)
    force = particle.magnetic_force(B, grad_Bsq)

    force_mag = np.linalg.norm(force)
    if force_mag > EPS:
        gamma_eff, _, _ = particle._effective_drag(force_mag)
        v_mag = force / gamma_eff
    else:
        v_mag = np.zeros(3, dtype=float)

    if flow is None:
        v_flow = np.zeros(3, dtype=float)
    else:
        v_flow = flow.velocity_at(particle_pos, sim_time)

    step_vec = (v_flow + v_mag) * dt
    next_pos = workspace.clip(particle_pos + step_vec)

    return next_pos, force, step_vec


def _wall_penalty(
    pos,
    workspace,
    wall_buffer_fraction: float = 0.15,
    wall_buffer_m: float | None = None,
):
    """
    Vessel-aware wall proximity penalty.

    Returns a positive penalty when the predicted position is too close
    to the vessel wall. Falls back to cubic workspace edge penalty when
    no vessel network is attached.
    """
    pos = np.asarray(pos, dtype=float)

    if hasattr(workspace, "network"):
        network = workspace.network
        seg, radial_dist = network.find_segment(pos)

        if seg is None:
            return 0.0

        buffer = wall_buffer_fraction * seg.radius
        if wall_buffer_m is not None:
            buffer = max(buffer, float(wall_buffer_m))
        margin = seg.radius - radial_dist

        return max(0.0, buffer - margin)

    # Legacy cubic workspace fallback.
    margins = workspace.half - np.abs(pos)
    edge_buffer = 0.006

    return max(0.0, edge_buffer - float(np.min(margins)))


def _overshoot_penalty(step_vec, current_distance, fraction=0.65):
    """
    Penalize predicted steps that are too large relative to remaining distance.
    """
    step_len = float(np.linalg.norm(step_vec))
    return max(0.0, step_len - fraction * current_distance)


def _heuristic_starts(target_pos, particle_pos, magnet_array, max_current):
    """
    Cheap nonzero initial guesses to seed the optimizer.

    Since the allocator is bidirectional, this returns both positive and
    negative polarity seeds. The optimizer is still free to find mixed-polarity
    current patterns.
    """
    particle_pos = np.asarray(particle_pos, dtype=float)
    target_pos = np.asarray(target_pos, dtype=float)

    desired = target_pos - particle_pos
    dist = float(np.linalg.norm(desired))
    n = magnet_array.n_magnets

    zero = np.zeros(n, dtype=float)

    if dist < EPS:
        return [zero]

    desired_hat = desired / dist

    pull_dirs = magnet_array.positions - particle_pos
    norms = np.linalg.norm(pull_dirs, axis=1)
    pull_dirs = pull_dirs / (norms[:, np.newaxis] + EPS)

    proj = pull_dirs @ desired_hat
    proj_pos = np.clip(proj, 0.0, None)

    if float(np.max(proj_pos)) > EPS:
        full_seed = max_current * proj_pos / (np.max(proj_pos) + EPS)
    else:
        full_seed = zero.copy()

    half_seed = 0.5 * full_seed

    best_idx = int(np.argmax(proj_pos)) if np.max(proj_pos) > EPS else 0
    single = zero.copy()
    single[best_idx] = max_current

    uniform = np.ones(n, dtype=float) * (0.5 * max_current)

    # Signed projection seed can initialize mixed-polarity current patterns.
    max_abs_proj = float(np.max(np.abs(proj)))
    if max_abs_proj > EPS:
        signed_seed = max_current * proj / (max_abs_proj + EPS)
    else:
        signed_seed = zero.copy()

    half_signed_seed = 0.5 * signed_seed

    negative_full_seed = -full_seed
    negative_half_seed = -half_seed
    negative_single = -single
    negative_uniform = -uniform

    starts = [
        zero,
        full_seed.astype(float),
        half_seed.astype(float),
        single.astype(float),
        uniform.astype(float),
        signed_seed.astype(float),
        half_signed_seed.astype(float),
        negative_full_seed.astype(float),
        negative_half_seed.astype(float),
        negative_single.astype(float),
        negative_uniform.astype(float),
    ]

    return [
        np.clip(s, -max_current, max_current).astype(float)
        for s in starts
    ]


def _mpc_cost(
    currents,
    particle,
    particle_pos,
    target_pos,
    magnet_array,
    workspace,
    dt,
    w_progress,
    w_align,
    w_dist,
    w_curr,
    w_smooth,
    prev_currents,
    w_edge,
    w_overshoot,
    flow,
    sim_time,
    w_route=0.0,
    route_tangent=None,
    wall_buffer_m=None,
):
    """Scalar cost for a candidate current vector. Lower is better."""
    particle_pos = np.asarray(particle_pos, dtype=float)
    target_pos = np.asarray(target_pos, dtype=float)
    currents = np.asarray(currents, dtype=float)

    desired = target_pos - particle_pos
    current_distance = float(np.linalg.norm(desired))

    if current_distance < EPS:
        return 0.0

    desired_hat = desired / current_distance

    next_pos, force, step_vec = _simulate_one_step(
        particle=particle,
        particle_pos=particle_pos,
        currents=currents,
        magnet_array=magnet_array,
        workspace=workspace,
        dt=dt,
        flow=flow,
        sim_time=sim_time,
    )

    # Use the unclipped predicted position for wall penalty so attempted
    # wall exits are penalized even if workspace.clip keeps the prediction inside.
    unclipped_pos = particle_pos + step_vec

    next_distance = float(np.linalg.norm(target_pos - next_pos))
    progress = current_distance - next_distance

    step_along_target = float(np.dot(step_vec, desired_hat))

    force_norm = float(np.linalg.norm(force))
    alignment = float(np.dot(desired_hat, force / (force_norm + EPS)))

    wall_cost = _wall_penalty(unclipped_pos, workspace, wall_buffer_m=wall_buffer_m)
    overshoot_cost = _overshoot_penalty(step_vec, current_distance)

    smoothness_cost = 0.0
    if prev_currents is not None:
        prev_currents = np.asarray(prev_currents, dtype=float)
        smoothness_cost = float(np.sum((currents - prev_currents) ** 2))

    # Route-tangent alignment term.
    # This rewards predicted displacement along the route direction.
    route_align_cost = 0.0
    if w_route > 0.0 and route_tangent is not None:
        rt = np.asarray(route_tangent, dtype=float)
        rtn = float(np.linalg.norm(rt))

        if rtn > EPS:
            rt_hat = rt / rtn
            route_align_cost = -float(np.dot(step_vec, rt_hat))

    return float(
        -w_progress * progress
        -w_align * step_along_target
        -0.5 * alignment
        + w_dist * (next_distance ** 2)
        + w_curr * np.sum(currents ** 2)
        + w_smooth * smoothness_cost
        + w_edge * wall_cost
        + w_overshoot * overshoot_cost
        + w_route * route_align_cost
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
    w_route: float = 0.0,
    route_tangent=None,
    wall_buffer_m=None,
    flow=None,
    sim_time: float = 0.0,
    fast_mode: bool = True,
    ftol: float = 1e-5,
    maxiter: int = 60,
    n_starts_to_optimize: int = 2,
):
    """
    1-step MPC current allocator.

    Finds signed bidirectional currents that minimize a weighted one-step cost.

    Current bounds:
        I_i ∈ [-max_current, max_current]

    Returns:
        best_currents, best_force, best_cost, optimizer_converged

    w_route / route_tangent:
        Optional route-tangent alignment term. This helps the predicted motion
        follow the planned vessel path rather than only minimizing Euclidean
        distance to the local target.
    """
    particle_pos = np.asarray(particle_pos, dtype=float)
    target_pos = np.asarray(target_pos, dtype=float)

    if flow is None:
        flow = _NO_FLOW

    if np.linalg.norm(target_pos - particle_pos) < EPS:
        z = np.zeros(magnet_array.n_magnets, dtype=float)
        return z, np.zeros(3, dtype=float), 0.0, True

    bounds = [(-max_current, max_current)] * magnet_array.n_magnets

    starts = _heuristic_starts(
        target_pos=target_pos,
        particle_pos=particle_pos,
        magnet_array=magnet_array,
        max_current=max_current,
    )

    if initial_guess is not None:
        ig = np.clip(
            np.asarray(initial_guess, dtype=float),
            -max_current,
            max_current,
        )
        starts.insert(0, ig)

    best_currents = np.zeros(magnet_array.n_magnets, dtype=float)
    best_cost = np.inf
    any_converged = False

    args_tuple = (
        particle,
        particle_pos,
        target_pos,
        magnet_array,
        workspace,
        dt,
        w_progress,
        w_align,
        w_dist,
        w_curr,
        w_smooth,
        initial_guess,
        w_edge,
        w_overshoot,
        flow,
        sim_time,
        w_route,
        route_tangent,
        wall_buffer_m,
    )

    if fast_mode:
        start_costs = []

        for x0 in starts:
            c = _mpc_cost(x0, *args_tuple)
            start_costs.append((c, x0))

        start_costs.sort(key=lambda x: x[0])
        starts_to_optimize = [
            sc[1] for sc in start_costs[:max(1, int(n_starts_to_optimize))]
        ]
    else:
        starts_to_optimize = starts

    for x0 in starts_to_optimize:
        result = minimize(
            fun=_mpc_cost,
            x0=x0,
            args=args_tuple,
            method="SLSQP",
            bounds=bounds,
            options={
                "maxiter": int(maxiter),
                "ftol": float(ftol),
                "disp": False,
            },
        )

        if result.success:
            any_converged = True

        cand = np.clip(
            result.x,
            -max_current,
            max_current,
        )

        cand_cost = _mpc_cost(cand, *args_tuple)

        if cand_cost < best_cost:
            best_cost = cand_cost
            best_currents = cand.copy()

    magnet_array.set_currents(best_currents)

    B = magnet_array.field_at(particle_pos)
    grad_Bsq = magnet_array.grad_B_squared(particle_pos)
    best_force = particle.magnetic_force(B, grad_Bsq)

    return best_currents, best_force, float(best_cost), any_converged
