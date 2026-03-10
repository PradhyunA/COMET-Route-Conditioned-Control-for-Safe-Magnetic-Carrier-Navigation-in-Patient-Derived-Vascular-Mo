import os
import csv
import itertools
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D  # noqa: F401 – enables 3D projection

# ============================================================
# GLOBAL CONSTANTS
# ============================================================

INCH_TO_METER = 0.0254
SIDE = 6.0 * INCH_TO_METER          # 0.1524 m
HALF = SIDE / 2.0                    # 0.0762 m

EPS = 1e-12

TEST_RANGE = 0.06   # metres – how far from centre to place start/target

# Maximum displacement per timestep (prevents wall-bounce convergence).
# Set to ~5 % of the workspace half-width so the particle needs many
# steps to cross the workspace, producing realistic trajectories.
MAX_STEP = 0.004  # metres


# ============================================================
# WORKSPACE
# ============================================================

class Workspace3D:
    """Cubic workspace clipping."""

    def __init__(self, half_side=HALF):
        self.half = float(half_side)

    def clip(self, p):
        p = np.array(p, dtype=float)
        p[0] = np.clip(p[0], -self.half, self.half)
        p[1] = np.clip(p[1], -self.half, self.half)
        p[2] = np.clip(p[2], -self.half, self.half)
        return p


# ============================================================
# ELECTROMAGNET ARRAY  (8 corners of a cube)
# ============================================================

class ElectromagnetArray3D:
    """
    8 corner electromagnets around a cubic workspace.

    Each magnet has a fixed orientation vector pointing inward toward
    the cube centre at 45° from each adjacent face (i.e. along the
    body diagonal).

    Field model (per magnet):
        B_i = k * N * I / (r + z0)^p  *  orientation_hat_i

    The net field is the superposition of all 8 contributions.
    Force is derived from grad(0.5 * |B|^2).
    """

    def __init__(
        self,
        turns_per_magnet=None,
        max_current=8.0,
        field_constant=2.0e-7,
        decay_power=3.0,
        standoff=0.01,
    ):
        # All 8 corners of the cube
        signs = list(itertools.product((-1, 1), repeat=3))
        self.positions = np.array(
            [[s[0] * HALF, s[1] * HALF, s[2] * HALF] for s in signs],
            dtype=float,
        )

        # Orientation: each magnet points inward (toward centre).
        # For a corner at (sx*H, sy*H, sz*H), inward direction is
        # (-sx, -sy, -sz) / sqrt(3).
        inv_sqrt3 = 1.0 / np.sqrt(3.0)
        self.orientations = np.array(
            [[-s[0] * inv_sqrt3, -s[1] * inv_sqrt3, -s[2] * inv_sqrt3]
             for s in signs],
            dtype=float,
        )

        n_magnets = 8

        if turns_per_magnet is None:
            turns_per_magnet = [250] * n_magnets

        self.turns = np.array(turns_per_magnet, dtype=float)
        self.max_current = float(max_current)
        self.currents = np.zeros(n_magnets, dtype=float)

        # Field model parameters (unchanged from 2D)
        self.field_constant = float(field_constant)
        self.decay_power = float(decay_power)
        self.standoff = float(standoff)

    # ---- helpers used by the vectorised controller ----

    def set_currents(self, currents):
        currents = np.array(currents, dtype=float)
        self.currents = np.clip(currents, -self.max_current, self.max_current)

    # ---- field & force ----

    def field_at(self, point):
        """
        Net magnetic field vector at a 3D point.

        Each magnet emits field along its fixed inward orientation.
        Magnitude decays with distance from the magnet:
            |B_i| = k * N * I / (r + z0)^p
        """
        point = np.array(point, dtype=float)
        B = np.zeros(3, dtype=float)

        for i in range(8):
            r_vec = point - self.positions[i]
            r = np.linalg.norm(r_vec) + EPS

            magnitude = (
                self.field_constant
                * self.turns[i]
                * self.currents[i]
                / ((r + self.standoff) ** self.decay_power)
            )

            B += magnitude * self.orientations[i]

        return B

    def energy_proxy_at(self, point):
        """phi = 0.5 * |B|^2"""
        B = self.field_at(point)
        return 0.5 * np.dot(B, B)

    def force_at(self, point, grad_step=1e-4, force_gain=1.0):
        """
        Force from grad(0.5*|B|^2) via central finite differences.
        """
        point = np.array(point, dtype=float)
        h = float(grad_step)

        ex = np.array([h, 0.0, 0.0])
        ey = np.array([0.0, h, 0.0])
        ez = np.array([0.0, 0.0, h])

        dphi_dx = (self.energy_proxy_at(point + ex) -
                   self.energy_proxy_at(point - ex)) / (2.0 * h)
        dphi_dy = (self.energy_proxy_at(point + ey) -
                   self.energy_proxy_at(point - ey)) / (2.0 * h)
        dphi_dz = (self.energy_proxy_at(point + ez) -
                   self.energy_proxy_at(point - ez)) / (2.0 * h)

        return force_gain * np.array([dphi_dx, dphi_dy, dphi_dz], dtype=float)


# ============================================================
# PARTICLE MODEL
# ============================================================

class Particle3D:
    """
    Overdamped particle in 3D:
        v = mobility * F
        x_{t+dt} = x_t + v * dt   (clamped to MAX_STEP)
    """

    def __init__(self, position, mobility=1.0):
        self.pos = np.array(position, dtype=float)
        self.vel = np.zeros(3, dtype=float)
        self.mobility = float(mobility)

    def update(self, force, dt, max_step=MAX_STEP):
        """Move the particle, clamping displacement to *max_step*."""
        self.vel = self.mobility * np.array(force, dtype=float)
        step = self.vel * dt
        step_mag = np.linalg.norm(step)
        if step_mag > max_step:
            step = step * (max_step / step_mag)
        self.pos = self.pos + step


# ============================================================
# CANDIDATE SET HELPERS
# ============================================================

def precompute_current_combos_3d(candidate_currents):
    """Pre-compute all 8-magnet current combinations."""
    return np.array(
        list(itertools.product(candidate_currents, repeat=8)),
        dtype=float,
    )


def choose_candidate_set_3d(current_distance):
    """
    Adaptive resolution for 8 magnets.

    With 8 magnets, N^8 grows fast, so we keep the sets coarse.
    """
    if current_distance > 0.04:
        return (-8.0, 0.0, 8.0)               # 3^8 = 6 561
    elif current_distance > 0.015:
        return (-8.0, -4.0, 0.0, 4.0, 8.0)    # 5^8 = 390 625
    else:
        return (-4.0, 0.0, 4.0)                # 3^8 = 6 561


def adaptive_lambda_current_3d(base_lambda, current_distance):
    """Scale current-penalty: less penalisation when far from target."""
    if current_distance > 0.05:
        return 0.15 * base_lambda
    if current_distance > 0.02:
        return 0.35 * base_lambda
    if current_distance > 0.008:
        return 0.70 * base_lambda
    return base_lambda


# ============================================================
# CONTROLLERS
# ============================================================

def brute_force_lookahead_controller_3d(
    particle_pos,
    target_pos,
    magnet_array,
    workspace,
    dt,
    mobility,
    all_combos,
    force_gain=1.0,
    grad_step=1e-4,
    lambda_current=5e-5,
    edge_buffer=0.006,
    max_step=MAX_STEP,
):
    """
    Hybrid lookahead controller in 3D.

    Evaluates every candidate current combination, simulates one step
    (with step-size clamping), and scores it.
    """
    particle_pos = np.array(particle_pos, dtype=float)
    target_pos = np.array(target_pos, dtype=float)

    desired = target_pos - particle_pos
    current_distance = np.linalg.norm(desired)

    if current_distance < EPS:
        return np.zeros(8), np.zeros(3), 0.0, 0.0

    desired_hat = desired / current_distance
    lambda_eff = adaptive_lambda_current_3d(lambda_current, current_distance)

    best_score = -np.inf
    best_currents = None
    best_force = None
    best_step_along = -np.inf

    for combo in all_combos:
        magnet_array.set_currents(combo)
        F = magnet_array.force_at(particle_pos, grad_step=grad_step,
                                  force_gain=force_gain)

        # Compute proposed step WITH the same clamping the particle uses
        raw_step = mobility * F * dt
        raw_mag = np.linalg.norm(raw_step)
        if raw_mag > max_step:
            step_vec = raw_step * (max_step / raw_mag)
        else:
            step_vec = raw_step

        step_mag = np.linalg.norm(step_vec)
        step_along = np.dot(step_vec, desired_hat)

        x_raw = particle_pos + step_vec
        x_next = workspace.clip(x_raw)

        clipped_amount = np.linalg.norm(x_raw - x_next)
        distance_next = np.linalg.norm(x_next - target_pos)
        progress = current_distance - distance_next

        F_norm = np.linalg.norm(F)
        alignment = np.dot(desired_hat, F / (F_norm + EPS))

        margin_x = workspace.half - abs(x_next[0])
        margin_y = workspace.half - abs(x_next[1])
        margin_z = workspace.half - abs(x_next[2])
        min_margin = min(margin_x, margin_y, margin_z)
        edge_penalty = max(0.0, edge_buffer - min_margin)

        overshoot_penalty = max(0.0, step_mag - 0.65 * current_distance)

        nonzero_bonus = 0.002 if (np.any(np.abs(combo) > 1e-12) and
                                  step_along > 0) else 0.0

        score = (
            10.0 * progress
            + 3.0 * step_along
            - 1.2 * distance_next
            + 0.10 * alignment
            + nonzero_bonus
            - lambda_eff * np.sum(combo ** 2)
            - 30.0 * clipped_amount
            - 5.0 * edge_penalty
            - 1.2 * overshoot_penalty
        )

        if score > best_score:
            best_score = score
            best_currents = combo.copy()
            best_force = F.copy()
            best_step_along = step_along

    return best_currents, best_force, best_score, best_step_along


def brute_force_directional_fallback_controller_3d(
    particle_pos,
    target_pos,
    magnet_array,
    all_combos,
    dt,
    mobility,
    force_gain=1.0,
    grad_step=1e-4,
    lambda_current=1e-5,
    max_step=MAX_STEP,
):
    """Fallback controller: maximise forward motion toward target."""
    particle_pos = np.array(particle_pos, dtype=float)
    target_pos = np.array(target_pos, dtype=float)

    desired = target_pos - particle_pos
    current_distance = np.linalg.norm(desired)

    if current_distance < EPS:
        return np.zeros(8), np.zeros(3), 0.0

    desired_hat = desired / current_distance

    best_score = -np.inf
    best_currents = None
    best_force = None

    for combo in all_combos:
        magnet_array.set_currents(combo)
        F = magnet_array.force_at(particle_pos, grad_step=grad_step,
                                  force_gain=force_gain)

        raw_step = mobility * F * dt
        raw_mag = np.linalg.norm(raw_step)
        if raw_mag > max_step:
            step_vec = raw_step * (max_step / raw_mag)
        else:
            step_vec = raw_step

        step_along = np.dot(step_vec, desired_hat)
        perp_vec = step_vec - step_along * desired_hat
        perp_mag = np.linalg.norm(perp_vec)

        score = (
            5.0 * step_along
            - 1.5 * perp_mag
            - lambda_current * np.sum(combo ** 2)
        )

        if score > best_score:
            best_score = score
            best_currents = combo.copy()
            best_force = F.copy()

    return best_currents, best_force, best_score


# ============================================================
# SIMULATION
# ============================================================

def simulate_trial_3d(
    start,
    target,
    dt=0.001,
    steps=6000,
    tolerance=0.003,
    turns_per_magnet=None,
    max_current=8.0,
    field_constant=2.0e-7,
    decay_power=3.0,
    standoff=0.01,
    force_gain=1.0,
    mobility=1.0,
    lambda_current=5e-5,
    max_step=MAX_STEP,
):
    workspace = Workspace3D()
    magnets = ElectromagnetArray3D(
        turns_per_magnet=turns_per_magnet,
        max_current=max_current,
        field_constant=field_constant,
        decay_power=decay_power,
        standoff=standoff,
    )
    particle = Particle3D(start, mobility=mobility)

    start = np.array(start, dtype=float)
    target = np.array(target, dtype=float)

    trajectory = [particle.pos.copy()]
    distance_log = []
    current_log = []
    field_log = []
    force_log = []
    score_log = []

    reached = False

    # Pre-compute candidate combo tables (cached per resolution tier)
    combo_cache = {}

    for step in range(steps):
        current_distance = np.linalg.norm(particle.pos - target)
        current_set = choose_candidate_set_3d(current_distance)

        if current_set not in combo_cache:
            combo_cache[current_set] = precompute_current_combos_3d(current_set)
        all_combos = combo_cache[current_set]

        best_currents, best_force, best_score, best_step_along = (
            brute_force_lookahead_controller_3d(
                particle_pos=particle.pos,
                target_pos=target,
                magnet_array=magnets,
                workspace=workspace,
                dt=dt,
                mobility=particle.mobility,
                all_combos=all_combos,
                force_gain=force_gain,
                grad_step=1e-4,
                lambda_current=lambda_current,
                edge_buffer=0.006,
                max_step=max_step,
            )
        )

        # Fallback if the lookahead controller stalls
        if current_distance > tolerance:
            if (best_currents is None
                    or np.all(np.abs(best_currents) < 1e-12)
                    or best_step_along < 1e-8):
                fb_currents, fb_force, fb_score = (
                    brute_force_directional_fallback_controller_3d(
                        particle_pos=particle.pos,
                        target_pos=target,
                        magnet_array=magnets,
                        all_combos=all_combos,
                        dt=dt,
                        mobility=particle.mobility,
                        force_gain=force_gain,
                        grad_step=1e-4,
                        lambda_current=1e-5,
                        max_step=max_step,
                    )
                )
                best_currents = fb_currents
                best_force = fb_force
                best_score = fb_score

        magnets.set_currents(best_currents)

        B = magnets.field_at(particle.pos)
        F = best_force

        particle.update(F, dt, max_step=max_step)
        particle.pos = workspace.clip(particle.pos)

        trajectory.append(particle.pos.copy())
        distance = np.linalg.norm(particle.pos - target)

        distance_log.append(distance)
        current_log.append(magnets.currents.copy())
        field_log.append(B.copy())
        force_log.append(F.copy())
        score_log.append(best_score)

        if distance < tolerance:
            reached = True
            break

        # Progress indicator every 500 steps
        if (step + 1) % 500 == 0:
            print(f"  step {step + 1:>5d}/{steps}  dist={distance:.5f} m")

    trajectory = np.array(trajectory)
    distance_log = np.array(distance_log)
    current_log = np.array(current_log)
    field_log = np.array(field_log)
    force_log = np.array(force_log)
    score_log = np.array(score_log)

    final_error = np.linalg.norm(trajectory[-1] - target)
    max_current_used = (np.max(np.abs(current_log))
                        if len(current_log) > 0 else 0.0)

    return {
        "start": start,
        "target": target,
        "trajectory": trajectory,
        "distance_log": distance_log,
        "current_log": current_log,
        "field_log": field_log,
        "force_log": force_log,
        "score_log": score_log,
        "magnets": magnets.positions.copy(),
        "orientations": magnets.orientations.copy(),
        "reached": reached,
        "steps_taken": len(trajectory) - 1,
        "final_position": trajectory[-1],
        "final_error": final_error,
        "max_current_used": max_current_used,
        "workspace_half": HALF,
        "params": {
            "dt": dt,
            "steps": steps,
            "tolerance": tolerance,
            "turns_per_magnet": (turns_per_magnet
                                 if turns_per_magnet is not None
                                 else [250] * 8),
            "max_current": max_current,
            "field_constant": field_constant,
            "decay_power": decay_power,
            "standoff": standoff,
            "force_gain": force_gain,
            "mobility": mobility,
            "lambda_current": lambda_current,
            "max_step": max_step,
        },
    }


# ============================================================
# REPORTING & VISUALISATION
# ============================================================

def print_summary(result, trial_name):
    print(f"\n=== {trial_name} ===")
    print("Start:", result["start"])
    print("Target:", result["target"])
    print("Final position:", result["final_position"])
    print("Steps taken:", result["steps_taken"])
    print("Final error (m):", result["final_error"])
    print("Final error (mm):", result["final_error"] * 1000.0)
    print("Reached target:", result["reached"])
    print("Max current used (A):", result["max_current_used"])


def make_output_dir(path="simulation_outputs_3d"):
    os.makedirs(path, exist_ok=True)
    return path


def save_trajectory_plot_3d(result, trial_name, output_dir):
    """3D scatter + line trajectory plot."""
    traj = result["trajectory"]
    target = result["target"]
    magnets = result["magnets"]
    half = result["workspace_half"]

    fig = plt.figure(figsize=(10, 10))
    ax = fig.add_subplot(111, projection="3d")

    ax.plot(traj[:, 0], traj[:, 1], traj[:, 2], linewidth=2, label="Path")
    ax.scatter(*traj[0], s=120, marker="o", label="Start")
    ax.scatter(*target, s=120, marker="*", label="Target")
    ax.scatter(magnets[:, 0], magnets[:, 1], magnets[:, 2],
               s=80, marker="^", label="Magnets")

    corner_labels = ["---", "+--", "-+-", "++-",
                     "--+", "+-+", "-++", "+++"]
    for pos, label in zip(magnets, corner_labels):
        ax.text(pos[0], pos[1], pos[2], f" {label}", fontsize=7)

    ax.set_xlim(-half, half)
    ax.set_ylim(-half, half)
    ax.set_zlim(-half, half)
    ax.set_xlabel("X (m)")
    ax.set_ylabel("Y (m)")
    ax.set_zlabel("Z (m)")
    ax.set_title(f"{trial_name} – 3D Trajectory")
    ax.legend(fontsize=8)

    filename = os.path.join(output_dir, f"{trial_name}_trajectory_3d.png")
    plt.savefig(filename, dpi=300, bbox_inches="tight")
    plt.close()


def save_distance_plot(result, trial_name, output_dir):
    dist = result["distance_log"]

    plt.figure(figsize=(8, 5))
    plt.plot(dist, linewidth=2)
    plt.grid(True)
    plt.xlabel("Simulation Step")
    plt.ylabel("Distance to Target (m)")
    plt.title(f"{trial_name} – Distance to Target")

    filename = os.path.join(output_dir, f"{trial_name}_distance.png")
    plt.savefig(filename, dpi=300, bbox_inches="tight")
    plt.close()


def save_force_plot(result, trial_name, output_dir):
    force_log = result["force_log"]
    if len(force_log) == 0:
        return

    force_mag = np.linalg.norm(force_log, axis=1)

    plt.figure(figsize=(8, 5))
    plt.plot(force_mag, linewidth=2)
    plt.grid(True)
    plt.xlabel("Simulation Step")
    plt.ylabel("Force Magnitude (arb. units)")
    plt.title(f"{trial_name} – Force Magnitude")

    filename = os.path.join(output_dir, f"{trial_name}_force.png")
    plt.savefig(filename, dpi=300, bbox_inches="tight")
    plt.close()


def save_field_map_3d(result, trial_name, output_dir, grid_points=15):
    """Field quiver plot on the z=0 midplane using first-step currents."""
    if len(result["current_log"]) == 0:
        return

    half = result["workspace_half"]
    first_currents = result["current_log"][0]
    params = result["params"]

    array = ElectromagnetArray3D(
        turns_per_magnet=params["turns_per_magnet"],
        max_current=params["max_current"],
        field_constant=params["field_constant"],
        decay_power=params["decay_power"],
        standoff=params["standoff"],
    )
    array.set_currents(first_currents)

    xs = np.linspace(-half, half, grid_points)
    ys = np.linspace(-half, half, grid_points)
    X, Y = np.meshgrid(xs, ys)

    U = np.zeros_like(X)
    V = np.zeros_like(Y)
    M = np.zeros_like(X)

    for i in range(grid_points):
        for j in range(grid_points):
            p = np.array([X[i, j], Y[i, j], 0.0])  # z = 0 slice
            B = array.field_at(p)
            U[i, j] = B[0]
            V[i, j] = B[1]
            M[i, j] = np.linalg.norm(B)

    if np.all(M < 1e-12):
        return

    mask = M > 1e-14
    U_plot = np.zeros_like(U)
    V_plot = np.zeros_like(V)
    U_plot[mask] = U[mask] / M[mask]
    V_plot[mask] = V[mask] / M[mask]

    plt.figure(figsize=(8, 8))
    plt.quiver(X, Y, U_plot, V_plot, M)

    magnets = result["magnets"]
    plt.scatter(magnets[:, 0], magnets[:, 1], s=80, marker="^",
                label="Magnets (proj.)")
    plt.scatter(result["start"][0], result["start"][1], s=100,
                label="Start (proj.)")
    plt.scatter(result["target"][0], result["target"][1], s=100,
                label="Target (proj.)")

    plt.xlim(-half, half)
    plt.ylim(-half, half)
    plt.gca().set_aspect("equal")
    plt.grid(True)
    plt.legend()
    plt.xlabel("X (m)")
    plt.ylabel("Y (m)")
    plt.title(f"{trial_name} – Field Map (z=0 slice, first-step currents)")

    filename = os.path.join(output_dir, f"{trial_name}_fieldmap_z0.png")
    plt.savefig(filename, dpi=300, bbox_inches="tight")
    plt.close()


def save_results_csv(results, output_dir, filename="trial_summary.csv"):
    path = os.path.join(output_dir, filename)
    with open(path, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow([
            "trial_name",
            "start_x", "start_y", "start_z",
            "target_x", "target_y", "target_z",
            "final_x", "final_y", "final_z",
            "steps_taken",
            "final_error_m",
            "final_error_mm",
            "reached_target",
            "max_current_used_A",
        ])

        for trial_name, r in results.items():
            writer.writerow([
                trial_name,
                r["start"][0], r["start"][1], r["start"][2],
                r["target"][0], r["target"][1], r["target"][2],
                r["final_position"][0], r["final_position"][1],
                r["final_position"][2],
                r["steps_taken"],
                r["final_error"],
                r["final_error"] * 1000.0,
                r["reached"],
                r["max_current_used"],
            ])


# ============================================================
# TRIAL RUNNER
# ============================================================

def run_all_trials_3d():
    EDGE = TEST_RANGE

    trials = {
        "trial_1_corner_to_corner": {
            "start": (-EDGE, -EDGE, -EDGE),
            "target": (EDGE, EDGE, EDGE),
        },
        "trial_2_opposite_edges": {
            "start": (-EDGE, EDGE, -EDGE),
            "target": (EDGE, -EDGE, EDGE),
        },
        "trial_3_center_to_corner": {
            "start": (0.0, 0.0, 0.0),
            "target": (EDGE, EDGE, EDGE),
        },
        "trial_4_single_axis_x": {
            "start": (0.0, 0.0, 0.0),
            "target": (EDGE, 0.0, 0.0),
        },
        "trial_5_single_axis_z": {
            "start": (0.0, 0.0, 0.0),
            "target": (0.0, 0.0, EDGE),
        },
    }

    output_dir = make_output_dir("simulation_outputs_3d")
    results = {}

    sim_kwargs = {
        "dt": 0.001,
        "steps": 6000,
        "tolerance": 0.003,
        "turns_per_magnet": [250] * 8,
        "max_current": 8.0,
        "field_constant": 2.0e-7,
        "decay_power": 3.0,
        "standoff": 0.01,
        "force_gain": 1.0,
        "mobility": 1.0,
        "lambda_current": 5e-5,
        "max_step": MAX_STEP,
    }

    for trial_name, cfg in trials.items():
        print(f"\n>>> Running {trial_name} …")
        result = simulate_trial_3d(
            start=cfg["start"],
            target=cfg["target"],
            **sim_kwargs,
        )
        results[trial_name] = result

        print_summary(result, trial_name)
        save_trajectory_plot_3d(result, trial_name, output_dir)
        save_distance_plot(result, trial_name, output_dir)
        save_force_plot(result, trial_name, output_dir)
        save_field_map_3d(result, trial_name, output_dir)

    save_results_csv(results, output_dir)
    print(f"\nSaved plots and CSV to: {output_dir}")
    return results


if __name__ == "__main__":
    run_all_trials_3d()
