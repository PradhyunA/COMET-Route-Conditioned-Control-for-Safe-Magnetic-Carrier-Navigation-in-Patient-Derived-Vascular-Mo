import os
import csv
import itertools
import numpy as np
import matplotlib.pyplot as plt

INCH_TO_METER = 0.0254
SIDE = 6.0 * INCH_TO_METER
HALF = SIDE / 2.0
EPS = 1e-12
TEST_RANGE = 0.075

class Workspace2D:
    def __init__(self, half_side=HALF):
        self.half = float(half_side)

    def clip(self, p):
        p = np.array(p, dtype=float)
        p[0] = np.clip(p[0], -self.half, self.half)
        p[1] = np.clip(p[1], -self.half, self.half)
        return p


class ElectromagnetArray2D:
    """
    Higher-physics 2D model:
    - 4 corner magnets
    - field decays ~ 1 / (r + z0)^p
    - particle force approximated by grad(0.5 * |B|^2)
    """

    def __init__(
        self,
        turns_per_magnet=None,
        max_current=8.0,
        field_constant=2.0e-7,
        decay_power=3.0,
        standoff=0.01,
    ):
        self.positions = np.array([
            [-HALF, -HALF],   # bottom-left
            [ HALF, -HALF],   # bottom-right
            [-HALF,  HALF],   # top-left
            [ HALF,  HALF],   # top-right
        ], dtype=float)

        if turns_per_magnet is None:
            turns_per_magnet = [250, 250, 250, 250]

        self.turns = np.array(turns_per_magnet, dtype=float)
        self.max_current = float(max_current)
        self.currents = np.zeros(4, dtype=float)

        self.field_constant = float(field_constant)
        self.decay_power = float(decay_power)
        self.standoff = float(standoff)

    def set_currents(self, currents):
        currents = np.array(currents, dtype=float)
        self.currents = np.clip(currents, -self.max_current, self.max_current)

    def field_at(self, point):
        """
        Simplified net magnetic field vector.

        B_i ~ k * N * I / (r + z0)^p * r_hat
        """
        point = np.array(point, dtype=float)
        B = np.zeros(2, dtype=float)

        for i in range(4):
            r_vec = point - self.positions[i]
            r = np.linalg.norm(r_vec) + EPS
            r_hat = r_vec / r

            magnitude = (
                self.field_constant
                * self.turns[i]
                * self.currents[i]
                / ((r + self.standoff) ** self.decay_power)
            )

            B += magnitude * r_hat

        return B

    def energy_proxy_at(self, point):
        """
        Magnetic-energy-like proxy:
            phi = 0.5 * |B|^2
        """
        B = self.field_at(point)
        return 0.5 * np.dot(B, B)

    def force_at(self, point, grad_step=1e-4, force_gain=1.0):
        """
        Force approximation:
            F ~ grad(0.5 * |B|^2)

        Computed by central finite differences.
        """
        point = np.array(point, dtype=float)
        h = float(grad_step)

        ex = np.array([h, 0.0])
        ey = np.array([0.0, h])

        phi_x_plus = self.energy_proxy_at(point + ex)
        phi_x_minus = self.energy_proxy_at(point - ex)
        phi_y_plus = self.energy_proxy_at(point + ey)
        phi_y_minus = self.energy_proxy_at(point - ey)

        dphi_dx = (phi_x_plus - phi_x_minus) / (2.0 * h)
        dphi_dy = (phi_y_plus - phi_y_minus) / (2.0 * h)

        F = force_gain * np.array([dphi_dx, dphi_dy], dtype=float)
        return F



class Particle2D:
    """
    Overdamped particle:
        v = mobility * F
        x_(t+dt) = x_t + v * dt
    """
    def __init__(self, position, mobility=2.5e4):
        self.pos = np.array(position, dtype=float)
        self.vel = np.zeros(2, dtype=float)
        self.mobility = float(mobility)

    def update(self, force, dt):
        self.vel = self.mobility * np.array(force, dtype=float)
        self.pos = self.pos + self.vel * dt




def precompute_current_combos(candidate_currents):
    return np.array(list(itertools.product(candidate_currents, repeat=4)), dtype=float)


def choose_candidate_set(current_distance):
    """
    Coarse far away, finer near target.
    """
    if current_distance > 0.04:
        return (-8.0, -4.0, 0.0, 4.0, 8.0)
    elif current_distance > 0.015:
        return (-8.0, -4.0, -2.0, 0.0, 2.0, 4.0, 8.0)
    else:
        return (-4.0, -2.0, -1.0, -0.5, 0.0, 0.5, 1.0, 2.0, 4.0)


def adaptive_lambda_current(base_lambda, current_distance):
    """
    Penalize current less when far away so the controller does not
    choose zero-current at center / weak-field states.
    """
    if current_distance > 0.05:
        return 0.15 * base_lambda
    if current_distance > 0.02:
        return 0.35 * base_lambda
    if current_distance > 0.008:
        return 0.70 * base_lambda
    return base_lambda



def brute_force_lookahead_controller(
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
):
    """
    Hybrid controller:
    - reward progress
    - reward moving in target direction
    - penalize distance remaining
    - penalize clipping / edge hugging / oversize steps
    - lightly penalize current

    Returns:
        best_currents, best_force, best_score, best_step_along
    """
    particle_pos = np.array(particle_pos, dtype=float)
    target_pos = np.array(target_pos, dtype=float)

    desired = target_pos - particle_pos
    current_distance = np.linalg.norm(desired)

    if current_distance < EPS:
        zero = np.zeros(4, dtype=float)
        return zero, np.zeros(2, dtype=float), 0.0, 0.0

    desired_hat = desired / current_distance
    lambda_eff = adaptive_lambda_current(lambda_current, current_distance)

    best_score = -np.inf
    best_currents = None
    best_force = None
    best_step_along = -np.inf

    for combo in all_combos:
        magnet_array.set_currents(combo)

        F = magnet_array.force_at(
            particle_pos,
            grad_step=grad_step,
            force_gain=force_gain,
        )

        step_vec = mobility * F * dt
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
        min_margin = min(margin_x, margin_y)
        edge_penalty = max(0.0, edge_buffer - min_margin)

        overshoot_penalty = max(0.0, step_mag - 0.65 * current_distance)

        nonzero_bonus = 0.0
        if np.any(np.abs(combo) > 1e-12) and step_along > 0:
            nonzero_bonus = 0.002

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


def brute_force_directional_fallback_controller(
    particle_pos,
    target_pos,
    magnet_array,
    all_combos,
    dt,
    mobility,
    force_gain=1.0,
    grad_step=1e-4,
    lambda_current=1e-5,
):
    """
    Fallback used when the lookahead controller still prefers to stall.

    Pushes hard for motion in the correct direction and lightly penalizes
    sideways motion and current usage.
    """
    particle_pos = np.array(particle_pos, dtype=float)
    target_pos = np.array(target_pos, dtype=float)

    desired = target_pos - particle_pos
    current_distance = np.linalg.norm(desired)

    if current_distance < EPS:
        zero = np.zeros(4, dtype=float)
        return zero, np.zeros(2, dtype=float), 0.0

    desired_hat = desired / current_distance

    best_score = -np.inf
    best_currents = None
    best_force = None

    for combo in all_combos:
        magnet_array.set_currents(combo)

        F = magnet_array.force_at(
            particle_pos,
            grad_step=grad_step,
            force_gain=force_gain,
        )

        step_vec = mobility * F * dt
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



def simulate_trial_model_b(
    start,
    target,
    dt=0.0007,
    steps=8000,
    tolerance=0.003,
    turns_per_magnet=None,
    max_current=8.0,
    field_constant=2.0e-7,
    decay_power=3.0,
    standoff=0.01,
    force_gain=1.0,
    mobility=2.5e4,
    lambda_current=5e-5,
):
    workspace = Workspace2D()
    magnets = ElectromagnetArray2D(
        turns_per_magnet=turns_per_magnet,
        max_current=max_current,
        field_constant=field_constant,
        decay_power=decay_power,
        standoff=standoff,
    )
    particle = Particle2D(start, mobility=mobility)

    start = np.array(start, dtype=float)
    target = np.array(target, dtype=float)

    trajectory = [particle.pos.copy()]
    distance_log = []
    current_log = []
    field_log = []
    force_log = []
    score_log = []

    reached = False

    combo_cache = {
        (-8.0, -4.0, 0.0, 4.0, 8.0): precompute_current_combos((-8.0, -4.0, 0.0, 4.0, 8.0)),
        (-8.0, -4.0, -2.0, 0.0, 2.0, 4.0, 8.0): precompute_current_combos((-8.0, -4.0, -2.0, 0.0, 2.0, 4.0, 8.0)),
        (-4.0, -2.0, -1.0, -0.5, 0.0, 0.5, 1.0, 2.0, 4.0): precompute_current_combos(
            (-4.0, -2.0, -1.0, -0.5, 0.0, 0.5, 1.0, 2.0, 4.0)
        ),
    }

    for _ in range(steps):
        current_distance = np.linalg.norm(particle.pos - target)
        current_set = choose_candidate_set(current_distance)
        all_combos = combo_cache[current_set]

        best_currents, best_force, best_score, best_step_along = brute_force_lookahead_controller(
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
        )

        # Fallback if the lookahead controller still stalls or chooses
        # effectively zero forward motion while far from the target.
        if current_distance > tolerance:
            if (
                best_currents is None
                or np.all(np.abs(best_currents) < 1e-12)
                or best_step_along < 1e-8
            ):
                fb_currents, fb_force, fb_score = brute_force_directional_fallback_controller(
                    particle_pos=particle.pos,
                    target_pos=target,
                    magnet_array=magnets,
                    all_combos=all_combos,
                    dt=dt,
                    mobility=particle.mobility,
                    force_gain=force_gain,
                    grad_step=1e-4,
                    lambda_current=1e-5,
                )
                best_currents = fb_currents
                best_force = fb_force
                best_score = fb_score

        magnets.set_currents(best_currents)

        B = magnets.field_at(particle.pos)
        F = best_force

        particle.update(F, dt)
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

    trajectory = np.array(trajectory)
    distance_log = np.array(distance_log)
    current_log = np.array(current_log)
    field_log = np.array(field_log)
    force_log = np.array(force_log)
    score_log = np.array(score_log)

    final_error = np.linalg.norm(trajectory[-1] - target)
    max_current_used = np.max(np.abs(current_log)) if len(current_log) > 0 else 0.0

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
            "turns_per_magnet": turns_per_magnet if turns_per_magnet is not None else [250, 250, 250, 250],
            "max_current": max_current,
            "field_constant": field_constant,
            "decay_power": decay_power,
            "standoff": standoff,
            "force_gain": force_gain,
            "mobility": mobility,
            "lambda_current": lambda_current,
        }
    }


#plots
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


def make_output_dir(path="simulation_outputs_model_b"):
    os.makedirs(path, exist_ok=True)
    return path


def save_trajectory_plot(result, trial_name, output_dir):
    traj = result["trajectory"]
    target = result["target"]
    magnets = result["magnets"]
    half = result["workspace_half"]

    plt.figure(figsize=(8, 8))
    plt.plot(traj[:, 0], traj[:, 1], linewidth=2, label="Path")
    plt.scatter(traj[0, 0], traj[0, 1], s=120, label="Start")
    plt.scatter(target[0], target[1], s=120, label="Target")
    plt.scatter(magnets[:, 0], magnets[:, 1], s=120, label="Magnets")

    labels = ["BL", "BR", "TL", "TR"]
    for pos, label in zip(magnets, labels):
        plt.text(pos[0] + 0.002, pos[1] + 0.002, label)

    plt.xlim(-half, half)
    plt.ylim(-half, half)
    plt.gca().set_aspect("equal")
    plt.grid(True)
    plt.legend()
    plt.xlabel("X Position (m)")
    plt.ylabel("Y Position (m)")
    plt.title(f"{trial_name} - Trajectory")

    filename = os.path.join(output_dir, f"{trial_name}_trajectory.png")
    plt.savefig(filename, dpi=300, bbox_inches="tight")
    plt.close()


def save_distance_plot(result, trial_name, output_dir):
    dist = result["distance_log"]

    plt.figure(figsize=(8, 5))
    plt.plot(dist, linewidth=2)
    plt.grid(True)
    plt.xlabel("Simulation Step")
    plt.ylabel("Distance to Target (m)")
    plt.title(f"{trial_name} - Distance to Target")

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
    plt.title(f"{trial_name} - Force Magnitude")

    filename = os.path.join(output_dir, f"{trial_name}_force.png")
    plt.savefig(filename, dpi=300, bbox_inches="tight")
    plt.close()


def save_field_map(result, trial_name, output_dir, grid_points=21):
    if len(result["current_log"]) == 0:
        return

    half = result["workspace_half"]
    first_currents = result["current_log"][0]
    params = result["params"]

    array = ElectromagnetArray2D(
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
            p = np.array([X[i, j], Y[i, j]])
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
    plt.scatter(result["magnets"][:, 0], result["magnets"][:, 1], s=120, label="Magnets")
    plt.scatter(result["start"][0], result["start"][1], s=100, label="Start")
    plt.scatter(result["target"][0], result["target"][1], s=100, label="Target")

    plt.xlim(-half, half)
    plt.ylim(-half, half)
    plt.gca().set_aspect("equal")
    plt.grid(True)
    plt.legend()
    plt.xlabel("X Position (m)")
    plt.ylabel("Y Position (m)")
    plt.title(f"{trial_name} - Field Map (First-Step Currents)")

    filename = os.path.join(output_dir, f"{trial_name}_fieldmap.png")
    plt.savefig(filename, dpi=300, bbox_inches="tight")
    plt.close()


def save_results_csv(results, output_dir, filename="trial_summary.csv"):
    path = os.path.join(output_dir, filename)
    with open(path, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow([
            "trial_name",
            "start_x", "start_y",
            "target_x", "target_y",
            "final_x", "final_y",
            "steps_taken",
            "final_error_m",
            "final_error_mm",
            "reached_target",
            "max_current_used_A",
        ])

        for trial_name, r in results.items():
            writer.writerow([
                trial_name,
                r["start"][0], r["start"][1],
                r["target"][0], r["target"][1],
                r["final_position"][0], r["final_position"][1],
                r["steps_taken"],
                r["final_error"],
                r["final_error"] * 1000.0,
                r["reached"],
                r["max_current_used"],
            ])



def run_all_trials_model_b():
    rng = np.random.default_rng(42)

    # Keep benchmark trials for comparability, but expand the random test
    # to near-full workspace coverage.
    trials = {
        "trial_1_bottomleft_to_topright": {
            "start": (-0.05, -0.05),
            "target": (0.05, 0.05),
        },
        "trial_2_topleft_to_bottomright": {
            "start": (-0.05, 0.05),
            "target": (0.05, -0.05),
        },
        "trial_3_center_to_topright": {
            "start": (0.0, 0.0),
            "target": (0.05, 0.05),
        },
        "trial_4_random_to_random_near_full_workspace": {
            "start": tuple(rng.uniform(-TEST_RANGE, TEST_RANGE, size=2)),
            "target": tuple(rng.uniform(-TEST_RANGE, TEST_RANGE, size=2)),
        },
    }

    output_dir = make_output_dir("simulation_outputs_model_b")
    results = {}

    sim_kwargs = {
        "dt": 0.0007,
        "steps": 8000,
        "tolerance": 0.003,
        "turns_per_magnet": [250, 250, 250, 250],
        "max_current": 8.0,
        "field_constant": 2.0e-7,
        "decay_power": 3.0,
        "standoff": 0.01,
        "force_gain": 1.0,
        "mobility": 2.5e4,
        "lambda_current": 5e-5,
    }

    for trial_name, cfg in trials.items():
        result = simulate_trial_model_b(
            start=cfg["start"],
            target=cfg["target"],
            **sim_kwargs,
        )
        results[trial_name] = result

        print_summary(result, trial_name)
        save_trajectory_plot(result, trial_name, output_dir)
        save_distance_plot(result, trial_name, output_dir)
        save_force_plot(result, trial_name, output_dir)
        save_field_map(result, trial_name, output_dir)

    save_results_csv(results, output_dir)
    print(f"\nSaved plots and CSV to: {output_dir}")
    return results


if __name__ == "__main__":
    run_all_trials_model_b()