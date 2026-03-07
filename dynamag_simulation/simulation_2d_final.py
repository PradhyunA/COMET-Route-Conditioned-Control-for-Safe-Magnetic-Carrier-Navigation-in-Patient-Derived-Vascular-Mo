import os
import csv
import numpy as np
import matplotlib.pyplot as plt

# ============================================================
# GLOBAL CONSTANTS
# ============================================================

INCH_TO_METER = 0.0254
SIDE = 6.0 * INCH_TO_METER
HALF = SIDE / 2.0

# Numerical epsilon
EPS = 1e-12


# ============================================================
# WORKSPACE
# ============================================================

class Workspace2D:
    def __init__(self, half_side=HALF):
        self.half = float(half_side)

    def clip(self, p):
        p = np.array(p, dtype=float)
        p[0] = np.clip(p[0], -self.half, self.half)
        p[1] = np.clip(p[1], -self.half, self.half)
        return p


# ============================================================
# ELECTROMAGNET ARRAY
# ============================================================

class ElectromagnetArray2D:
    """
    4 corner electromagnets around a square workspace.

    This class computes a simplified magnetic field-like quantity B(x,y).
    The particle force is later derived from grad(0.5*|B|^2).
    """

    def __init__(
        self,
        turns_per_magnet=None,
        max_current=8.0,
        field_constant=2.0e-7,
        decay_power=3.0,
        standoff=0.01,
    ):
        # Corner magnets
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

        # Simplified field parameters
        self.field_constant = float(field_constant)
        self.decay_power = float(decay_power)
        self.standoff = float(standoff)

    def set_currents(self, currents):
        currents = np.array(currents, dtype=float)
        self.currents = np.clip(currents, -self.max_current, self.max_current)

    def field_at(self, point):
        """
        Compute a simplified net magnetic field vector at a 2D point.

        Model:
            B_i ~ k * N * I / (r + z0)^p * r_hat

        Using p=3 is closer to dipole-style decay than inverse distance.
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

    def field_energy_density_proxy(self, point):
        """
        Returns 0.5 * |B|^2.
        This acts as a magnetic potential-like quantity whose gradient
        drives the particle.
        """
        B = self.field_at(point)
        return 0.5 * np.dot(B, B)

    def force_at(self, point, grad_step=1e-4, force_gain=1.0):
        """
        Compute force from gradient of 0.5*|B|^2 via central differences.

        F ~ grad(0.5*|B|^2)

        This is a CPU-friendly approximation to magnetic gradient force.
        """
        point = np.array(point, dtype=float)
        h = grad_step

        ex = np.array([h, 0.0])
        ey = np.array([0.0, h])

        phi_x_plus = self.field_energy_density_proxy(point + ex)
        phi_x_minus = self.field_energy_density_proxy(point - ex)
        phi_y_plus = self.field_energy_density_proxy(point + ey)
        phi_y_minus = self.field_energy_density_proxy(point - ey)

        dphi_dx = (phi_x_plus - phi_x_minus) / (2.0 * h)
        dphi_dy = (phi_y_plus - phi_y_minus) / (2.0 * h)

        F = force_gain * np.array([dphi_dx, dphi_dy], dtype=float)
        return F


# ============================================================
# PARTICLE MODEL
# ============================================================

class Particle2D:
    """
    Overdamped particle:
        v = mobility * F
        x_{t+dt} = x_t + v*dt
    """

    def __init__(self, position, mobility=2.0e5):
        self.pos = np.array(position, dtype=float)
        self.vel = np.zeros(2, dtype=float)
        self.mobility = float(mobility)

    def update(self, force, dt):
        self.vel = self.mobility * np.array(force, dtype=float)
        self.pos = self.pos + self.vel * dt


# ============================================================
# CONTROLLER
# ============================================================

def target_alignment_controller(particle_pos, target_pos, magnet_positions, max_current):
    """
    Simple controller:
    - compute desired direction to target
    - score each magnet by alignment with that direction
    - assign current proportional to alignment

    This is still simple, but works well for steering.
    """
    particle_pos = np.array(particle_pos, dtype=float)
    target_pos = np.array(target_pos, dtype=float)

    desired = target_pos - particle_pos
    norm = np.linalg.norm(desired)

    if norm < EPS:
        return np.zeros(4, dtype=float)

    desired_hat = desired / norm
    currents = np.zeros(4, dtype=float)

    for i, mp in enumerate(magnet_positions):
        # Vector from magnet toward particle
        influence_vec = particle_pos - mp
        influence_norm = np.linalg.norm(influence_vec) + EPS
        influence_hat = influence_vec / influence_norm

        score = np.dot(influence_hat, desired_hat)
        currents[i] = score * max_current

    return np.clip(currents, -max_current, max_current)


# ============================================================
# SIMULATION
# ============================================================

def simulate_trial(
    start,
    target,
    dt=0.01,
    steps=2500,
    tolerance=0.003,
    turns_per_magnet=None,
    max_current=8.0,
    field_constant=2.0e-7,
    decay_power=3.0,
    standoff=0.01,
    force_gain=1.0,
    mobility=2.0e5,
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

    reached = False

    for _ in range(steps):
        currents = target_alignment_controller(
            particle_pos=particle.pos,
            target_pos=target,
            magnet_positions=magnets.positions,
            max_current=magnets.max_current,
        )
        magnets.set_currents(currents)

        B = magnets.field_at(particle.pos)
        F = magnets.force_at(particle.pos, grad_step=1e-4, force_gain=force_gain)

        particle.update(F, dt)
        particle.pos = workspace.clip(particle.pos)

        trajectory.append(particle.pos.copy())
        distance = np.linalg.norm(particle.pos - target)

        distance_log.append(distance)
        current_log.append(magnets.currents.copy())
        field_log.append(B.copy())
        force_log.append(F.copy())

        if distance < tolerance:
            reached = True
            break

    trajectory = np.array(trajectory)
    distance_log = np.array(distance_log)
    current_log = np.array(current_log)
    field_log = np.array(field_log)
    force_log = np.array(force_log)

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
        }
    }


# ============================================================
# REPORTING
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


def make_output_dir(path="simulation_outputs_final"):
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

    mag = np.linalg.norm(force_log, axis=1)

    plt.figure(figsize=(8, 5))
    plt.plot(mag, linewidth=2)
    plt.grid(True)
    plt.xlabel("Simulation Step")
    plt.ylabel("Force Magnitude (arb. units)")
    plt.title(f"{trial_name} - Force Magnitude")

    filename = os.path.join(output_dir, f"{trial_name}_force.png")
    plt.savefig(filename, dpi=300, bbox_inches="tight")
    plt.close()


def save_field_map(result, trial_name, output_dir, grid_points=21):
    """
    Uses the FIRST control step currents and computes the field map over the workspace.
    """
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

    U_plot = U / (M + EPS)
    V_plot = V / (M + EPS)

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


# ============================================================
# TRIAL RUNNER
# ============================================================

def run_all_trials():
    rng = np.random.default_rng(42)

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
        "trial_4_random_to_random": {
            "start": tuple(rng.uniform(-0.05, 0.05, size=2)),
            "target": tuple(rng.uniform(-0.05, 0.05, size=2)),
        },
    }

    output_dir = make_output_dir("simulation_outputs_final")
    results = {}

    # Tuned baseline parameters for this final 2D model
    sim_kwargs = {
        "dt": 0.01,
        "steps": 2500,
        "tolerance": 0.003,
        "turns_per_magnet": [250, 250, 250, 250],
        "max_current": 8.0,
        "field_constant": 2.0e-7,
        "decay_power": 3.0,
        "standoff": 0.01,
        "force_gain": 1.0,
        "mobility": 2.0e5,
    }

    for trial_name, cfg in trials.items():
        result = simulate_trial(
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
    run_all_trials()