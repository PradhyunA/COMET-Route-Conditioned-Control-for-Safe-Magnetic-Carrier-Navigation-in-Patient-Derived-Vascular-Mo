import os
import numpy as np
import matplotlib.pyplot as plt

INCH_TO_METER = 0.0254
SIDE = 6 * INCH_TO_METER
HALF = SIDE / 2


class Workspace2D:
    def __init__(self):
        self.half = HALF

    def clip(self, p):
        p[0] = np.clip(p[0], -self.half, self.half)
        p[1] = np.clip(p[1], -self.half, self.half)
        return p


class ElectromagnetArray:
    def __init__(self):
        # 4 corner magnets
        self.positions = np.array([
            [-HALF, -HALF],   # bottom-left
            [ HALF, -HALF],   # bottom-right
            [-HALF,  HALF],   # top-left
            [ HALF,  HALF],   # top-right
        ], dtype=float)

        # Placeholder until finalized
        self.turns = np.array([250, 250, 250, 250], dtype=float)

        self.max_current = 8.0
        self.currents = np.zeros(4, dtype=float)

        # Tunable simplified model parameters
        self.field_constant = 0.002
        self.decay_power = 2.0
        self.standoff = 0.01

    def set_currents(self, c):
        self.currents = np.clip(np.array(c, dtype=float), -self.max_current, self.max_current)

    def field(self, point):
        point = np.array(point, dtype=float)
        B = np.zeros(2, dtype=float)

        for i in range(4):
            r = point - self.positions[i]
            d = np.linalg.norm(r)
            direction = r / (d + 1e-9)

            magnitude = (
                self.field_constant
                * self.turns[i]
                * self.currents[i]
                / ((d + self.standoff) ** self.decay_power)
            )

            B += magnitude * direction

        return B


class Particle:
    def __init__(self, pos):
        self.pos = np.array(pos, dtype=float)
        self.vel = np.zeros(2, dtype=float)

        #mobility parameter for moving in liquid
        self.mobility = 0.0008

    def update(self, field, dt):
        self.vel = self.mobility * field
        self.pos = self.pos + self.vel * dt


def controller(particle, target, magnets, max_current=8.0):
    """
    controller for corner magnets
    """
    desired = np.array(target, dtype=float) - np.array(particle, dtype=float)
    norm = np.linalg.norm(desired)

    if norm < 1e-9:
        return np.zeros(4, dtype=float)

    desired = desired / norm
    currents = np.zeros(4, dtype=float)

    for i in range(4):
        r = np.array(particle, dtype=float) - np.array(magnets[i], dtype=float)
        r = r / (np.linalg.norm(r) + 1e-9)

        score = np.dot(r, desired)
        currents[i] = score * max_current

    return currents


def simulate(start, target, dt=0.02, steps=1500, tolerance=0.003):
    workspace = Workspace2D()
    magnets = ElectromagnetArray()
    particle = Particle(start)

    start = np.array(start, dtype=float)
    target = np.array(target, dtype=float)

    traj = [particle.pos.copy()]
    dist_log = []
    field_log = []
    current_log = []

    reached = False

    for step in range(steps):
        currents = controller(
            particle=particle.pos,
            target=target,
            magnets=magnets.positions,
            max_current=magnets.max_current,
        )

        magnets.set_currents(currents)
        B = magnets.field(particle.pos)

        particle.update(B, dt)
        particle.pos = workspace.clip(particle.pos)

        traj.append(particle.pos.copy())
        dist = np.linalg.norm(particle.pos - target)

        dist_log.append(dist)
        field_log.append(B.copy())
        current_log.append(magnets.currents.copy())

        if dist < tolerance:
            reached = True
            break

    traj = np.array(traj)
    dist_log = np.array(dist_log)
    field_log = np.array(field_log)
    current_log = np.array(current_log)

    final_error = np.linalg.norm(traj[-1] - target)
    max_current_used = np.max(np.abs(current_log)) if len(current_log) > 0 else 0.0

    return {
        "start": start,
        "target": target,
        "trajectory": traj,
        "distance_log": dist_log,
        "field_log": field_log,
        "current_log": current_log,
        "magnets": magnets.positions.copy(),
        "reached": reached,
        "steps_taken": len(traj) - 1,
        "final_position": traj[-1],
        "final_error": final_error,
        "max_current_used": max_current_used,
        "workspace_half": HALF,
    }


def print_summary(result, trial_name):
    print(f"\n=== {trial_name} ===")
    print("Start:", result["start"])
    print("Target:", result["target"])
    print("Final position:", result["final_position"])
    print("Steps taken:", result["steps_taken"])
    print("Final error (m):", result["final_error"])
    print("Final error (mm):", result["final_error"] * 1000)
    print("Reached target:", result["reached"])
    print("Max current used (A):", result["max_current_used"])


def make_output_dir(path="simulation_outputs"):
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


def save_field_map(result, trial_name, output_dir, grid_points=17):
    """
    Quiver plot of the field

    uses the currents from the first
    control step of the trial and computes the resulting field over a grid.
    """
    half = result["workspace_half"]
    magnets_pos = result["magnets"]

    if len(result["current_log"]) == 0:
        return

    first_currents = result["current_log"][0]

    # rebuild electromagnet array with same parameters
    array = ElectromagnetArray()
    array.set_currents(first_currents)

    xs = np.linspace(-half, half, grid_points)
    ys = np.linspace(-half, half, grid_points)
    X, Y = np.meshgrid(xs, ys)

    U = np.zeros_like(X)
    V = np.zeros_like(Y)

    for i in range(grid_points):
        for j in range(grid_points):
            point = np.array([X[i, j], Y[i, j]])
            field = array.field(point)
            U[i, j] = field[0]
            V[i, j] = field[1]

    magnitude = np.sqrt(U**2 + V**2)


    U_plot = U / (magnitude + 1e-9)
    V_plot = V / (magnitude + 1e-9)

    plt.figure(figsize=(8, 8))
    plt.quiver(X, Y, U_plot, V_plot, magnitude)
    plt.scatter(magnets_pos[:, 0], magnets_pos[:, 1], s=120, label="Magnets")
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

    output_dir = make_output_dir("simulation_outputs")
    results = {}

    for trial_name, cfg in trials.items():
        result = simulate(
            start=cfg["start"],
            target=cfg["target"],
            dt=0.02,
            steps=1500,
            tolerance=0.003,
        )

        results[trial_name] = result
        print_summary(result, trial_name)

        save_trajectory_plot(result, trial_name, output_dir)
        save_distance_plot(result, trial_name, output_dir)
        save_field_map(result, trial_name, output_dir)

    print(f"\nSaved plots to: {output_dir}")
    return results


if __name__ == "__main__":
    run_all_trials()