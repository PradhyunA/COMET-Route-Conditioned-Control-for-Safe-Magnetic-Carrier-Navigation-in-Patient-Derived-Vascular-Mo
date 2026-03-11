import os
import csv
import time
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D  # noqa: F401

from simulation_3d_realistic import (
    HALF,
    TEST_RANGE,
    MAX_STEP,
    Workspace3D,
    ElectromagnetArray3D_Real,
    MagneticParticle3D,
)

from mpc_controller_3d import continuous_mpc_controller_3d


def simulate_trial_realistic_mpc(
    start,
    target,
    dt=0.02,
    steps=3000,
    tolerance=0.003,
    coil_radius=0.03,
    coil_length=0.12,
    turns=1500,
    max_current=8.0,
    particle_diameter=500e-6,
    chi_eff=3.0,
    M_sat=4.5e5,
    fluid_viscosity=8.9e-4,
    temperature=298.0,
    enable_brownian=False,   # OFF for controller validation
    max_step=MAX_STEP,
    rng_seed=42,
):
    workspace = Workspace3D()
    magnets = ElectromagnetArray3D_Real(
        coil_radius=coil_radius,
        coil_length=coil_length,
        turns=turns,
        max_current=max_current,
    )
    particle = MagneticParticle3D(
        position=start,
        diameter=particle_diameter,
        chi_eff=chi_eff,
        M_sat=M_sat,
        fluid_viscosity=fluid_viscosity,
        temperature=temperature,
        enable_brownian=enable_brownian,
        rng_seed=rng_seed,
    )

    start = np.asarray(start, dtype=float)
    target = np.asarray(target, dtype=float)

    trajectory = [particle.pos.copy()]
    distance_log = []
    current_log = []
    field_log = []
    force_log = []
    velocity_log = []
    cost_log = []
    step_time_log = []

    reached = False
    prev_currents = np.zeros(magnets.n_magnets, dtype=float)

    for step in range(steps):
        t0 = time.perf_counter()

        best_currents, best_force, best_cost = continuous_mpc_controller_3d(
            particle=particle,
            particle_pos=particle.pos,
            target_pos=target,
            magnet_array=magnets,
            workspace=workspace,
            dt=dt,
            max_current=max_current,
            max_step=max_step,
            initial_guess=prev_currents,
            w_progress=80.0,
            w_align=8.0,
            w_dist=2.0,
            w_curr=1e-5,        # Penalty for absolute current usage
            w_smooth=1e-4,      # Very gentle penalty on changing current rapidly
            w_edge=20.0,
            w_overshoot=2.0,
        )

        t1 = time.perf_counter()

        prev_currents = best_currents.copy()
        magnets.set_currents(best_currents)
        B = magnets.field_at(particle.pos)

        particle.update(best_force, dt, max_step=max_step)
        particle.pos = workspace.clip(particle.pos)

        trajectory.append(particle.pos.copy())
        distance = np.linalg.norm(particle.pos - target)

        distance_log.append(distance)
        current_log.append(magnets.currents.copy())
        field_log.append(np.linalg.norm(B))
        force_log.append(np.linalg.norm(best_force))
        velocity_log.append(np.linalg.norm(particle.vel))
        cost_log.append(best_cost)
        step_time_log.append(t1 - t0)

        if distance < tolerance:
            reached = True
            break

        if (step + 1) % 50 == 0:
            print(
                f"  step {step+1:>4d}/{steps}  "
                f"dist={distance:.5f} m  "
                f"|I|max={np.max(np.abs(best_currents)):.3f} A  "
                f"|B|={np.linalg.norm(B):.3e} T  "
                f"|F|={np.linalg.norm(best_force):.3e} N"
            )

    trajectory = np.array(trajectory)
    distance_log = np.array(distance_log)
    current_log = np.array(current_log)
    field_log = np.array(field_log)
    force_log = np.array(force_log)
    velocity_log = np.array(velocity_log)
    cost_log = np.array(cost_log)
    step_time_log = np.array(step_time_log)

    final_error = np.linalg.norm(trajectory[-1] - target)
    max_current_used = np.max(np.abs(current_log)) if len(current_log) > 0 else 0.0
    avg_step_time = np.mean(step_time_log) if len(step_time_log) > 0 else 0.0

    return {
        "start": start,
        "target": target,
        "trajectory": trajectory,
        "distance_log": distance_log,
        "current_log": current_log,
        "field_log": field_log,
        "force_log": force_log,
        "velocity_log": velocity_log,
        "cost_log": cost_log,
        "step_time_log": step_time_log,
        "avg_step_time": avg_step_time,
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
            "coil_radius": coil_radius,
            "coil_length": coil_length,
            "turns": turns,
            "max_current": max_current,
            "particle_diameter": particle_diameter,
            "chi_eff": chi_eff,
            "M_sat": M_sat,
            "fluid_viscosity": fluid_viscosity,
            "temperature": temperature,
            "enable_brownian": enable_brownian,
            "max_step": max_step,
        },
    }


def print_summary(result, trial_name):
    print(f"\n{'='*60}")
    print(f"  {trial_name}")
    print(f"{'='*60}")
    print(f"  Start           : {result['start']}")
    print(f"  Target          : {result['target']}")
    print(f"  Final position  : {result['final_position']}")
    print(f"  Steps taken     : {result['steps_taken']}")
    print(f"  Final error (m) : {result['final_error']:.6f}")
    print(f"  Final error (mm): {result['final_error'] * 1000:.3f}")
    print(f"  Reached target  : {result['reached']}")
    print(f"  Max current (A) : {result['max_current_used']:.3f}")
    print(f"  Avg ctrl time(s): {result['avg_step_time']:.4f}")
    if len(result["force_log"]) > 0:
        print(f"  Peak |F|  (N)   : {np.max(result['force_log']):.3e}")
        print(f"  Peak |v|  (m/s) : {np.max(result['velocity_log']):.3e}")
        print(f"  Peak |B|  (T)   : {np.max(result['field_log']):.3e}")


def make_output_dir(path="simulation_outputs_3d_mpc"):
    os.makedirs(path, exist_ok=True)
    return path


def save_trajectory_plot_3d(result, trial_name, output_dir):
    traj = result["trajectory"]
    target = result["target"]
    magnets = result["magnets"]
    half = result["workspace_half"]

    fig = plt.figure(figsize=(10, 10))
    ax = fig.add_subplot(111, projection="3d")

    ax.plot(traj[:, 0], traj[:, 1], traj[:, 2], lw=2, label="Path")
    ax.scatter(*traj[0], s=120, marker="o", label="Start")
    ax.scatter(*target, s=120, marker="*", label="Target")
    ax.scatter(magnets[:, 0], magnets[:, 1], magnets[:, 2], s=80, marker="^", label="Magnets")

    ax.set_xlim(-half, half)
    ax.set_ylim(-half, half)
    ax.set_zlim(-half, half)
    ax.set_xlabel("X (m)")
    ax.set_ylabel("Y (m)")
    ax.set_zlabel("Z (m)")
    ax.set_title(f"{trial_name} - 3D MPC Trajectory")
    ax.legend(fontsize=8)

    fn = os.path.join(output_dir, f"{trial_name}_trajectory_3d.png")
    plt.savefig(fn, dpi=300, bbox_inches="tight")
    plt.close()


def save_distance_plot(result, trial_name, output_dir):
    if len(result["distance_log"]) == 0:
        return

    steps = np.arange(len(result["distance_log"]))
    t_sec = steps * result["params"]["dt"]

    plt.figure(figsize=(10, 5))
    plt.plot(t_sec, result["distance_log"] * 1000.0, lw=2)
    plt.grid(True, alpha=0.3)
    plt.xlabel("Time (s)")
    plt.ylabel("Distance to Target (mm)")
    plt.title(f"{trial_name} - Distance to Target")

    fn = os.path.join(output_dir, f"{trial_name}_distance.png")
    plt.savefig(fn, dpi=300, bbox_inches="tight")
    plt.close()


def save_physics_plot(result, trial_name, output_dir):
    if len(result["force_log"]) == 0:
        return

    steps = np.arange(len(result["force_log"]))
    t_sec = steps * result["params"]["dt"]

    fig, axes = plt.subplots(3, 1, figsize=(10, 10), sharex=True)

    axes[0].plot(t_sec, result["force_log"], lw=1.5)
    axes[0].set_ylabel("|F| (N)")
    axes[0].grid(True, alpha=0.3)
    axes[0].ticklabel_format(axis="y", style="scientific", scilimits=(0, 0))
    axes[0].set_title(f"{trial_name} - Physics Over Time")

    axes[1].plot(t_sec, np.array(result["velocity_log"]) * 1e6, lw=1.5)
    axes[1].set_ylabel("|v| (μm/s)")
    axes[1].grid(True, alpha=0.3)

    axes[2].plot(t_sec, np.array(result["field_log"]) * 1e3, lw=1.5)
    axes[2].set_ylabel("|B| (mT)")
    axes[2].set_xlabel("Time (s)")
    axes[2].grid(True, alpha=0.3)

    plt.tight_layout()
    fn = os.path.join(output_dir, f"{trial_name}_physics.png")
    plt.savefig(fn, dpi=300, bbox_inches="tight")
    plt.close()


def save_current_plot(result, trial_name, output_dir):
    current_log = result["current_log"]
    if len(current_log) == 0:
        return

    plt.figure(figsize=(10, 6))
    for i in range(current_log.shape[1]):
        plt.plot(current_log[:, i], linewidth=1.5, label=f"I{i+1}")
    plt.grid(True, alpha=0.3)
    plt.xlabel("Simulation Step")
    plt.ylabel("Current (A)")
    plt.title(f"{trial_name} - 3D MPC Currents")
    plt.legend(ncol=2, fontsize=8)

    fn = os.path.join(output_dir, f"{trial_name}_currents.png")
    plt.savefig(fn, dpi=300, bbox_inches="tight")
    plt.close()


def save_results_csv(results, output_dir, filename="trial_summary.csv"):
    path = os.path.join(output_dir, filename)
    with open(path, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow([
            "trial_name",
            "start_x", "start_y", "start_z",
            "target_x", "target_y", "target_z",
            "final_x", "final_y", "final_z",
            "steps_taken", "sim_time_s",
            "final_error_m", "final_error_mm",
            "reached_target", "max_current_A",
            "avg_control_step_time_s",
            "peak_force_N", "peak_velocity_m_s", "peak_field_T",
        ])

        for name, r in results.items():
            w.writerow([
                name,
                r["start"][0], r["start"][1], r["start"][2],
                r["target"][0], r["target"][1], r["target"][2],
                r["final_position"][0], r["final_position"][1], r["final_position"][2],
                r["steps_taken"],
                r["steps_taken"] * r["params"]["dt"],
                r["final_error"],
                r["final_error"] * 1000.0,
                r["reached"],
                r["max_current_used"],
                r["avg_step_time"],
                np.max(r["force_log"]) if len(r["force_log"]) > 0 else 0.0,
                np.max(r["velocity_log"]) if len(r["velocity_log"]) > 0 else 0.0,
                np.max(r["field_log"]) if len(r["field_log"]) > 0 else 0.0,
            ])


def run_all_trials():
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
    }

    output_dir = make_output_dir("simulation_outputs_3d_mpc")
    results = {}

    sim_kwargs = {
        "dt": 0.02,
        "steps": 3000,
        "tolerance": 0.003,
        "coil_radius": 0.03,
        "coil_length": 0.12,
        "turns": 1500,
        "max_current": 8.0,
        "particle_diameter": 500e-6,
        "chi_eff": 3.0,
        "M_sat": 4.5e5,
        "fluid_viscosity": 8.9e-4,
        "temperature": 298.0,
        "enable_brownian": False,   # turn on later after control works
        "max_step": MAX_STEP,
        "rng_seed": 42,
    }

    for trial_name, cfg in trials.items():
        print(f"\n>>> Running {trial_name} ...")
        result = simulate_trial_realistic_mpc(
            start=cfg["start"],
            target=cfg["target"],
            **sim_kwargs,
        )
        results[trial_name] = result

        print_summary(result, trial_name)
        save_trajectory_plot_3d(result, trial_name, output_dir)
        save_distance_plot(result, trial_name, output_dir)
        save_physics_plot(result, trial_name, output_dir)
        save_current_plot(result, trial_name, output_dir)

    save_results_csv(results, output_dir)
    print(f"\nSaved plots and CSV to: {output_dir}")
    return results


if __name__ == "__main__":
    run_all_trials()