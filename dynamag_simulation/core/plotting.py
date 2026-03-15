"""
Consolidated plotting utilities for all simulation variants.
"""

import os
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D  # noqa: F401


def save_trajectory_plot_3d(result, output_dir):
    traj = result.trajectory
    target = result.target
    magnets = result.magnets
    half = result.workspace_half

    fig = plt.figure(figsize=(10, 10))
    ax = fig.add_subplot(111, projection="3d")

    ax.plot(traj[:, 0], traj[:, 1], traj[:, 2], lw=2, label="Path")
    ax.scatter(*traj[0], s=120, marker="o", label="Start")
    ax.scatter(*target, s=120, marker="*", label="Target")
    if magnets is not None:
        ax.scatter(magnets[:, 0], magnets[:, 1], magnets[:, 2],
                   s=80, marker="^", label="Magnets")

    ax.set_xlim(-half, half)
    ax.set_ylim(-half, half)
    ax.set_zlim(-half, half)
    ax.set_xlabel("X (m)")
    ax.set_ylabel("Y (m)")
    ax.set_zlabel("Z (m)")
    ax.set_title(f"{result.trial_name} \u2013 3D Trajectory")
    ax.legend(fontsize=8)

    fn = os.path.join(output_dir, f"{result.trial_name}_trajectory_3d.png")
    plt.savefig(fn, dpi=300, bbox_inches="tight")
    plt.close()


def save_distance_plot(result, output_dir):
    if len(result.distance_log) == 0:
        return

    steps = np.arange(len(result.distance_log))
    t_sec = steps * result.dt

    plt.figure(figsize=(10, 5))
    plt.plot(t_sec, result.distance_log * 1000, lw=2, color="tab:blue")
    plt.xlabel("Time (s)")
    plt.ylabel("Distance to Target (mm)")
    plt.grid(True, alpha=0.3)
    plt.title(f"{result.trial_name} \u2013 Distance to Target")

    fn = os.path.join(output_dir, f"{result.trial_name}_distance.png")
    plt.savefig(fn, dpi=300, bbox_inches="tight")
    plt.close()


def save_physics_plot(result, output_dir):
    if len(result.force_log) == 0:
        return

    steps = np.arange(len(result.force_log))
    t_sec = steps * result.dt

    fig, axes = plt.subplots(3, 1, figsize=(10, 10), sharex=True)

    axes[0].plot(t_sec, result.force_log, lw=1.5, color="tab:red")
    axes[0].set_ylabel("|F| (N)")
    axes[0].set_title(f"{result.trial_name} \u2013 Physics Over Time")
    axes[0].grid(True, alpha=0.3)
    axes[0].ticklabel_format(axis="y", style="scientific", scilimits=(0, 0))

    axes[1].plot(t_sec, np.array(result.velocity_log) * 1e6,
                 lw=1.5, color="tab:green")
    axes[1].set_ylabel("|v| (\u03bcm/s)")
    axes[1].grid(True, alpha=0.3)

    axes[2].plot(t_sec, np.array(result.field_log) * 1e3,
                 lw=1.5, color="tab:purple")
    axes[2].set_ylabel("|B| (mT)")
    axes[2].set_xlabel("Time (s)")
    axes[2].grid(True, alpha=0.3)

    plt.tight_layout()
    fn = os.path.join(output_dir, f"{result.trial_name}_physics.png")
    plt.savefig(fn, dpi=300, bbox_inches="tight")
    plt.close()


def save_current_plot(result, output_dir):
    if len(result.current_log) == 0:
        return

    plt.figure(figsize=(10, 6))
    for i in range(result.current_log.shape[1]):
        plt.plot(result.current_log[:, i], lw=1.5, label=f"I{i + 1}")
    plt.grid(True, alpha=0.3)
    plt.xlabel("Simulation Step")
    plt.ylabel("Current (A)")
    plt.title(f"{result.trial_name} \u2013 Coil Currents")
    plt.legend(ncol=2, fontsize=8)

    fn = os.path.join(output_dir, f"{result.trial_name}_currents.png")
    plt.savefig(fn, dpi=300, bbox_inches="tight")
    plt.close()


def save_field_map_3d(result, output_dir, magnet_array, grid_points=15):
    """z=0 midplane field map using first-step currents."""
    if len(result.current_log) == 0:
        return

    half = result.workspace_half
    magnet_array.set_currents(result.current_log[0])

    xs = np.linspace(-half, half, grid_points)
    ys = np.linspace(-half, half, grid_points)
    X, Y = np.meshgrid(xs, ys)

    U = np.zeros_like(X)
    V = np.zeros_like(Y)
    M = np.zeros_like(X)

    for i in range(grid_points):
        for j in range(grid_points):
            B = magnet_array.field_at(np.array([X[i, j], Y[i, j], 0.0]))
            U[i, j] = B[0]
            V[i, j] = B[1]
            M[i, j] = np.linalg.norm(B)

    if np.all(M < 1e-20):
        return

    mask = M > 1e-20
    U_n = np.zeros_like(U)
    V_n = np.zeros_like(V)
    U_n[mask] = U[mask] / M[mask]
    V_n[mask] = V[mask] / M[mask]

    plt.figure(figsize=(8, 8))
    plt.quiver(X, Y, U_n, V_n, M * 1e3)
    plt.colorbar(label="|B| (mT)")

    if result.magnets is not None:
        plt.scatter(result.magnets[:, 0], result.magnets[:, 1],
                    s=80, marker="^", label="Magnets (proj.)")
    plt.scatter(result.start[0], result.start[1], s=100, label="Start (proj.)")
    plt.scatter(result.target[0], result.target[1], s=100, label="Target (proj.)")

    plt.xlim(-half, half)
    plt.ylim(-half, half)
    plt.gca().set_aspect("equal")
    plt.grid(True, alpha=0.3)
    plt.legend()
    plt.xlabel("X (m)")
    plt.ylabel("Y (m)")
    plt.title(f"{result.trial_name} \u2013 Field Map z=0 (mT)")

    fn = os.path.join(output_dir, f"{result.trial_name}_fieldmap_z0.png")
    plt.savefig(fn, dpi=300, bbox_inches="tight")
    plt.close()


def save_all_plots(result, output_dir, magnet_array=None):
    """Convenience: generate every standard plot for a trial."""
    save_trajectory_plot_3d(result, output_dir)
    save_distance_plot(result, output_dir)
    save_physics_plot(result, output_dir)
    save_current_plot(result, output_dir)
    if magnet_array is not None:
        save_field_map_3d(result, output_dir, magnet_array)
