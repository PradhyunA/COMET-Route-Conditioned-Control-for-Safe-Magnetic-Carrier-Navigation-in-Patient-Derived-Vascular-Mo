"""
Consolidated plotting utilities for all simulation variants.
Includes trajectory, vessel tree, planned route, currents, physics, and field plots.
"""

import os
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D  # noqa: F401


def _set_axes_equal(ax):
    """Make 3D plot axes equal scale."""
    x_limits = ax.get_xlim3d()
    y_limits = ax.get_ylim3d()
    z_limits = ax.get_zlim3d()

    x_range = abs(x_limits[1] - x_limits[0])
    y_range = abs(y_limits[1] - y_limits[0])
    z_range = abs(z_limits[1] - z_limits[0])

    max_range = max(x_range, y_range, z_range)

    x_mid = np.mean(x_limits)
    y_mid = np.mean(y_limits)
    z_mid = np.mean(z_limits)

    ax.set_xlim3d([x_mid - max_range / 2, x_mid + max_range / 2])
    ax.set_ylim3d([y_mid - max_range / 2, y_mid + max_range / 2])
    ax.set_zlim3d([z_mid - max_range / 2, z_mid + max_range / 2])


def save_trajectory_plot_3d(result, output_dir):
    traj = result.trajectory
    target = result.target
    magnets = result.magnets
    half = result.workspace_half

    fig = plt.figure(figsize=(10, 10))
    ax = fig.add_subplot(111, projection="3d")

    ax.plot(traj[:, 0], traj[:, 1], traj[:, 2], lw=2, label="Particle trajectory")
    ax.scatter(*traj[0], s=120, marker="o", label="Start")
    ax.scatter(*target, s=150, marker="*", label="Target")

    if magnets is not None:
        ax.scatter(
            magnets[:, 0],
            magnets[:, 1],
            magnets[:, 2],
            s=80,
            marker="^",
            label="Magnets",
        )

    ax.set_xlim(-half, half)
    ax.set_ylim(-half, half)
    ax.set_zlim(-half, half)
    ax.set_xlabel("X (m)")
    ax.set_ylabel("Y (m)")
    ax.set_zlabel("Z (m)")
    ax.set_title(f"{result.trial_name} – 3D Particle Trajectory")
    ax.legend(fontsize=8)

    fn = os.path.join(output_dir, f"{result.trial_name}_trajectory_3d.png")
    plt.savefig(fn, dpi=300, bbox_inches="tight")
    plt.close()


def save_vessel_tree_with_path(result, output_dir, network, max_segments=None):
    """
    Plot vessel centerline tree, planned waypoint route, and actual trajectory.
    This is the most important visualization for real vessel navigation.
    """
    if network is None:
        return

    traj = result.trajectory

    fig = plt.figure(figsize=(12, 12))
    ax = fig.add_subplot(111, projection="3d")

    segments = network.segments
    if max_segments is not None and len(segments) > max_segments:
        step = max(1, len(segments) // max_segments)
        segments_to_plot = segments[::step]
    else:
        segments_to_plot = segments

    # Plot full vessel tree centerlines
    for seg in segments_to_plot:
        xs = [seg.start[0], seg.end[0]]
        ys = [seg.start[1], seg.end[1]]
        zs = [seg.start[2], seg.end[2]]

        lw = max(0.3, min(2.5, seg.radius * 1000))
        ax.plot(xs, ys, zs, linewidth=lw, alpha=0.25)

    # Planned waypoint route
    try:
        waypoints = network.plan_waypoints(result.start, result.target)
        waypoints = np.array(waypoints)

        if len(waypoints) > 0:
            route = np.vstack([result.start, waypoints])
            ax.plot(
                route[:, 0],
                route[:, 1],
                route[:, 2],
                linestyle="--",
                linewidth=3,
                label="Planned vessel route",
            )
            ax.scatter(
                route[:, 0],
                route[:, 1],
                route[:, 2],
                s=35,
                marker="x",
                label="Waypoints",
            )
    except Exception as e:
        print(f"[plotting] Could not plot planned waypoints: {e}")

    # Actual particle trajectory
    ax.plot(
        traj[:, 0],
        traj[:, 1],
        traj[:, 2],
        linewidth=3,
        label="Actual particle trajectory",
    )

    ax.scatter(*result.start, s=120, marker="o", label="Start")
    ax.scatter(*result.target, s=160, marker="*", label="Target")

    if result.magnets is not None:
        ax.scatter(
            result.magnets[:, 0],
            result.magnets[:, 1],
            result.magnets[:, 2],
            s=50,
            marker="^",
            label="Magnets",
        )

    ax.set_xlabel("X (m)")
    ax.set_ylabel("Y (m)")
    ax.set_zlabel("Z (m)")
    ax.set_title(f"{result.trial_name} – Vessel Tree + Planned Path + Actual Trajectory")
    ax.legend(fontsize=8)

    _set_axes_equal(ax)

    fn = os.path.join(output_dir, f"{result.trial_name}_vessel_tree_path_trajectory.png")
    plt.savefig(fn, dpi=300, bbox_inches="tight")
    plt.close()


def save_vessel_tree_only(output_dir, network, filename="real_vessel_tree.png", max_segments=None):
    """Plot just the full vessel centerline tree."""
    if network is None:
        return

    fig = plt.figure(figsize=(12, 12))
    ax = fig.add_subplot(111, projection="3d")

    segments = network.segments
    if max_segments is not None and len(segments) > max_segments:
        step = max(1, len(segments) // max_segments)
        segments_to_plot = segments[::step]
    else:
        segments_to_plot = segments

    for seg in segments_to_plot:
        xs = [seg.start[0], seg.end[0]]
        ys = [seg.start[1], seg.end[1]]
        zs = [seg.start[2], seg.end[2]]

        lw = max(0.3, min(2.5, seg.radius * 1000))
        ax.plot(xs, ys, zs, linewidth=lw, alpha=0.35)

    ax.set_xlabel("X (m)")
    ax.set_ylabel("Y (m)")
    ax.set_zlabel("Z (m)")
    ax.set_title(f"Real Vessel Centerline Tree ({network.n_segments} segments)")

    _set_axes_equal(ax)

    fn = os.path.join(output_dir, filename)
    plt.savefig(fn, dpi=300, bbox_inches="tight")
    plt.close()


def save_distance_plot(result, output_dir):
    if len(result.distance_log) == 0:
        return

    steps = np.arange(len(result.distance_log))
    t_sec = steps * result.dt

    plt.figure(figsize=(10, 5))
    plt.plot(t_sec, result.distance_log * 1000, lw=2)
    plt.xlabel("Time (s)")
    plt.ylabel("Distance to Final Target (mm)")
    plt.grid(True, alpha=0.3)
    plt.title(f"{result.trial_name} – Distance to Target")

    fn = os.path.join(output_dir, f"{result.trial_name}_distance.png")
    plt.savefig(fn, dpi=300, bbox_inches="tight")
    plt.close()


def save_physics_plot(result, output_dir):
    if len(result.force_log) == 0:
        return

    steps = np.arange(len(result.force_log))
    t_sec = steps * result.dt

    fig, axes = plt.subplots(4, 1, figsize=(10, 12), sharex=True)

    axes[0].plot(t_sec, result.force_log, lw=1.5)
    axes[0].set_ylabel("|F| (N)")
    axes[0].set_title(f"{result.trial_name} – Physics Over Time")
    axes[0].grid(True, alpha=0.3)
    axes[0].ticklabel_format(axis="y", style="scientific", scilimits=(0, 0))

    axes[1].plot(t_sec, np.array(result.velocity_log) * 1e6, lw=1.5)
    axes[1].set_ylabel("|v| (μm/s)")
    axes[1].grid(True, alpha=0.3)

    axes[2].plot(t_sec, np.array(result.field_log) * 1e3, lw=1.5)
    axes[2].set_ylabel("|B| (mT)")
    axes[2].grid(True, alpha=0.3)

    axes[3].plot(t_sec, result.reynolds_log, lw=1.5)
    axes[3].set_ylabel("Re")
    axes[3].set_xlabel("Time (s)")
    axes[3].grid(True, alpha=0.3)
    axes[3].ticklabel_format(axis="y", style="scientific", scilimits=(0, 0))

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
    plt.title(f"{result.trial_name} – Coil Currents")
    plt.legend(ncol=2, fontsize=8)

    fn = os.path.join(output_dir, f"{result.trial_name}_currents.png")
    plt.savefig(fn, dpi=300, bbox_inches="tight")
    plt.close()


def save_current_heatmap(result, output_dir):
    """Useful for seeing which coils dominate over time."""
    if len(result.current_log) == 0:
        return

    plt.figure(figsize=(12, 5))
    plt.imshow(
        result.current_log.T,
        aspect="auto",
        interpolation="nearest",
        origin="lower",
    )
    plt.colorbar(label="Current (A)")
    plt.xlabel("Simulation Step")
    plt.ylabel("Coil Index")
    plt.title(f"{result.trial_name} – Coil Current Heatmap")

    fn = os.path.join(output_dir, f"{result.trial_name}_current_heatmap.png")
    plt.savefig(fn, dpi=300, bbox_inches="tight")
    plt.close()


def save_error_components_plot(result, output_dir):
    """Shows x/y/z error between particle and target over time."""
    traj = result.trajectory
    if len(traj) < 2:
        return

    error = traj - result.target
    t_sec = np.arange(len(traj)) * result.dt

    plt.figure(figsize=(10, 6))
    plt.plot(t_sec, error[:, 0] * 1000, label="x error")
    plt.plot(t_sec, error[:, 1] * 1000, label="y error")
    plt.plot(t_sec, error[:, 2] * 1000, label="z error")

    plt.axhline(0, linewidth=1)
    plt.grid(True, alpha=0.3)
    plt.xlabel("Time (s)")
    plt.ylabel("Position Error (mm)")
    plt.title(f"{result.trial_name} – XYZ Error Components")
    plt.legend()

    fn = os.path.join(output_dir, f"{result.trial_name}_xyz_error.png")
    plt.savefig(fn, dpi=300, bbox_inches="tight")
    plt.close()


def save_speed_plot(result, output_dir):
    if len(result.velocity_log) == 0:
        return

    t_sec = np.arange(len(result.velocity_log)) * result.dt

    plt.figure(figsize=(10, 5))
    plt.plot(t_sec, np.array(result.velocity_log) * 1e6, lw=2)
    plt.xlabel("Time (s)")
    plt.ylabel("Speed (μm/s)")
    plt.title(f"{result.trial_name} – Particle Speed")
    plt.grid(True, alpha=0.3)

    fn = os.path.join(output_dir, f"{result.trial_name}_speed.png")
    plt.savefig(fn, dpi=300, bbox_inches="tight")
    plt.close()


def save_wall_clearance_plot(result, output_dir, network):
    """
    Approximate wall clearance from centerline-radius model.
    clearance = segment radius - radial distance.
    Positive = inside vessel.
    Near zero = near wall.
    """
    if network is None:
        return

    clearances = []

    for pos in result.trajectory:
        seg, _ = network.find_segment(pos)
        if seg is None:
            clearances.append(np.nan)
            continue

        _, _, radial_dist, _ = seg.closest_point_and_radial(pos)
        clearances.append(seg.radius - radial_dist)

    clearances = np.array(clearances)
    t_sec = np.arange(len(clearances)) * result.dt

    plt.figure(figsize=(10, 5))
    plt.plot(t_sec, clearances * 1000, lw=2)
    plt.axhline(0, linestyle="--", linewidth=1, label="Wall boundary")
    plt.xlabel("Time (s)")
    plt.ylabel("Approx. Wall Clearance (mm)")
    plt.title(f"{result.trial_name} – Distance From Vessel Wall")
    plt.grid(True, alpha=0.3)
    plt.legend()

    fn = os.path.join(output_dir, f"{result.trial_name}_wall_clearance.png")
    plt.savefig(fn, dpi=300, bbox_inches="tight")
    plt.close()


def save_field_map_3d(result, output_dir, magnet_array, grid_points=15):
    """Midplane magnetic field map using first-step currents."""
    if len(result.current_log) == 0:
        return

    half = result.workspace_half
    magnet_array.set_currents(result.current_log[0])
    magnet_center = result.params.get("magnet_center") if result.params else None
    z_slice = float(magnet_center[2]) if magnet_center is not None else 0.0

    xs = np.linspace(-half, half, grid_points)
    ys = np.linspace(-half, half, grid_points)
    X, Y = np.meshgrid(xs, ys)

    U = np.zeros_like(X)
    V = np.zeros_like(Y)
    M = np.zeros_like(X)

    for i in range(grid_points):
        for j in range(grid_points):
            B = magnet_array.field_at(np.array([X[i, j], Y[i, j], z_slice]))
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
        plt.scatter(
            result.magnets[:, 0],
            result.magnets[:, 1],
            s=80,
            marker="^",
            label="Magnets projected",
        )

    plt.scatter(result.start[0], result.start[1], s=100, label="Start projected")
    plt.scatter(result.target[0], result.target[1], s=100, label="Target projected")

    plt.xlim(-half, half)
    plt.ylim(-half, half)
    plt.gca().set_aspect("equal")
    plt.grid(True, alpha=0.3)
    plt.legend()
    plt.xlabel("X (m)")
    plt.ylabel("Y (m)")
    plt.title(f"{result.trial_name} – Field Map z={z_slice:.3f} m")

    fn = os.path.join(output_dir, f"{result.trial_name}_fieldmap_z0.png")
    plt.savefig(fn, dpi=300, bbox_inches="tight")
    plt.close()


def save_all_plots(result, output_dir, magnet_array=None, network=None):
    """Generate every useful plot for a trial."""
    os.makedirs(output_dir, exist_ok=True)

    save_trajectory_plot_3d(result, output_dir)
    save_distance_plot(result, output_dir)
    save_physics_plot(result, output_dir)
    save_current_plot(result, output_dir)
    save_current_heatmap(result, output_dir)
    save_error_components_plot(result, output_dir)
    save_speed_plot(result, output_dir)

    if network is not None:
        save_vessel_tree_with_path(result, output_dir, network)
        save_wall_clearance_plot(result, output_dir, network)

    if magnet_array is not None:
        save_field_map_3d(result, output_dir, magnet_array)
