"""
generate_poster_figures.py
==========================
Create 5 publication-quality figures summarizing the 100-trial flow dataset.

  1. Success & convergence: histogram of steps-to-target + final error boxplot
  2. Overlaid 3D trajectories (all 100 trials in one cube)
  3. Distance-to-target curves (all 100, with median + IQR band)
  4. Peak physics distributions: force, velocity, B-field violin/box plots
  5. Representative single trial: trajectory + distance + currents (3-panel)
"""

import os
import sys
# Add parent directory to path to allow importing dynamag_simulation when run directly
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

import csv

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D  # noqa: F401

DATA_DIR = "outputs/flow_100_trials"
OUT_DIR = "outputs/poster_figures"
os.makedirs(OUT_DIR, exist_ok=True)

CSV_PATH = os.path.join(DATA_DIR, "trial_summary.csv")


def load_csv():
    rows = []
    with open(CSV_PATH) as f:
        reader = csv.DictReader(f)
        for r in reader:
            rows.append(r)
    return rows


def load_npz(trial_name):
    path = os.path.join(DATA_DIR, f"{trial_name}_data.npz")
    return dict(np.load(path, allow_pickle=True))


rows = load_csv()

steps = np.array([int(r["steps_taken"]) for r in rows])
sim_time = np.array([float(r["sim_time_s"]) for r in rows])
final_err_mm = np.array([float(r["final_error_mm"]) for r in rows])
peak_force = np.array([float(r["peak_force_N"]) for r in rows])
peak_vel = np.array([float(r["peak_velocity_m_s"]) for r in rows])
peak_field = np.array([float(r["peak_field_T"]) for r in rows])
mean_re = np.array([float(r["mean_reynolds"]) for r in rows])
max_re = np.array([float(r["max_reynolds"]) for r in rows])
start_dists = np.array([
    np.linalg.norm([
        float(r["target_x"]) - float(r["start_x"]),
        float(r["target_y"]) - float(r["start_y"]),
        float(r["target_z"]) - float(r["start_z"]),
    ]) for r in rows
])

# ── Figure 1: Convergence overview ──────────────────────────────────────
fig, axes = plt.subplots(1, 3, figsize=(16, 4.5))

ax = axes[0]
ax.hist(sim_time, bins=20, color="steelblue", edgecolor="white", linewidth=0.6)
ax.set_xlabel("Time to reach target (s)", fontsize=11)
ax.set_ylabel("Count", fontsize=11)
ax.set_title("Convergence Time Distribution", fontsize=12, fontweight="bold")
ax.axvline(np.median(sim_time), color="tomato", ls="--", lw=1.5,
           label=f"Median = {np.median(sim_time):.1f} s")
ax.legend(fontsize=9)
ax.grid(True, alpha=0.2)

ax = axes[1]
ax.hist(final_err_mm, bins=20, color="seagreen", edgecolor="white", linewidth=0.6)
ax.set_xlabel("Final positioning error (mm)", fontsize=11)
ax.set_ylabel("Count", fontsize=11)
ax.set_title("Final Error Distribution", fontsize=12, fontweight="bold")
ax.axvline(np.median(final_err_mm), color="tomato", ls="--", lw=1.5,
           label=f"Median = {np.median(final_err_mm):.2f} mm")
ax.legend(fontsize=9)
ax.grid(True, alpha=0.2)

ax = axes[2]
ax.scatter(start_dists * 1000, sim_time, s=28, alpha=0.7, c="steelblue", edgecolors="k", linewidth=0.3)
ax.set_xlabel("Initial start–target distance (mm)", fontsize=11)
ax.set_ylabel("Time to reach target (s)", fontsize=11)
ax.set_title("Convergence vs. Travel Distance", fontsize=12, fontweight="bold")
ax.grid(True, alpha=0.2)

plt.tight_layout()
fig.savefig(os.path.join(OUT_DIR, "fig1_convergence_overview.png"), dpi=300, bbox_inches="tight")
plt.close()
print("  [1/5] fig1_convergence_overview.png")


# ── Figure 2: All 100 trajectories overlaid in 3D ──────────────────────
fig = plt.figure(figsize=(9, 9))
ax = fig.add_subplot(111, projection="3d")

HALF = 0.0762
cmap = plt.cm.viridis

for i, r in enumerate(rows):
    d = load_npz(r["trial_name"])
    traj = d["trajectory"]
    color = cmap(i / len(rows))
    ax.plot(traj[:, 0] * 1000, traj[:, 1] * 1000, traj[:, 2] * 1000,
            lw=0.6, alpha=0.55, color=color)

for i, r in enumerate(rows):
    sx, sy, sz = float(r["start_x"]), float(r["start_y"]), float(r["start_z"])
    tx, ty, tz = float(r["target_x"]), float(r["target_y"]), float(r["target_z"])
    ax.scatter(sx * 1000, sy * 1000, sz * 1000, s=10, marker="o",
               color="lime", alpha=0.5, zorder=5)
    ax.scatter(tx * 1000, ty * 1000, tz * 1000, s=10, marker="*",
               color="red", alpha=0.5, zorder=5)

ax.set_xlim(-HALF * 1000, HALF * 1000)
ax.set_ylim(-HALF * 1000, HALF * 1000)
ax.set_zlim(-HALF * 1000, HALF * 1000)
ax.set_xlabel("X (mm)", fontsize=10)
ax.set_ylabel("Y (mm)", fontsize=10)
ax.set_zlabel("Z (mm)", fontsize=10)
ax.set_title("100 MPC Trajectories Under Uniform Flow (2 mm/s +x)",
             fontsize=12, fontweight="bold", pad=15)

start_proxy = plt.Line2D([0], [0], marker="o", color="w", markerfacecolor="lime",
                         markersize=7, label="Start")
target_proxy = plt.Line2D([0], [0], marker="*", color="w", markerfacecolor="red",
                          markersize=9, label="Target")
ax.legend(handles=[start_proxy, target_proxy], fontsize=9, loc="upper left")

fig.savefig(os.path.join(OUT_DIR, "fig2_all_trajectories_3d.png"), dpi=300, bbox_inches="tight")
plt.close()
print("  [2/5] fig2_all_trajectories_3d.png")


# ── Figure 3: Distance curves with median + IQR band ───────────────────
MAX_STEPS_PLOT = max(steps) + 10
dt = 0.02

all_dist = []
for r in rows:
    d = load_npz(r["trial_name"])
    dl = d["distance_log"] * 1000  # mm
    padded = np.full(MAX_STEPS_PLOT, np.nan)
    padded[:len(dl)] = dl
    all_dist.append(padded)

all_dist = np.array(all_dist)
t_axis = np.arange(MAX_STEPS_PLOT) * dt

fig, ax = plt.subplots(figsize=(12, 5))
for i in range(len(rows)):
    ax.plot(t_axis, all_dist[i], lw=0.35, alpha=0.25, color="steelblue")

median_d = np.nanmedian(all_dist, axis=0)
q25 = np.nanpercentile(all_dist, 25, axis=0)
q75 = np.nanpercentile(all_dist, 75, axis=0)

ax.plot(t_axis, median_d, lw=2.2, color="navy", label="Median")
ax.fill_between(t_axis, q25, q75, alpha=0.25, color="steelblue", label="IQR (25th–75th)")

ax.axhline(0.5, color="red", ls="--", lw=1.2, label="Tolerance (0.5 mm)")
ax.set_xlabel("Simulation time (s)", fontsize=11)
ax.set_ylabel("Distance to target (mm)", fontsize=11)
ax.set_title("Distance-to-Target Across 100 Trials", fontsize=12, fontweight="bold")
ax.legend(fontsize=9)
ax.set_xlim(0, np.max(sim_time) * 1.05)
ax.grid(True, alpha=0.2)

fig.savefig(os.path.join(OUT_DIR, "fig3_distance_curves.png"), dpi=300, bbox_inches="tight")
plt.close()
print("  [3/5] fig3_distance_curves.png")


# ── Figure 4: Physics distributions (violin + box) ─────────────────────
fig, axes = plt.subplots(1, 4, figsize=(16, 4.5))

data_labels = [
    (peak_force * 1e6, "Peak force (μN)", "tab:red"),
    (peak_vel * 1e3, "Peak velocity (mm/s)", "tab:green"),
    (peak_field * 1e3, "Peak |B| (mT)", "tab:purple"),
    (mean_re, "Mean Reynolds number", "tab:orange"),
]

for ax, (data, label, color) in zip(axes, data_labels):
    parts = ax.violinplot(data, positions=[0], showmeans=True, showextrema=False)
    for pc in parts["bodies"]:
        pc.set_facecolor(color)
        pc.set_alpha(0.35)
    parts["cmeans"].set_color(color)
    bp = ax.boxplot(data, positions=[0], widths=0.25, patch_artist=True,
                    boxprops=dict(facecolor=color, alpha=0.4),
                    medianprops=dict(color="black", lw=1.5),
                    flierprops=dict(marker="o", markersize=3, alpha=0.5))
    ax.set_ylabel(label, fontsize=10)
    ax.set_xticks([])
    ax.grid(True, alpha=0.2, axis="y")

axes[0].set_title("Force", fontsize=11, fontweight="bold")
axes[1].set_title("Velocity", fontsize=11, fontweight="bold")
axes[2].set_title("B-field", fontsize=11, fontweight="bold")
axes[3].set_title("Reynolds", fontsize=11, fontweight="bold")

fig.suptitle("Physics Regime Summary (100 Trials, Uniform Flow)", fontsize=13, fontweight="bold", y=1.02)
plt.tight_layout()
fig.savefig(os.path.join(OUT_DIR, "fig4_physics_distributions.png"), dpi=300, bbox_inches="tight")
plt.close()
print("  [4/5] fig4_physics_distributions.png")


# ── Figure 5: Representative single trial (median convergence time) ────
median_steps = int(np.median(steps))
closest_idx = int(np.argmin(np.abs(steps - median_steps)))
rep_name = rows[closest_idx]["trial_name"]
rep = load_npz(rep_name)

fig = plt.figure(figsize=(16, 5))
gs = fig.add_gridspec(1, 3, width_ratios=[1.2, 1, 1], wspace=0.32)

# 5a: 3D trajectory
ax3d = fig.add_subplot(gs[0], projection="3d")
traj = rep["trajectory"]
ax3d.plot(traj[:, 0] * 1000, traj[:, 1] * 1000, traj[:, 2] * 1000,
          lw=2, color="steelblue")
ax3d.scatter(*rep["start"] * 1000, s=90, marker="o", color="lime",
             edgecolors="k", linewidth=0.5, label="Start", zorder=5)
ax3d.scatter(*rep["target"] * 1000, s=110, marker="*", color="red",
             edgecolors="k", linewidth=0.5, label="Target", zorder=5)
ax3d.set_xlabel("X (mm)", fontsize=9)
ax3d.set_ylabel("Y (mm)", fontsize=9)
ax3d.set_zlabel("Z (mm)", fontsize=9)
ax3d.set_title(f"{rep_name} — Trajectory", fontsize=11, fontweight="bold")
ax3d.legend(fontsize=8)

# 5b: distance vs time
ax_d = fig.add_subplot(gs[1])
dl = rep["distance_log"] * 1000
t = np.arange(len(dl)) * dt
ax_d.plot(t, dl, lw=2, color="steelblue")
ax_d.axhline(3.0, color="red", ls="--", lw=1, label="Tolerance")
ax_d.set_xlabel("Time (s)", fontsize=10)
ax_d.set_ylabel("Distance (mm)", fontsize=10)
ax_d.set_title("Distance to Target", fontsize=11, fontweight="bold")
ax_d.legend(fontsize=8)
ax_d.grid(True, alpha=0.2)

# 5c: coil currents
ax_c = fig.add_subplot(gs[2])
cl = rep["current_log"]
for j in range(cl.shape[1]):
    ax_c.plot(np.arange(cl.shape[0]) * dt, cl[:, j], lw=1.2, label=f"I{j+1}")
ax_c.set_xlabel("Time (s)", fontsize=10)
ax_c.set_ylabel("Current (A)", fontsize=10)
ax_c.set_title("Coil Currents", fontsize=11, fontweight="bold")
ax_c.legend(fontsize=7, ncol=2, loc="lower left")
ax_c.grid(True, alpha=0.2)

fig.suptitle(f"Representative Trial ({rep_name}, median convergence)",
             fontsize=13, fontweight="bold", y=1.02)
fig.savefig(os.path.join(OUT_DIR, "fig5_representative_trial.png"), dpi=300, bbox_inches="tight")
plt.close()
print("  [5/5] fig5_representative_trial.png")


print(f"\nAll 5 poster figures saved to: {OUT_DIR}/")
print(f"  Trials: {len(rows)}, Reached: {sum(1 for r in rows if r['reached_target'] == 'True')}/100")
print(f"  Median convergence: {np.median(sim_time):.1f} s ({int(np.median(steps))} steps)")
print(f"  Median final error: {np.median(final_err_mm):.2f} mm")
