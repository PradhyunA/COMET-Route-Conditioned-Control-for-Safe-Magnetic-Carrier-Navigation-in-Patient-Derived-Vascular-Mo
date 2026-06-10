"""
generate_poster_figures_50um.py
===============================
Publication-quality poster figures for the 50 um / flow dataset.

Produces 5 main figures plus a stats text file:

  1. Convergence overview  (3-panel: time histogram, error histogram, time vs distance)
  2. All 100 trajectories overlaid in 3D
  3. Distance curves with median + IQR + tolerance line
  4. Physics regime  (Re, force, velocity, B-field violin/box)
  5. Representative trial  (trajectory + distance + currents, 3-panel)

  + stats_summary.txt  (key numbers for poster text / captions)
"""

import os
import sys
# Add parent directory to path to allow importing dynamag_simulation when run directly
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

import csv
import json
import textwrap

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D  # noqa: F401
from matplotlib.lines import Line2D

DATA_DIR = "outputs/flow_50um_100trials"
OUT_DIR = "outputs/poster_figures_50um"
os.makedirs(OUT_DIR, exist_ok=True)

CSV_PATH = os.path.join(DATA_DIR, "trial_summary.csv")
JSON_PATH = os.path.join(DATA_DIR, "batch_summary.json")

TOLERANCE_MM = 0.5


def load_csv():
    with open(CSV_PATH) as f:
        return list(csv.DictReader(f))


def load_npz(trial_name):
    return dict(np.load(os.path.join(DATA_DIR, f"{trial_name}_data.npz"),
                        allow_pickle=True))


def load_batch_json():
    with open(JSON_PATH) as f:
        return json.load(f)


# ═══════════════════════════════════════════════════════════════════
#  Load data
# ═══════════════════════════════════════════════════════════════════
rows = load_csv()
batch = load_batch_json()
N = len(rows)

steps      = np.array([int(r["steps_taken"]) for r in rows])
sim_time   = np.array([float(r["sim_time_s"]) for r in rows])
err_mm     = np.array([float(r["final_error_mm"]) for r in rows])
peak_force = np.array([float(r["peak_force_N"]) for r in rows])
peak_vel   = np.array([float(r["peak_velocity_m_s"]) for r in rows])
peak_field = np.array([float(r["peak_field_T"]) for r in rows])
mean_re    = np.array([float(r["mean_reynolds"]) for r in rows])
max_re     = np.array([float(r["max_reynolds"]) for r in rows])
reached    = np.array([r["reached_target"] == "True" for r in rows])

start_pts  = np.array([[float(r["start_x"]), float(r["start_y"]), float(r["start_z"])] for r in rows])
target_pts = np.array([[float(r["target_x"]), float(r["target_y"]), float(r["target_z"])] for r in rows])
start_dist = np.linalg.norm(target_pts - start_pts, axis=1)

dt = float(rows[0].get("sim_time_s", 0)) / max(int(rows[0]["steps_taken"]), 1) if int(rows[0]["steps_taken"]) > 0 else 1.0
# fallback
try:
    dt = batch["dt"]
except KeyError:
    pass

HALF = 0.0762  # m

# ═══════════════════════════════════════════════════════════════════
#  Colour palette
# ═══════════════════════════════════════════════════════════════════
C_BLUE    = "#3575b5"
C_GREEN   = "#2ca02c"
C_RED     = "#d62728"
C_PURPLE  = "#9467bd"
C_ORANGE  = "#e67e22"
C_TEAL    = "#17becf"

plt.rcParams.update({
    "font.size": 11,
    "axes.titlesize": 13,
    "axes.labelsize": 11,
    "figure.dpi": 150,
})


# ═══════════════════════════════════════════════════════════════════
#  Figure 1 — Convergence overview (3-panel)
# ═══════════════════════════════════════════════════════════════════
fig, axes = plt.subplots(1, 3, figsize=(17, 5))

ax = axes[0]
ax.hist(sim_time, bins=25, color=C_BLUE, edgecolor="white", lw=0.5)
ax.axvline(np.median(sim_time), color=C_RED, ls="--", lw=1.5,
           label=f"Median = {np.median(sim_time):.0f} s")
ax.set_xlabel("Time to reach target (s)")
ax.set_ylabel("Count")
ax.set_title("Convergence Time", fontweight="bold")
ax.legend(fontsize=9)
ax.grid(True, alpha=0.15)

ax = axes[1]
ax.hist(err_mm, bins=25, color=C_GREEN, edgecolor="white", lw=0.5)
ax.axvline(np.median(err_mm), color=C_RED, ls="--", lw=1.5,
           label=f"Median = {np.median(err_mm):.2f} mm")
ax.axvline(TOLERANCE_MM, color="black", ls=":", lw=1.2, label=f"Tolerance = {TOLERANCE_MM} mm")
ax.set_xlabel("Final positioning error (mm)")
ax.set_ylabel("Count")
ax.set_title("Final Error", fontweight="bold")
ax.legend(fontsize=9)
ax.grid(True, alpha=0.15)

ax = axes[2]
sc = ax.scatter(start_dist * 1e3, sim_time, s=30, c=err_mm, cmap="RdYlGn_r",
                edgecolors="k", lw=0.3, vmin=0, vmax=TOLERANCE_MM)
cb = plt.colorbar(sc, ax=ax, label="Final error (mm)")
ax.set_xlabel("Start → target distance (mm)")
ax.set_ylabel("Convergence time (s)")
ax.set_title("Time vs. Travel Distance", fontweight="bold")
ax.grid(True, alpha=0.15)

plt.tight_layout()
fig.savefig(os.path.join(OUT_DIR, "fig1_convergence_overview.png"), dpi=300, bbox_inches="tight")
plt.close()
print("  [1/5] fig1_convergence_overview.png")


# ═══════════════════════════════════════════════════════════════════
#  Figure 2 — All trajectories overlaid in 3D
# ═══════════════════════════════════════════════════════════════════
fig = plt.figure(figsize=(10, 10))
ax = fig.add_subplot(111, projection="3d")

cmap = plt.cm.viridis
for i, r in enumerate(rows):
    d = load_npz(r["trial_name"])
    traj = d["trajectory"]
    ax.plot(traj[:, 0] * 1e3, traj[:, 1] * 1e3, traj[:, 2] * 1e3,
            lw=0.5, alpha=0.45, color=cmap(i / N))

ax.scatter(start_pts[:, 0] * 1e3, start_pts[:, 1] * 1e3, start_pts[:, 2] * 1e3,
           s=12, marker="o", color="lime", alpha=0.6, zorder=5, label="Start")
ax.scatter(target_pts[:, 0] * 1e3, target_pts[:, 1] * 1e3, target_pts[:, 2] * 1e3,
           s=12, marker="*", color="red", alpha=0.6, zorder=5, label="Target")

ax.set_xlim(-HALF * 1e3, HALF * 1e3)
ax.set_ylim(-HALF * 1e3, HALF * 1e3)
ax.set_zlim(-HALF * 1e3, HALF * 1e3)
ax.set_xlabel("X (mm)")
ax.set_ylabel("Y (mm)")
ax.set_zlabel("Z (mm)")
ax.set_title(f"{N} MPC Trajectories  |  50 µm particle  |  Uniform flow 0.2 mm/s (+x)",
             fontweight="bold", pad=15)
ax.legend(fontsize=9, loc="upper left")

fig.savefig(os.path.join(OUT_DIR, "fig2_all_trajectories_3d.png"), dpi=300, bbox_inches="tight")
plt.close()
print("  [2/5] fig2_all_trajectories_3d.png")


# ═══════════════════════════════════════════════════════════════════
#  Figure 3 — Distance curves (all + median + IQR)
# ═══════════════════════════════════════════════════════════════════
max_steps_plot = int(np.max(steps)) + 20

all_dist = []
for r in rows:
    d = load_npz(r["trial_name"])
    dl = d["distance_log"] * 1e3
    padded = np.full(max_steps_plot, np.nan)
    padded[:len(dl)] = dl
    all_dist.append(padded)
all_dist = np.array(all_dist)

t_axis = np.arange(max_steps_plot) * dt

fig, ax = plt.subplots(figsize=(13, 5))
for i in range(N):
    ax.plot(t_axis, all_dist[i], lw=0.3, alpha=0.2, color=C_BLUE)

med = np.nanmedian(all_dist, axis=0)
q25 = np.nanpercentile(all_dist, 25, axis=0)
q75 = np.nanpercentile(all_dist, 75, axis=0)
ax.plot(t_axis, med, lw=2.5, color="navy", label="Median", zorder=4)
ax.fill_between(t_axis, q25, q75, alpha=0.2, color=C_BLUE, label="IQR (25th–75th)")
ax.axhline(TOLERANCE_MM, color=C_RED, ls="--", lw=1.3, label=f"Tolerance ({TOLERANCE_MM} mm)")

ax.set_xlabel("Simulation time (s)")
ax.set_ylabel("Distance to target (mm)")
ax.set_title("Distance-to-Target Convergence  |  50 µm particle under flow", fontweight="bold")
ax.legend(fontsize=9)
ax.set_xlim(0, np.nanmax(sim_time) * 1.05)
ax.set_ylim(bottom=0)
ax.grid(True, alpha=0.15)

fig.savefig(os.path.join(OUT_DIR, "fig3_distance_curves.png"), dpi=300, bbox_inches="tight")
plt.close()
print("  [3/5] fig3_distance_curves.png")


# ═══════════════════════════════════════════════════════════════════
#  Figure 4 — Physics regime (violin + box)
# ═══════════════════════════════════════════════════════════════════
fig, axes = plt.subplots(1, 4, figsize=(17, 5))

specs = [
    (max_re,              "Max Reynolds number",    C_ORANGE),
    (peak_force * 1e9,    "Peak force (nN)",        C_RED),
    (peak_vel * 1e6,      "Peak velocity (µm/s)",   C_GREEN),
    (peak_field * 1e3,    "Peak |B| (mT)",          C_PURPLE),
]

for ax, (data, label, color) in zip(axes, specs):
    parts = ax.violinplot(data, positions=[0], showmeans=True, showextrema=False)
    for pc in parts["bodies"]:
        pc.set_facecolor(color)
        pc.set_alpha(0.3)
    parts["cmeans"].set_color(color)
    bp = ax.boxplot(data, positions=[0], widths=0.3, patch_artist=True,
                    boxprops=dict(facecolor=color, alpha=0.35),
                    medianprops=dict(color="black", lw=1.5),
                    flierprops=dict(marker="o", markersize=3, alpha=0.4))
    ax.set_ylabel(label)
    ax.set_xticks([])
    ax.grid(True, alpha=0.15, axis="y")

axes[0].axhline(0.1, color="black", ls=":", lw=1.2, label="Re = 0.1 (Stokes limit)")
axes[0].legend(fontsize=8)
axes[0].set_title("Reynolds", fontweight="bold")
axes[1].set_title("Force", fontweight="bold")
axes[2].set_title("Velocity", fontweight="bold")
axes[3].set_title("B-field", fontweight="bold")

fig.suptitle("Physics Regime Summary  |  50 µm particle, uniform flow, 100 trials",
             fontweight="bold", fontsize=13, y=1.02)
plt.tight_layout()
fig.savefig(os.path.join(OUT_DIR, "fig4_physics_regime.png"), dpi=300, bbox_inches="tight")
plt.close()
print("  [4/5] fig4_physics_regime.png")


# ═══════════════════════════════════════════════════════════════════
#  Figure 5 — Representative trial (median convergence)
# ═══════════════════════════════════════════════════════════════════
median_steps = int(np.median(steps))
rep_idx = int(np.argmin(np.abs(steps - median_steps)))
rep_name = rows[rep_idx]["trial_name"]
rep = load_npz(rep_name)

fig = plt.figure(figsize=(17, 5.5))
gs = fig.add_gridspec(1, 3, width_ratios=[1.3, 1, 1], wspace=0.30)

# 5a: 3D trajectory
ax3d = fig.add_subplot(gs[0], projection="3d")
traj = rep["trajectory"]
ax3d.plot(traj[:, 0] * 1e3, traj[:, 1] * 1e3, traj[:, 2] * 1e3,
          lw=2, color=C_BLUE)
ax3d.scatter(*rep["start"] * 1e3, s=100, marker="o", color="lime",
             edgecolors="k", lw=0.5, label="Start", zorder=5)
ax3d.scatter(*rep["target"] * 1e3, s=120, marker="*", color=C_RED,
             edgecolors="k", lw=0.5, label="Target", zorder=5)
ax3d.set_xlabel("X (mm)", fontsize=9)
ax3d.set_ylabel("Y (mm)", fontsize=9)
ax3d.set_zlabel("Z (mm)", fontsize=9)
ax3d.set_title(f"{rep_name} — Trajectory", fontweight="bold")
ax3d.legend(fontsize=8)

# 5b: distance curve
ax_d = fig.add_subplot(gs[1])
dl = rep["distance_log"] * 1e3
t = np.arange(len(dl)) * dt
ax_d.plot(t, dl, lw=2, color=C_BLUE)
ax_d.axhline(TOLERANCE_MM, color=C_RED, ls="--", lw=1.2, label="Tolerance")
ax_d.set_xlabel("Time (s)")
ax_d.set_ylabel("Distance to target (mm)")
ax_d.set_title("Convergence", fontweight="bold")
ax_d.legend(fontsize=8)
ax_d.grid(True, alpha=0.15)

# 5c: coil currents
ax_c = fig.add_subplot(gs[2])
cl = rep["current_log"]
colors_8 = plt.cm.tab10(np.linspace(0, 1, 8))
for j in range(cl.shape[1]):
    ax_c.plot(np.arange(cl.shape[0]) * dt, cl[:, j], lw=1, color=colors_8[j], label=f"I{j+1}")
ax_c.set_xlabel("Time (s)")
ax_c.set_ylabel("Current (A)")
ax_c.set_title("Coil Currents", fontweight="bold")
ax_c.legend(fontsize=7, ncol=2, loc="lower right")
ax_c.grid(True, alpha=0.15)

fig.suptitle(f"Representative Trial ({rep_name}, median convergence time)",
             fontweight="bold", fontsize=13, y=1.02)
fig.savefig(os.path.join(OUT_DIR, "fig5_representative_trial.png"), dpi=300, bbox_inches="tight")
plt.close()
print("  [5/5] fig5_representative_trial.png")


# ═══════════════════════════════════════════════════════════════════
#  Stats summary text file (for poster text / captions)
# ═══════════════════════════════════════════════════════════════════
stats_text = textwrap.dedent(f"""\
    DynaMag 50 µm Particle — 100 Trial Summary
    ============================================
    Particle diameter      : 50 µm
    Magnetic susceptibility: χ_eff = 3.0
    Saturation magnetisation: M_sat = 4.5×10⁵ A/m
    Fluid                  : water at 25°C (η = 8.9×10⁻⁴ Pa·s)
    Background flow        : uniform 0.2 mm/s along +x
    Controller             : 1-step MPC (SLSQP, 5 seeds)
    Time step              : dt = {dt} s
    Tolerance              : {TOLERANCE_MM} mm

    Results
    -------
    Trials                 : {N}
    Success rate           : {np.sum(reached)}/{N} ({np.sum(reached)/N*100:.0f}%)
    Median convergence time: {np.median(sim_time):.0f} s  ({int(np.median(steps))} steps)
    Mean convergence time  : {np.mean(sim_time):.0f} s
    Median final error     : {np.median(err_mm):.2f} mm
    Max final error        : {np.max(err_mm):.2f} mm

    Physics Regime
    --------------
    Max Reynolds (median)  : {np.median(max_re):.4f}
    Max Reynolds (worst)   : {np.max(max_re):.4f}
    Low-Re valid (all)     : {bool(np.all(max_re < 0.1))}
    Peak |B| median        : {np.median(peak_field)*1e3:.2f} mT
    Peak |F| median        : {np.median(peak_force)*1e9:.2f} nN
    Peak |v| median        : {np.median(peak_vel)*1e6:.1f} µm/s

    Batch Info
    ----------
    Wall-clock time        : {batch.get('batch_wall_time_s', 0)/3600:.1f} h
    Avg control step time  : {np.mean([float(r.get('avg_control_step_time_s','0')) for r in rows]):.2f} s
""")

stats_path = os.path.join(OUT_DIR, "stats_summary.txt")
with open(stats_path, "w") as f:
    f.write(stats_text)

print(f"\n{stats_text}")
print(f"All outputs saved to: {OUT_DIR}/")
