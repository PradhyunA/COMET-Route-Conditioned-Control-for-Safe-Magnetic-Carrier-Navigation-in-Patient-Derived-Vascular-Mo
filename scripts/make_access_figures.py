"""Journal figures for the COMET IEEE Access manuscript (static, print)."""
import csv, collections, pickle, sys
import numpy as np, matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.collections import LineCollection

REPO = "/Users/amrit/Projects/DynaMag-Magnetic-Nanoparticle-Steering-Simulation"
OUT = "/Users/amrit/Projects/COMETManuscript/Figures"
PS = f"{REPO}/outputs/physics_sensitivity"
CTRL = ["comet_v4", "nmpc_tuned", "nmpc_default", "mpc_tuned", "acbc_tuned"]
LABEL = {"comet_v4": "COMET", "nmpc_tuned": "NMPC (tuned)", "nmpc_default": "NMPC (default)",
         "mpc_tuned": "Heuristic MPC", "acbc_tuned": "ACBC"}
COLOR = dict(zip(CTRL, ["#2a78d6", "#eb6834", "#1baf7a", "#eda100", "#e87ba4"]))
INK, INK2, GRID = "#1f1f1e", "#5f5e58", "#e4e3de"
plt.rcParams.update({"font.family": "serif", "font.serif": ["Times New Roman", "Times", "DejaVu Serif"],
    "font.size": 8, "axes.edgecolor": INK2, "axes.labelcolor": INK, "xtick.color": INK2, "ytick.color": INK2,
    "axes.spines.top": False, "axes.spines.right": False, "axes.linewidth": 0.6, "legend.frameon": False,
    "pdf.fonttype": 42, "savefig.dpi": 600})
tb = lambda v: str(v).strip().lower() in ("true", "1", "1.0")
def safe(r): return tb(r["reached_target"]) and tb(r["clearance_target_met"]) and not tb(r["wall_exit"])

def load(csvpath):
    R = collections.defaultdict(dict)
    for r in csv.DictReader(open(csvpath)):
        R[r["controller_key"]][(r["anatomy"], r["pair_id"], float(r["flow_scale"]))] = r
    return R
MAIN = load(f"{PS}/five_anatomy_final_comet_v4_20261103/all_controllers_by_vessel_network.csv")

def save(fig, name):
    fig.savefig(f"{OUT}/{name}.pdf", bbox_inches="tight"); fig.savefig(f"{OUT}/{name}.png", bbox_inches="tight", dpi=300)
    plt.close(fig)

# 1. Safe successes by flow scale, 400 mT/m.
fig, ax = plt.subplots(figsize=(3.5, 2.2))
w = 0.16
for i, c in enumerate(CTRL):
    vals = [sum(safe(r) for k, r in MAIN[c].items() if k[2] == f) for f in (1, 3, 5)]
    xs = np.arange(3) + (i - 2) * (w + 0.012)
    ax.bar(xs, vals, w, color=COLOR[c], label=LABEL[c], linewidth=0)
    for x, v in zip(xs, vals):
        ax.text(x, v + 0.4, str(v), ha="center", va="bottom", fontsize=6, color=INK2)
ax.set_xticks(range(3), ["1$\\times$ flow", "3$\\times$ flow", "5$\\times$ flow"])
ax.set_ylabel("Safe successes (of 25)"); ax.set_ylim(0, 25)
ax.yaxis.grid(True, color=GRID, linewidth=0.5); ax.set_axisbelow(True)
ax.legend(ncol=3, fontsize=6.5, loc="upper center", bbox_to_anchor=(0.5, 1.22), handlelength=1, columnspacing=1)
save(fig, "fig_safe_by_flow")

# 2. Safe rate vs actuator gradient with route-bootstrap 95% CIs.
def ci(path):
    return {r["controller"]: (float(r["safe_rate"]), float(r["safe_rate_route_ci_low"]), float(r["safe_rate_route_ci_high"]))
            for r in csv.DictReader(open(path))}
G = {300: ci(f"{PS}/actuator_300_all5_ci_summary.csv"), 400: ci(f"{PS}/five_anatomy_final_comet_v4_20261103/ci_summary.csv"),
     600: ci(f"{PS}/actuator_600_all5_ci_summary.csv")}
fig, ax = plt.subplots(figsize=(3.5, 2.3))
ax.axvspan(500, 650, color="#f1f0ec", linewidth=0)
ax.text(625, 3, "beyond current\nhuman-scale systems", ha="right", va="bottom", fontsize=6, color=INK2)
for i, c in enumerate(CTRL):
    xs = np.array([300, 400, 600]) + (i - 2) * 5
    m = [100 * G[g][c][0] for g in (300, 400, 600)]
    lo = [100 * (G[g][c][0] - G[g][c][1]) for g in (300, 400, 600)]
    hi = [100 * (G[g][c][2] - G[g][c][0]) for g in (300, 400, 600)]
    ax.errorbar(xs, m, yerr=[lo, hi], color=COLOR[c], marker="o", ms=4, lw=2 if c == "comet_v4" else 1.2,
                capsize=0, elinewidth=0.8, mec="white", mew=0.6, label=LABEL[c])
ax.set_xlim(280, 650); ax.set_xticks([300, 400, 600]); ax.set_ylim(0, 100)
ax.set_xlabel("Peak array gradient at workspace centre (mT/m)"); ax.set_ylabel("Safe success rate (%)")
ax.yaxis.grid(True, color=GRID, linewidth=0.5); ax.set_axisbelow(True)
ax.legend(fontsize=6, ncol=3, loc="upper center", bbox_to_anchor=(0.5, 1.25), handlelength=1.5, columnspacing=1)
save(fig, "fig_gradient_sensitivity")

# 3. Minimum surface clearance of arrivals.
fig, ax = plt.subplots(figsize=(3.5, 2.3))
rng = np.random.default_rng(0)
for i, c in enumerate(CTRL):
    rs = [r for r in MAIN[c].values() if tb(r["reached_target"])]
    y = np.array([float(r["min_surface_clearance_mm"]) for r in rs])
    ok = np.array([safe(r) for r in rs])
    x = i + rng.uniform(-0.22, 0.22, len(y))
    ax.scatter(x[ok], y[ok], s=9, color=COLOR[c], edgecolor="white", linewidth=0.4, zorder=3)
    ax.scatter(x[~ok], y[~ok], s=9, facecolor="white", edgecolor=COLOR[c], linewidth=0.8, zorder=3)
    ax.text(i, 2.05, f"{ok.sum()}/{len(y)}", ha="center", fontsize=6, color=INK2)
ax.axhline(0.45, color=INK2, lw=0.8, ls="--"); ax.text(4.35, 0.5, "0.45 mm\ntarget", fontsize=6, ha="left", va="bottom", color=INK2)
ax.axhline(0, color=INK, lw=0.6); ax.text(4.35, -0.07, "wall\ncontact", fontsize=6, ha="left", va="top", color=INK2)
ax.set_xlim(-0.5, 5.0)
ax.set_xticks(range(5), [LABEL[c].replace(" (", "\n(") for c in CTRL], fontsize=6.5); ax.set_ylim(-1.0, 2.2)
ax.set_ylabel("Min. surface clearance (mm)")
ax.scatter([], [], s=9, color=INK2, label="safe success"); ax.scatter([], [], s=9, facecolor="white", edgecolor=INK2, label="reached, unsafe")
ax.legend(fontsize=6, loc="lower left", handletextpad=0.2)
save(fig, "fig_clearance")

# 4. The five anatomies with benchmark start/target pairs (coronal projection).
AN = [("Abdominal aorta", "abdominal", "data/real_vessel_segments_cm_corrected.pkl"),
      ("Aortic root", "aortic_root", "data/vessels/aortic_root_cerebral_40Left85Right/real_vessel_segments.pkl"),
      ("Pulmonary", "pulmonary", "data/vessels/pulmonary_0082_H_PULM_H/real_vessel_segments_cm_corrected.pkl"),
      ("Thoracic aorta", "thoracic_aorta", "data/vessels/aorta_0024_H_AO_H/real_vessel_segments.pkl"),
      ("Aortofemoral", "aortofemoral", "data/vessels/aortofemoral_0049_H_ABAO_AIOD/real_vessel_segments.pkl")]
fig, axs = plt.subplots(1, 5, figsize=(7.16, 2.6))
for ax, (title, key, f) in zip(axs, AN):
    segs = pickle.load(open(f"{REPO}/{f}", "rb"))
    lines = [[(1e3 * s["start"][0], 1e3 * s["start"][2]), (1e3 * s["end"][0], 1e3 * s["end"][2])] for s in segs]
    widths = np.clip([0.15 + 0.08 * 1e3 * s["radius"] for s in segs], 0.15, 2.0)
    ax.add_collection(LineCollection(lines, colors="#b9b8b0", linewidths=widths, capstyle="round"))
    pairs = {}
    for k, r in MAIN["comet_v4"].items():
        if k[0] == key: pairs[k[1]] = r
    for r in pairs.values():
        s = (1e3 * float(r["start_x_m"]), 1e3 * float(r["start_z_m"])); t = (1e3 * float(r["target_x_m"]), 1e3 * float(r["target_z_m"]))
        ax.plot(*s, "o", ms=3.5, color="#2a78d6", mec="white", mew=0.4, zorder=3)
        ax.plot(*t, "*", ms=6, color="#eb6834", mec="white", mew=0.3, zorder=3)
    ax.autoscale(); ax.set_aspect("equal"); ax.set_anchor("N"); ax.set_title(title, fontsize=7.5, color=INK)
    ax.set_xticks([]); ax.set_yticks([]); [sp.set_visible(False) for sp in ax.spines.values()]
    x0, x1 = ax.get_xlim(); y0, _ = ax.get_ylim()
    ax.plot([x0, x0 + 20], [y0 - 4] * 2, color=INK, lw=1); ax.text(x0 + 10, y0 - 7, "20 mm", ha="center", va="top", fontsize=5.5, color=INK2)
axs[0].plot([], [], "o", ms=3.5, color="#2a78d6", label="start"); axs[0].plot([], [], "*", ms=6, color="#eb6834", label="target")
fig.legend(loc="lower center", ncol=2, fontsize=6.5, bbox_to_anchor=(0.5, -0.04))
save(fig, "fig_anatomies")
print("ok")
