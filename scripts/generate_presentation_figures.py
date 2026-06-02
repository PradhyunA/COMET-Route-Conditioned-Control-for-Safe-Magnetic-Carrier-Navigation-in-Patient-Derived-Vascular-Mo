"""
generate_presentation_figures.py
================================
Presentation-quality figures for the DynaMag 50 µm nanoparticle steering
simulation (100-trial dataset).

Produces 11 polished figures suitable for slides / posters:

  IMPROVED (from prior scripts)
    1. Single-trial convergence (gradient + phase annotations)
    2. Convergence-time histogram (KDE overlay, quartile annotations)
    3. Error distribution (ECDF + log-scale histogram, dual panel)
    4. Time vs distance scatter (regression, marginals, outcome coding)

  NEW
    5. Multi-trial convergence (median + IQR with active-trial guardrail)
    6. Force vs velocity (colored by Reynolds, Stokes reference)
    7. Control smoothness (current-derivative analysis)
    8. Success rate by distance quartile (bar + Wilson CI)

  BONUS
    9. Path efficiency vs initial distance
   10. Energy proxy distribution (violin, reached vs not)
   11. MPC cost evolution (representative + multi-trial)

Usage:
    .venv/bin/python3 generate_presentation_figures.py \
        --data-dir flow_50um_100trials \
        --out-dir  presentation_figures
"""

import argparse
import csv
import json
import os
import sys
# Add parent directory to path to allow importing dynamag_simulation when run directly
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

import textwrap
from dataclasses import dataclass, field

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import numpy as np
from matplotlib.colors import Normalize
from matplotlib.lines import Line2D
from scipy.ndimage import uniform_filter1d
from scipy.stats import linregress, gaussian_kde


# ═══════════════════════════════════════════════════════════════════════
#  Design tokens  (clean white-background theme)
# ═══════════════════════════════════════════════════════════════════════
BG        = "#ffffff"
BG_AX     = "#ffffff"
GRID_CLR  = "#cccccc"
TXT_CLR   = "#222222"
ACCENT_1  = "#1f77b4"   # blue
ACCENT_2  = "#2ca02c"   # green
ACCENT_3  = "#ff7f0e"   # orange
ACCENT_4  = "#9467bd"   # purple
ACCENT_5  = "#d62728"   # red
ACCENT_6  = "#b8860b"   # dark goldenrod
WHITE     = "#222222"   # re-used as "emphasis" text color (dark on white bg)
TOLERANCE_MM = 3.0
LOW_RE = 0.1


def apply_theme():
    """Set global matplotlib rcParams for a clean, standard white look."""
    plt.rcParams.update({
        "figure.facecolor":   BG,
        "axes.facecolor":     BG_AX,
        "axes.edgecolor":     "#333333",
        "axes.labelcolor":    TXT_CLR,
        "axes.titlecolor":    TXT_CLR,
        "text.color":         TXT_CLR,
        "xtick.color":        TXT_CLR,
        "ytick.color":        TXT_CLR,
        "grid.color":         GRID_CLR,
        "grid.alpha":         0.5,
        "legend.facecolor":   "#ffffff",
        "legend.edgecolor":   "#999999",
        "legend.labelcolor":  TXT_CLR,
        "figure.dpi":         160,
        "savefig.dpi":        300,
        "savefig.facecolor":  BG,
        "font.size":          11,
        "axes.titlesize":     14,
        "axes.labelsize":     11,
        "axes.titleweight":   "bold",
        "font.family":        "sans-serif",
        "font.sans-serif":    ["Helvetica Neue", "Arial", "sans-serif"],
    })


# ═══════════════════════════════════════════════════════════════════════
#  Data loading
# ═══════════════════════════════════════════════════════════════════════
@dataclass
class TrialBundle:
    """All data from the 100-trial batch."""
    rows:           list
    batch:          dict
    trial_names:    np.ndarray
    sim_time_s:     np.ndarray
    steps:          np.ndarray
    final_error_mm: np.ndarray
    peak_force:     np.ndarray
    peak_vel:       np.ndarray
    peak_field:     np.ndarray
    mean_re:        np.ndarray
    max_re:         np.ndarray
    reached:        np.ndarray
    start_pts:      np.ndarray
    target_pts:     np.ndarray
    start_dist:     np.ndarray
    dt:             float
    data_dir:       str


def load_batch(data_dir: str) -> TrialBundle:
    csv_path = os.path.join(data_dir, "trial_summary.csv")
    json_path = os.path.join(data_dir, "batch_summary.json")

    with open(csv_path, newline="") as f:
        rows = list(csv.DictReader(f))
    with open(json_path) as f:
        batch = json.load(f)

    trial_names    = np.array([r["trial_name"] for r in rows], dtype=object)
    sim_time_s     = np.array([float(r["sim_time_s"]) for r in rows])
    steps          = np.array([int(r["steps_taken"]) for r in rows])
    final_error_mm = np.array([float(r["final_error_mm"]) for r in rows])
    peak_force     = np.array([float(r["peak_force_N"]) for r in rows])
    peak_vel       = np.array([float(r["peak_velocity_m_s"]) for r in rows])
    peak_field     = np.array([float(r["peak_field_T"]) for r in rows])
    mean_re        = np.array([float(r["mean_reynolds"]) for r in rows])
    max_re         = np.array([float(r["max_reynolds"]) for r in rows])
    reached        = np.array([r["reached_target"] == "True" for r in rows])

    start_pts  = np.array([[float(r["start_x"]), float(r["start_y"]), float(r["start_z"])] for r in rows])
    target_pts = np.array([[float(r["target_x"]), float(r["target_y"]), float(r["target_z"])] for r in rows])
    start_dist = np.linalg.norm(target_pts - start_pts, axis=1)

    dt = batch.get("dt", 1.0)

    return TrialBundle(
        rows=rows, batch=batch, trial_names=trial_names,
        sim_time_s=sim_time_s, steps=steps, final_error_mm=final_error_mm,
        peak_force=peak_force, peak_vel=peak_vel, peak_field=peak_field,
        mean_re=mean_re, max_re=max_re, reached=reached,
        start_pts=start_pts, target_pts=target_pts, start_dist=start_dist,
        dt=dt, data_dir=data_dir,
    )


def load_npz(data_dir, trial_name):
    return dict(np.load(os.path.join(data_dir, f"{trial_name}_data.npz"),
                        allow_pickle=True))


def _load_distance_matrix(tb: TrialBundle) -> np.ndarray:
    logs = []
    for name in tb.trial_names:
        d = load_npz(tb.data_dir, name)
        logs.append(np.asarray(d["distance_log"], dtype=float) * 1e3)
    max_len = max(len(l) for l in logs)
    mat = np.full((len(logs), max_len), np.nan)
    for i, dl in enumerate(logs):
        mat[i, :len(dl)] = dl
    return mat


def _wilson_ci(successes: int, n: int, z: float = 1.96):
    if n <= 0:
        return 0.0, 0.0
    p = successes / n
    den = 1.0 + z * z / n
    center = (p + z * z / (2.0 * n)) / den
    radius = (z / den) * np.sqrt((p * (1 - p) + z * z / (4 * n)) / n)
    return max(0.0, center - radius), min(1.0, center + radius)


# ═══════════════════════════════════════════════════════════════════════
#  🟡  Figure 1 — Single-Trial Convergence (gradient + phases)
# ═══════════════════════════════════════════════════════════════════════
def fig01_single_trial_convergence(tb: TrialBundle, out_dir: str):
    # Pick the trial closest to median convergence
    med_steps = int(np.median(tb.steps))
    idx = int(np.argmin(np.abs(tb.steps - med_steps)))
    rep_name = tb.trial_names[idx]
    d = load_npz(tb.data_dir, rep_name)

    dist_mm = np.asarray(d["distance_log"], dtype=float) * 1e3
    t = np.arange(len(dist_mm)) * tb.dt

    fig, ax = plt.subplots(figsize=(12, 5))

    # Gradient coloring segments
    norm = Normalize(vmin=0, vmax=float(np.max(dist_mm)) * 1.05)
    cmap = plt.cm.RdYlGn_r
    for i in range(len(t) - 1):
        ax.plot(t[i:i+2], dist_mm[i:i+2], lw=2.8, color=cmap(norm(dist_mm[i])),
                solid_capstyle="round")

    # Tolerance band
    ax.axhspan(0, TOLERANCE_MM, color=ACCENT_2, alpha=0.08, zorder=0)
    ax.axhline(TOLERANCE_MM, color=ACCENT_2, ls="--", lw=1.4, alpha=0.7,
               label=f"Tolerance = {TOLERANCE_MM} mm")

    # Convergence marker
    converged_idx = np.where(dist_mm <= TOLERANCE_MM)[0]
    if len(converged_idx) > 0:
        tc = t[converged_idx[0]]
        ax.axvline(tc, color=ACCENT_6, ls=":", lw=1.3, alpha=0.8,
                   label=f"Converged at t = {tc:.0f} s")
        ax.scatter([tc], [dist_mm[converged_idx[0]]], s=80, color=ACCENT_6,
                   edgecolors="#666666", linewidths=0.8, zorder=6)

    # Phase annotations
    max_d = float(np.max(dist_mm))
    ax.annotate("Approach", xy=(t[0] + 1, max_d * 0.85),
                fontsize=9, color=ACCENT_3, fontstyle="italic", alpha=0.9)
    if len(converged_idx) > 0 and converged_idx[0] > 5:
        mid = converged_idx[0] // 2
        ax.annotate("Fine control", xy=(t[mid], dist_mm[mid] + max_d * 0.04),
                    fontsize=9, color=ACCENT_1, fontstyle="italic", alpha=0.9)
        ax.annotate("Within tolerance", xy=(t[converged_idx[0]] + 2, TOLERANCE_MM * 0.5),
                    fontsize=9, color=ACCENT_2, fontstyle="italic", alpha=0.9)

    ax.set_xlabel("Time (s)")
    ax.set_ylabel("Distance to target (mm)")
    ax.set_title(f"Single-Trial Convergence — {rep_name}")
    ax.set_xlim(0, t[-1] * 1.03)
    ax.set_ylim(bottom=0)
    ax.legend(fontsize=9, loc="upper right")
    ax.grid(True, alpha=0.2)

    # Colorbar
    sm = plt.cm.ScalarMappable(cmap=cmap, norm=norm)
    sm.set_array([])
    cb = fig.colorbar(sm, ax=ax, pad=0.02, aspect=30)
    cb.set_label("Distance (mm)", fontsize=9)

    fig.tight_layout()
    fig.savefig(os.path.join(out_dir, "fig01_single_trial_convergence.png"),
                bbox_inches="tight")
    plt.close(fig)


# ═══════════════════════════════════════════════════════════════════════
#  🟡  Figure 2 — Convergence Time Histogram with KDE
# ═══════════════════════════════════════════════════════════════════════
def fig02_convergence_histogram(tb: TrialBundle, out_dir: str):
    fig, ax = plt.subplots(figsize=(11, 5.5))

    times = tb.sim_time_s
    q1, med, q3 = np.percentile(times, [25, 50, 75])

    # Histogram
    n_bins = 30
    counts, edges, patches = ax.hist(
        times, bins=n_bins, color=ACCENT_1, alpha=0.55, edgecolor=BG_AX,
        linewidth=0.5, zorder=3, label="Trials"
    )

    # Color bins by reach status: green for fast, orange for slow
    bin_centers = (edges[:-1] + edges[1:]) / 2
    for patch, center in zip(patches, bin_centers):
        if center <= med:
            patch.set_facecolor(ACCENT_2)
            patch.set_alpha(0.6)
        else:
            patch.set_facecolor(ACCENT_3)
            patch.set_alpha(0.6)

    # KDE overlay
    try:
        kde = gaussian_kde(times, bw_method=0.3)
        x_kde = np.linspace(float(np.min(times)) * 0.8, float(np.max(times)) * 1.1, 300)
        y_kde = kde(x_kde)
        # Scale kde to match histogram
        y_kde_scaled = y_kde * len(times) * (edges[1] - edges[0])
        ax.plot(x_kde, y_kde_scaled, lw=2.5, color=WHITE, alpha=0.85, zorder=4,
                label="KDE")
    except Exception:
        pass

    # Quartile lines
    for val, lbl, clr in [(q1, "Q1", ACCENT_4), (med, "Median", ACCENT_6),
                           (q3, "Q3", ACCENT_4)]:
        ax.axvline(val, color=clr, ls="--", lw=1.8, alpha=0.9,
                   label=f"{lbl} = {val:.0f} s")

    # Mean line
    mean_t = float(np.mean(times))
    ax.axvline(mean_t, color=ACCENT_5, ls=":", lw=1.5, alpha=0.8,
               label=f"Mean = {mean_t:.0f} s")

    ax.set_xlabel("Convergence time (s)")
    ax.set_ylabel("Count")
    ax.set_title("Convergence Time Distribution — 100 Trials")
    ax.legend(fontsize=9, loc="upper right")
    ax.grid(True, axis="y", alpha=0.2)

    fig.tight_layout()
    fig.savefig(os.path.join(out_dir, "fig02_convergence_histogram.png"),
                bbox_inches="tight")
    plt.close(fig)


# ═══════════════════════════════════════════════════════════════════════
#  🟡  Figure 3 — Error Distribution (ECDF + log histogram)
# ═══════════════════════════════════════════════════════════════════════
def fig03_error_distribution(tb: TrialBundle, out_dir: str):
    all_err = tb.final_error_mm
    succ_err = all_err[tb.reached]
    fail_err = all_err[~tb.reached]

    def ecdf(vals):
        x = np.sort(vals)
        y = np.arange(1, len(x) + 1) / len(x)
        return x, y

    fig, (ax_l, ax_r) = plt.subplots(1, 2, figsize=(15, 5.5))

    # Left: ECDF
    x_a, y_a = ecdf(all_err)
    x_s, y_s = ecdf(succ_err)
    ax_l.step(x_a, y_a, where="post", lw=2.5, color=ACCENT_1, label="All trials")
    ax_l.step(x_s, y_s, where="post", lw=2.5, color=ACCENT_2, label="Reached only")
    ax_l.axvline(TOLERANCE_MM, color=ACCENT_5, ls="--", lw=1.5, alpha=0.8,
                 label=f"Tolerance = {TOLERANCE_MM} mm")
    # Shade left of tolerance
    ax_l.axvspan(0, TOLERANCE_MM, color=ACCENT_2, alpha=0.05, zorder=0)

    pct_within = float(np.mean(all_err <= TOLERANCE_MM)) * 100
    ax_l.annotate(f"{pct_within:.0f}% within tolerance",
                  xy=(TOLERANCE_MM * 0.5, 0.5), fontsize=10, color=ACCENT_2,
                  ha="center", fontstyle="italic")

    ax_l.set_xlim(0, min(15.0, float(np.max(all_err)) * 1.05))
    ax_l.set_ylim(0, 1.02)
    ax_l.set_xlabel("Final error (mm)")
    ax_l.set_ylabel("ECDF")
    ax_l.set_title("Cumulative Error Distribution")
    ax_l.grid(True, alpha=0.2)
    ax_l.legend(fontsize=9, loc="lower right")

    # Right: log-scale histogram
    safe_min = max(0.05, float(np.min(all_err[all_err > 0])))
    bins_log = np.logspace(np.log10(safe_min), np.log10(float(np.max(all_err)) * 1.05), 25)
    ax_r.hist(succ_err, bins=bins_log, color=ACCENT_2, alpha=0.55, label="Reached",
              edgecolor=BG_AX, linewidth=0.4)
    if len(fail_err) > 0:
        ax_r.hist(fail_err, bins=bins_log, color=ACCENT_5, alpha=0.65, label="Not reached",
                  edgecolor=BG_AX, linewidth=0.4)
    ax_r.axvline(TOLERANCE_MM, color=ACCENT_5, ls="--", lw=1.5, alpha=0.8)
    ax_r.set_xscale("log")
    ax_r.set_xlabel("Final error (mm, log scale)")
    ax_r.set_ylabel("Count")
    ax_r.set_title("Error Tail Visibility")
    ax_r.grid(True, alpha=0.2, which="both")
    ax_r.legend(fontsize=9, loc="upper right")

    fig.tight_layout()
    fig.savefig(os.path.join(out_dir, "fig03_error_distribution.png"),
                bbox_inches="tight")
    plt.close(fig)


# ═══════════════════════════════════════════════════════════════════════
#  🟡  Figure 4 — Time vs Distance Scatter (regression + marginals)
# ═══════════════════════════════════════════════════════════════════════
def fig04_time_vs_distance(tb: TrialBundle, out_dir: str):
    dist_mm = tb.start_dist * 1e3
    times = tb.sim_time_s
    err = tb.final_error_mm
    reached = tb.reached

    fig = plt.figure(figsize=(12, 8))
    gs = fig.add_gridspec(2, 2, width_ratios=[4, 1], height_ratios=[1, 4],
                          wspace=0.03, hspace=0.03)

    ax_main = fig.add_subplot(gs[1, 0])
    ax_top  = fig.add_subplot(gs[0, 0], sharex=ax_main)
    ax_right = fig.add_subplot(gs[1, 1], sharey=ax_main)

    # Main scatter
    norm = Normalize(vmin=0, vmax=min(10.0, float(np.max(err))))
    cmap = plt.cm.RdYlGn_r

    # Not-reached markers (X)
    if np.any(~reached):
        ax_main.scatter(dist_mm[~reached], times[~reached],
                        c=err[~reached], cmap=cmap, norm=norm,
                        s=90, marker="X", alpha=0.9, edgecolors="#666666",
                        linewidths=0.6, zorder=5, label="Not reached")

    # Reached markers (o)
    sc = ax_main.scatter(dist_mm[reached], times[reached],
                         c=err[reached], cmap=cmap, norm=norm,
                         s=55, marker="o", alpha=0.85, edgecolors="#666666",
                         linewidths=0.4, zorder=4, label="Reached")

    # Regression line
    slope, intercept, r, p, se = linregress(dist_mm, times)
    x_fit = np.linspace(float(np.min(dist_mm)) * 0.9, float(np.max(dist_mm)) * 1.1, 100)
    ax_main.plot(x_fit, slope * x_fit + intercept, lw=2, color=ACCENT_6, ls="--",
                 alpha=0.8, label=f"OLS  R² = {r**2:.3f}")

    ax_main.set_xlabel("Start-to-target distance (mm)")
    ax_main.set_ylabel("Convergence time (s)")
    ax_main.legend(fontsize=9, loc="upper left")
    ax_main.grid(True, alpha=0.2)

    cb = fig.colorbar(sc, ax=ax_main, pad=0.12, aspect=30, shrink=0.8)
    cb.set_label("Final error (mm)", fontsize=9)

    # Top marginal (distance histogram)
    ax_top.hist(dist_mm, bins=25, color=ACCENT_1, alpha=0.5, edgecolor=BG_AX,
                linewidth=0.4)
    ax_top.set_ylabel("Count", fontsize=8)
    ax_top.tick_params(labelbottom=False)
    ax_top.grid(True, alpha=0.15, axis="y")
    ax_top.set_title("Time vs. Start-Target Distance", pad=10)

    # Right marginal (time histogram)
    ax_right.hist(times, bins=25, orientation="horizontal", color=ACCENT_1,
                  alpha=0.5, edgecolor=BG_AX, linewidth=0.4)
    ax_right.set_xlabel("Count", fontsize=8)
    ax_right.tick_params(labelleft=False)
    ax_right.grid(True, alpha=0.15, axis="x")

    try:
        fig.tight_layout()
    except Exception:
        pass
    fig.savefig(os.path.join(out_dir, "fig04_time_vs_distance.png"),
                bbox_inches="tight")
    plt.close(fig)


# ═══════════════════════════════════════════════════════════════════════
#  🔴  Figure 5 — Multi-Trial Convergence (median + IQR + active count)
# ═══════════════════════════════════════════════════════════════════════
def fig05_multi_trial_convergence(tb: TrialBundle, out_dir: str, min_active: int = 20):
    mat = _load_distance_matrix(tb)
    active = np.sum(~np.isnan(mat), axis=0)
    t = np.arange(mat.shape[1]) * tb.dt

    med  = np.nanmedian(mat, axis=0)
    q25  = np.nanpercentile(mat, 25, axis=0)
    q75  = np.nanpercentile(mat, 75, axis=0)
    q10  = np.nanpercentile(mat, 10, axis=0)
    q90  = np.nanpercentile(mat, 90, axis=0)

    mask = active >= min_active

    fig, (ax_top, ax_bot) = plt.subplots(
        2, 1, figsize=(14, 7.5), sharex=True,
        gridspec_kw={"height_ratios": [3.5, 1]}
    )

    # 90–10 band (wider)
    ax_top.fill_between(t, np.where(mask, q10, np.nan), np.where(mask, q90, np.nan),
                        alpha=0.1, color=ACCENT_1, label="10th–90th pctl")
    # IQR band
    ax_top.fill_between(t, np.where(mask, q25, np.nan), np.where(mask, q75, np.nan),
                        alpha=0.22, color=ACCENT_1, label=f"IQR (≥{min_active} active)")
    # Median
    ax_top.plot(t, np.where(mask, med, np.nan), lw=2.8, color=ACCENT_1,
                label="Median (reliable window)", zorder=4)
    ax_top.plot(t, med, lw=1.0, ls="--", color="#888888", alpha=0.6,
                label="Median (all available)")

    # Tolerance
    ax_top.axhline(TOLERANCE_MM, color=ACCENT_5, ls="--", lw=1.5, alpha=0.7,
                   label=f"Tolerance = {TOLERANCE_MM} mm")

    # Last reached trial
    max_success_t = float(np.max(tb.sim_time_s[tb.reached])) if np.any(tb.reached) else np.nan
    if np.isfinite(max_success_t):
        ax_top.axvline(max_success_t, color=ACCENT_6, ls=":", lw=1.3, alpha=0.7,
                       label=f"Last reached at {max_success_t:.0f} s")

    ax_top.set_ylabel("Distance to target (mm)")
    ax_top.set_title("Multi-Trial Distance-to-Target Convergence")
    ax_top.set_ylim(bottom=0)
    ax_top.grid(True, alpha=0.2)
    ax_top.legend(fontsize=8, loc="upper right", ncol=2)

    # Active-trial count subplot
    ax_bot.fill_between(t, 0, active, color=ACCENT_1, alpha=0.25, step="mid")
    ax_bot.plot(t, active, color=ACCENT_1, lw=1.8, drawstyle="steps-mid")
    ax_bot.axhline(min_active, color=ACCENT_4, ls="--", lw=1.4, alpha=0.8,
                   label=f"Min active = {min_active}")
    ax_bot.set_xlabel("Time (s)")
    ax_bot.set_ylabel("Active trials")
    ax_bot.grid(True, alpha=0.2)
    ax_bot.legend(fontsize=8, loc="upper right")

    ax_bot.set_xlim(0, float(np.max(tb.sim_time_s)) * 1.03)

    fig.tight_layout()
    fig.savefig(os.path.join(out_dir, "fig05_multi_trial_convergence.png"),
                bbox_inches="tight")
    plt.close(fig)


# ═══════════════════════════════════════════════════════════════════════
#  🔴  Figure 6 — Force vs Velocity (colored by Reynolds)
# ═══════════════════════════════════════════════════════════════════════
def fig06_force_vs_velocity(tb: TrialBundle, out_dir: str):
    force_nN = tb.peak_force * 1e9
    vel_um_s = tb.peak_vel * 1e6
    re = tb.max_re

    fig = plt.figure(figsize=(12, 8))
    gs = fig.add_gridspec(2, 2, width_ratios=[4, 1], height_ratios=[1, 4],
                          wspace=0.03, hspace=0.03)

    ax_main  = fig.add_subplot(gs[1, 0])
    ax_top   = fig.add_subplot(gs[0, 0], sharex=ax_main)
    ax_right = fig.add_subplot(gs[1, 1], sharey=ax_main)

    # Main scatter
    norm = Normalize(vmin=float(np.min(re)), vmax=float(np.max(re)))
    sc = ax_main.scatter(vel_um_s, force_nN, c=re, cmap="plasma", norm=norm,
                         s=60, alpha=0.85, edgecolors="#666666", linewidths=0.4,
                         zorder=4)

    # Stokes drag reference: F = 6πηrv
    eta = 8.9e-4        # Pa·s
    r = 25e-6            # 50 µm diameter → 25 µm radius
    v_range = np.linspace(0, float(np.max(vel_um_s)) * 1.1, 200)
    f_stokes = 6 * np.pi * eta * r * (v_range * 1e-6) * 1e9  # nN
    ax_main.plot(v_range, f_stokes, lw=2, color=ACCENT_6, ls="--", alpha=0.8,
                 label="Stokes drag (6πηrv)")

    ax_main.set_xlabel("Peak velocity (µm/s)")
    ax_main.set_ylabel("Peak force (nN)")
    ax_main.legend(fontsize=9, loc="upper left")
    ax_main.grid(True, alpha=0.2)

    cb = fig.colorbar(sc, ax=ax_main, pad=0.12, aspect=30, shrink=0.8)
    cb.set_label("Max Reynolds", fontsize=9)

    # Marginals
    ax_top.hist(vel_um_s, bins=25, color=ACCENT_4, alpha=0.5, edgecolor=BG_AX, lw=0.4)
    ax_top.set_ylabel("Count", fontsize=8)
    ax_top.tick_params(labelbottom=False)
    ax_top.grid(True, alpha=0.15, axis="y")
    ax_top.set_title("Peak Force vs. Peak Velocity", pad=10)

    ax_right.hist(force_nN, bins=25, orientation="horizontal", color=ACCENT_4,
                  alpha=0.5, edgecolor=BG_AX, lw=0.4)
    ax_right.set_xlabel("Count", fontsize=8)
    ax_right.tick_params(labelleft=False)
    ax_right.grid(True, alpha=0.15, axis="x")

    try:
        fig.tight_layout()
    except Exception:
        pass
    fig.savefig(os.path.join(out_dir, "fig06_force_vs_velocity.png"),
                bbox_inches="tight")
    plt.close(fig)


# ═══════════════════════════════════════════════════════════════════════
#  🔴  Figure 7 — Control Smoothness (current derivatives)
# ═══════════════════════════════════════════════════════════════════════
def fig07_control_smoothness(tb: TrialBundle, out_dir: str):
    fig, (ax_l, ax_r) = plt.subplots(1, 2, figsize=(16, 5.5))

    # Left panel — representative trial current derivatives
    med_steps = int(np.median(tb.steps))
    idx = int(np.argmin(np.abs(tb.steps - med_steps)))
    rep_name = tb.trial_names[idx]
    d = load_npz(tb.data_dir, rep_name)
    cl = d["current_log"]  # (N, 8)
    dI = np.diff(cl, axis=0) / tb.dt  # (N-1, 8) – current derivative
    t = np.arange(dI.shape[0]) * tb.dt

    coil_colors = plt.cm.Set2(np.linspace(0, 1, cl.shape[1]))
    for j in range(dI.shape[1]):
        ax_l.plot(t, dI[:, j], lw=1.0, alpha=0.7, color=coil_colors[j],
                  label=f"Coil {j+1}")

    ax_l.set_xlabel("Time (s)")
    ax_l.set_ylabel("dI/dt (A/s)")
    ax_l.set_title(f"Current Rate of Change — {rep_name}")
    ax_l.legend(fontsize=7, ncol=2, loc="upper right")
    ax_l.grid(True, alpha=0.2)

    # Right panel — distribution of ||ΔI|| across all trials
    jerk_norms = []
    for name in tb.trial_names:
        dd = load_npz(tb.data_dir, name)
        cc = dd["current_log"]
        if len(cc) > 1:
            dI_trial = np.diff(cc, axis=0) / tb.dt
            norms = np.linalg.norm(dI_trial, axis=1)
            jerk_norms.append(float(np.mean(norms)))  # mean jerk per trial

    jerk_norms = np.array(jerk_norms)

    parts = ax_r.violinplot(jerk_norms, positions=[0], showmeans=True, showextrema=False)
    for pc in parts["bodies"]:
        pc.set_facecolor(ACCENT_4)
        pc.set_alpha(0.4)
    parts["cmeans"].set_color(ACCENT_6)

    bp = ax_r.boxplot(jerk_norms, positions=[0], widths=0.35, patch_artist=True,
                      boxprops=dict(facecolor=ACCENT_4, alpha=0.3),
                      medianprops=dict(color=WHITE, lw=1.5),
                      whiskerprops=dict(color=TXT_CLR),
                      capprops=dict(color=TXT_CLR),
                      flierprops=dict(marker="o", markersize=4, alpha=0.5,
                                      markerfacecolor=ACCENT_5))

    ax_r.set_ylabel("Mean ‖ΔI/Δt‖ (A/s)")
    ax_r.set_title("Control Smoothness — All Trials")
    ax_r.set_xticks([0], ["100 trials"])
    ax_r.grid(True, alpha=0.2, axis="y")

    # Annotate stats
    med_j = float(np.median(jerk_norms))
    ax_r.annotate(f"Median = {med_j:.2f} A/s",
                  xy=(0.25, med_j), fontsize=9, color=ACCENT_6)

    fig.tight_layout()
    fig.savefig(os.path.join(out_dir, "fig07_control_smoothness.png"),
                bbox_inches="tight")
    plt.close(fig)


# ═══════════════════════════════════════════════════════════════════════
#  🔴  Figure 8 — Success Rate by Distance Quartile (bar + Wilson CI)
# ═══════════════════════════════════════════════════════════════════════
def fig08_success_rate_box(tb: TrialBundle, out_dir: str):
    dist_mm = tb.start_dist * 1e3

    # Quartile thresholds
    q_vals = np.percentile(dist_mm, [0, 25, 50, 75, 100])
    labels, rates, ci_lo, ci_hi, counts = [], [], [], [], []

    for i in range(4):
        lo, hi = q_vals[i], q_vals[i + 1]
        if i == 3:
            mask = (dist_mm >= lo) & (dist_mm <= hi)
        else:
            mask = (dist_mm >= lo) & (dist_mm < hi)
        n = int(np.sum(mask))
        s = int(np.sum(tb.reached[mask]))
        rate = s / n if n > 0 else 0
        w_lo, w_hi = _wilson_ci(s, n)
        labels.append(f"{lo:.0f}–{hi:.0f} mm\n(n={n})")
        rates.append(rate * 100)
        ci_lo.append(rate * 100 - w_lo * 100)
        ci_hi.append(w_hi * 100 - rate * 100)
        counts.append(f"{s}/{n}")

    fig, ax = plt.subplots(figsize=(10, 6))
    x = np.arange(4)
    colors = [ACCENT_2, ACCENT_1, ACCENT_3, ACCENT_5]

    bars = ax.bar(x, rates, width=0.6, color=colors, alpha=0.75,
                  edgecolor="#333333", linewidth=0.5, zorder=3)

    # Error bars (Wilson CI)
    ax.errorbar(x, rates, yerr=[ci_lo, ci_hi], fmt="none",
                ecolor=WHITE, elinewidth=1.8, capsize=6, capthick=1.5, zorder=5)

    # Count annotations
    for xi, bar, cnt in zip(x, bars, counts):
        ax.text(xi, bar.get_height() + max(ci_hi) + 2, cnt,
                ha="center", va="bottom", fontsize=11, color=WHITE, fontweight="bold")

    ax.set_xticks(x, labels, fontsize=10)
    ax.set_ylabel("Success rate (%)")
    ax.set_ylim(0, 115)
    ax.set_title("Success Rate by Start–Target Distance Quartile")
    ax.grid(True, axis="y", alpha=0.2)

    # Overall rate annotation
    overall = float(np.mean(tb.reached)) * 100
    ax.axhline(overall, color=ACCENT_6, ls=":", lw=1.5, alpha=0.7,
               label=f"Overall = {overall:.0f}%")
    ax.legend(fontsize=9, loc="lower left")

    fig.tight_layout()
    fig.savefig(os.path.join(out_dir, "fig08_success_rate_box.png"),
                bbox_inches="tight")
    plt.close(fig)


# ═══════════════════════════════════════════════════════════════════════
#  💡  Figure 9 — Path Efficiency vs Initial Distance
# ═══════════════════════════════════════════════════════════════════════
def fig09_path_efficiency(tb: TrialBundle, out_dir: str):
    dist_mm = tb.start_dist * 1e3
    times = tb.sim_time_s

    eff = []
    for name in tb.trial_names:
        d = load_npz(tb.data_dir, name)
        eff.append(float(d.get("path_efficiency", np.nan)))
    eff = np.array(eff)

    fig, ax = plt.subplots(figsize=(11, 6))
    norm = Normalize(vmin=float(np.min(times)), vmax=float(np.max(times)))

    sc = ax.scatter(dist_mm, eff, c=times, cmap="cool", norm=norm,
                    s=55, alpha=0.85, edgecolors="#666666", linewidths=0.4, zorder=4)

    ax.axhline(1.0, color=ACCENT_6, ls="--", lw=1.5, alpha=0.7,
               label="Perfect efficiency = 1.0")

    ax.set_xlabel("Start-to-target distance (mm)")
    ax.set_ylabel("Path efficiency (straight / actual)")
    ax.set_title("Path Efficiency vs. Initial Distance")
    ax.set_ylim(0, 1.1)
    ax.grid(True, alpha=0.2)
    ax.legend(fontsize=9)

    cb = fig.colorbar(sc, ax=ax, pad=0.02, aspect=30)
    cb.set_label("Convergence time (s)", fontsize=9)

    fig.tight_layout()
    fig.savefig(os.path.join(out_dir, "fig09_path_efficiency.png"),
                bbox_inches="tight")
    plt.close(fig)


# ═══════════════════════════════════════════════════════════════════════
#  💡  Figure 10 — Energy Proxy Distribution (violin, reached vs not)
# ═══════════════════════════════════════════════════════════════════════
def fig10_energy_proxy(tb: TrialBundle, out_dir: str):
    energy = []
    for name in tb.trial_names:
        d = load_npz(tb.data_dir, name)
        energy.append(float(d.get("energy_proxy", np.nan)))
    energy = np.array(energy)

    e_reached = energy[tb.reached]
    e_not     = energy[~tb.reached]

    fig, ax = plt.subplots(figsize=(9, 6))
    datasets, positions, colors_v, labels_v = [], [], [], []
    if len(e_reached) > 0:
        datasets.append(e_reached)
        positions.append(0)
        colors_v.append(ACCENT_2)
        labels_v.append("Reached")
    if len(e_not) > 0:
        datasets.append(e_not)
        positions.append(1)
        colors_v.append(ACCENT_5)
        labels_v.append("Not reached")

    for data, pos, clr in zip(datasets, positions, colors_v):
        parts = ax.violinplot(data, positions=[pos], showmeans=True, showextrema=False)
        for pc in parts["bodies"]:
            pc.set_facecolor(clr)
            pc.set_alpha(0.4)
        parts["cmeans"].set_color(ACCENT_6)

        ax.boxplot(data, positions=[pos], widths=0.35, patch_artist=True,
                   boxprops=dict(facecolor=clr, alpha=0.3),
                   medianprops=dict(color=WHITE, lw=1.5),
                   whiskerprops=dict(color=TXT_CLR),
                   capprops=dict(color=TXT_CLR),
                   flierprops=dict(marker="o", markersize=4, alpha=0.5,
                                   markerfacecolor=clr))

    ax.set_xticks(positions, labels_v)
    ax.set_ylabel("Energy proxy  Σ I² · dt  (A²·s)")
    ax.set_title("Control Energy Cost by Outcome")
    ax.grid(True, alpha=0.2, axis="y")

    fig.tight_layout()
    fig.savefig(os.path.join(out_dir, "fig10_energy_proxy.png"),
                bbox_inches="tight")
    plt.close(fig)


# ═══════════════════════════════════════════════════════════════════════
#  💡  Figure 11 — MPC Cost Evolution
# ═══════════════════════════════════════════════════════════════════════
def fig11_mpc_cost(tb: TrialBundle, out_dir: str):
    fig, (ax_l, ax_r) = plt.subplots(1, 2, figsize=(16, 5.5))

    # Left: representative trial cost log
    med_steps = int(np.median(tb.steps))
    idx = int(np.argmin(np.abs(tb.steps - med_steps)))
    rep_name = tb.trial_names[idx]
    d = load_npz(tb.data_dir, rep_name)
    cost = np.asarray(d["cost_log"], dtype=float)
    t = np.arange(len(cost)) * tb.dt

    ax_l.plot(t, cost, lw=2.2, color=ACCENT_1, zorder=4)
    ax_l.fill_between(t, 0, cost, alpha=0.12, color=ACCENT_1)
    ax_l.set_xlabel("Time (s)")
    ax_l.set_ylabel("MPC cost (objective value)")
    ax_l.set_title(f"MPC Cost — {rep_name}")
    ax_l.grid(True, alpha=0.2)

    # Right: multi-trial overlay
    max_len = int(np.max(tb.steps)) + 5
    for name in tb.trial_names:
        dd = load_npz(tb.data_dir, name)
        cc = np.asarray(dd["cost_log"], dtype=float)
        tt = np.arange(len(cc)) * tb.dt
        ax_r.plot(tt, cc, lw=0.5, alpha=0.2, color=ACCENT_1)

    # Median cost curve
    cost_mat = np.full((len(tb.trial_names), max_len), np.nan)
    for i, name in enumerate(tb.trial_names):
        dd = load_npz(tb.data_dir, name)
        cc = np.asarray(dd["cost_log"], dtype=float)
        cost_mat[i, :len(cc)] = cc
    import warnings
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", RuntimeWarning)
        med_cost = np.nanmedian(cost_mat, axis=0)
    t_med = np.arange(max_len) * tb.dt
    active_cost = np.sum(~np.isnan(cost_mat), axis=0)
    med_cost_masked = np.where(active_cost >= 20, med_cost, np.nan)
    ax_r.plot(t_med, med_cost_masked, lw=2.5, color=ACCENT_6, zorder=5,
              label="Median (≥20 active)")

    ax_r.set_xlabel("Time (s)")
    ax_r.set_ylabel("MPC cost")
    ax_r.set_title("MPC Cost Evolution — All 100 Trials")
    ax_r.set_xlim(0, float(np.max(tb.sim_time_s)) * 1.03)
    ax_r.grid(True, alpha=0.2)
    ax_r.legend(fontsize=9)

    fig.tight_layout()
    fig.savefig(os.path.join(out_dir, "fig11_mpc_cost.png"),
                bbox_inches="tight")
    plt.close(fig)


# ═══════════════════════════════════════════════════════════════════════
#  Stats summary
# ═══════════════════════════════════════════════════════════════════════
def save_stats_summary(tb: TrialBundle, out_dir: str):
    n = len(tb.reached)
    nr = int(np.sum(tb.reached))
    w_lo, w_hi = _wilson_ci(nr, n)

    txt = textwrap.dedent(f"""\
        DynaMag 50 µm Particle — Presentation Statistics
        ==================================================
        Trials               : {n}
        Success rate          : {nr}/{n} ({nr/n*100:.0f}%)   95% CI [{w_lo:.3f}, {w_hi:.3f}]
        Median conv. time     : {np.median(tb.sim_time_s):.0f} s  ({int(np.median(tb.steps))} steps)
        Mean conv. time       : {np.mean(tb.sim_time_s):.0f} s
        IQR conv. time        : [{np.percentile(tb.sim_time_s, 25):.0f}, {np.percentile(tb.sim_time_s, 75):.0f}] s
        Median final error    : {np.median(tb.final_error_mm):.2f} mm
        95th pctl final error : {np.percentile(tb.final_error_mm, 95):.2f} mm
        Max final error       : {np.max(tb.final_error_mm):.2f} mm

        Physics Regime
        --------------
        Max Reynolds (median) : {np.median(tb.max_re):.5f}
        Max Reynolds (worst)  : {np.max(tb.max_re):.5f}
        Low-Re valid (all)    : {bool(np.all(tb.max_re < 0.1))}
        Peak |B| median       : {np.median(tb.peak_field)*1e3:.2f} mT
        Peak |F| median       : {np.median(tb.peak_force)*1e9:.2f} nN
        Peak |v| median       : {np.median(tb.peak_vel)*1e6:.1f} µm/s

        Batch info
        ----------
        Wall-clock time       : {tb.batch.get('batch_wall_time_s', 0)/3600:.1f} h
    """)

    with open(os.path.join(out_dir, "stats_summary.txt"), "w") as f:
        f.write(txt)


# ═══════════════════════════════════════════════════════════════════════
#  Main
# ═══════════════════════════════════════════════════════════════════════
def main():
    parser = argparse.ArgumentParser(
        description="Generate presentation-quality figures for DynaMag."
    )
    parser.add_argument("--data-dir", default="outputs/flow_50um_100trials",
                        help="Directory containing trial NPZ + CSV data")
    parser.add_argument("--out-dir", default="outputs/presentation_figures",
                        help="Output directory for figures")
    parser.add_argument("--min-active", type=int, default=20,
                        help="Min active trials for reliable distance quantiles")
    args = parser.parse_args()

    os.makedirs(args.out_dir, exist_ok=True)
    apply_theme()

    print("Loading data …")
    tb = load_batch(args.data_dir)
    print(f"  {len(tb.rows)} trials loaded  (dt={tb.dt} s, reached={int(np.sum(tb.reached))}/{len(tb.rows)})")

    generators = [
        ("01", "Single-trial convergence",         lambda: fig01_single_trial_convergence(tb, args.out_dir)),
        ("02", "Convergence histogram",             lambda: fig02_convergence_histogram(tb, args.out_dir)),
        ("03", "Error distribution",                lambda: fig03_error_distribution(tb, args.out_dir)),
        ("04", "Time vs distance scatter",          lambda: fig04_time_vs_distance(tb, args.out_dir)),
        ("05", "Multi-trial convergence",           lambda: fig05_multi_trial_convergence(tb, args.out_dir, args.min_active)),
        ("06", "Force vs velocity",                 lambda: fig06_force_vs_velocity(tb, args.out_dir)),
        ("07", "Control smoothness",                lambda: fig07_control_smoothness(tb, args.out_dir)),
        ("08", "Success rate box",                  lambda: fig08_success_rate_box(tb, args.out_dir)),
        ("09", "Path efficiency",                   lambda: fig09_path_efficiency(tb, args.out_dir)),
        ("10", "Energy proxy",                      lambda: fig10_energy_proxy(tb, args.out_dir)),
        ("11", "MPC cost evolution",                lambda: fig11_mpc_cost(tb, args.out_dir)),
    ]

    for num, name, gen_fn in generators:
        print(f"  [{num}/11] {name} …", end=" ", flush=True)
        gen_fn()
        print("✓")

    save_stats_summary(tb, args.out_dir)
    print(f"\n✅  All 11 figures + stats saved to: {args.out_dir}/")


if __name__ == "__main__":
    main()
