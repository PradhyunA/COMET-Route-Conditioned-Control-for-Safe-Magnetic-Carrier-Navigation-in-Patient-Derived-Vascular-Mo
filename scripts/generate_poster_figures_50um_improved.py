"""
generate_poster_figures_50um_improved.py
=======================================
Improved poster figures for the 50 um flow dataset.

This script focuses on:
1) Right-censored convergence reporting (Kaplan-Meier style CDF)
2) Distance-curve summaries with active-trial counts (avoids late-time bias)
3) Reynolds-vs-outcome visualization with low-Re threshold context
4) Final-error distribution with explicit tail visibility
"""

import argparse
import csv
import json
import os
import sys
# Add parent directory to path to allow importing dynamag_simulation when run directly
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from dataclasses import dataclass

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np


TOLERANCE_MM = 3.0
LOW_RE_THRESHOLD = 0.1


@dataclass
class BatchData:
    rows: list
    batch: dict
    trial_names: np.ndarray
    sim_time_s: np.ndarray
    steps: np.ndarray
    final_error_mm: np.ndarray
    max_re: np.ndarray
    reached: np.ndarray
    dt: float


def _load_batch(data_dir: str) -> BatchData:
    csv_path = os.path.join(data_dir, "trial_summary.csv")
    json_path = os.path.join(data_dir, "batch_summary.json")

    with open(csv_path, newline="") as f:
        rows = list(csv.DictReader(f))

    with open(json_path) as f:
        batch = json.load(f)

    trial_names = np.array([r["trial_name"] for r in rows], dtype=object)
    sim_time_s = np.array([float(r["sim_time_s"]) for r in rows], dtype=float)
    steps = np.array([int(r["steps_taken"]) for r in rows], dtype=int)
    final_error_mm = np.array([float(r["final_error_mm"]) for r in rows], dtype=float)
    max_re = np.array([float(r["max_reynolds"]) for r in rows], dtype=float)
    reached = np.array([r["reached_target"] == "True" for r in rows], dtype=bool)

    if rows and int(rows[0]["steps_taken"]) > 0:
        dt = float(rows[0]["sim_time_s"]) / float(rows[0]["steps_taken"])
    else:
        dt = 1.0
    if "dt" in batch:
        dt = float(batch["dt"])

    return BatchData(
        rows=rows,
        batch=batch,
        trial_names=trial_names,
        sim_time_s=sim_time_s,
        steps=steps,
        final_error_mm=final_error_mm,
        max_re=max_re,
        reached=reached,
        dt=dt,
    )


def _wilson_interval(successes: int, n: int, z: float = 1.96) -> tuple[float, float]:
    if n <= 0:
        return 0.0, 0.0
    p = successes / n
    den = 1.0 + (z * z) / n
    center = (p + (z * z) / (2.0 * n)) / den
    radius = (z / den) * np.sqrt((p * (1.0 - p) + (z * z) / (4.0 * n)) / n)
    return max(0.0, float(center - radius)), min(1.0, float(center + radius))


def _kaplan_meier(times: np.ndarray, events: np.ndarray):
    """
    Return step points for Kaplan-Meier survival S(t) with Greenwood CI.
    times: observed times (event or censor)
    events: True for event, False for censor
    """
    times = np.asarray(times, dtype=float)
    events = np.asarray(events, dtype=bool)

    event_times = np.unique(times[events])
    x = [0.0]
    s = [1.0]
    lo = [1.0]
    hi = [1.0]

    surv = 1.0
    greenwood_sum = 0.0

    for t in event_times:
        at_risk = int(np.sum(times >= t))
        d = int(np.sum((times == t) & events))
        if at_risk <= 0:
            continue

        surv *= (1.0 - d / at_risk)
        if at_risk > d:
            greenwood_sum += d / (at_risk * (at_risk - d))
        var = (surv * surv) * greenwood_sum
        se = np.sqrt(max(var, 0.0))

        x.append(float(t))
        s.append(float(surv))
        lo.append(float(max(0.0, surv - 1.96 * se)))
        hi.append(float(min(1.0, surv + 1.96 * se)))

    return np.array(x), np.array(s), np.array(lo), np.array(hi)


def _survival_at(step_x: np.ndarray, step_s: np.ndarray, t: np.ndarray) -> np.ndarray:
    idx = np.searchsorted(step_x, t, side="right") - 1
    idx = np.clip(idx, 0, len(step_s) - 1)
    return step_s[idx]


def _load_distance_matrix(data_dir: str, trial_names: np.ndarray) -> np.ndarray:
    lengths = []
    logs = []
    for name in trial_names:
        npz = np.load(os.path.join(data_dir, f"{name}_data.npz"), allow_pickle=True)
        d_mm = np.asarray(npz["distance_log"], dtype=float) * 1e3
        logs.append(d_mm)
        lengths.append(len(d_mm))

    max_len = int(max(lengths)) if lengths else 0
    mat = np.full((len(logs), max_len), np.nan, dtype=float)
    for i, d_mm in enumerate(logs):
        mat[i, : len(d_mm)] = d_mm
    return mat


def _save_fig1_km(data: BatchData, out_dir: str):
    times = data.sim_time_s
    events = data.reached

    x, s, s_lo, s_hi = _kaplan_meier(times, events)
    cdf = 1.0 - s
    cdf_lo = 1.0 - s_hi
    cdf_hi = 1.0 - s_lo

    cens_t = times[~events]
    cens_cdf = 1.0 - _survival_at(x, s, cens_t)

    median_success = float(np.median(times[events])) if np.any(events) else np.nan
    success_rate = float(np.mean(events))

    fig, ax = plt.subplots(figsize=(10, 6))
    ax.fill_between(x, cdf_lo, cdf_hi, step="post", alpha=0.18, color="#4c78a8",
                    label="95% CI (Greenwood)")
    ax.step(x, cdf, where="post", lw=2.5, color="#1f4e79",
            label="Converged fraction (KM)")
    if len(cens_t) > 0:
        ax.scatter(cens_t, cens_cdf, marker="x", s=55, color="#b22222",
                   label="Right-censored (not reached)")

    if np.isfinite(median_success):
        ax.axvline(median_success, ls="--", lw=1.5, color="#444444",
                   label=f"Median reached time = {median_success:.0f} s")
    ax.axhline(success_rate, ls=":", lw=1.5, color="#2e7d32",
               label=f"Final success fraction = {success_rate * 100:.1f}%")

    ax.set_xlim(0, float(np.max(times) * 1.02))
    ax.set_ylim(0, 1.02)
    ax.set_xlabel("Time (s)")
    ax.set_ylabel("Fraction converged")
    ax.set_title("Convergence Under Flow (Right-Censor Aware)")
    ax.grid(True, alpha=0.2)
    ax.legend(fontsize=9, loc="lower right")
    fig.tight_layout()
    fig.savefig(os.path.join(out_dir, "fig1_km_convergence.png"), dpi=300, bbox_inches="tight")
    plt.close(fig)


def _save_fig2_distance_active(data: BatchData, data_dir: str, out_dir: str, min_active: int):
    mat = _load_distance_matrix(data_dir, data.trial_names)
    active = np.sum(~np.isnan(mat), axis=0)
    t = np.arange(mat.shape[1], dtype=float) * data.dt

    med = np.nanmedian(mat, axis=0)
    q25 = np.nanpercentile(mat, 25, axis=0)
    q75 = np.nanpercentile(mat, 75, axis=0)

    mask = active >= min_active
    med_masked = np.where(mask, med, np.nan)
    q25_masked = np.where(mask, q25, np.nan)
    q75_masked = np.where(mask, q75, np.nan)

    max_success_time = float(np.max(data.sim_time_s[data.reached])) if np.any(data.reached) else np.nan

    fig, (ax_top, ax_bot) = plt.subplots(
        2, 1, figsize=(12, 7), sharex=True, gridspec_kw={"height_ratios": [3, 1]}
    )

    ax_top.fill_between(t, q25_masked, q75_masked, alpha=0.25, color="#4c78a8",
                        label=f"IQR with >= {min_active} active trials")
    ax_top.plot(t, med_masked, lw=2.6, color="#0b3c6f", label="Median (reliable window)")
    ax_top.plot(t, med, lw=1.2, ls="--", color="#7f7f7f", alpha=0.8,
                label="Median (all available trials)")
    ax_top.axhline(TOLERANCE_MM, color="#d62728", ls="--", lw=1.5,
                   label=f"Tolerance = {TOLERANCE_MM:.1f} mm")
    if np.isfinite(max_success_time):
        ax_top.axvline(max_success_time, color="#333333", ls=":", lw=1.3,
                       label=f"Last reached trial at {max_success_time:.0f} s")

    ax_top.set_ylabel("Distance to target (mm)")
    ax_top.set_title("Distance-to-Target Summary With Active-Trial Guardrail")
    ax_top.grid(True, alpha=0.2)
    ax_top.legend(fontsize=8, loc="upper right")

    ax_bot.plot(t, active, color="#444444", lw=2.0)
    ax_bot.axhline(min_active, color="#9467bd", ls="--", lw=1.4,
                   label=f"Min active threshold = {min_active}")
    ax_bot.set_ylabel("Active trials")
    ax_bot.set_xlabel("Time (s)")
    ax_bot.grid(True, alpha=0.2)
    ax_bot.legend(fontsize=8, loc="upper right")

    ax_bot.set_xlim(0, float(np.max(data.sim_time_s) * 1.02))
    fig.tight_layout()
    fig.savefig(os.path.join(out_dir, "fig2_distance_with_active_counts.png"),
                dpi=300, bbox_inches="tight")
    plt.close(fig)


def _save_fig3_reynolds_outcome(data: BatchData, out_dir: str):
    success = data.reached
    low_re = data.max_re < LOW_RE_THRESHOLD
    fail = ~success

    fig, (ax_sc, ax_bar) = plt.subplots(1, 2, figsize=(14, 5), gridspec_kw={"width_ratios": [2, 1]})

    err_clip = np.clip(data.final_error_mm, 0.0, 15.0)
    sc1 = ax_sc.scatter(
        data.max_re[success], data.sim_time_s[success],
        c=err_clip[success], cmap="viridis", s=42, alpha=0.85,
        edgecolors="k", linewidths=0.25, marker="o", label="Reached"
    )
    if np.any(fail):
        ax_sc.scatter(
            data.max_re[fail], data.sim_time_s[fail],
            c=err_clip[fail], cmap="viridis", s=100, alpha=0.95,
            edgecolors="k", linewidths=0.4, marker="X", label="Not reached"
        )

    ax_sc.axvline(LOW_RE_THRESHOLD, color="#d62728", ls="--", lw=1.6,
                  label=f"Re threshold = {LOW_RE_THRESHOLD}")
    ax_sc.set_yscale("log")
    ax_sc.set_xlabel("Max Reynolds number per trial")
    ax_sc.set_ylabel("Convergence time (s, log scale)")
    ax_sc.set_title("Outcome vs. Physics Regime")
    ax_sc.grid(True, alpha=0.2, which="both")
    ax_sc.legend(fontsize=9, loc="upper left")

    cbar = fig.colorbar(sc1, ax=ax_sc, pad=0.01)
    cbar.set_label("Final error (mm, clipped at 15)")

    categories = [
        ("Reached\nRe < 0.1", int(np.sum(success & low_re)), "#2ca02c"),
        ("Reached\nRe >= 0.1", int(np.sum(success & ~low_re)), "#ff7f0e"),
        ("Not reached\nRe < 0.1", int(np.sum(fail & low_re)), "#1f77b4"),
        ("Not reached\nRe >= 0.1", int(np.sum(fail & ~low_re)), "#d62728"),
    ]
    x = np.arange(len(categories))
    y = np.array([c[1] for c in categories], dtype=int)
    colors = [c[2] for c in categories]
    bars = ax_bar.bar(x, y, color=colors, alpha=0.9)
    ax_bar.set_xticks(x, [c[0] for c in categories], rotation=0)
    ax_bar.set_ylabel("Trials")
    ax_bar.set_ylim(0, max(5, int(np.max(y) + 5)))
    ax_bar.set_title("Count Breakdown")
    ax_bar.grid(True, alpha=0.2, axis="y")

    for b, count in zip(bars, y):
        ax_bar.text(b.get_x() + b.get_width() / 2.0, b.get_height() + 0.5, str(int(count)),
                    ha="center", va="bottom", fontsize=10)

    fig.tight_layout()
    fig.savefig(os.path.join(out_dir, "fig3_reynolds_outcomes.png"), dpi=300, bbox_inches="tight")
    plt.close(fig)


def _ecdf(values: np.ndarray):
    x = np.sort(np.asarray(values, dtype=float))
    y = np.arange(1, len(x) + 1, dtype=float) / float(len(x))
    return x, y


def _save_fig4_error_tail(data: BatchData, out_dir: str):
    all_err = data.final_error_mm
    succ_err = all_err[data.reached]
    fail_err = all_err[~data.reached]

    x_all, y_all = _ecdf(all_err)
    x_succ, y_succ = _ecdf(succ_err)

    fig, (ax_l, ax_r) = plt.subplots(1, 2, figsize=(14, 5))

    ax_l.plot(x_all, y_all, lw=2.2, color="#1f77b4", label="All trials")
    ax_l.plot(x_succ, y_succ, lw=2.2, color="#2ca02c", label="Reached only")
    ax_l.axvline(TOLERANCE_MM, color="#d62728", ls="--", lw=1.4, label="Tolerance")
    ax_l.set_xlim(0, 10.0)
    ax_l.set_ylim(0, 1.02)
    ax_l.set_xlabel("Final error (mm)")
    ax_l.set_ylabel("ECDF")
    ax_l.set_title("Final Error ECDF (Zoom: 0-10 mm)")
    ax_l.grid(True, alpha=0.2)
    ax_l.legend(fontsize=9, loc="lower right")

    bins = np.logspace(np.log10(max(0.05, float(np.min(all_err)))), np.log10(float(np.max(all_err))), 30)
    ax_r.hist(all_err, bins=bins, color="#1f77b4", alpha=0.45, label="All")
    ax_r.hist(succ_err, bins=bins, color="#2ca02c", alpha=0.55, label="Reached")
    if len(fail_err) > 0:
        ax_r.hist(fail_err, bins=bins, color="#d62728", alpha=0.65, label="Not reached")
    ax_r.axvline(TOLERANCE_MM, color="#d62728", ls="--", lw=1.4)
    ax_r.set_xscale("log")
    ax_r.set_xlabel("Final error (mm, log scale)")
    ax_r.set_ylabel("Count")
    ax_r.set_title("Error Tail Visibility")
    ax_r.grid(True, alpha=0.2, which="both")
    ax_r.legend(fontsize=9, loc="upper right")

    fig.tight_layout()
    fig.savefig(os.path.join(out_dir, "fig4_error_distribution_tail.png"), dpi=300, bbox_inches="tight")
    plt.close(fig)


def _save_summary(data: BatchData, out_dir: str):
    n = len(data.reached)
    reached_n = int(np.sum(data.reached))
    low_re_n = int(np.sum(data.max_re < LOW_RE_THRESHOLD))
    reached_low_re_n = int(np.sum(data.reached & (data.max_re < LOW_RE_THRESHOLD)))

    s_lo, s_hi = _wilson_interval(reached_n, n)
    r_lo, r_hi = _wilson_interval(low_re_n, n)

    profile_name = str(data.rows[0].get("profile_name", "unknown_profile")) if data.rows else "unknown_profile"
    diameter_um = data.batch.get("particle_diameter_um", None)
    if diameter_um is None:
        if "50um" in profile_name:
            diameter_um = 50.0
        elif "500um" in profile_name:
            diameter_um = 500.0

    header = f"Improved Poster Metrics ({profile_name})"
    if diameter_um is not None:
        header += f" [{float(diameter_um):.1f} um]"

    summary = (
        f"{header}\n"
        f"{'=' * len(header)}\n"
        f"Trials: {n}\n"
        f"Success: {reached_n}/{n} ({100.0 * reached_n / n:.1f}%)"
        f"  95% CI [{s_lo:.3f}, {s_hi:.3f}]\n"
        f"Low-Re valid (Re<0.1): {low_re_n}/{n} ({100.0 * low_re_n / n:.1f}%)"
        f"  95% CI [{r_lo:.3f}, {r_hi:.3f}]\n"
        f"Reached and low-Re valid: {reached_low_re_n}/{n} ({100.0 * reached_low_re_n / n:.1f}%)\n"
        f"Median convergence time (reached): {np.median(data.sim_time_s[data.reached]):.1f} s\n"
        f"Mean convergence time (reached): {np.mean(data.sim_time_s[data.reached]):.1f} s\n"
        f"Median final error (all): {np.median(data.final_error_mm):.3f} mm\n"
        f"95th percentile final error: {np.percentile(data.final_error_mm, 95):.3f} mm\n"
        f"Worst final error: {np.max(data.final_error_mm):.3f} mm\n"
        f"Median max Re: {np.median(data.max_re):.5f}\n"
        f"Worst max Re: {np.max(data.max_re):.5f}\n"
    )

    with open(os.path.join(out_dir, "stats_summary_improved.txt"), "w") as f:
        f.write(summary)


def main():
    parser = argparse.ArgumentParser(description="Generate improved 50 um poster figures.")
    parser.add_argument("--data-dir", default="outputs/flow_50um_100trials")
    parser.add_argument("--out-dir", default="outputs/poster_figures_50um_improved")
    parser.add_argument("--min-active", type=int, default=20,
                        help="Minimum active trials for reliable distance quantiles")
    args = parser.parse_args()

    os.makedirs(args.out_dir, exist_ok=True)
    data = _load_batch(args.data_dir)

    plt.rcParams.update({
        "font.size": 11,
        "axes.titlesize": 13,
        "axes.labelsize": 11,
        "figure.dpi": 140,
    })

    _save_fig1_km(data, args.out_dir)
    _save_fig2_distance_active(data, args.data_dir, args.out_dir, args.min_active)
    _save_fig3_reynolds_outcome(data, args.out_dir)
    _save_fig4_error_tail(data, args.out_dir)
    _save_summary(data, args.out_dir)

    print(f"Saved improved figures to: {args.out_dir}")
    print("  - fig1_km_convergence.png")
    print("  - fig2_distance_with_active_counts.png")
    print("  - fig3_reynolds_outcomes.png")
    print("  - fig4_error_distribution_tail.png")
    print("  - stats_summary_improved.txt")


if __name__ == "__main__":
    main()
