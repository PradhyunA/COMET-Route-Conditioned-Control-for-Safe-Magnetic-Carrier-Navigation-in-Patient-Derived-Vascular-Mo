"""
Generate publication figures and manuscript tables for the 1 mm ACBC study.

Input is a completed run directory from run_publication_benchmark_1mm.py.
"""

from __future__ import annotations

import argparse
import csv
import glob
import json
import os
import tempfile
from collections import defaultdict

os.environ.setdefault(
    "MPLCONFIGDIR",
    os.path.join(tempfile.gettempdir(), "dynamag_matplotlib"),
)

import matplotlib.pyplot as plt
import numpy as np


COLORS = {
    "mpc_default": "#7a7a7a",
    "mpc_tuned": "#1f77b4",
    "acbc_same_weights": "#2ca02c",
    "acbc_tuned": "#d62728",
}

STATE_COLORS = {
    "FREE_NAVIGATION": "#8c8c8c",
    "VIABILITY_ASSESSMENT": "#1f77b4",
    "BRANCH_COMMITMENT": "#2ca02c",
    "RECOVERY_MODE": "#ff7f0e",
    "STALL_ESCAPE": "#d62728",
}


def _read_rows(path):
    rows = []
    with open(path, newline="") as f:
        for row in csv.DictReader(f):
            rows.append(row)
    return rows


def _as_float(row, key):
    try:
        return float(row[key])
    except Exception:
        return np.nan


def _as_bool(row, key):
    return str(row.get(key, "")).lower() == "true"


def _controllers(rows):
    keys = []
    for row in rows:
        key = row["controller_key"]
        if key not in keys:
            keys.append(key)
    return keys


def _labels(rows):
    return {
        row["controller_key"]: row["controller_label"]
        for row in rows
    }


def _savefig(path):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    plt.tight_layout()
    plt.savefig(path, dpi=300)
    plt.close()


def fig_distributions(rows, out_dir):
    labels = _labels(rows)
    controllers = _controllers(rows)
    metrics = [
        ("sim_time_s", "Time to target / cutoff (s)"),
        ("final_error_mm", "Final error (mm)"),
        ("path_efficiency", "Path efficiency"),
    ]

    fig, axes = plt.subplots(1, 3, figsize=(13, 4))
    for ax, (metric, ylabel) in zip(axes, metrics):
        data = [
            [_as_float(r, metric) for r in rows if r["controller_key"] == key]
            for key in controllers
        ]
        ax.boxplot(data, labels=[labels[k] for k in controllers], showfliers=False)
        ax.set_ylabel(ylabel)
        ax.tick_params(axis="x", rotation=25)
        ax.grid(alpha=0.25)

    _savefig(os.path.join(out_dir, "fig01_controller_distributions.png"))


def fig_paired_improvement(rows, out_dir, baseline="mpc_tuned", comparator="acbc_tuned"):
    by_pair = defaultdict(dict)
    for row in rows:
        by_pair[(row["flow_scale"], row["pair_id"])][row["controller_key"]] = row

    flows = sorted({row["flow_scale"] for row in rows}, key=float)
    fig, axes = plt.subplots(1, len(flows), figsize=(5 * len(flows), 4), squeeze=False)

    for ax, flow in zip(axes[0], flows):
        diffs = []
        for (row_flow, _), controllers in by_pair.items():
            if row_flow != flow:
                continue
            if baseline in controllers and comparator in controllers:
                base = _as_float(controllers[baseline], "final_error_mm")
                comp = _as_float(controllers[comparator], "final_error_mm")
                diffs.append(base - comp)
        ax.axhline(0, color="#333333", lw=1)
        ax.scatter(range(len(diffs)), diffs, s=22, color="#d62728", alpha=0.8)
        ax.set_title(f"Flow {flow}x")
        ax.set_ylabel("MPC-tuned error - ACBC-tuned error (mm)")
        ax.set_xlabel("Paired trial")
        ax.grid(alpha=0.25)

    _savefig(os.path.join(out_dir, "fig02_paired_error_improvement.png"))


def fig_kaplan_meier(rows, out_dir):
    labels = _labels(rows)
    controllers = _controllers(rows)

    plt.figure(figsize=(7, 5))
    for key in controllers:
        group = [r for r in rows if r["controller_key"] == key]
        times = np.array([_as_float(r, "sim_time_s") for r in group], dtype=float)
        reached = np.array([_as_bool(r, "reached_target") for r in group], dtype=bool)
        order = np.argsort(times)
        times = times[order]
        reached = reached[order]
        n = len(times)
        cumulative = []
        for t in times:
            cumulative.append(float(np.sum((times <= t) & reached) / n) if n else 0.0)
        plt.step(times, cumulative, where="post", label=labels[key],
                 color=COLORS.get(key), lw=2)

    plt.xlabel("Simulation time (s)")
    plt.ylabel("Fraction reached")
    plt.ylim(-0.02, 1.02)
    plt.grid(alpha=0.25)
    plt.legend()
    _savefig(os.path.join(out_dir, "fig03_time_to_target_curve.png"))


def fig_wall_clearance(rows, out_dir):
    labels = _labels(rows)
    controllers = _controllers(rows)
    data = [
        [_as_float(r, "min_wall_clearance_mm") for r in rows if r["controller_key"] == key]
        for key in controllers
    ]
    plt.figure(figsize=(7, 5))
    plt.boxplot(data, labels=[labels[k] for k in controllers], showfliers=False)
    plt.axhline(0, color="#333333", lw=1)
    plt.ylabel("Minimum wall clearance (mm)")
    plt.xticks(rotation=25)
    plt.grid(alpha=0.25)
    _savefig(os.path.join(out_dir, "fig04_wall_clearance.png"))


def fig_current_thermal(rows, out_dir):
    labels = _labels(rows)
    controllers = _controllers(rows)
    fig, axes = plt.subplots(1, 3, figsize=(13, 4))
    metrics = [
        ("current_rms_A", "RMS current (A)"),
        ("energy_proxy_A2s", "Energy proxy (A^2 s)"),
        ("max_power_W", "Max coil power (W)"),
    ]
    for ax, (metric, ylabel) in zip(axes, metrics):
        data = [
            [_as_float(r, metric) for r in rows if r["controller_key"] == key]
            for key in controllers
        ]
        ax.boxplot(data, labels=[labels[k] for k in controllers], showfliers=False)
        ax.set_ylabel(ylabel)
        ax.tick_params(axis="x", rotation=25)
        ax.grid(alpha=0.25)
    _savefig(os.path.join(out_dir, "fig05_current_power_thermal.png"))


def fig_ablation(rows, out_dir):
    ablation_rows = [
        r for r in rows
        if r["controller_key"].startswith("acbc_")
    ]
    if not ablation_rows:
        return

    grouped = defaultdict(list)
    for row in ablation_rows:
        grouped[row["controller_key"]].append(row)

    keys = sorted(grouped)
    medians = [np.median([_as_float(r, "final_error_mm") for r in grouped[k]]) for k in keys]
    labels = {
        rows[0]["controller_key"]: rows[0]["controller_label"]
        for rows in grouped.values()
    }

    plt.figure(figsize=(9, 5))
    plt.bar(range(len(keys)), medians, color=[COLORS.get(k, "#8c8c8c") for k in keys])
    plt.xticks(range(len(keys)), [labels[k] for k in keys], rotation=30, ha="right")
    plt.ylabel("Median final error (mm)")
    plt.grid(axis="y", alpha=0.25)
    _savefig(os.path.join(out_dir, "fig06_ablation_summary.png"))


def fig_finite_re(rows, out_dir):
    controllers = _controllers(rows)
    labels = _labels(rows)
    plt.figure(figsize=(8, 5))
    for key in controllers:
        group = [r for r in rows if r["controller_key"] == key]
        x = np.array([_as_float(r, "max_reynolds") for r in group], dtype=float)
        y = np.array([_as_float(r, "sim_time_s") for r in group], dtype=float)
        plt.scatter(x, y, s=24, alpha=0.7, label=labels[key], color=COLORS.get(key))
    plt.axvline(0.1, color="#666666", ls="--", lw=1.2, label="Stokes threshold")
    plt.axvline(800.0, color="#111111", ls=":", lw=1.5, label="Schiller-Naumann range")
    plt.xlabel("Max Reynolds number")
    plt.ylabel("Simulation time (s)")
    plt.xscale("log")
    plt.grid(alpha=0.25)
    plt.legend()
    _savefig(os.path.join(out_dir, "fig07_finite_re_regime.png"))


def fig_state_timeline(run_dir, out_dir):
    paths = sorted(glob.glob(os.path.join(run_dir, "**", "acbc_diagnostics.csv"), recursive=True))
    if not paths:
        return

    rows = _read_rows(paths[0])
    if not rows:
        return

    first_trial = rows[0]["trial_name"]
    rows = [r for r in rows if r["trial_name"] == first_trial]
    times = [_as_float(r, "time_s") for r in rows]
    states = [r["state"] for r in rows]

    plt.figure(figsize=(10, 2.8))
    for i, (t, state) in enumerate(zip(times, states)):
        t_next = times[i + 1] if i + 1 < len(times) else t + 0.02
        plt.axvspan(t, t_next, color=STATE_COLORS.get(state, "#cccccc"), alpha=0.9)

    handles = [
        plt.Rectangle((0, 0), 1, 1, color=color)
        for color in STATE_COLORS.values()
    ]
    plt.legend(handles, list(STATE_COLORS), loc="upper center", ncol=3, fontsize=8)
    plt.yticks([])
    plt.xlabel("Simulation time (s)")
    plt.title(f"ACBC state timeline: {first_trial}")
    _savefig(os.path.join(out_dir, "fig08_acbc_state_timeline.png"))


def _write_csv(path, header, rows):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=header)
        writer.writeheader()
        for row in rows:
            writer.writerow(row)


def write_tables(rows, run_config, out_dir):
    labels = _labels(rows)
    grouped = defaultdict(list)
    for row in rows:
        grouped[row["controller_key"]].append(row)

    summary_rows = []
    for key, group in sorted(grouped.items()):
        summary_rows.append({
            "controller": labels[key],
            "n": len(group),
            "success_rate": np.mean([_as_bool(r, "reached_target") for r in group]),
            "median_final_error_mm": np.median([_as_float(r, "final_error_mm") for r in group]),
            "median_sim_time_s": np.median([_as_float(r, "sim_time_s") for r in group]),
            "median_min_wall_clearance_mm": np.median([
                _as_float(r, "min_wall_clearance_mm") for r in group
            ]),
            "finite_re_drag_valid_rate": np.mean([
                _as_bool(r, "finite_re_drag_valid") for r in group
            ]),
        })

    _write_csv(
        os.path.join(out_dir, "table_benchmark_summary.csv"),
        list(summary_rows[0]) if summary_rows else ["controller"],
        summary_rows,
    )

    variant_rows = []
    for key, variant in run_config.get("variants", {}).items():
        row = {
            "controller_key": key,
            "label": variant["label"],
            "kind": variant["kind"],
        }
        row.update(variant.get("kwargs", {}))
        variant_rows.append(row)

    headers = sorted({k for row in variant_rows for k in row})
    _write_csv(os.path.join(out_dir, "table_controller_parameters.csv"), headers, variant_rows)

    limitation_rows = [
        {"item": "Carrier type", "assumption": "1 mm magnetic carrier, not nanoparticles"},
        {"item": "Drag", "assumption": "Schiller-Naumann finite-Re drag; valid target Re <= 800"},
        {"item": "Hydrodynamics", "assumption": "No added mass, history force, rolling/sliding, or near-wall correction"},
        {"item": "Blood", "assumption": "Newtonian viscosity approximation"},
        {"item": "Magnetics", "assumption": "Regularized dipole model, not FEM-calibrated hardware"},
        {"item": "Safety", "assumption": "Simulation benchmark only; no human/animal/device validation"},
    ]
    _write_csv(
        os.path.join(out_dir, "table_limitations_validity.csv"),
        ["item", "assumption"],
        limitation_rows,
    )


def generate(run_dir, out_dir):
    paired_path = os.path.join(run_dir, "paired_results.csv")
    config_path = os.path.join(run_dir, "run_config.json")
    rows = _read_rows(paired_path)
    with open(config_path) as f:
        run_config = json.load(f)

    os.makedirs(out_dir, exist_ok=True)
    fig_distributions(rows, out_dir)
    fig_paired_improvement(rows, out_dir)
    fig_kaplan_meier(rows, out_dir)
    fig_wall_clearance(rows, out_dir)
    fig_current_thermal(rows, out_dir)
    fig_ablation(rows, out_dir)
    fig_finite_re(rows, out_dir)
    fig_state_timeline(run_dir, out_dir)
    write_tables(rows, run_config, out_dir)

    print(f"Publication figures/tables saved to: {out_dir}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("run_dir", help="Benchmark run directory")
    parser.add_argument("--out-dir", default=None)
    args = parser.parse_args()

    out_dir = args.out_dir or os.path.join(args.run_dir, "publication_figures")
    generate(args.run_dir, out_dir)


if __name__ == "__main__":
    main()
