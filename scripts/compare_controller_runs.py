"""
Compare controllers across stored paired-benchmark runs.

Merges paired_results.csv files from multiple run directories (e.g. the
stored MPC/ACBC baselines and a new COMET run), matches trials by
(flow_scale, pair_id), and reports per-controller aggregates plus paired
COMET-vs-baseline statistics. Baseline result files are only read, never
modified.

Usage:
    python scripts/compare_controller_runs.py \
        --runs outputs/publication_benchmark_1mm/overnight_efficiency_500um \
               outputs/publication_benchmark_1mm/efficiency_500um_flow3_5 \
               outputs/publication_benchmark_1mm/comet_efficiency_500um \
        --comparator comet
"""

from __future__ import annotations

import argparse
import csv
import json
import os
from collections import defaultdict

import numpy as np


METRICS = [
    ("reached_target", "higher"),
    ("sim_time_s", "lower"),
    ("steps_taken", "lower"),
    ("final_error_mm", "lower"),
    ("path_efficiency", "higher"),
    ("energy_proxy_A2s", "lower"),
    ("current_rms_A", "lower"),
    ("min_wall_clearance_mm", "higher"),
    ("clearance_target_met", "higher"),
    ("avg_control_step_time_s", "lower"),
    ("acbc_transition_count", "lower"),
]


def _to_float(value):
    if isinstance(value, str):
        v = value.strip().lower()
        if v in ("true",):
            return 1.0
        if v in ("false",):
            return 0.0
        if v in ("", "nan", "inf"):
            return np.nan
    try:
        return float(value)
    except (TypeError, ValueError):
        return np.nan


def load_rows(run_dirs):
    rows = []
    for run_dir in run_dirs:
        path = os.path.join(run_dir, "paired_results.csv")
        if not os.path.exists(path):
            raise FileNotFoundError(path)
        with open(path) as f:
            for row in csv.DictReader(f):
                row["_run"] = os.path.basename(os.path.normpath(run_dir))
                rows.append(row)
    return rows


def aggregate(rows):
    grouped = defaultdict(list)
    for row in rows:
        grouped[(row["flow_scale"], row["controller_key"])].append(row)

    out = []
    for (flow, controller), group in sorted(grouped.items()):
        entry = {"flow_scale": flow, "controller": controller, "n": len(group)}
        for metric, _ in METRICS:
            values = np.array([_to_float(r.get(metric)) for r in group])
            valid = values[np.isfinite(values)]
            if metric in ("reached_target", "clearance_target_met"):
                entry[metric] = float(np.mean(valid)) if len(valid) else np.nan
            else:
                entry[metric] = float(np.median(valid)) if len(valid) else np.nan
        out.append(entry)
    return out


def paired_stats(rows, comparator):
    try:
        from scipy.stats import wilcoxon
    except Exception:
        wilcoxon = None

    by_key = defaultdict(dict)
    for row in rows:
        by_key[(row["flow_scale"], row["pair_id"])][row["controller_key"]] = row

    baselines = sorted({
        r["controller_key"] for r in rows if r["controller_key"] != comparator
    })

    stats = []
    for baseline in baselines:
        for flow in sorted({r["flow_scale"] for r in rows}):
            pairs = [
                ctrls for (f, _), ctrls in by_key.items()
                if f == flow and baseline in ctrls and comparator in ctrls
            ]
            if not pairs:
                continue
            for metric, better in METRICS:
                base = np.array([_to_float(p[baseline].get(metric)) for p in pairs])
                comp = np.array([_to_float(p[comparator].get(metric)) for p in pairs])
                mask = np.isfinite(base) & np.isfinite(comp)
                if not np.any(mask):
                    continue
                diff = comp[mask] - base[mask]
                sign = 1.0 if better == "higher" else -1.0
                wins = int(np.sum(sign * diff > 0))
                losses = int(np.sum(sign * diff < 0))
                p_value = None
                if wilcoxon is not None and np.any(diff != 0):
                    try:
                        p_value = float(wilcoxon(comp[mask], base[mask]).pvalue)
                    except Exception:
                        p_value = None
                stats.append({
                    "flow_scale": flow,
                    "baseline": baseline,
                    "metric": metric,
                    "better": better,
                    "n_pairs": int(np.sum(mask)),
                    "median_diff": float(np.median(diff)),
                    "comet_wins": wins,
                    "comet_losses": losses,
                    "wilcoxon_p": p_value,
                })
    return stats


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--runs", nargs="+", required=True)
    parser.add_argument("--comparator", default="comet")
    parser.add_argument("--json-out", default=None)
    args = parser.parse_args()

    rows = load_rows(args.runs)

    agg = aggregate(rows)
    print("\n=== Per-controller aggregates (medians; rates for booleans) ===")
    header = ["flow", "controller", "n"] + [m for m, _ in METRICS]
    print(
        f"{'flow':>5} {'controller':<20} {'n':>3} "
        f"{'succ':>5} {'simT':>6} {'steps':>6} {'err_mm':>7} {'patheff':>7} "
        f"{'energy':>8} {'I_rms':>6} {'clr_mm':>6} {'clr_ok':>6} "
        f"{'ctrl_s':>7} {'trans':>5}"
    )
    for e in agg:
        print(
            f"{e['flow_scale']:>5} {e['controller']:<20} {e['n']:>3} "
            f"{e['reached_target']:>5.2f} {e['sim_time_s']:>6.2f} "
            f"{e['steps_taken']:>6.0f} {e['final_error_mm']:>7.2f} "
            f"{e['path_efficiency']:>7.3f} {e['energy_proxy_A2s']:>8.1f} "
            f"{e['current_rms_A']:>6.2f} {e['min_wall_clearance_mm']:>6.2f} "
            f"{e['clearance_target_met']:>6.2f} "
            f"{e['avg_control_step_time_s']:>7.4f} "
            f"{e['acbc_transition_count']:>5.0f}"
        )

    stats = paired_stats(rows, args.comparator)
    print(f"\n=== Paired {args.comparator} vs baselines ===")
    print(
        f"{'flow':>5} {'baseline':<20} {'metric':<26} {'n':>3} "
        f"{'med_diff':>10} {'W/L':>7} {'p':>8}"
    )
    for s in stats:
        p = f"{s['wilcoxon_p']:.1e}" if s["wilcoxon_p"] is not None else "-"
        print(
            f"{s['flow_scale']:>5} {s['baseline']:<20} {s['metric']:<26} "
            f"{s['n_pairs']:>3} {s['median_diff']:>10.3f} "
            f"{s['comet_wins']:>3}/{s['comet_losses']:<3} {p:>8}"
        )

    if args.json_out:
        with open(args.json_out, "w") as f:
            json.dump({"aggregate": agg, "paired": stats}, f, indent=2)
        print(f"\nSaved JSON comparison to {args.json_out}")


if __name__ == "__main__":
    main()
