"""
Summarize a run_physics_sensitivity_sweep.sh output directory.

Prints success counts per condition x controller (overall, per flow scale,
per anatomy) plus safe success (reached, clearance criterion met, never
left the vessel), median final error, early stops, wall exits and peak
carrier speed, and writes a combined CSV next to the sweep.
"""

import argparse
import csv
import json
from collections import defaultdict
from pathlib import Path
from statistics import median


def _load_rows(sweep_dir):
    rows = []
    for path in sorted(Path(sweep_dir).glob("*/*/paired_results.csv")):
        condition, anatomy = path.parent.parent.name, path.parent.name
        config = json.loads((path.parent / "run_config.json").read_text())
        for row in csv.DictReader(path.open()):
            row["condition"] = condition
            row["anatomy"] = anatomy
            row["core_mu_r"] = config["common_kwargs"]["core_relative_permeability"]
            row["particle_density_kg_m3"] = config["profile"]["particle_density_kg_m3"]
            row["gravity_m_s2"] = config["profile"]["gravity_m_s2"]
            rows.append(row)
    return rows


def _safe_success(row):
    return (
        row["reached_target"] == "True"
        and row.get("clearance_target_met") == "True"
        and row.get("wall_exit") != "True"
    )


def _fmt_counts(rows, key):
    groups = defaultdict(list)
    for row in rows:
        groups[row[key]].append(row["reached_target"] == "True")
    return " ".join(
        f"{k}:{sum(v)}/{len(v)}" for k, v in sorted(groups.items())
    )


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("sweep_dir")
    args = parser.parse_args()

    rows = _load_rows(args.sweep_dir)
    if not rows:
        raise SystemExit(f"No paired_results.csv under {args.sweep_dir}")

    by_cond = defaultdict(list)
    for row in rows:
        by_cond[(row["condition"], row["controller_key"])].append(row)

    header = (
        f"{'condition':14s} {'ctrl':10s} {'mu_r':>5s} {'rho_p':>6s} "
        f"{'success':>8s} {'safe':>6s} {'med err mm':>10s} {'early':>5s} {'exit':>4s} "
        f"{'peak v mm/s':>11s}  by flow | by anatomy"
    )
    print(header)
    print("-" * len(header))
    for (condition, controller), group in sorted(by_cond.items()):
        ok = sum(r["reached_target"] == "True" for r in group)
        safe = sum(_safe_success(r) for r in group)
        errors = [float(r["final_error_mm"]) for r in group]
        early = sum(r["early_stopped"] == "True" for r in group)
        exits = sum(r["wall_exit"] == "True" for r in group)
        peak_v = median(float(r["peak_velocity_m_s"]) * 1e3 for r in group)
        rho = group[0]["particle_density_kg_m3"]
        print(
            f"{condition:14s} {controller:10s} {group[0]['core_mu_r']:5.0f} "
            f"{(f'{rho:.0f}' if rho else '-'):>6s} {ok:3d}/{len(group):<4d} {safe:6d} "
            f"{median(errors):10.3f} {early:5d} {exits:4d} {peak_v:11.2f}  "
            f"{_fmt_counts(group, 'flow_scale')} | {_fmt_counts(group, 'anatomy')}"
        )

    out = Path(args.sweep_dir) / "combined_results.csv"
    fields = list(rows[0].keys())
    with out.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)
    print(f"\nCombined CSV: {out}")


if __name__ == "__main__":
    main()
