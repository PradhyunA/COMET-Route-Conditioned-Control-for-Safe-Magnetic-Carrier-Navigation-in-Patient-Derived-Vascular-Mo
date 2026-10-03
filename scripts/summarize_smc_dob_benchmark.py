"""Summarize a completed SMC-DOB-inspired three-anatomy benchmark."""

from __future__ import annotations

import argparse
import csv
import json
import statistics
from pathlib import Path


CONTROLLER_KEY = "smc_dob_inspired"


def _truth(value):
    return str(value).strip().lower() in {"true", "1", "yes"}


def _metrics(rows):
    errors = [float(row["final_error_mm"]) for row in rows]
    steps = [int(float(row["steps_taken"])) for row in rows]
    successes = sum(_truth(row["reached_target"]) for row in rows)
    return {
        "n_cases": len(rows),
        "success_count": successes,
        "success_rate": successes / len(rows) if rows else 0.0,
        "median_final_error_mm": statistics.median(errors) if errors else None,
        "max_final_error_mm": max(errors) if errors else None,
        "median_steps": statistics.median(steps) if steps else None,
        "mean_effective_steps": statistics.fmean(steps) if steps else None,
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("benchmark_root", type=Path)
    parser.add_argument("--expect-cases", type=int, default=45)
    args = parser.parse_args()

    paths = sorted(args.benchmark_root.glob("*/paired_results.csv"))
    rows = []
    for path in paths:
        with path.open(newline="") as f:
            for row in csv.DictReader(f):
                row["anatomy_run_id"] = path.parent.name
                rows.append(row)

    if len(rows) != args.expect_cases:
        raise RuntimeError(
            f"Expected {args.expect_cases} cases, found {len(rows)} in "
            f"{args.benchmark_root}"
        )
    controllers = {row["controller_key"] for row in rows}
    if controllers != {CONTROLLER_KEY}:
        raise RuntimeError(f"Expected only {CONTROLLER_KEY}, found {controllers}")

    flow_5x = [row for row in rows if float(row["flow_scale"]) == 5.0]
    by_anatomy = {
        anatomy: _metrics([row for row in rows if row["anatomy_run_id"] == anatomy])
        for anatomy in sorted({row["anatomy_run_id"] for row in rows})
    }
    report = {
        "controller_key": CONTROLLER_KEY,
        "controller_description": (
            "Adapted published-style SMC-DOB baseline based on Bhattacharjee "
            "et al. (2026); not an exact reproduction. The source paper used "
            "a different 2-D in-vitro phantom and actuation/localization system."
        ),
        "overall": _metrics(rows),
        "flow_5x": _metrics(flow_5x),
        "by_anatomy": by_anatomy,
    }

    output_path = args.benchmark_root / "smc_dob_inspired_summary.json"
    output_path.write_text(json.dumps(report, indent=2) + "\n")

    overall = report["overall"]
    high_flow = report["flow_5x"]
    print("SMC-DOB-inspired benchmark summary")
    print(
        f"Overall: success {overall['success_count']}/{overall['n_cases']}; "
        f"median final error {overall['median_final_error_mm']:.3f} mm; "
        f"max final error {overall['max_final_error_mm']:.3f} mm; "
        f"median steps {overall['median_steps']:.1f}; "
        f"mean effective steps {overall['mean_effective_steps']:.1f}"
    )
    print(
        f"5x flow: success {high_flow['success_count']}/{high_flow['n_cases']}; "
        f"median final error {high_flow['median_final_error_mm']:.3f} mm; "
        f"max final error {high_flow['max_final_error_mm']:.3f} mm; "
        f"median steps {high_flow['median_steps']:.1f}; "
        f"mean effective steps {high_flow['mean_effective_steps']:.1f}"
    )
    print(f"Report: {output_path}")


if __name__ == "__main__":
    main()
