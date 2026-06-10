"""Tests for paired publication benchmark bookkeeping."""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))

from scripts.run_publication_benchmark import (
    CONTROLLER_VARIANTS,
    _paired_difference_rows,
)


def _row(controller, reached, steps):
    return {
        "flow_scale": 1.0,
        "pair_id": "pair_1",
        "trial_kind": "random",
        "controller_key": controller,
        "reached_target": reached,
        "steps_taken": steps,
        "sim_time_s": steps * 0.02,
        "wall_time_s": float(steps),
        "avg_control_step_time_s": 1.0,
        "path_efficiency": 0.9,
        "current_rms_A": 2.0,
        "energy_proxy_A2s": 3.0,
        "min_wall_clearance_mm": 0.7,
        "min_surface_clearance_mm": 0.45,
    }


def test_five_primary_controller_variants_are_available():
    assert {
        "naive_mpc",
        "mpc_default",
        "mpc_tuned",
        "acbc_same_weights",
        "acbc_tuned",
    }.issubset(CONTROLLER_VARIANTS)


def test_paired_differences_match_trials_by_flow_and_pair():
    rows = [
        _row("mpc_default", True, 100),
        _row("acbc_same_weights", True, 90),
    ]
    differences = _paired_difference_rows(rows)
    step_difference = next(
        row for row in differences if row["metric"] == "steps_taken"
    )

    assert step_difference["baseline"] == "mpc_default"
    assert step_difference["comparator"] == "acbc_same_weights"
    assert step_difference["paired_difference"] == -10.0
