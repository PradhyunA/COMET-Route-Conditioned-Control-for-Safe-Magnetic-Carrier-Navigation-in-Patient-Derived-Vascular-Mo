"""Tests for result reporting schema compatibility."""

import csv

import numpy as np

from dynamag_simulation.core.reporting import TrialResult, save_results_csv


def test_trial_summary_includes_legacy_and_finite_re_fields(tmp_path):
    result = TrialResult(
        trial_name="trial",
        start=np.zeros(3),
        target=np.ones(3),
        final_position=np.ones(3),
        trajectory=np.array([np.zeros(3), np.ones(3)]),
        distance_log=np.array([1.0, 0.0]),
        current_log=np.zeros((1, 9)),
        field_log=np.array([0.1]),
        force_log=np.array([1e-6]),
        velocity_log=np.array([0.1]),
        reynolds_log=np.array([190.0]),
        cost_log=np.array([0.0]),
        step_time_log=np.array([0.01]),
        reached=True,
        steps_taken=1,
        final_error=0.0,
        max_current_used=0.0,
        mean_reynolds=190.0,
        max_reynolds=190.0,
        low_re_valid=False,
        stokes_regime=False,
        finite_re_drag_valid=True,
        drag_model="schiller_naumann",
        drag_regime_label="finite_re_schiller_naumann",
    )

    save_results_csv([result], str(tmp_path))

    with open(tmp_path / "trial_summary.csv", newline="") as f:
        row = next(csv.DictReader(f))

    assert row["low_re_valid"] == "False"
    assert row["stokes_regime"] == "False"
    assert row["finite_re_drag_valid"] == "True"
    assert row["drag_model"] == "schiller_naumann"
    assert row["drag_regime_label"] == "finite_re_schiller_naumann"
