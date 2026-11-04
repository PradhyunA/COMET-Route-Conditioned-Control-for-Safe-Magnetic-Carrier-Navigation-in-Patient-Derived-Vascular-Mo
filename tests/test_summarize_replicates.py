"""Tests for the replicate confidence-interval summary."""

import numpy as np
import pytest

from scripts.summarize_replicates import outcome_tensor, replicate_ci, route_bootstrap, summarize


def _results(a_safe, b_safe):
    results = {}
    for flow, a, b in zip((1.0, 3.0), a_safe, b_safe):
        results[("a", "abdominal", "route_1", flow)] = a
        results[("b", "abdominal", "route_1", flow)] = b
        results[("a", "pulmonary", "route_2", flow)] = a
        results[("b", "pulmonary", "route_2", flow)] = False
    return results


def test_outcome_tensor_groups_flow_scales_into_one_route():
    outcomes, route_of_case, cases = outcome_tensor({"seed_1": _results((True, True), (False, True))}, ["a", "b"])
    assert outcomes.shape == (2, 1, 4)
    assert route_of_case.tolist() == [0, 0, 1, 1]
    assert outcomes[0].sum() == 4
    assert outcomes[1].sum() == 1


def test_outcome_tensor_refuses_a_missing_case():
    results = _results((True, True), (True, True))
    del results[("b", "pulmonary", "route_2", 3.0)]
    with pytest.raises(ValueError, match="Missing"):
        outcome_tensor({"seed_1": results}, ["a", "b"])


def test_route_bootstrap_has_no_spread_when_every_route_agrees():
    outcomes = np.ones((1, 2, 6))
    boot = route_bootstrap(outcomes, np.array([0, 0, 1, 1, 2, 2]), n_boot=100, seed=0)
    assert boot.shape == (1, 100)
    assert np.all(boot == 1.0)


def test_replicate_ci_matches_the_t_interval():
    mean, half = replicate_ci([10, 12, 14])
    assert mean == 12
    assert half == pytest.approx(4.303 * 2 / np.sqrt(3), abs=1e-2)


def test_summarize_reports_the_paired_gain():
    outcomes, route_of_case, _ = outcome_tensor({"seed_1": _results((True, True), (False, True))}, ["a", "b"])
    rows = summarize(outcomes, route_of_case, ["a", "b"], "a", n_boot=200, seed=0)
    assert rows[1]["reference_only"] == 3
    assert rows[1]["other_only"] == 0
    assert rows[1]["rate_gain_of_reference"] == pytest.approx(0.75)
