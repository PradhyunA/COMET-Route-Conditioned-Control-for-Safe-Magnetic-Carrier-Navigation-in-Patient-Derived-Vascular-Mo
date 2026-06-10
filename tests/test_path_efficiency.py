"""Tests for vessel-route-relative path efficiency."""

import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))

from scripts.run_publication_benchmark import _path_efficiency_metrics


class _RouteNetwork:
    def plan_waypoints(self, start, target):
        return [
            np.array([1.0, 0.0, 0.0]),
            np.asarray(target, dtype=float),
        ]


def test_path_efficiency_uses_planned_vessel_route():
    metrics = _path_efficiency_metrics(
        network=_RouteNetwork(),
        start=np.array([0.0, 0.0, 0.0]),
        target=np.array([1.0, 1.0, 0.0]),
        trajectory=np.array([
            [0.0, 0.0, 0.0],
            [1.0, 0.0, 0.0],
            [1.0, 1.0, 0.0],
        ]),
    )

    assert np.isclose(metrics["planned_route_length_m"], 2.0)
    assert np.isclose(metrics["path_length_m"], 2.0)
    assert np.isclose(metrics["path_efficiency"], 1.0)
    assert np.isclose(metrics["euclidean_path_efficiency"], np.sqrt(2.0) / 2.0)


def test_path_efficiency_penalizes_distance_beyond_planned_route():
    metrics = _path_efficiency_metrics(
        network=_RouteNetwork(),
        start=np.array([0.0, 0.0, 0.0]),
        target=np.array([1.0, 1.0, 0.0]),
        trajectory=np.array([
            [0.0, 0.0, 0.0],
            [0.0, -0.5, 0.0],
            [1.0, -0.5, 0.0],
            [1.0, 1.0, 0.0],
        ]),
    )

    assert np.isclose(metrics["path_length_m"], 3.0)
    assert np.isclose(metrics["route_length_ratio"], 2.0 / 3.0)
    assert np.isclose(metrics["path_efficiency"], 2.0 / 3.0)
