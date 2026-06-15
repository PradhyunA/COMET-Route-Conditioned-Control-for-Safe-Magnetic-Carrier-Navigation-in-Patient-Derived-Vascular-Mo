"""Regression tests for physically feasible vessel-network routes."""

import pickle
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))

from dynamag_simulation.core.comet_controller import COMETController
from dynamag_simulation.core.real_vessel_network import (
    RealVesselNetwork,
    RoutePlanningOptions,
)
from scripts.run_publication_benchmark import _wall_clearance_stats


def _network(tmp_path):
    node = np.array([0.0, 0.0, 0.0])
    upper = np.array([0.0, 0.010, 0.0])
    end = np.array([0.010, 0.0, 0.0])
    segments = [
        {"seg_id": 0, "start": [-0.010, 0.0, 0.0], "end": node, "radius": 0.0012},
        {"seg_id": 1, "start": node, "end": end, "radius": 0.0006},
        {"seg_id": 2, "start": node, "end": upper, "radius": 0.0012},
        {"seg_id": 3, "start": upper, "end": end, "radius": 0.0012},
    ]
    path = tmp_path / "network.pkl"
    with path.open("wb") as handle:
        pickle.dump(segments, handle)
    return RealVesselNetwork(path)


def test_route_rejects_short_branch_below_clearance(tmp_path):
    network = _network(tmp_path)
    options = RoutePlanningOptions(wall_clearance_m=0.0007)
    target = np.array([0.009, 0.001, 0.0])

    plan = network.plan_route(np.array([-0.009, 0.0, 0.0]), target, options)

    assert plan.feasible
    assert plan.segment_ids == [0, 2, 3]
    assert plan.min_center_clearance_m == 0.0012
    np.testing.assert_allclose(plan.waypoints[0], [0.0, 0.0, 0.0])
    np.testing.assert_allclose(plan.waypoints[1], [0.0, 0.010, 0.0])


def test_carrier_surface_clearance_can_be_more_restrictive(tmp_path):
    network = _network(tmp_path)
    options = RoutePlanningOptions(
        wall_clearance_m=0.0007,
        particle_radius_m=0.0005,
        minimum_surface_clearance_m=0.0008,
    )

    plan = network.plan_route(
        np.array([-0.009, 0.0, 0.0]),
        np.array([0.009, 0.001, 0.0]),
        options,
    )

    assert not plan.feasible
    assert options.required_center_clearance_m == 0.0013


def test_route_rejects_adverse_branch_beyond_magnetic_authority(tmp_path):
    node = np.array([0.0, 0.0, 0.0])
    upper = np.array([0.0, 0.010, 0.0])
    end = np.array([0.010, 0.0, 0.0])
    segments = [
        {"seg_id": 0, "start": [-0.010, 0.0, 0.0], "end": node, "radius": 0.0012},
        {"seg_id": 1, "start": end, "end": node, "radius": 0.0012, "v_mean": 0.10},
        {"seg_id": 2, "start": node, "end": upper, "radius": 0.0012},
        {"seg_id": 3, "start": upper, "end": end, "radius": 0.0012},
    ]
    path = tmp_path / "authority_network.pkl"
    with path.open("wb") as handle:
        pickle.dump(segments, handle)
    network = RealVesselNetwork(path)
    options = RoutePlanningOptions(
        wall_clearance_m=0.0007,
        magnetic_authority_m_s=0.05,
        minimum_authority_reserve_m_s=0.0,
    )

    plan = network.plan_route(
        np.array([-0.009, 0.0, 0.0]),
        np.array([0.009, 0.001, 0.0]),
        options,
    )

    assert plan.feasible
    assert plan.segment_ids == [0, 2, 3]
    assert plan.min_authority_reserve_m_s >= 0.0


def test_comet_bifurcations_follow_selected_route_topology(tmp_path):
    network = _network(tmp_path)
    navigator = network.waypoint_navigator(
        np.array([-0.009, 0.0, 0.0]),
        np.array([0.009, 0.001, 0.0]),
        planning_options=RoutePlanningOptions(wall_clearance_m=0.0007),
    )

    class Magnets:
        n_magnets = 1
        max_current = 1.0

    controller = COMETController(network, Magnets(), workspace=None)
    controller._ensure_route(navigator, np.array([-0.009, 0.0, 0.0]))

    assert len(controller._route_nodes) == 1
    assert controller._route_nodes[0][2] == [0, 1, 2]


def test_wall_clearance_uses_local_segment(tmp_path):
    network = _network(tmp_path)
    trajectory = np.array([
        [-0.009, 0.0, 0.0],
        [-0.005, 0.0002, 0.0],
    ])

    assert np.isclose(_wall_clearance_stats(network, trajectory), 0.0010)
