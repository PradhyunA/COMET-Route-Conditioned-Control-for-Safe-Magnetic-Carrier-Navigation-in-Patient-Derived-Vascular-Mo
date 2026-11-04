"""Tests for route de-spurring, along-route progress, and the tracking NMPC."""

import pickle
import sys
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))

from dynamag_simulation.core.comet_controller import COMETConfig, COMETController
from dynamag_simulation.core.magnets import ElectromagnetArray3D
from dynamag_simulation.core.particle import MagneticParticle3D
from dynamag_simulation.core.profiles import SPION_CARRIER_500UM
from dynamag_simulation.core.real_vessel_network import RealVesselNetwork
from dynamag_simulation.core.route_progress import RouteRemaining
from dynamag_simulation.core.tracking_mpc import TrackingMPC, TrackingMPCConfig
from scripts.run_publication_benchmark import ACTUATOR_PRESETS, _jitter_start
from simulation_3d_comet import (
    FIXED_COIL_LENGTH,
    FIXED_COIL_RADIUS,
    FIXED_CORE_RADIUS,
    FIXED_CORE_SATURATION_T,
    FIXED_MAX_CURRENT,
    FIXED_PLACEMENT_RADIUS,
    FIXED_TURNS,
)


def _straight_network(tmp_path, radius=0.004, n=40, spacing=0.002):
    segments = [
        {"seg_id": i, "start": np.array([0.0, 0.0, i * spacing]),
         "end": np.array([0.0, 0.0, (i + 1) * spacing]), "radius": radius,
         "v_mean": 0.0, "level": 0}
        for i in range(n)
    ]
    path = tmp_path / "straight.pkl"
    with open(path, "wb") as f:
        pickle.dump(segments, f)
    return RealVesselNetwork(str(path))


def test_backtrack_removal_cuts_out_and_back_spur(tmp_path):
    network = _straight_network(tmp_path)
    z = lambda v: np.array([0.0, 0.0, v])  # noqa: E731
    # Route runs to 40 mm, doubles back to 20 mm, then on to 60 mm.
    route = [z(0.0), z(0.01), z(0.02), z(0.03), z(0.04), z(0.03), z(0.02), z(0.03), z(0.05), z(0.06)]
    cleaned = network.remove_route_backtracks(route, required_clearance=0.001)
    steps = np.diff(np.asarray(cleaned)[:, 2])
    assert np.all(steps > 0)
    assert cleaned[0][2] == 0.0 and cleaned[-1][2] == pytest.approx(0.06)


def test_backtrack_removal_keeps_route_when_chord_leaves_lumen(tmp_path):
    network = _straight_network(tmp_path, radius=0.0012)
    route = [np.array([0.0, 0.0, 0.01]), np.array([0.0, 0.0, 0.02]), np.array([0.0, 0.0, 0.01])]
    cleaned = network.remove_route_backtracks(route, required_clearance=0.005)
    assert len(cleaned) == len(route)


def test_route_remaining_follows_the_polyline():
    remaining = RouteRemaining([[0, 0, 0], [0.01, 0, 0], [0.01, 0.01, 0]])
    assert remaining([0.0, 0.0, 0.0]) == pytest.approx(0.02)
    assert remaining([0.01, 0.005, 0.0]) == pytest.approx(0.005)
    # Straight-line distance to the end is ~0.0112 at (0.005, 0, 0); route left is 0.015.
    assert remaining([0.005, 0.0, 0.0]) == pytest.approx(0.015)


def test_tracking_nmpc_moves_carrier_along_route(tmp_path):
    network = _straight_network(tmp_path, radius=0.006)
    magnets = ElectromagnetArray3D(
        coil_radius=FIXED_COIL_RADIUS, coil_length=FIXED_COIL_LENGTH, turns=FIXED_TURNS,
        max_current=FIXED_MAX_CURRENT, placement_radius=FIXED_PLACEMENT_RADIUS,
        center=np.array([0.0, 0.0, 0.01]), core_radius=FIXED_CORE_RADIUS,
        core_saturation_tesla=FIXED_CORE_SATURATION_T, **ACTUATOR_PRESETS["high_gradient_400"],
    )
    particle = MagneticParticle3D(np.array([0.0, 0.0, 0.005]), profile=SPION_CARRIER_500UM)
    mpc = TrackingMPC(network, [[0, 0, 0.005], [0, 0, 0.05]], wall_clearance_m=0.0007,
                      current_limit=FIXED_MAX_CURRENT,
                      config=TrackingMPCConfig(reference_speed_m_s=0.05))
    currents, force, _, _ = mpc.command(particle=particle, particle_pos=particle.pos,
                                        magnet_array=magnets, dt=0.02, flow=network)
    assert np.all(np.abs(currents) <= FIXED_MAX_CURRENT + 1e-9)
    assert force[2] > 0.0
    assert abs(force[2]) > np.linalg.norm(force[:2])


def test_start_jitter_is_reproducible_and_stays_clear_of_the_wall(tmp_path):
    network = _straight_network(tmp_path, radius=0.004)
    pair = {"pair_id": "p", "start": np.array([0.0, 0.0, 0.03]), "target": np.array([0.0, 0.0, 0.06])}
    draw = lambda seed: _jitter_start(network, pair, 0.0005, 0.0007, np.random.default_rng(seed))  # noqa: E731
    first, again, other = draw(1), draw(1), draw(2)
    assert np.array_equal(first["start"], again["start"])
    assert not np.array_equal(first["start"], other["start"])
    assert first["start"] is not pair["start"] and np.array_equal(pair["start"], [0.0, 0.0, 0.03])
    assert network.lumen_clearance(first["start"]) >= 0.0007


def test_start_jitter_keeps_the_release_point_when_no_draw_is_clear(tmp_path):
    network = _straight_network(tmp_path, radius=0.001)
    pair = {"pair_id": "p", "start": np.array([0.0, 0.0, 0.03]), "target": np.array([0.0, 0.0, 0.06])}
    assert _jitter_start(network, pair, 0.0005, 0.005, np.random.default_rng(0)) is pair


def _fold_controller(network, route, **config):
    controller = COMETController.__new__(COMETController)
    controller.cfg = COMETConfig(route_fold_shortcut=True, **config)
    controller.network = network
    controller._particle_diameter = 0.0005
    pts = np.asarray(route, dtype=float)
    controller._route_pts = pts
    controller._route_cum = np.concatenate([[0.0], np.cumsum(np.linalg.norm(np.diff(pts, axis=0), axis=1))])
    return controller


def _hook_route():
    # 20 mm out along +z, 10 mm across, then back past the carrier along -z.
    out = [[0.0, 0.0, z] for z in np.arange(0.0, 0.020, 0.0005)]
    across = [[x, 0.0, 0.020] for x in np.arange(0.0, 0.010, 0.0005)]
    back = [[0.010, 0.0, z] for z in np.arange(0.020, -0.020, -0.0005)]
    return out + across + back


def _two_tube_network(tmp_path, radius, name):
    # Parallel tubes joined at the far end; with a large radius they merge into one lumen.
    segments = []
    for x in (0.0, 0.010):
        for i in range(80):
            segments.append({"seg_id": len(segments), "start": np.array([x, 0.0, -0.020 + i * 0.0005]),
                             "end": np.array([x, 0.0, -0.020 + (i + 1) * 0.0005]), "radius": radius,
                             "v_mean": 0.0, "level": 0})
    path = tmp_path / name
    with open(path, "wb") as f:
        pickle.dump(segments, f)
    return RealVesselNetwork(str(path))


def test_fold_shortcut_skips_a_visible_fold(tmp_path):
    network = _two_tube_network(tmp_path, 0.008, "wide.pkl")
    controller = _fold_controller(network, _hook_route())
    pos = np.array([0.0, 0.0, 0.005])
    skip = controller._fold_skip_index(pos, arc_now=0.005, lookahead=0.012)
    assert skip is not None
    assert controller._route_cum[skip] > 0.005 + 1.5 * 0.012
    assert np.linalg.norm(controller._route_pts[skip] - pos) <= 0.012


def test_fold_shortcut_will_not_cross_a_wall(tmp_path):
    network = _two_tube_network(tmp_path, 0.002, "narrow.pkl")
    controller = _fold_controller(network, _hook_route())
    assert controller._fold_skip_index(np.array([0.0, 0.0, 0.005]), arc_now=0.005, lookahead=0.012) is None


def test_fold_shortcut_ignores_a_straight_route(tmp_path):
    network = _straight_network(tmp_path, radius=0.006)
    route = [[0.0, 0.0, z] for z in np.arange(0.0, 0.08, 0.0005)]
    controller = _fold_controller(network, route)
    assert controller._fold_skip_index(np.array([0.0, 0.0, 0.01]), arc_now=0.01, lookahead=0.008) is None


def test_high_flow_aim_uses_the_closest_axis_test(tmp_path):
    # A thin tube overlapping a wide one: the aim has clearance in the wide lumen
    # but not from the thin tube's axis, which is the closer of the two.
    segments = []
    for x, radius in ((0.0, 0.006), (0.0062, 0.001)):
        for i in range(40):
            segments.append({"seg_id": len(segments), "start": np.array([x, 0.0, -0.01 + i * 0.0005]),
                             "end": np.array([x, 0.0, -0.01 + (i + 1) * 0.0005]), "radius": radius,
                             "v_mean": 0.0, "level": 0})
    path = tmp_path / "side_by_side.pkl"
    with open(path, "wb") as f:
        pickle.dump(segments, f)
    network = RealVesselNetwork(str(path))
    controller = _fold_controller(network, _hook_route(), visible_axis_high_flow=True)
    pos, aim = np.zeros(3), np.array([0.0054, 0.0, 0.0])
    assert controller._aim_in_view(pos, aim, high_flow=False)
    assert not controller._aim_in_view(pos, aim, high_flow=True)
    controller.cfg.visible_axis_high_flow = False
    assert controller._aim_in_view(pos, aim, high_flow=True)
