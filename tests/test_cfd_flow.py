"""CFD import geometry, SI flow sampling, and benchmark factory regressions."""

import json
import pickle
from pathlib import Path

import numpy as np
import pytest

from dynamag_simulation.core.cfd_flow_field import SteadyCFDFlowField
from dynamag_simulation.core.patched_vessel_network import RealVesselNetworkCFD
from dynamag_simulation.core.real_vessel_network import RealVesselNetwork, RoutePlanningOptions
from dynamag_simulation.core.cem_mpc_controller import _BatchRealVesselModel
from scripts.import_simscale_flow import (
    boundary_values, cell_geometry, check_geometry, face_geometry, parse_array,
)


def _field(tmp_path, velocity=(0.01, 0.0, 0.0)):
    path = tmp_path / "flow.npz"
    points = np.array([[0.0, 0.0, 0.0], [0.002, 0.0, 0.0], [0.0, 0.002, 0.0], [0.0, 0.0, 0.002]])
    np.savez(path, points_m=points, velocity_m_s=np.tile(velocity, (4, 1)),
             support_radius_m=np.full(4, 0.01), metadata_json=json.dumps({"source_sha256": "test"}))
    return path


def test_cfd_constant_field_is_steady_scaled_once_and_has_no_distant_fallback(tmp_path):
    field = SteadyCFDFlowField(_field(tmp_path), flow_scale=3)
    np.testing.assert_allclose(field.velocity_at([0.001, 0.001, 0.001], time=0.25), [0.03, 0, 0])
    np.testing.assert_allclose(field.velocity_at([0, 0, 0], time=100), [0.03, 0, 0])
    with pytest.raises(ValueError, match="outside local sample support"):
        field.velocity_at([1, 1, 1])


def test_cfd_network_uses_imported_direction_for_particles_and_route_authority(tmp_path):
    path = tmp_path / "vessel.pkl"
    with path.open("wb") as stream:
        pickle.dump([{"seg_id": 0, "start": [-0.002, 0, 0], "end": [0.004, 0, 0], "radius": 0.001, "v_mean": 0.5}], stream)
    field = SteadyCFDFlowField(_field(tmp_path, (-0.01, 0, 0)), flow_scale=3)
    network = RealVesselNetworkCFD(path, field, flow_scale=3)
    np.testing.assert_allclose(network.velocity_at([0.001, 0, 0], 0.25), [-0.03, 0, 0])
    np.testing.assert_array_equal(network.velocity_at([0.001, 1, 0]), [0, 0, 0])
    metrics = network._route_edge_metrics(network.segments[0], np.array([1, 0, 0]), RoutePlanningOptions(magnetic_authority_m_s=0.1))
    assert metrics[1] == pytest.approx(0.03)
    assert network.params_dict()["flow_model"] == "real_vessel_network_steady_cfd"
    with pytest.raises(ValueError, match="required"):
        RealVesselNetworkCFD(path, None)
    assert isinstance(RealVesselNetwork(path), RealVesselNetwork)


def test_cem_batch_flow_uses_cfd_direction_scale_and_steady_profile(tmp_path):
    path = tmp_path / "vessel.pkl"
    with path.open("wb") as stream:
        pickle.dump([{"seg_id": 0, "start": [-0.002, 0, 0], "end": [0.004, 0, 0],
                      "radius": 0.001, "v_mean": 0.5}], stream)
    field = SteadyCFDFlowField(_field(tmp_path, (-0.01, 0.02, 0)), flow_scale=3)
    network = RealVesselNetworkCFD(path, field, flow_scale=3)
    model = _BatchRealVesselModel(network)
    positions = np.array([[0, 0, 0], [0.001, 0.0009, 0], [0.001, 1, 0]])
    expected = np.array([network.velocity_at(p, 0.25) for p in positions])
    count = field.query_count
    np.testing.assert_allclose(model.flow_velocity(positions, 0.25), expected)
    np.testing.assert_allclose(model.flow_velocity(positions, 100), expected)
    assert field.query_count - count == 4
    np.testing.assert_allclose(expected[:2], [[-0.03, 0.06, 0]] * 2)
    # Ordinary analytical rollouts retain their radial profile and pulsation.
    analytical = RealVesselNetwork(path, flow_scale=3)
    scalar = np.array([analytical.velocity_at(p, .25) for p in positions])
    np.testing.assert_allclose(_BatchRealVesselModel(analytical).flow_velocity(positions, .25), scalar)


def test_openfoam_cube_centroid_volume_and_orientation():
    points = np.array([[0, 0, 0], [1, 0, 0], [1, 1, 0], [0, 1, 0],
                       [0, 0, 1], [1, 0, 1], [1, 1, 1], [0, 1, 1]], dtype=float)
    vertices = np.array([[0, 3, 2, 1], [4, 5, 6, 7], [0, 1, 5, 4],
                         [1, 2, 6, 5], [2, 3, 7, 6], [3, 0, 4, 7]]).ravel()
    sizes = np.full(6, 4)
    offsets = np.arange(7) * 4
    fc, sf, fr = face_geometry(points, sizes, offsets, vertices)
    centres, radii, volumes = cell_geometry(fc, sf, fr, np.zeros(6, dtype=int), np.array([], dtype=int), 1)
    np.testing.assert_allclose(centres, [[0.5, 0.5, 0.5]])
    np.testing.assert_allclose(volumes, [1.0])
    np.testing.assert_allclose(sf.sum(axis=0), [0, 0, 0])
    assert radii[0] >= np.sqrt(3) / 2


def test_face_geometry_matches_known_faces_across_chunk_boundaries():
    # Large mixed-polygon batches catch errors hidden by a tiny cube fixture.
    points = np.array([[-0.1, 0.03, -0.4], [-0.099, 0.03, -0.4],
                       [-0.099, 0.031, -0.4], [-0.1, 0.031, -0.4]])
    sizes = np.tile([3, 4], 50001)
    offsets = np.r_[0, np.cumsum(sizes)]
    vertices = np.tile([0, 1, 2, 0, 1, 2, 3], 50001)
    fc, sf, _ = face_geometry(points, sizes, offsets, vertices)
    np.testing.assert_allclose(fc[::2], np.broadcast_to(points[:3].mean(0), fc[::2].shape), atol=1e-14)
    np.testing.assert_allclose(fc[1::2], np.broadcast_to(points.mean(0), fc[1::2].shape), atol=1e-14)
    np.testing.assert_allclose(sf[:, 2], np.tile([0.5e-6, 1e-6], 50001))


def test_geometry_gate_rejects_postsolve_unit_fix():
    points = np.array([[0, 0, 0], [0.1, 0.1, 0.1], [0.05, 0.05, 0.05]])
    segments = [{"start": np.zeros(3), "end": np.full(3, 0.1), "radius": 0.01}]
    assert check_geometry(points, segments)["p95_nearest_vertex_distance_over_radius"] == 0
    with pytest.raises(ValueError, match="scale mismatch"):
        check_geometry(points * 0.1, segments)


def test_ascii_field_parser_preserves_order_and_boundary_values():
    text = "internalField nonuniform List<vector>\n2\n(\n(1 2 3)\n(4 5 6)\n)\n;\nboundaryField { inlet { value uniform (0 0 -0.04); } }"
    np.testing.assert_array_equal(parse_array(text, True, True), [[1, 2, 3], [4, 5, 6]])
    np.testing.assert_allclose(boundary_values(text, "inlet", 2, True), [[0, 0, -0.04], [0, 0, -0.04]])
