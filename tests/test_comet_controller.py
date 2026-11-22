"""Tests for the COMET measure-transport controller."""

import sys
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from dynamag_simulation.core.comet_controller import (
    COMETConfig,
    COMETController,
    _LocalFieldModel,
    _magnetic_velocity_batch,
)
from dynamag_simulation.core.magnets import ElectromagnetArray3D
from dynamag_simulation.core.particle import MagneticParticle3D
from dynamag_simulation.core.profiles import SPION_CARRIER_500UM
from dynamag_simulation.core.real_vessel_network import RealVesselNetwork
from dynamag_simulation.core.workspace import WorkspaceVessel


VESSEL_FILE = "vessels/abdominal_0033_H_ABAO_AAA/real_vessel_segments.pkl"


@pytest.fixture(scope="module")
def network():
    return RealVesselNetwork(VESSEL_FILE, flow_scale=1.0)


@pytest.fixture(scope="module")
def magnets(network):
    bounds_min, bounds_max = network.bounds(include_radius=True)
    center = 0.5 * (bounds_min + bounds_max)
    return ElectromagnetArray3D(
        coil_radius=0.070,
        coil_length=0.160,
        turns=600.0,
        max_current=16.0,
        placement_radius=0.300,
        center=center,
        core_radius=0.030,
        core_relative_permeability=70.0,
        core_saturation_tesla=1.8,
    )


def test_local_field_model_matches_magnet_array(network, magnets):
    """The surrogate must reproduce field_at and grad_B_squared exactly."""
    rng = np.random.default_rng(3)
    pos = network.inlet_start() + np.array([0.0, 0.0, 0.002])
    model = _LocalFieldModel(magnets, pos, h=1e-4)

    for _ in range(5):
        currents = rng.uniform(-16.0, 16.0, magnets.n_magnets)

        previous = magnets.currents.copy()
        magnets.set_currents(currents)
        # set_currents may derate currents via the hardware model;
        # compare the surrogate against the currents actually applied.
        applied = magnets.currents.copy()
        b_exact = magnets.field_at(pos)
        grad_exact = magnets.grad_B_squared(pos)
        magnets.set_currents(previous)

        b_model, grad_model = model.field_and_grad(applied[None, :])

        np.testing.assert_allclose(b_model[0], b_exact, rtol=1e-9, atol=1e-15)
        np.testing.assert_allclose(
            grad_model[0], grad_exact, rtol=1e-9, atol=1e-15,
        )


def test_velocity_batch_matches_particle_drag(network, magnets):
    """Vectorized drag matches MagneticParticle3D._effective_drag."""
    particle = MagneticParticle3D(
        position=network.inlet_start(),
        profile=SPION_CARRIER_500UM,
    )
    pos = network.inlet_start()
    model = _LocalFieldModel(magnets, pos, h=1e-4)

    currents = np.full((1, magnets.n_magnets), 12.0)
    b, grad = model.field_and_grad(currents)
    velocity, force = _magnetic_velocity_batch(particle, b, grad)

    f_exact = particle.magnetic_force(b[0], grad[0])
    np.testing.assert_allclose(force[0], f_exact, rtol=1e-9)

    f_mag = np.linalg.norm(f_exact)
    if f_mag > 0:
        gamma_eff, _, _ = particle._effective_drag(f_mag)
        np.testing.assert_allclose(
            velocity[0], f_exact / gamma_eff, rtol=1e-3,
        )


def test_comet_step_produces_bounded_currents_and_intent(network, magnets):
    particle = MagneticParticle3D(
        position=network.inlet_start(),
        profile=SPION_CARRIER_500UM,
    )
    workspace = WorkspaceVessel(network)
    start = network.inlet_start()
    target = network.outlet_end()
    navigator = network.waypoint_navigator(start, target)

    controller = COMETController(
        network=network,
        magnet_array=magnets,
        workspace=workspace,
        max_current=16.0,
        config=COMETConfig(),
        wall_clearance_threshold_m=0.0007,
    )

    for step in range(5):
        currents, force, cost, converged, info = controller.step(
            particle=particle,
            final_target=target,
            navigator=navigator,
            dt=0.02,
            flow=network,
            sim_time=step * 0.02,
        )

        assert currents.shape == (magnets.n_magnets,)
        assert np.all(np.abs(currents) <= 16.0 + 1e-9)
        assert np.all(np.isfinite(force))
        assert converged
        assert info.mode.startswith("COMET_")

        if info.q:
            total = sum(info.q.values())
            assert abs(total - 1.0) < 1e-6
            assert all(v >= 0.0 for v in info.q.values())

        # Advance the particle so subsequent steps see motion.
        particle.update(force, 0.02, flow_velocity=network.velocity_at(
            particle.pos, step * 0.02,
        ))
        particle.pos = workspace.clip(particle.pos)


def test_downstream_distances_prefer_target_branch(network, magnets):
    workspace = WorkspaceVessel(network)
    controller = COMETController(
        network=network,
        magnet_array=magnets,
        workspace=workspace,
        max_current=16.0,
    )
    target = network.outlet_end()
    dist = controller._downstream_distances(target)

    target_seg, _ = network.find_segment(target)
    assert dist[target_seg.seg_id] == 0.0
    assert any(np.isfinite(v) and v > 0 for v in dist.values())
