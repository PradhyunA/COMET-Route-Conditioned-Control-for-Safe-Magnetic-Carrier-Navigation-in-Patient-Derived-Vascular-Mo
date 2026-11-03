"""Tests for the actuator calibration and patient positioning planner."""

import itertools
import sys
from dataclasses import replace
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))

from dynamag_simulation.core.constants import STANDARD_GRAVITY_M_S2
from dynamag_simulation.core.magnets import ElectromagnetArray3D
from dynamag_simulation.core.particle import MagneticParticle3D
from dynamag_simulation.core.patient_positioning import (
    choose_patient_orientation,
    table_orientations,
)
from dynamag_simulation.core.profiles import SPION_CARRIER_500UM
from scripts.run_publication_benchmark import ACTUATOR_PRESETS
from simulation_3d_comet import (
    FIXED_COIL_LENGTH,
    FIXED_COIL_RADIUS,
    FIXED_CORE_RADIUS,
    FIXED_CORE_SATURATION_T,
    FIXED_MAX_CURRENT,
    FIXED_PLACEMENT_RADIUS,
    FIXED_TURNS,
)


class _UniformFlow:
    def __init__(self, velocity):
        self.velocity = np.asarray(velocity, dtype=float)

    def velocity_at(self, pos, time=0.0):
        return self.velocity


def _carrier():
    profile = replace(
        SPION_CARRIER_500UM,
        particle_density=3542.0,
        gravity_m_s2=(0.0, 0.0, -STANDARD_GRAVITY_M_S2),
    )
    particle = MagneticParticle3D(np.zeros(3), profile=profile)
    return particle, float(np.linalg.norm(particle.body_force))


def _array(**overrides):
    kwargs = dict(
        coil_radius=FIXED_COIL_RADIUS,
        coil_length=FIXED_COIL_LENGTH,
        turns=FIXED_TURNS,
        max_current=FIXED_MAX_CURRENT,
        placement_radius=FIXED_PLACEMENT_RADIUS,
        core_radius=FIXED_CORE_RADIUS,
        core_saturation_tesla=FIXED_CORE_SATURATION_T,
    )
    kwargs.update(overrides)
    return ElectromagnetArray3D(**kwargs)


def test_table_orientations_cover_roll_and_bounded_tilt():
    poses = list(table_orientations(max_tilt_deg=20.0))
    assert len(poses) == 8 * 5
    axis = np.array([0.0, 0.0, 1.0])
    for _, tilt, g_dir in poses:
        assert np.linalg.norm(g_dir) == pytest.approx(1.0)
        assert np.degrees(np.arcsin(g_dir @ axis)) == pytest.approx(tilt, abs=1e-9)


def test_positioning_tilts_gravity_along_upstream_travel():
    particle, weight = _carrier()
    network = _UniformFlow((0.0, 0.0, 0.02))
    route = [np.array([0.0, 0.0, -0.01 * i]) for i in range(5)]  # travel -z against flow

    choice = choose_patient_orientation(network, route, particle, weight, max_tilt_deg=20.0)

    assert choice.tilt_deg == pytest.approx(-20.0)
    assert choice.gravity_m_s2[2] < 0.0
    assert choice.peak_required_force_n < choice.worst_orientation_peak_force_n


def test_positioning_without_tilt_cannot_use_gravity_along_axis():
    particle, weight = _carrier()
    network = _UniformFlow((0.0, 0.0, 0.0))
    route = [np.array([0.0, 0.0, 0.01 * i]) for i in range(3)]

    choice = choose_patient_orientation(network, route, particle, weight, max_tilt_deg=0.0)

    assert choice.tilt_deg == 0.0
    assert choice.gravity_m_s2[2] == pytest.approx(0.0, abs=1e-12)
    assert choice.peak_required_force_n >= weight


def test_ampere_turn_scale_scales_dipole_moment():
    base = _array(core_relative_permeability=1.0)
    scaled = _array(core_relative_permeability=1.0, ampere_turn_scale=3.0)
    assert scaled.moment_per_amp == pytest.approx(3.0 * base.moment_per_amp)
    assert scaled.params_dict()["ampere_turn_scale"] == 3.0
    with pytest.raises(ValueError):
        _array(ampere_turn_scale=0.0)


@pytest.mark.parametrize("preset, gradient_t_m", [
    ("navion_calibrated", 0.25),
    ("high_gradient_400", 0.40),
])
def test_actuator_presets_match_published_gradient(preset, gradient_t_m):
    particle = MagneticParticle3D(np.zeros(3), profile=SPION_CARRIER_500UM)
    array = _array(**ACTUATOR_PRESETS[preset])
    centre = np.zeros(3)
    best_force, best_grad = 0.0, 0.0
    for signs in itertools.product((-1.0, 1.0), repeat=array.n_magnets):
        array.set_currents(np.array(signs) * array.max_current)
        field = array.field_at(centre)
        if np.linalg.norm(field) < 1e-9:
            continue
        grad_sq = array.grad_B_squared(centre)
        force = float(np.linalg.norm(particle.magnetic_force(field, grad_sq)))
        if force > best_force:
            best_force = force
            best_grad = float(np.linalg.norm(grad_sq)) / (2.0 * np.linalg.norm(field))
    assert best_grad == pytest.approx(gradient_t_m, rel=0.02)
