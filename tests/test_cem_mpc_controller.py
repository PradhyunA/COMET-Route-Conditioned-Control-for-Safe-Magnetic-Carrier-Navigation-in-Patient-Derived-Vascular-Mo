"""Tests for the published PETS-style CEM-MPC baseline."""

import numpy as np
import pytest

from dynamag_simulation.core.cem_mpc_controller import (
    _batch_magnetic_force,
    _powerlaw_psd_gaussian,
    ICEMConfig,
    ICEMMPCController,
    PETSCEMConfig,
    PETSCEMMPCController,
    cross_entropy_optimize,
    icem_population_schedule,
    improved_cross_entropy_optimize,
)
from dynamag_simulation.core.magnets import ElectromagnetArray3D
from dynamag_simulation.core.particle import MagneticParticle3D
from dynamag_simulation.core.profiles import SPION_CARRIER_1MM


def _solve_quadratic(seed):
    target = np.array([0.6, -0.35, 0.2])

    def costs(samples):
        return np.sum((samples - target) ** 2, axis=1)

    return cross_entropy_optimize(
        costs,
        initial_mean=np.zeros(3),
        initial_variance=np.ones(3),
        lower_bound=-np.ones(3),
        upper_bound=np.ones(3),
        population_size=400,
        num_elites=40,
        max_iterations=5,
        alpha=0.1,
        epsilon=1e-6,
        rng=np.random.default_rng(seed),
    )


def test_cem_converges_on_bounded_quadratic():
    result = _solve_quadratic(7)
    assert result.converged
    assert result.cost < 1e-3
    assert np.all(result.solution >= -1.0)
    assert np.all(result.solution <= 1.0)


def test_cem_is_reproducible_for_a_fixed_seed():
    first = _solve_quadratic(11)
    second = _solve_quadratic(11)
    assert np.array_equal(first.solution, second.solution)
    assert first.cost == second.cost


def test_cem_config_rejects_more_elites_than_samples():
    with pytest.raises(ValueError, match="num_elites"):
        PETSCEMConfig(population_size=8, num_elites=9).validate()


def test_icem_blitz_population_schedule_matches_authors_repository():
    assert icem_population_schedule(ICEMConfig()) == (40, 32, 25)


def test_icem_colored_noise_is_seeded_and_temporally_correlated():
    first = _powerlaw_psd_gaussian(
        3.0,
        (128, 2, 30),
        rng=np.random.default_rng(2021),
    )
    second = _powerlaw_psd_gaussian(
        3.0,
        (128, 2, 30),
        rng=np.random.default_rng(2021),
    )
    assert np.array_equal(first, second)
    lag_one = np.corrcoef(first[..., :-1].reshape(-1), first[..., 1:].reshape(-1))[0, 1]
    assert lag_one > 0.8


def test_icem_uses_decayed_populations_and_reuses_three_elites():
    config = ICEMConfig(horizon=4)
    call_sizes = []

    def costs(samples):
        call_sizes.append(samples.shape[0])
        return np.sum(np.square(samples - 0.4), axis=1)

    result = improved_cross_entropy_optimize(
        costs,
        initial_mean=np.zeros((4, 1)),
        initial_std=np.full((4, 1), 0.5),
        lower_bound=-np.ones((4, 1)),
        upper_bound=np.ones((4, 1)),
        config=config,
        rng=np.random.default_rng(7),
    )

    assert result.population_sizes == (40, 32, 25)
    # New samples are evaluated in 40/32/25 batches; the latter two iterations
    # additionally retain int(10 * 0.3) prior elites, as in upstream MpcICem.
    assert call_sizes == [40, 32, 3, 25, 3]
    assert result.elites.shape == (10, 4, 1)
    assert result.solution.shape == (4,)
    assert result.converged


def test_vectorized_magnetic_rollout_matches_shared_scalar_physics():
    magnet = ElectromagnetArray3D()
    particle = MagneticParticle3D(np.zeros(3), profile=SPION_CARRIER_1MM)
    positions = np.array([
        [0.001, 0.002, 0.003],
        [-0.004, 0.002, 0.001],
    ])
    currents = np.array([
        [1, -2, 3, -4, 5, -6, 7, -8, 2],
        [-3, 1, 5, 2, -7, 4, 6, -2, 8],
    ], dtype=float)

    batch_force = _batch_magnetic_force(
        particle,
        positions,
        currents,
        magnet,
    )
    scalar_force = []
    for position, command in zip(positions, currents):
        magnet.set_currents(command)
        field = magnet.field_at(position)
        gradient = magnet.grad_B_squared(position)
        scalar_force.append(particle.magnetic_force(field, gradient))

    assert np.allclose(batch_force, scalar_force, rtol=1e-11, atol=1e-20)


class _LinearMagnet:
    n_magnets = 1

    def __init__(self):
        self.currents = np.zeros(1)

    def set_currents(self, currents):
        self.currents = np.asarray(currents, dtype=float).copy()

    def field_at(self, _position):
        return np.array([1.0, 0.0, 0.0])

    def grad_B_squared(self, _position):
        return np.array([self.currents[0], 0.0, 0.0])


class _LinearParticle:
    def magnetic_force(self, _field, gradient):
        return np.asarray(gradient, dtype=float)

    def _effective_drag(self, force_magnitude):
        return 1.0, float(force_magnitude), float(force_magnitude)


class _OpenWorkspace:
    half = 100.0

    def clip(self, position):
        return np.asarray(position, dtype=float)


def test_controller_executes_first_action_and_shifts_warm_start():
    config = PETSCEMConfig(
        horizon=3,
        population_size=120,
        num_elites=12,
        max_iterations=4,
        alpha=0.1,
        epsilon=1e-6,
    )
    controller = PETSCEMMPCController(
        action_dim=1,
        max_current=1.0,
        config=config,
        rng_seed=5,
    )
    magnet = _LinearMagnet()

    currents, force, cost, converged = controller.command(
        particle=_LinearParticle(),
        particle_pos=np.zeros(3),
        target_pos=np.array([1.0, 0.0, 0.0]),
        magnet_array=magnet,
        workspace=_OpenWorkspace(),
        dt=0.1,
        w_progress=100.0,
        w_align=10.0,
        w_dist=1.0,
        w_curr=1e-4,
        w_smooth=1e-4,
    )

    solution = controller.last_result.solution.reshape(config.horizon, 1)
    assert converged
    assert np.isfinite(cost)
    assert currents[0] > 0.0
    assert force[0] > 0.0
    assert currents[0] == pytest.approx(solution[0, 0])
    assert np.allclose(controller.previous_solution[:-1], solution[1:].reshape(-1))
    assert controller.previous_solution[-1] == 0.0


def test_icem_controller_shifts_fitted_mean_and_retains_elites():
    config = ICEMConfig(
        horizon=4,
        population_size=20,
        num_elites=5,
        max_iterations=2,
    )
    controller = ICEMMPCController(
        action_dim=1,
        max_current=1.0,
        config=config,
        rng_seed=5,
    )
    magnet = _LinearMagnet()

    currents, _force, cost, converged = controller.command(
        particle=_LinearParticle(),
        particle_pos=np.zeros(3),
        target_pos=np.array([1.0, 0.0, 0.0]),
        magnet_array=magnet,
        workspace=_OpenWorkspace(),
        dt=0.1,
        w_progress=100.0,
        w_align=10.0,
        w_dist=1.0,
        w_curr=1e-4,
        w_smooth=1e-4,
    )

    mean = controller.last_result.mean
    assert converged
    assert np.isfinite(cost)
    assert currents[0] > 0.0
    assert controller.previous_elites.shape == (5, 4, 1)
    assert np.allclose(controller.previous_solution[:-1], mean[1:].reshape(-1))
    assert controller.previous_solution[-1] == pytest.approx(mean[-1, 0])
