"""Reference behavior and DynaMag safety-channel regression tests."""
import numpy as np
import pytest

from dynamag_simulation.core.constrained_cem_controller import (
    ConstrainedCEMConfig, ConstrainedCEMMPCController,
    constrained_cross_entropy_optimize, select_constrained_elites,
)
from dynamag_simulation.core.magnets import ElectromagnetArray3D
from dynamag_simulation.core.particle import MagneticParticle3D
from dynamag_simulation.core.profiles import SPION_CARRIER_500UM
from dynamag_simulation.core.workspace import Workspace3D


def test_elites_ignore_objective_until_enough_feasible():
    objective = np.array([-100., 7., 4., -200.])
    # One feasible (<K): rank solely by constraint cost.
    np.testing.assert_array_equal(select_constrained_elites(objective, np.array([3., 0., 1., 2.]), 2), [1, 2])
    # At least K feasible: ignore the infeasible, even with better objective.
    np.testing.assert_array_equal(select_constrained_elites(objective, np.array([3., 0., 0., 0.]), 2), [3, 2])


class FixedNoise:
    def __init__(self, batches):
        self.batches = iter(batches)

    def normal(self, *, loc=0.0, scale=1.0, size=None):
        batch = np.asarray(next(self.batches), dtype=float).reshape(size)
        return np.asarray(loc) + batch * np.asarray(scale)


def test_unbounded_gaussian_unbiased_std_and_final_sample_selection():
    seen = []

    def evaluate(samples):
        seen.append(samples.copy())
        # First update elites are 1 and 3 -> mean=2, unbiased std=sqrt(2).
        return np.array([0., 1., 2.]), np.zeros(3)

    result = constrained_cross_entropy_optimize(
        evaluate, config=ConstrainedCEMConfig(1, 3, 2, 2), action_dim=1,
        rng=FixedNoise([[1., 3., 9.], [0., 1., 2.]]),
    )
    np.testing.assert_allclose(seen[1].ravel(), [2, 2 + np.sqrt(2), 2 + 2*np.sqrt(2)])
    assert seen[0].max() == 9  # no sample clipping
    assert result.solution.item() == 2  # sampled best, not refitted mean
    assert result.feasible_counts == (3, 3)


def test_final_batch_infeasible_does_not_reuse_old_feasible_plan():
    calls = []

    def evaluate(samples):
        calls.append(1)
        return np.arange(3.), np.zeros(3) if len(calls) == 1 else np.ones(3)

    result = constrained_cross_entropy_optimize(
        evaluate, config=ConstrainedCEMConfig(1, 3, 2, 2), action_dim=1,
        rng=np.random.default_rng(4),
    )
    assert not result.converged
    assert result.feasible_counts == (3, 0)
    assert np.all(result.solution == 0)


def test_nan_objective_is_never_reported_feasible():
    result = constrained_cross_entropy_optimize(
        lambda s: (np.full(3, np.nan), np.zeros(3)),
        config=ConstrainedCEMConfig(1, 3, 2, 1), action_dim=1,
        rng=np.random.default_rng(4),
    )
    assert not result.converged


def test_tuned_sampler_stays_inside_action_box():
    seen = []

    def evaluate(samples):
        seen.append(samples.copy())
        return np.sum(samples**2, axis=(1, 2)), np.zeros(len(samples))

    constrained_cross_entropy_optimize(
        evaluate,
        config=ConstrainedCEMConfig(
            horizon=10, population_size=100, num_elites=10, max_iterations=8,
            initial_std=0.75, sample_within_action_bounds=True,
        ),
        action_dim=9,
        rng=np.random.default_rng(7),
    )
    assert all(np.all(np.abs(batch) <= 2.0) for batch in seen)


def test_best_effort_returns_least_violating_sample():
    result = constrained_cross_entropy_optimize(
        lambda s: (np.sum(s**2, axis=(1, 2)), np.full(len(s), 3.0)),
        config=ConstrainedCEMConfig(
            horizon=1, population_size=10, num_elites=2, max_iterations=2,
            best_effort_on_infeasible=True,
        ),
        action_dim=1,
        rng=np.random.default_rng(8),
    )
    assert not result.converged
    assert result.constraint_cost == 3.0
    assert np.any(result.solution != 0.0)


def test_constraints_detect_unprojected_wall_and_raw_action_violation():
    c = ConstrainedCEMMPCController(2, 8)
    c._workspace = Workspace3D(half_side=0.1)
    c._clearance = 0.001
    c._current_limits = np.array([8., 4.])
    values = c._constraint_costs(
        np.array([[0, 0, 0], [0.101, 0, 0], [0, 0, 0], [0.101, 0, 0]]),
        np.array([[0, 0], [0, 0], [0, 5], [9, 0]]),
    )
    np.testing.assert_array_equal(values, [0, 3, 3, 6])


def test_controller_hardware_bounds_and_explicit_infeasible_fallback():
    magnets = ElectromagnetArray3D()
    particle = MagneticParticle3D(position=[0., 0., 0.], profile=SPION_CARRIER_500UM)
    c = ConstrainedCEMMPCController(magnets.n_magnets, 8, ConstrainedCEMConfig(1, 20, 4, 3))
    kw = dict(particle=particle, particle_pos=particle.pos, target_pos=np.array([.03, 0, 0]),
              magnet_array=magnets, workspace=Workspace3D(half_side=.5), dt=.02,
              wall_buffer_m=.0007)
    currents, force, cost, feasible = c.command(**kw)
    assert feasible
    assert np.all(np.abs(currents) <= c._current_limits)
    assert np.all(np.isfinite(force))
    assert np.isfinite(cost)
    # Initial state outside safe set: all full trajectories are infeasible.
    kw['particle_pos'] = np.array([.6, 0, 0])
    currents, _, _, feasible = c.command(**kw)
    assert not feasible
    assert np.all(currents == 0)
    assert c.diagnostics[-1]['zero_current_fallback']
    # Reaching a local waypoint must not bypass feasibility in background flow.
    kw['target_pos'] = kw['particle_pos'].copy()
    currents, _, _, feasible = c.command(**kw)
    assert not feasible
    assert np.all(currents == 0)


def test_benchmark_variants_share_sampler_and_only_change_objective():
    from scripts.run_publication_benchmark import CONTROLLER_VARIANTS, MPC_DEFAULT_WEIGHTS, ACBC_TUNED_WEIGHTS
    default = CONTROLLER_VARIANTS['constrained_cem_mpc'].kwargs
    tuned = CONTROLLER_VARIANTS['constrained_cem_mpc_tuned_shared'].kwargs
    for k in MPC_DEFAULT_WEIGHTS:
        assert default[k] == MPC_DEFAULT_WEIGHTS[k]
        assert tuned[k] == ACBC_TUNED_WEIGHTS[k]
    assert {k:v for k,v in default.items() if k not in MPC_DEFAULT_WEIGHTS} == {
        k:v for k,v in tuned.items() if k not in ACBC_TUNED_WEIGHTS}


def test_recovery_variant_uses_existing_shared_tuned_objective():
    from scripts.run_publication_benchmark import CONTROLLER_VARIANTS, ACBC_TUNED_WEIGHTS
    recovery = CONTROLLER_VARIANTS['constrained_cem_mpc_tuned_recovery'].kwargs
    for key, value in ACBC_TUNED_WEIGHTS.items():
        assert recovery[key] == value
    assert recovery['constrained_cem_sample_within_action_bounds']
    assert not recovery['constrained_cem_include_initial_state_constraint']
    assert recovery['constrained_cem_best_effort_on_infeasible']
    assert recovery['constrained_cem_action_blocks'] == 2


def test_action_blocking_expands_solution_to_full_prediction_horizon():
    from dynamag_simulation.core.constrained_cem_controller import ConstrainedCEMResult

    controller = ConstrainedCEMMPCController(
        1,
        8,
        ConstrainedCEMConfig(horizon=10, action_blocks=2),
    )
    controller._action_scale = np.array([4.0])
    controller.rng = np.random.default_rng(3)

    def fake_optimizer(evaluate, **kwargs):
        # Verify evaluation receives ten expanded actions: five of each block.
        objectives, constraints = evaluate(np.array([[[1.0], [-1.0]]]))
        np.testing.assert_array_equal(objectives, [0.0])
        np.testing.assert_array_equal(constraints, [0.0])
        return ConstrainedCEMResult(
            solution=np.array([1.0, -1.0]), variance=np.array([1.0, 4.0]),
            cost=0.0, iterations=1, converged=True, feasible_count=1,
            min_constraint_cost=0.0, constraint_cost=0.0, feasible_counts=(1,),
        )

    import dynamag_simulation.core.constrained_cem_controller as module
    original = module.constrained_cross_entropy_optimize
    module.constrained_cross_entropy_optimize = fake_optimizer
    try:
        result = controller._optimize(
            lambda flat: (
                np.sum(flat.reshape(-1, 10, 1), axis=(1, 2)),
                np.zeros(len(flat)),
            )
        )
    finally:
        module.constrained_cross_entropy_optimize = original
    np.testing.assert_array_equal(result.solution, [4.0] * 5 + [-4.0] * 5)
    np.testing.assert_array_equal(result.variance, [16.0] * 5 + [64.0] * 5)


def test_invalid_elite_count_rejected():
    with pytest.raises(ValueError):
        ConstrainedCEMConfig(num_elites=1).validate()
