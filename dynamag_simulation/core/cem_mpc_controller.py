"""Published PETS and improved-CEM model-predictive controllers.

This module implements the trajectory optimizer and receding-horizon behavior
released with:

    Chua, Calandra, McAllister, and Levine, "Deep Reinforcement Learning in a
    Handful of Trials using Probabilistic Dynamics Models," NeurIPS 2018.
    https://proceedings.neurips.cc/paper/2018/hash/
    3de568f8597b94bda53149c7d7f5958c-Abstract.html

The CEM update, bounded truncated-normal sampling, initial variance, and
shifted warm start follow the authors' implementation:
https://github.com/kchua/handful-of-trials

PETS learned an ensemble dynamics model.  For a controlled DynaMag baseline,
this implementation instead rolls out the same deterministic magnetic/flow
physics model used by every controller in this repository.  Thus the published
CEM-MPC planning method is the independent variable, not the plant model.

The iCEM implementation follows Pinneri et al., "Sample-efficient Cross-Entropy
Method for Real-time Planning," CoRL 2020 (published in PMLR 2021), and the
authors' ``MpcICem`` implementation:
https://github.com/martius-lab/iCEM
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Callable

import numpy as np
from scipy.stats import truncnorm

from .constants import EPS
from .controllers import _overshoot_penalty, _simulate_one_step, _wall_penalty
from .flow import NoFlow


@dataclass(frozen=True)
class PETSCEMConfig:
    """CEM settings from the published PETS 3-D reacher experiment."""

    horizon: int = 25
    population_size: int = 400
    num_elites: int = 40
    max_iterations: int = 5
    alpha: float = 0.1
    epsilon: float = 1e-3

    def validate(self) -> None:
        if self.horizon < 1:
            raise ValueError("horizon must be at least 1")
        if self.population_size < 1:
            raise ValueError("population_size must be at least 1")
        if not 1 <= self.num_elites <= self.population_size:
            raise ValueError("num_elites must be in [1, population_size]")
        if self.max_iterations < 1:
            raise ValueError("max_iterations must be at least 1")
        if not 0.0 <= self.alpha <= 1.0:
            raise ValueError("alpha must be in [0, 1]")
        if self.epsilon < 0.0:
            raise ValueError("epsilon must be nonnegative")


@dataclass(frozen=True)
class ICEMConfig:
    """Official ``i-cem-blitz`` settings from Pinneri et al.'s repository."""

    horizon: int = 30
    population_size: int = 40
    num_elites: int = 10
    max_iterations: int = 3
    alpha: float = 0.1
    init_std: float = 0.5
    population_decay_factor: float = 1.25
    fraction_elites_reused: float = 0.3
    noise_beta: float = 3.0
    use_mean_action: bool = True
    keep_previous_elites: bool = True
    shift_elites_over_time: bool = True

    def validate(self) -> None:
        if self.horizon < 1:
            raise ValueError("horizon must be at least 1")
        if self.population_size < 2:
            raise ValueError("population_size must be at least 2")
        if not 2 <= self.num_elites <= self.population_size // 2:
            raise ValueError("num_elites must be in [2, population_size // 2]")
        if self.max_iterations < 1:
            raise ValueError("max_iterations must be at least 1")
        if not 0.0 <= self.alpha <= 1.0:
            raise ValueError("alpha must be in [0, 1]")
        if self.init_std <= 0.0:
            raise ValueError("init_std must be positive")
        if self.population_decay_factor < 1.0:
            raise ValueError("population_decay_factor must be at least 1")
        if not 0.0 <= self.fraction_elites_reused <= 1.0:
            raise ValueError("fraction_elites_reused must be in [0, 1]")
        if self.noise_beta < 0.0:
            raise ValueError("noise_beta must be nonnegative")


@dataclass(frozen=True)
class CEMOptimizationResult:
    solution: np.ndarray
    variance: np.ndarray
    cost: float
    iterations: int
    converged: bool


@dataclass(frozen=True)
class ICEMOptimizationResult(CEMOptimizationResult):
    """iCEM result plus state required by its two elite-reuse mechanisms."""

    mean: np.ndarray
    elites: np.ndarray
    population_sizes: tuple[int, ...]


def _powerlaw_psd_gaussian(
    exponent: float,
    size,
    *,
    rng: np.random.Generator,
) -> np.ndarray:
    """Generate the exact colored-noise distribution used by upstream iCEM.

    This is a dependency-free transcription of ``colorednoise``'s
    ``powerlaw_psd_gaussian`` routine.  Supplying the same NumPy Generator gives
    bit-for-bit identical samples to the package used by the authors.
    """

    shape = [int(size)] if isinstance(size, (int, np.integer)) else list(size)
    samples = int(shape[-1])
    frequencies = np.fft.rfftfreq(samples)
    fmin = 1.0 / samples
    scale = frequencies
    cutoff = int(np.sum(scale < fmin))
    if cutoff and cutoff < len(scale):
        scale[:cutoff] = scale[cutoff]
    scale = scale ** (-float(exponent) / 2.0)

    weights = scale[1:].copy()
    weights[-1] *= (1 + (samples % 2)) / 2.0
    sigma = 2.0 * np.sqrt(np.sum(weights**2)) / samples

    shape[-1] = len(frequencies)
    scale = scale[(np.newaxis,) * (len(shape) - 1) + (Ellipsis,)]
    real = rng.normal(scale=scale, size=shape)
    imaginary = rng.normal(scale=scale, size=shape)
    if not samples % 2:
        imaginary[..., -1] = 0.0
        real[..., -1] *= np.sqrt(2.0)
    imaginary[..., 0] = 0.0
    real[..., 0] *= np.sqrt(2.0)
    return np.fft.irfft(real + 1j * imaginary, n=samples, axis=-1) / sigma


def icem_population_schedule(config: ICEMConfig) -> tuple[int, ...]:
    """Return the per-iteration trajectory counts used by upstream iCEM."""

    config.validate()
    count = int(config.population_size)
    schedule = []
    for iteration in range(config.max_iterations):
        if iteration > 0:
            count = max(
                config.num_elites * 2,
                int(count / config.population_decay_factor),
            )
        schedule.append(count)
    return tuple(schedule)


def improved_cross_entropy_optimize(
    cost_function: Callable[[np.ndarray], np.ndarray],
    initial_mean,
    initial_std,
    lower_bound,
    upper_bound,
    *,
    config: ICEMConfig,
    rng: np.random.Generator,
    previous_elites=None,
) -> ICEMOptimizationResult:
    """Minimize a batched cost using Pinneri et al.'s published iCEM sampler."""

    config.validate()
    mean = np.asarray(initial_mean, dtype=float).copy()
    std = np.asarray(initial_std, dtype=float).copy()
    if mean.ndim != 2:
        raise ValueError("initial_mean must have shape (horizon, action_dim)")
    if mean.shape[0] != config.horizon or std.shape != mean.shape:
        raise ValueError("mean/std must match the configured horizon")
    lower = np.broadcast_to(np.asarray(lower_bound, dtype=float), mean.shape)
    upper = np.broadcast_to(np.asarray(upper_bound, dtype=float), mean.shape)
    if np.any(lower >= upper):
        raise ValueError("every lower bound must be below its upper bound")

    mean = np.clip(mean, lower, upper)
    std = np.maximum(std, 0.0)
    retained_elites = None
    if previous_elites is not None:
        retained_elites = np.asarray(previous_elites, dtype=float)
        if retained_elites.ndim != 3 or retained_elites.shape[1:] != mean.shape:
            raise ValueError("previous_elites must have shape (elites, horizon, action_dim)")

    population_sizes = icem_population_schedule(config)
    best_sequence = mean.copy()
    best_cost = np.inf

    for iteration, population_size in enumerate(population_sizes):
        if config.noise_beta > 0.0:
            noise = _powerlaw_psd_gaussian(
                config.noise_beta,
                (population_size, mean.shape[1], mean.shape[0]),
                rng=rng,
            ).transpose(0, 2, 1)
        else:
            noise = rng.standard_normal((population_size, *mean.shape))
        samples = np.clip(noise * std + mean, lower, upper)

        if config.use_mean_action and iteration == config.max_iterations - 1:
            samples[0] = mean

        if (
            iteration == 0
            and config.shift_elites_over_time
            and retained_elites is not None
            and retained_elites.shape[0] > 0
        ):
            reuse_count = int(retained_elites.shape[0] * config.fraction_elites_reused)
            if reuse_count > 0:
                shifted = retained_elites[:reuse_count, 1:].copy()
                if config.noise_beta > 0.0:
                    last_noise = _powerlaw_psd_gaussian(
                        config.noise_beta,
                        (reuse_count, mean.shape[1], mean.shape[0]),
                        rng=rng,
                    ).transpose(0, 2, 1)[:, -1:]
                else:
                    last_noise = rng.standard_normal((reuse_count, 1, mean.shape[1]))
                last_actions = np.clip(
                    last_noise * std[-1:] + mean[-1:],
                    lower[-1:],
                    upper[-1:],
                )
                samples = np.concatenate(
                    [samples, np.concatenate([shifted, last_actions], axis=1)],
                    axis=0,
                )

        costs = np.asarray(cost_function(samples.reshape(samples.shape[0], -1)), dtype=float)
        if costs.shape != (samples.shape[0],):
            raise ValueError(
                f"cost_function must return shape ({samples.shape[0]},), got {costs.shape}"
            )
        costs = np.where(np.isfinite(costs), costs, np.inf)

        if (
            iteration > 0
            and config.keep_previous_elites
            and retained_elites is not None
            and retained_elites.shape[0] > 0
        ):
            reuse_count = int(retained_elites.shape[0] * config.fraction_elites_reused)
            if reuse_count > 0:
                reused = retained_elites[:reuse_count]
                reused_costs = np.asarray(
                    cost_function(reused.reshape(reuse_count, -1)),
                    dtype=float,
                )
                samples = np.concatenate([samples, reused], axis=0)
                costs = np.concatenate([costs, np.where(np.isfinite(reused_costs), reused_costs, np.inf)])

        best_index = int(np.argmin(costs))
        best_sequence = samples[best_index].copy()
        best_cost = float(costs[best_index])
        elite_indices = np.argsort(costs)[: config.num_elites]
        retained_elites = samples[elite_indices].copy()
        elite_mean = retained_elites.mean(axis=0)
        elite_std = retained_elites.std(axis=0)
        mean = (1.0 - config.alpha) * elite_mean + config.alpha * mean
        std = (1.0 - config.alpha) * elite_std + config.alpha * std

    return ICEMOptimizationResult(
        solution=best_sequence.reshape(-1),
        variance=np.square(std.reshape(-1)),
        cost=best_cost,
        iterations=config.max_iterations,
        converged=bool(np.isfinite(best_cost)),
        mean=mean.copy(),
        elites=retained_elites.copy(),
        population_sizes=population_sizes,
    )


class _BatchRealVesselModel:
    """Vectorized equivalent of RealVesselNetwork lookup, flow, and clipping."""

    def __init__(self, network):
        self.network = network
        self._candidate_cache = {}

    def _cell_indices(self, positions):
        denominator = self.network._bbox_max - self.network._bbox_min
        denominator = np.where(denominator < EPS, EPS, denominator)
        cells = (
            (positions - self.network._bbox_min)
            / denominator
            * self.network._grid_res
        ).astype(int)
        return np.clip(cells, 0, self.network._grid_res - 1)

    def _candidate_arrays(self, key):
        key = tuple(int(value) for value in key)
        cached = self._candidate_cache.get(key)
        if cached is not None:
            return cached

        candidates = self.network._grid.get(key) or self.network.segments
        cached = (
            np.asarray([segment.start for segment in candidates], dtype=float),
            np.asarray([segment._axis for segment in candidates], dtype=float),
            np.asarray([segment._length for segment in candidates], dtype=float),
            np.asarray([segment.radius for segment in candidates], dtype=float),
            np.asarray([segment.v_mean for segment in candidates], dtype=float),
        )
        self._candidate_cache[key] = cached
        return cached

    def geometry(self, positions):
        positions = np.asarray(positions, dtype=float)
        count = positions.shape[0]
        nearest_distance = np.empty(count, dtype=float)
        nearest_closest = np.empty((count, 3), dtype=float)
        nearest_radial = np.empty((count, 3), dtype=float)
        nearest_axis = np.empty((count, 3), dtype=float)
        nearest_radius = np.empty(count, dtype=float)
        nearest_v_mean = np.empty(count, dtype=float)
        inside_any = np.empty(count, dtype=bool)

        cells = self._cell_indices(positions)
        unique_cells, inverse = np.unique(cells, axis=0, return_inverse=True)
        for cell_index, cell in enumerate(unique_cells):
            row_indices = np.flatnonzero(inverse == cell_index)
            points = positions[row_indices]
            starts, axes, lengths, radii, v_means = self._candidate_arrays(cell)

            offsets = points[:, None, :] - starts[None, :, :]
            fractions = np.einsum("gci,ci->gc", offsets, axes)
            fractions /= lengths[None, :] + EPS
            fractions = np.clip(fractions, 0.0, 1.0)
            closest = (
                starts[None, :, :]
                + fractions[:, :, None] * lengths[None, :, None] * axes[None, :, :]
            )
            radial = points[:, None, :] - closest
            distances = np.linalg.norm(radial, axis=2)
            local_nearest = np.argmin(distances, axis=1)
            local_rows = np.arange(points.shape[0])

            nearest_distance[row_indices] = distances[local_rows, local_nearest]
            nearest_closest[row_indices] = closest[local_rows, local_nearest]
            nearest_radial[row_indices] = radial[local_rows, local_nearest]
            nearest_axis[row_indices] = axes[local_nearest]
            nearest_radius[row_indices] = radii[local_nearest]
            nearest_v_mean[row_indices] = v_means[local_nearest]
            inside_any[row_indices] = np.any(
                distances <= radii[None, :] + 1e-9,
                axis=1,
            )

        return {
            "distance": nearest_distance,
            "closest": nearest_closest,
            "radial": nearest_radial,
            "axis": nearest_axis,
            "radius": nearest_radius,
            "v_mean": nearest_v_mean,
            "inside": inside_any,
        }

    def clip(self, positions, geometry=None):
        positions = np.asarray(positions, dtype=float)
        geometry = self.geometry(positions) if geometry is None else geometry
        clipped = positions.copy()
        outside = ~geometry["inside"]
        if np.any(outside):
            distance = geometry["distance"][outside]
            radial = geometry["radial"][outside]
            radial_hat = np.divide(
                radial,
                distance[:, None],
                out=np.tile(np.array([1.0, 0.0, 0.0]), (distance.size, 1)),
                where=distance[:, None] >= EPS,
            )
            clipped[outside] = (
                geometry["closest"][outside]
                + radial_hat * geometry["radius"][outside, None] * 0.98
            )
        return clipped

    def flow_velocity(self, positions, time):
        geometry = self.geometry(positions)
        velocity = np.zeros_like(positions, dtype=float)
        inside = geometry["distance"] <= geometry["radius"]
        if np.any(inside):
            radius = geometry["radius"][inside]
            profile = 1.0 - np.square(geometry["distance"][inside] / radius)
            pulsation = 1.0 + 0.15 * np.sin(2.0 * np.pi * float(time))
            speed = 2.0 * np.abs(geometry["v_mean"][inside]) * profile * pulsation
            velocity[inside] = speed[:, None] * geometry["axis"][inside]
        return velocity

    def wall_penalty(self, positions, wall_buffer_m):
        geometry = self.geometry(positions)
        buffer = 0.15 * geometry["radius"]
        if wall_buffer_m is not None:
            buffer = np.maximum(buffer, float(wall_buffer_m))
        margin = geometry["radius"] - geometry["distance"]
        return np.maximum(0.0, buffer - margin)


def _batch_magnetic_force(particle, positions, requested_currents, magnet_array):
    """Evaluate the repository's regularized dipole model for a sample batch."""

    positions = np.asarray(positions, dtype=float)
    requested = np.asarray(requested_currents, dtype=float)
    if magnet_array.enforce_hardware_limits:
        resistances = magnet_array._coil_resistances_now()
        limits = np.minimum.reduce([
            magnet_array.max_current * magnet_array._thermal_current_derating(),
            magnet_array.supply_voltage_limit / np.maximum(resistances, EPS),
            np.sqrt(magnet_array.max_coil_power_w / np.maximum(resistances, EPS)),
        ])
        currents = np.clip(requested, -limits, limits)
    else:
        currents = np.clip(requested, -magnet_array.max_current, magnet_array.max_current)

    def fields_at(points):
        relative = points[:, None, :] - magnet_array.positions[None, :, :]
        distance_sq = np.sum(np.square(relative), axis=2)
        distance = np.sqrt(distance_sq)
        direction = np.divide(
            relative,
            distance[:, :, None],
            out=np.zeros_like(relative),
            where=distance[:, :, None] >= 1e-10,
        )
        moments = (
            magnet_array.moment_per_amp
            * currents[:, :, None]
            * magnet_array.orientations[None, :, :]
        )
        moment_projection = np.sum(moments * direction, axis=2)
        denominator = np.power(distance_sq + magnet_array._a_sq, 1.5)
        contributions = magnet_array._mu_over_4pi * (
            3.0 * moment_projection[:, :, None] * direction - moments
        ) / denominator[:, :, None]
        air_field = np.sum(contributions, axis=1)

        magnitude = np.linalg.norm(air_field, axis=1)
        effective_magnitude = magnet_array.core_saturation_tesla * np.tanh(
            magnet_array.core_relative_permeability
            * magnitude
            / (magnet_array.core_saturation_tesla + EPS)
        )
        return np.divide(
            air_field * effective_magnitude[:, None],
            magnitude[:, None],
            out=air_field.copy(),
            where=magnitude[:, None] >= EPS,
        )

    field = fields_at(positions)
    gradient = np.empty_like(positions)
    h = 1e-4
    for axis in range(3):
        offset = np.zeros(3, dtype=float)
        offset[axis] = h
        plus = fields_at(positions + offset)
        minus = fields_at(positions - offset)
        gradient[:, axis] = (
            np.sum(np.square(plus), axis=1) - np.sum(np.square(minus), axis=1)
        ) / (2.0 * h)

    field_magnitude = np.linalg.norm(field, axis=1)
    force = particle.force_prefactor_linear * gradient
    saturated = field_magnitude >= particle.B_sat
    if np.any(saturated):
        force[saturated] = (
            particle.force_prefactor_sat
            * gradient[saturated]
            / (2.0 * field_magnitude[saturated, None])
        )
    return force


def _batch_magnetic_velocity(particle, force):
    force_magnitude = np.linalg.norm(force, axis=1)
    gamma = np.full(force_magnitude.shape, particle.drag_stokes, dtype=float)
    finite_re = force_magnitude >= EPS
    if np.any(finite_re):
        active_force = force_magnitude[finite_re]
        reynolds = (
            particle.rho
            * (active_force / particle.drag_stokes)
            * particle.diameter
            / particle.eta
        )
        for _ in range(15):
            correction = 1.0 + 0.15 * np.power(reynolds, 0.687)
            active_gamma = particle.drag_stokes * correction
            new_reynolds = (
                particle.rho
                * (active_force / active_gamma)
                * particle.diameter
                / particle.eta
            )
            if np.all(np.abs(new_reynolds - reynolds) < 1e-4 * (reynolds + 1e-30)):
                reynolds = new_reynolds
                break
            reynolds = new_reynolds
        gamma[finite_re] = particle.drag_stokes * (
            1.0 + 0.15 * np.power(reynolds, 0.687)
        )
    return np.divide(
        force,
        gamma[:, None],
        out=np.zeros_like(force),
        where=force_magnitude[:, None] >= EPS,
    )


def cross_entropy_optimize(
    cost_function: Callable[[np.ndarray], np.ndarray],
    initial_mean,
    initial_variance,
    lower_bound,
    upper_bound,
    *,
    population_size: int,
    num_elites: int,
    max_iterations: int,
    alpha: float,
    epsilon: float,
    rng: np.random.Generator,
) -> CEMOptimizationResult:
    """Minimize a batched cost using the CEM update released with PETS.

    ``cost_function`` receives an array shaped ``(population, solution_dim)``
    and returns one scalar cost per row.  Sampling is from a standard normal
    truncated to +/-2 standard deviations.  Variance is additionally limited
    by distance to the action bounds, exactly as in the authors' optimizer.
    """

    mean = np.asarray(initial_mean, dtype=float).copy()
    variance = np.asarray(initial_variance, dtype=float).copy()
    lower = np.broadcast_to(np.asarray(lower_bound, dtype=float), mean.shape)
    upper = np.broadcast_to(np.asarray(upper_bound, dtype=float), mean.shape)

    if mean.ndim != 1:
        raise ValueError("initial_mean must be one-dimensional")
    if variance.shape != mean.shape:
        raise ValueError("initial_variance must match initial_mean")
    if np.any(lower >= upper):
        raise ValueError("every lower bound must be below its upper bound")
    if not 1 <= int(num_elites) <= int(population_size):
        raise ValueError("num_elites must be in [1, population_size]")

    mean = np.clip(mean, lower, upper)
    variance = np.maximum(variance, 0.0)
    iterations = 0

    while iterations < int(max_iterations) and float(np.max(variance)) > float(epsilon):
        lower_distance = mean - lower
        upper_distance = upper - mean
        constrained_variance = np.minimum(
            np.minimum(np.square(lower_distance / 2.0), np.square(upper_distance / 2.0)),
            variance,
        )

        standardized = truncnorm.rvs(
            -2.0,
            2.0,
            size=(int(population_size), mean.size),
            random_state=rng,
        )
        samples = standardized * np.sqrt(constrained_variance) + mean
        costs = np.asarray(cost_function(samples), dtype=float)

        if costs.shape != (int(population_size),):
            raise ValueError(
                "cost_function must return shape "
                f"({int(population_size)},), got {costs.shape}"
            )

        costs = np.where(np.isfinite(costs), costs, np.inf)
        elite_indices = np.argpartition(costs, int(num_elites) - 1)[: int(num_elites)]
        elites = samples[elite_indices]
        elite_mean = np.mean(elites, axis=0)
        elite_variance = np.var(elites, axis=0)

        mean = float(alpha) * mean + (1.0 - float(alpha)) * elite_mean
        variance = float(alpha) * variance + (1.0 - float(alpha)) * elite_variance
        iterations += 1

    solution = np.clip(mean, lower, upper)
    solution_cost = np.asarray(cost_function(solution[None, :]), dtype=float)
    if solution_cost.shape != (1,):
        raise ValueError("cost_function must accept a one-row batch")

    cost = float(solution_cost[0])
    return CEMOptimizationResult(
        solution=solution,
        variance=variance,
        cost=cost,
        iterations=iterations,
        converged=bool(np.isfinite(cost)),
    )


class PETSCEMMPCController:
    """Stateful, warm-started CEM-MPC current controller.

    The controller optimizes a horizon of signed coil-current vectors, returns
    only the first vector, and shifts the remaining mean forward to seed the
    next control step.  This is the receding-horizon behavior in the released
    PETS controller.
    """

    def __init__(self, action_dim: int, max_current: float, config=None, rng_seed: int = 42):
        self.action_dim = int(action_dim)
        self.max_current = float(max_current)
        self.config = PETSCEMConfig() if config is None else config
        self.config.validate()
        if self.action_dim < 1:
            raise ValueError("action_dim must be at least 1")
        if self.max_current <= 0.0:
            raise ValueError("max_current must be positive")

        self.rng = np.random.default_rng(int(rng_seed))
        self.solution_dim = self.config.horizon * self.action_dim
        self.lower_bound = np.full(self.solution_dim, -self.max_current, dtype=float)
        self.upper_bound = np.full(self.solution_dim, self.max_current, dtype=float)
        self.initial_variance = np.square(self.upper_bound - self.lower_bound) / 16.0
        self.previous_solution = np.zeros(self.solution_dim, dtype=float)
        self.last_result: CEMOptimizationResult | None = None
        self._batch_vessel_model: _BatchRealVesselModel | None = None

    def reset(self) -> None:
        self.previous_solution.fill(0.0)
        self.last_result = None

    def _constraint_costs(self, positions, currents=None):
        """Optional separate constraint channel for constrained-CEM subclasses."""
        return np.zeros(len(positions), dtype=float)

    def _optimize(self, trajectory_costs):
        cfg = self.config
        return cross_entropy_optimize(
            trajectory_costs,
            self.previous_solution,
            self.initial_variance,
            self.lower_bound,
            self.upper_bound,
            population_size=cfg.population_size,
            num_elites=cfg.num_elites,
            max_iterations=cfg.max_iterations,
            alpha=cfg.alpha,
            epsilon=cfg.epsilon,
            rng=self.rng,
        )

    def _update_warm_start(self, result) -> None:
        solution = result.solution.reshape(self.config.horizon, self.action_dim)
        shifted = np.vstack([solution[1:], np.zeros((1, self.action_dim), dtype=float)])
        self.previous_solution = shifted.reshape(-1)

    @staticmethod
    def _stage_cost(
        currents,
        particle,
        particle_pos,
        target_pos,
        magnet_array,
        workspace,
        dt,
        w_progress,
        w_align,
        w_dist,
        w_curr,
        w_smooth,
        prev_currents,
        w_edge,
        w_overshoot,
        flow,
        sim_time,
        w_route,
        route_tangent,
        wall_buffer_m,
        return_unclipped=False,
    ):
        """Return the repository's shared stage cost and deterministic next state."""

        particle_pos = np.asarray(particle_pos, dtype=float)
        target_pos = np.asarray(target_pos, dtype=float)
        currents = np.asarray(currents, dtype=float)
        desired = target_pos - particle_pos
        current_distance = float(np.linalg.norm(desired))

        next_pos, force, step_vec = _simulate_one_step(
            particle=particle,
            particle_pos=particle_pos,
            currents=currents,
            magnet_array=magnet_array,
            workspace=workspace,
            dt=dt,
            flow=flow,
            sim_time=sim_time,
        )

        if current_distance < EPS and not return_unclipped:
            return 0.0, next_pos

        desired_hat = desired / max(current_distance, EPS)
        next_distance = float(np.linalg.norm(target_pos - next_pos))
        progress = current_distance - next_distance
        step_along_target = float(np.dot(step_vec, desired_hat))
        force_norm = float(np.linalg.norm(force))
        alignment = float(np.dot(desired_hat, force / (force_norm + EPS)))
        wall_cost = _wall_penalty(
            particle_pos + step_vec,
            workspace,
            wall_buffer_m=wall_buffer_m,
        )
        overshoot_cost = _overshoot_penalty(step_vec, current_distance)
        smoothness_cost = float(np.sum((currents - prev_currents) ** 2))

        route_align_cost = 0.0
        if w_route > 0.0 and route_tangent is not None:
            tangent = np.asarray(route_tangent, dtype=float)
            tangent_norm = float(np.linalg.norm(tangent))
            if tangent_norm > EPS:
                route_align_cost = -float(np.dot(step_vec, tangent / tangent_norm))

        cost = (
            -w_progress * progress
            -w_align * step_along_target
            -0.5 * alignment
            + w_dist * next_distance**2
            + w_curr * float(np.sum(currents**2))
            + w_smooth * smoothness_cost
            + w_edge * wall_cost
            + w_overshoot * overshoot_cost
            + w_route * route_align_cost
        )
        if return_unclipped:
            return (0.0 if current_distance < EPS else float(cost)), next_pos, particle_pos + step_vec
        return float(cost), next_pos

    def command(
        self,
        *,
        particle,
        particle_pos,
        target_pos,
        magnet_array,
        workspace,
        dt: float,
        previous_currents=None,
        w_progress: float = 120.0,
        w_align: float = 8.0,
        w_dist: float = 1.0,
        w_curr: float = 1e-5,
        w_smooth: float = 1e-4,
        w_edge: float = 50.0,
        w_overshoot: float = 2.0,
        w_route: float = 0.0,
        route_tangent=None,
        wall_buffer_m=None,
        flow=None,
        sim_time: float = 0.0,
    ):
        particle_pos = np.asarray(particle_pos, dtype=float)
        target_pos = np.asarray(target_pos, dtype=float)

        if (
            np.linalg.norm(target_pos - particle_pos) < EPS
            and not getattr(self, "separate_constraints", False)
        ):
            currents = np.zeros(self.action_dim, dtype=float)
            magnet_array.set_currents(currents)
            return currents, np.zeros(3, dtype=float), 0.0, True

        if flow is None:
            flow = NoFlow()
        if previous_currents is None:
            previous_currents = np.zeros(self.action_dim, dtype=float)
        else:
            previous_currents = np.asarray(previous_currents, dtype=float)

        network = getattr(workspace, "network", None)
        use_vectorized_rollout = bool(
            network is not None
            and hasattr(network, "_grid")
            and hasattr(particle, "force_prefactor_linear")
            and hasattr(magnet_array, "_a_sq")
        )
        if use_vectorized_rollout and (
            self._batch_vessel_model is None
            or self._batch_vessel_model.network is not network
        ):
            self._batch_vessel_model = _BatchRealVesselModel(network)

        def trajectory_costs(flat_sequences):
            sequences = np.asarray(flat_sequences, dtype=float).reshape(
                -1,
                self.config.horizon,
                self.action_dim,
            )
            constrained = getattr(self, "separate_constraints", False)
            violations = (
                self._constraint_costs(
                    np.repeat(particle_pos[None, :], len(sequences), axis=0)
                )
                if constrained and self.config.include_initial_state_constraint
                else np.zeros(len(sequences), dtype=float)
                if constrained
                else None
            )

            if use_vectorized_rollout:
                sample_count = sequences.shape[0]
                positions = np.repeat(particle_pos[None, :], sample_count, axis=0)
                prior = np.repeat(previous_currents[None, :], sample_count, axis=0)
                totals = np.zeros(sample_count, dtype=float)
                target = target_pos[None, :]

                for horizon_idx in range(self.config.horizon):
                    currents = sequences[:, horizon_idx, :]
                    force = _batch_magnetic_force(
                        particle,
                        positions,
                        currents,
                        magnet_array,
                    )
                    magnetic_velocity = _batch_magnetic_velocity(particle, force)
                    flow_velocity = self._batch_vessel_model.flow_velocity(
                        positions,
                        sim_time + horizon_idx * dt,
                    )
                    step_vectors = (flow_velocity + magnetic_velocity) * dt
                    unclipped = positions + step_vectors
                    if constrained:
                        violations += self._constraint_costs(unclipped, currents)
                    next_positions = self._batch_vessel_model.clip(unclipped)

                    desired = target - positions
                    current_distance = np.linalg.norm(desired, axis=1)
                    desired_hat = np.divide(
                        desired,
                        current_distance[:, None],
                        out=np.zeros_like(desired),
                        where=current_distance[:, None] >= EPS,
                    )
                    next_distance = np.linalg.norm(target - next_positions, axis=1)
                    progress = current_distance - next_distance
                    step_along_target = np.sum(step_vectors * desired_hat, axis=1)
                    force_norm = np.linalg.norm(force, axis=1)
                    alignment = np.sum(
                        desired_hat
                        * np.divide(
                            force,
                            force_norm[:, None] + EPS,
                        ),
                        axis=1,
                    )
                    wall_cost = self._batch_vessel_model.wall_penalty(
                        unclipped,
                        wall_buffer_m,
                    )
                    overshoot_cost = np.maximum(
                        0.0,
                        np.linalg.norm(step_vectors, axis=1) - 0.65 * current_distance,
                    )
                    smoothness_cost = np.sum(np.square(currents - prior), axis=1)

                    route_align_cost = np.zeros(sample_count, dtype=float)
                    if w_route > 0.0 and route_tangent is not None:
                        tangent = np.asarray(route_tangent, dtype=float)
                        tangent_norm = float(np.linalg.norm(tangent))
                        if tangent_norm > EPS:
                            route_align_cost = -np.sum(
                                step_vectors * (tangent / tangent_norm)[None, :],
                                axis=1,
                            )

                    stage = (
                        -w_progress * progress
                        -w_align * step_along_target
                        -0.5 * alignment
                        + w_dist * np.square(next_distance)
                        + w_curr * np.sum(np.square(currents), axis=1)
                        + w_smooth * smoothness_cost
                        + w_edge * wall_cost
                        + w_overshoot * overshoot_cost
                        + w_route * route_align_cost
                    )
                    stage = np.where(current_distance < EPS, 0.0, stage)
                    totals += stage
                    positions = next_positions
                    prior = currents

                return (totals, violations) if constrained else totals

            totals = np.empty(sequences.shape[0], dtype=float)
            for sample_idx, sequence in enumerate(sequences):
                pos = particle_pos.copy()
                prior = previous_currents
                total = 0.0
                for horizon_idx, currents in enumerate(sequence):
                    stage_result = self._stage_cost(
                        currents,
                        particle,
                        pos,
                        target_pos,
                        magnet_array,
                        workspace,
                        dt,
                        w_progress,
                        w_align,
                        w_dist,
                        w_curr,
                        w_smooth,
                        prior,
                        w_edge,
                        w_overshoot,
                        flow,
                        sim_time + horizon_idx * dt,
                        w_route,
                        route_tangent,
                        wall_buffer_m,
                        return_unclipped=constrained,
                    )
                    stage_cost, pos = stage_result[:2]
                    if constrained:
                        violations[sample_idx] += self._constraint_costs(
                            stage_result[2][None, :], currents[None, :]
                        )[0]
                    total += stage_cost
                    prior = currents
                totals[sample_idx] = total
            return (totals, violations) if constrained else totals

        cfg = self.config
        result = self._optimize(trajectory_costs)
        self.last_result = result

        solution = result.solution.reshape(cfg.horizon, self.action_dim)
        currents = solution[0].copy()
        self._update_warm_start(result)

        magnet_array.set_currents(currents)
        field = magnet_array.field_at(particle_pos)
        gradient = magnet_array.grad_B_squared(particle_pos)
        force = particle.magnetic_force(field, gradient)
        return currents, force, result.cost, result.converged


class ICEMMPCController(PETSCEMMPCController):
    """Pinneri et al. iCEM planner coupled to the shared DynaMag plant model."""

    def __init__(self, action_dim: int, max_current: float, config=None, rng_seed: int = 42):
        config = ICEMConfig() if config is None else config
        super().__init__(
            action_dim=action_dim,
            max_current=max_current,
            config=config,
            rng_seed=rng_seed,
        )
        half_action_range = 0.5 * (self.upper_bound - self.lower_bound)
        self.initial_std = half_action_range * config.init_std
        self.previous_elites: np.ndarray | None = None

    def reset(self) -> None:
        super().reset()
        self.previous_elites = None

    def _optimize(self, trajectory_costs):
        cfg = self.config
        return improved_cross_entropy_optimize(
            trajectory_costs,
            self.previous_solution.reshape(cfg.horizon, self.action_dim),
            self.initial_std.reshape(cfg.horizon, self.action_dim),
            self.lower_bound.reshape(cfg.horizon, self.action_dim),
            self.upper_bound.reshape(cfg.horizon, self.action_dim),
            config=cfg,
            rng=self.rng,
            previous_elites=self.previous_elites,
        )

    def _update_warm_start(self, result) -> None:
        # Upstream shifts the fitted mean and preserves its final action.
        shifted_mean = np.vstack([result.mean[1:], result.mean[-1:]])
        self.previous_solution = shifted_mean.reshape(-1)
        # Elites remain unshifted here; upstream shifts them when sampling the
        # next MPC timestep and draws a new final action from the distribution.
        self.previous_elites = result.elites.copy()
