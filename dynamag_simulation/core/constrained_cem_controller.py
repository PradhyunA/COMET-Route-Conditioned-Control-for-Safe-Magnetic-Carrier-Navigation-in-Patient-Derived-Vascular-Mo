"""Wen–Topcu-based constrained CEM, adapted from Oscar Key's MIT-licensed MPC.

Reference: oscarkey/constrained-cem-mpc, commit
aa162f9e242269ab66c42598e77c28a8d69f00a2. See docs/constrained_cem_mpc.md
and THIRD_PARTY_NOTICES/constrained-cem-mpc.txt for provenance and adaptation.
"""
from dataclasses import dataclass

import numpy as np

from .cem_mpc_controller import CEMOptimizationResult, PETSCEMMPCController
from .constants import EPS


REFERENCE_COMMIT = "aa162f9e242269ab66c42598e77c28a8d69f00a2"


@dataclass(frozen=True)
class ConstrainedCEMConfig:
    # Reference pendulum demo's search budget (no global defaults upstream).
    horizon: int = 10
    population_size: int = 50
    num_elites: int = 10
    max_iterations: int = 8
    initial_std: float = 1.0
    sample_within_action_bounds: bool = False
    include_initial_state_constraint: bool = True
    best_effort_on_infeasible: bool = False
    hard_clearance_m: float | None = None
    action_blocks: int | None = None

    def validate(self):
        if self.horizon < 1 or self.max_iterations < 1:
            raise ValueError("horizon and max_iterations must be positive")
        if not 2 <= self.num_elites <= self.population_size:
            raise ValueError("num_elites must be in [2, population_size]")
        if self.initial_std <= 0.0:
            raise ValueError("initial_std must be positive")
        if self.hard_clearance_m is not None and self.hard_clearance_m < 0.0:
            raise ValueError("hard_clearance_m must be nonnegative")
        if self.action_blocks is not None and not 1 <= self.action_blocks <= self.horizon:
            raise ValueError("action_blocks must be in [1, horizon]")


@dataclass(frozen=True)
class ConstrainedCEMResult(CEMOptimizationResult):
    feasible_count: int
    min_constraint_cost: float
    constraint_cost: float
    feasible_counts: tuple[int, ...]


def select_constrained_elites(objectives, constraints, num_elites):
    """Reference's two-branch selection, NOT a weighted penalty objective.

    When fewer than num_elites are feasible, rank ONLY by constraint cost.
    Stable input-order ties are the NumPy port's explicit tie convention.
    """
    feasible = np.flatnonzero(constraints == 0)
    if len(feasible) >= num_elites:
        return feasible[np.argsort(objectives[feasible], kind="stable")[:num_elites]]
    return np.argsort(constraints, kind="stable")[:num_elites]


def _sample_gaussian(rng, mean, std, population_size, *, lower=None, upper=None):
    """Sample a diagonal Gaussian, optionally by componentwise rejection."""
    shape = (int(population_size), *mean.shape)
    samples = rng.normal(loc=mean, scale=std, size=shape)
    if lower is None or upper is None:
        return samples

    lower = np.broadcast_to(np.asarray(lower, dtype=float), mean.shape)
    upper = np.broadcast_to(np.asarray(upper, dtype=float), mean.shape)
    invalid = (samples < lower) | (samples > upper)
    for _ in range(100):
        if not np.any(invalid):
            return samples
        rows, horizon, actions = np.nonzero(invalid)
        samples[rows, horizon, actions] = rng.normal(
            loc=mean[horizon, actions],
            scale=std[horizon, actions],
        )
        invalid = (samples < lower) | (samples > upper)
    raise RuntimeError("bounded Gaussian rejection sampler did not converge")


def constrained_cross_entropy_optimize(evaluate, *, config, action_dim, rng):
    """Fresh Gaussian CEM with unbiased refitting and final-batch selection.

    Actions here are dimensionless; the plant adapter supplies physical units.
    The reference variant is unbounded N(0,1). The tuned adaptation can use a
    smaller initial deviation and componentwise truncated sampling. There is no
    smoothing, warm start, elite reuse, or variance early stop. Infeasibility
    returns False with either zero action or a disclosed minimum-violation sample.
    """
    config.validate()
    shape = (config.action_blocks or config.horizon, action_dim)
    mean = np.zeros(shape)
    std = np.full(shape, config.initial_std, dtype=float)
    feasible_counts = []
    for _ in range(config.max_iterations):
        lower, upper = (-2.0, 2.0) if config.sample_within_action_bounds else (None, None)
        samples = _sample_gaussian(
            rng,
            mean,
            std,
            config.population_size,
            lower=lower,
            upper=upper,
        )
        objectives, constraints = (np.asarray(x, dtype=float) for x in evaluate(samples))
        if objectives.shape != (config.population_size,) or constraints.shape != objectives.shape:
            raise ValueError("evaluate must return two (population_size,) arrays")
        valid = np.isfinite(objectives) & np.isfinite(constraints) & (constraints >= 0)
        objectives = np.where(valid, objectives, np.inf)
        constraints = np.where(valid, constraints, np.inf)
        ids = select_constrained_elites(objectives, constraints, config.num_elites)
        mean = samples[ids].mean(axis=0)
        # torch.Tensor.std(dim=0) uses Bessel's correction (ddof=1).
        std = samples[ids].std(axis=0, ddof=1)
        feasible_counts.append(int(np.sum(constraints == 0)))
    feasible = np.flatnonzero(constraints == 0)
    found = len(feasible) > 0
    if found:
        best = feasible[np.argmin(objectives[feasible])]
    elif config.best_effort_on_infeasible:
        minimum = np.min(constraints)
        least_violating = np.flatnonzero(constraints == minimum)
        best = least_violating[np.argmin(objectives[least_violating])]
    else:
        best = None
    return ConstrainedCEMResult(
        solution=samples[best].reshape(-1).copy() if best is not None else np.zeros(np.prod(shape)),
        variance=np.square(std).reshape(-1),
        cost=float(objectives[best]) if best is not None else np.inf,
        iterations=config.max_iterations,
        converged=found,
        feasible_count=len(feasible),
        min_constraint_cost=float(np.min(constraints)),
        constraint_cost=float(constraints[best]) if best is not None else np.inf,
        feasible_counts=tuple(feasible_counts),
    )


class ConstrainedCEMMPCController(PETSCEMMPCController):
    """Shared DynaMag rollout/objective with a separate vessel/current constraint."""

    separate_constraints = True

    def __init__(self, action_dim, max_current, config=None, rng_seed=42):
        super().__init__(action_dim, max_current, config or ConstrainedCEMConfig(), rng_seed)
        self.diagnostics = []

    def reset(self):
        super().reset()
        self.diagnostics.clear()

    def _update_warm_start(self, result):
        # Upstream resets the Gaussian at every MPC step.
        pass

    def _constraint_costs(self, positions, currents=None):
        # Match the benchmark's nearest-segment center-clearance convention.
        # Check UNPROJECTED predictions: workspace clipping cannot hide a breach.
        if self._batch_vessel_model is not None:
            geometry = self._batch_vessel_model.geometry(positions)
            margins = geometry["radius"] - geometry["distance"]
        elif getattr(self._workspace, "network", None) is not None:
            margins = []
            for pos in positions:
                seg, distance = self._workspace.network.find_segment(pos)
                margins.append(-np.inf if seg is None else seg.radius - distance)
            margins = np.asarray(margins)
        else:
            margins = np.min(self._workspace.half - np.abs(positions), axis=1)
        costs = 3.0 * (margins < self._clearance)
        if currents is not None:
            costs += 3.0 * np.any(np.abs(currents) > self._current_limits, axis=1)
        return costs

    def _optimize(self, trajectory_costs):
        def evaluate(normalized):
            if self.config.action_blocks is not None:
                repeats = int(np.ceil(self.config.horizon / self.config.action_blocks))
                normalized = np.repeat(normalized, repeats, axis=1)[:, :self.config.horizon]
            return trajectory_costs(
                (normalized * self._action_scale).reshape(len(normalized), -1)
            )

        result = constrained_cross_entropy_optimize(
            evaluate, config=self.config, action_dim=self.action_dim, rng=self.rng,
        )
        from dataclasses import replace
        sampled_horizon = self.config.action_blocks or self.config.horizon
        sampled = result.solution.reshape(sampled_horizon, self.action_dim)
        sampled_variance = result.variance.reshape(sampled_horizon, self.action_dim)
        if self.config.action_blocks is not None:
            repeats = int(np.ceil(self.config.horizon / self.config.action_blocks))
            sampled = np.repeat(sampled, repeats, axis=0)[:self.config.horizon]
            sampled_variance = np.repeat(
                sampled_variance, repeats, axis=0
            )[:self.config.horizon]
        scales = np.broadcast_to(
            self._action_scale, (self.config.horizon, self.action_dim)
        )
        return replace(
            result,
            solution=(sampled * scales).reshape(-1),
            variance=(sampled_variance * scales**2).reshape(-1),
        )

    def command(self, **kwargs):
        magnets = kwargs["magnet_array"]
        self._workspace = kwargs["workspace"]
        particle_radius = 0.5 * kwargs["particle"].diameter
        if self.config.hard_clearance_m is None:
            self._clearance = max(
                particle_radius,
                float(kwargs.get("wall_buffer_m") or 0.0),
            )
        else:
            self._clearance = max(particle_radius, self.config.hard_clearance_m)
        self._current_limits = np.full(self.action_dim, min(self.max_current, magnets.max_current))
        if magnets.enforce_hardware_limits:
            resistance = magnets._coil_resistances_now()
            self._current_limits = np.minimum.reduce([
                self._current_limits,
                magnets.max_current * magnets._thermal_current_derating(),
                magnets.supply_voltage_limit / np.maximum(resistance, EPS),
                np.sqrt(magnets.max_coil_power_w / np.maximum(resistance, EPS)),
            ])
        # Dimensionless bounds [-2,2], as in the upstream pendulum action box.
        # This is a disclosed units adaptation, not a fit to benchmark outcomes.
        self._action_scale = self._current_limits / 2.0
        self.last_result = None
        output = super().command(**kwargs)
        result = self.last_result
        self.diagnostics.append({
            "feasible": bool(output[3]),
            "feasible_count": result.feasible_count if result is not None else 0,
            "min_constraint_cost": result.min_constraint_cost if result is not None else 0.0,
            "zero_current_fallback": bool(
                result is not None
                and not result.converged
                and not self.config.best_effort_on_infeasible
            ),
            "best_effort_fallback": bool(
                result is not None
                and not result.converged
                and self.config.best_effort_on_infeasible
            ),
            "selected_constraint_cost": (
                result.constraint_cost if result is not None else 0.0
            ),
            "feasible_counts_by_iteration": list(result.feasible_counts) if result is not None else [],
        })
        return output
