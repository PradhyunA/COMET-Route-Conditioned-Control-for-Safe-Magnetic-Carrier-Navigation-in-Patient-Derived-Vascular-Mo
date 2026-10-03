# Constrained CEM-MPC baseline

This is a DynaMag adaptation of Oscar Key's **Constrained CEM MPC**, which cites
Wen and Topcu, *Constrained Cross-Entropy Method for Safe Reinforcement Learning*,
NeurIPS 2018. The linked repository is a third-party MPC implementation of the
published idea; it is not represented here as the paper authors' reference code.

- Repository: https://github.com/oscarkey/constrained-cem-mpc
- Inspected commit: `aa162f9e242269ab66c42598e77c28a8d69f00a2`
- Source file: `constrained_cem_mpc/constrained_cem_mpc.py`
- Paper: https://papers.nips.cc/paper/7974-constrained-cross-entropy-method-for-safe-reinforcement-learning
- MIT attribution: `THIRD_PARTY_NOTICES/constrained-cem-mpc.txt`

## Preserved algorithm

At each control step, start a fresh diagonal Gaussian with zero mean and unit
standard deviation in normalized action coordinates. Draw unbounded independent
Gaussian action sequences. Compute objective cost and constraint cost separately.
If at least K samples have exactly zero constraint cost, choose the K lowest
objective-cost feasible samples. Otherwise choose the K lowest constraint-cost
samples, ignoring the objective in that branch. Refit mean and **unbiased**
standard deviation (NumPy ddof=1, corresponding to PyTorch's default std).
Repeat for the fixed iteration budget. Execute the first action of the
lowest-objective feasible sample in the **final** population.

There is no PETS truncated normal, iCEM colored noise, warm start, CEM smoothing,
population decay, COMET branch probability, downstream value, or terminal
precision mechanism. Objective ties/constraint ties use stable input order in
the NumPy port; random samples are seeded but not bit-identical to PyTorch.
Degenerate Gaussian coordinates are permitted, and nonfinite evaluations are
treated as infeasible. These are explicit numerical robustness choices.

The upstream library has no universal search defaults. The registered variants
use its **pendulum demo budget**: H=10, N=50, K=10, 8 iterations (4,000 predicted
transitions per command). These are inherited demo settings, not magnetic-task
optimal settings. See `demos/pendulum_demo.py` in the pinned repository.

## Plant and constraint adaptation

Actual simulation uses the existing dense Dijkstra waypoint navigator, vessel
network, pulsatile flow, 500 µm particle, finite-Re drag, nine-coil hardware,
moving adaptive enclosing gantry, current/voltage/power/thermal limits, and
adaptive numerical integration. The shared MPC prediction model, including its
Euler predictions and vessel projection, is reused. The predictor is not an
exact copy of the adaptive plant integrator, so sampled feasibility is not a
closed-loop safety certificate.

Action units are normalized: I = z × (effective per-coil current limit / 2).
The normalized admissible box is [-2,2], like the upstream pendulum action box.
Initial z ~ N(0,1). Raw proposed currents are checked BEFORE hardware saturation.
The per-coil box includes the existing current, voltage, power, and thermal
derating limits, frozen during that command's prediction horizon.

State constraints use the benchmark's nearest-segment center-to-wall clearance:
clearance must be at least max(particle radius, configured benchmark clearance).
The pilot threshold is 0.7 mm from the particle center. This also keeps the
500 µm particle surface inside the vessel. Each violating predicted state costs
3 and each violating action vector costs 3, matching upstream StateConstraint
and ActionConstraint counting. The initial state and every UNPROJECTED next
state are checked, preventing vessel projection from concealing a violation.
This replaces the source's polytopic safe region with DynaMag vessel geometry.
It checks sampled states, not continuous swept-volume containment.

If no final-population trajectory is feasible, upstream returns None. This
adapter records the failure, returns `converged=False`, and applies zero current
so the benchmark continues. **Zero current is not certified safe in flow.**
The fallback is not counted as a feasible plan. An initial state violating the
clearance constraint makes every full trajectory infeasible, as in upstream.
There is no terminal invariant safe set and no claim of exact reproduction or
transfer of the source paper's guarantees to this nonlinear plant.

## Objectives and tuning decision

- `constrained_cem_mpc`: existing default shared MPC weights.
- `constrained_cem_mpc_tuned_shared`: existing ACBC/MPC shared tuned weights.
- `constrained_cem_mpc_tuned_recovery`: dimension-aware feasibility adaptation
  plus the existing shared tuned weights. This is the recommended pilot variant.

The objective-only variant reuses already defined weights; no weights were fit
to these five cases. It could not repair the original feasibility collapse,
because objective cost is ignored while fewer than K trajectories are feasible.

The recovery-tuned variant makes four disclosed adaptations:

1. N=100, K=10, H=10 and eight CEM iterations; initial normalized std=0.75.
2. Componentwise rejection sampling inside normalized current bounds [-2,2],
   eliminating the exponentially unlikely 90-variable action-feasibility event.
3. Two action blocks over H=10: one nine-coil vector for predicted steps 1–5
   and one for steps 6–10. Receding-horizon execution still applies one action.
4. The hard state constraint is particle-center clearance >= particle radius
   (0.25 mm). It evaluates predicted states, excluding the already-realized
   initial state so recovery remains possible. The common 0.7 mm clearance target
   remains in the shared soft objective and benchmark safety metrics. If no
   feasible final sample exists, the least-violating final-population sample is
   applied and explicitly recorded; it is not counted as feasible or safe.

This is labeled a tuned adaptation. It retains separate constraint/objective
channels and the source feasible-first elite rule, but is less literal than
`constrained_cem_mpc`. The weights are the repository's pre-existing shared
tuned MPC weights, not values fitted to this pilot.

## Five-case pilot

The script reads exact coordinates from `benchmark_manifests/icem_45case_500um`:

| Anatomy | Frozen route | Flow |
|---|---|---|
| CM | canonical_001_inlet_to_outlet | 1×, 5× (2 cases) |
| Aortic 40Left85Right | canonical_001_medium_40Left85Right_endpoint03 | 1×, 5× (2 cases) |
| Pulmonary 0082 | canonical_001_inlet_to_outlet | 3× (1 case) |

Exactly five route-flow cases per controller. Repeated routes are paired stress
conditions, not five independent patients. The selection was fixed before smoke
testing and is not a full 45-case performance estimate. Seed=20260611, dt=0.02 s,
maximum 2000 steps, target tolerance=0.5 mm, stagnation early stopping disabled.

From the repository root:

```bash
bash run_constrained_cem_mpc_5trials.sh
# Original sampler plus existing shared tuned objective (still five runs):
OBJECTIVE=tuned bash run_constrained_cem_mpc_5trials.sh
# Recommended recovery/dimension-tuned variant:
bash run_constrained_cem_mpc_tuned_5trials.sh
# Paired objective comparison: same five cases each, ten runs total:
OBJECTIVE=both WORKERS=2 bash run_constrained_cem_mpc_5trials.sh
# Original versus recovery-tuned: ten runs total:
OBJECTIVE=compare-recovery bash run_constrained_cem_mpc_5trials.sh
# Two-step wiring smoke on the same cases (not performance results):
bash run_constrained_cem_mpc_5trials.sh --smoke --objective both
```

The launcher prevents Mac sleep while running. Outputs go to a timestamped
`outputs/constrained_cem/pilot_*` directory; a pre-existing output directory is
rejected to avoid overwriting results. `--output-root PATH` selects a fresh path.
Each anatomy has a `paired_results.csv`; each trial has trajectory NPZ and a
`*_constrained_cem.json` with per-iteration feasible counts, minimum constraint
cost, and fallback flags. `pilot_summary.json` reports success count, median/max
final error, median steps, mean effective steps (all used steps, including the
2000-step failures), wall-exit cases, and no-feasible-plan frequency.

Validation uses tests for the source selection branches, unbounded Gaussian
sampling, unbiased refitting, selection from the final sampled population,
infeasibility, nonfinite evaluation rejection, raw-state/action violations,
hardware-bounded applied currents, and unchanged objective variant settings.
Existing PETS/iCEM tests also run because the prediction machinery is shared.

Initial validation on 2026-09-04: 26 focused regression tests passed. Two-step smoke
rollouts completed on all five cases for both objective variants (10 rollouts).
Each variant had 6 no-feasible-plan fallbacks in 10 controller calls; the two
variants selected identical actions in this short check. This flags feasibility
as an issue for the pilot and does not establish tracking success or a tuned
variant advantage. Smoke output: `outputs/constrained_cem/smoke_20260904_225546`.

Recovery-tuned validation: 30 focused regression tests passed. The final
two-step smoke completed all five cases with 10/10 feasible controller calls,
zero best-effort fallbacks, zero zero-current fallbacks, and zero wall exits.
Smoke output: `outputs/constrained_cem/smoke_20260904_233931`. A ten-step CM 5x
probe also retained 10/10 feasible calls and ended with 1.80 mm wall clearance;
its local waypoint error was 3.40 mm at step 10. This verifies the original
feasibility collapse is removed, not that all five full trials will succeed.
