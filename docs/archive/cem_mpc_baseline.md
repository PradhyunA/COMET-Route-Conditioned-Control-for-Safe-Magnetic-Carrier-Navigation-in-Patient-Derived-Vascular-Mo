# Published CEM-MPC baseline

## Method and citation

The independent published baseline is the CEM-based receding-horizon MPC
controller used by Chua et al.:

> K. Chua, R. Calandra, R. McAllister, and S. Levine, “Deep Reinforcement
> Learning in a Handful of Trials using Probabilistic Dynamics Models,”
> *Advances in Neural Information Processing Systems 31*, 2018.

- [NeurIPS proceedings paper](https://proceedings.neurips.cc/paper/2018/hash/3de568f8597b94bda53149c7d7f5958c-Abstract.html)
- [Authors’ released implementation](https://github.com/kchua/handful-of-trials)

At each control step, the baseline samples bounded coil-current sequences from
a truncated Gaussian, predicts the cumulative horizon cost, selects the lowest
cost elite sequences, and refits the Gaussian mean and variance. It executes
only the first current vector and shifts the remaining optimized sequence to
warm-start the next MPC step.

## Reproduction versus domain adaptation

| Component | DynaMag baseline | Provenance |
|---|---|---|
| Optimizer | Cross-entropy method | Chua et al. (2018) and released code |
| Sampling | Gaussian truncated to +/-2 standard deviations | Released code |
| Bounded variance | Minimum of current variance and squared half-distance to each bound | Released code |
| Distribution update | `alpha * old + (1 - alpha) * elite statistic` | Released code |
| Receding horizon | Execute first action; shift remaining mean; append zeros | Released code |
| Horizon | 25 | Published 3-D reacher configuration |
| Population / elites | 400 / 40 | Published 3-D reacher configuration |
| CEM iterations / alpha | 5 / 0.1 | Published 3-D reacher configuration |
| Dynamics rollout | Shared deterministic DynaMag magnetic, drag, and flow model | Domain adaptation for controlled comparison |
| Action | Nine signed coil currents in `[-16, 16] A` | DynaMag hardware model |
| Cost | Same route, current, smoothness, and wall terms supplied to the other MPC comparison | DynaMag controlled-comparison protocol |

PETS itself learns a probabilistic ensemble dynamics model. This baseline does
not reproduce that learning component because the study provides a known
physics model and the comparison must keep plant predictions identical across
controllers. The paper’s CEM trajectory optimizer and receding-horizon policy
are reproduced; the shared DynaMag physics model replaces the learned neural
model. Results should therefore be named **PETS-style CEM-MPC (Chua et al.,
2018; deterministic-physics adaptation)**, not the full PETS learning system.

## Running the paired benchmark

The published baseline key is `pets_cem_mpc` and is included in the benchmark’s
default controller set.

```bash
source .venv/bin/activate
python scripts/run_publication_benchmark.py \
  --run-id pets_cem_comparison \
  --controllers pets_cem_mpc,mpc_tuned,acbc_tuned,comet \
  --flow-scales 1,3,5 \
  --n-random-trials 10
```

For a quick integration check with reduced computation:

```bash
python scripts/run_publication_benchmark.py \
  --run-id pets_cem_smoke \
  --controllers pets_cem_mpc \
  --smoke-test
```

The full paper settings require `400 * 25 * 5 = 50,000` deterministic model
steps per MPC update. Smoke mode caps the outer simulation at two steps while
deliberately leaving the published CEM settings intact.

## Suggested manuscript wording

“We additionally compared the proposed controllers with a published
sampling-based MPC baseline following the CEM trajectory optimizer and
receding-horizon action selection of Chua et al. (2018). We used the authors’
3-D reacher CEM settings (horizon 25, 400 samples, 40 elites, five iterations,
and smoothing coefficient 0.1). To isolate controller effects, candidate
trajectories were propagated with the same deterministic magnetic, drag, and
flow model used by all methods rather than PETS’s learned probabilistic
ensemble. All controllers used identical starts, targets, vessel geometry,
flow, hardware limits, and success criteria.”
