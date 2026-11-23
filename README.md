# DynaMag: COMET magnetic carrier navigation

Simulation code, benchmark and results for

> H. Bal, A. Renjith, and P. Akkala, "COMET: Route-Conditioned Control for
> Safe Magnetic Carrier Navigation in Patient-Derived Vascular Models"
> (submitted to *Micromachines*).

DynaMag simulates a nine-coil electromagnetic array steering a 500 µm
magnetic carrier through patient-derived vessel networks from the
[Vascular Model Repository](https://www.vascularmodel.com). The model
includes saturating dipole fields, gravity and buoyancy, Schiller–Naumann
drag, analytical pulsatile or imported steady CFD flow, coil current,
voltage, power and thermal limits, and per-route patient positioning.
COMET and four baseline controllers are compared on a frozen 75-case
benchmark.

## Setup

Python 3.13 is recommended.

```bash
python3 -m venv .venv-cfd
source .venv-cfd/bin/activate
pip install -r requirements-cfd.txt
python -m pytest tests/
```

`requirements.txt` is the lighter environment for runs without CFD. The CFD
study also needs `data/cfd/*.npz`, which are regenerated from the SimScale
exports (see [docs/cfd_validation.md](docs/cfd_validation.md)).

## Reproducing the paper

Every run writes to `outputs/physics_sensitivity/<RUN_ID>/` (not tracked).
The defaults of `run_physics_sensitivity_sweep.sh` reproduce the main
benchmark (400 mT/m, gravity, optimized positioning, shared-route
protocol):

```bash
# COMET (75 cases)
RUN_ID=main_comet ./run_physics_sensitivity_sweep.sh

# Baselines
RUN_ID=main_baselines CONTROLLERS=nmpc_tuned,nmpc_default,mpc_tuned,acbc_tuned \
  ./run_physics_sensitivity_sweep.sh

# Actuator sensitivity (300 mT/m; use 600 likewise)
RUN_ID=grad300 ACTUATOR=high_gradient_300 CONDITIONS=grad300_gopt:preset:3542:optimized \
  CONTROLLERS=comet_v4,nmpc_tuned,nmpc_default,mpc_tuned,acbc_tuned ./run_physics_sensitivity_sweep.sh

# Replicates with jittered starts (seeds 101-105)
RUN_ID=rep101 SEED=101 START_JITTER_MM=0.25 CONTROLLERS=comet_v4,nmpc_tuned,nmpc_default \
  ./run_physics_sensitivity_sweep.sh

# Mechanism ablations
RUN_ID=ablations CONTROLLERS=comet_v4_no_polyline,comet_v4_no_terminal_precision,comet_v4_no_branch_shaping,comet_v4_no_flow_compensation,comet_v4_slsqp_allocator \
  ./run_physics_sensitivity_sweep.sh

# Localization noise
RUN_ID=noise CONTROLLERS=comet_v4_loc_noise_100um,comet_v4_loc_noise_200um ./run_physics_sensitivity_sweep.sh

# Steady CFD (abdominal and pulmonary only)
RUN_ID=cfd CFD=1 PYTHON_BIN=.venv-cfd/bin/python \
  CONTROLLERS=comet_v4,nmpc_tuned,nmpc_default,mpc_tuned,acbc_tuned ./run_physics_sensitivity_sweep.sh
```

Trials are deterministic: each trial's seed depends only on the case, so a
rerun reproduces the reported trial exactly. Full runs take hours; the
SLSQP-based baselines are the slowest.

Confidence intervals and paired tests (this reproduces Table 3 from the
stored results; the baselines were run separately from COMET, so their run
directories are combined with `+`):

```bash
M=results/main_benchmark
python scripts/summarize_replicates.py \
  --replicate main=$M/five_anatomy_final_comet_v4_20261103+$M/five_anatomy_final_20261103+$M/five_anatomy_final_nmpc_20261103+$M/five_anatomy_final_0049_20261103 \
  --anatomies abdominal,aortic_root,pulmonary,thoracic_aorta,aortofemoral \
  --controllers comet_v4,nmpc_tuned,nmpc_default,mpc_tuned,acbc_tuned
```

Figures: `python scripts/make_paper_figures.py figures/`.

## Results

`results/` holds the trial-level results reported in the paper (one
`paired_results.csv` per anatomy and run, plus run configurations and
summaries). See [results/README.md](results/README.md) for which directory
backs which table.

## Repository layout

```
dynamag_simulation/core/   physics, vessel networks, flow, controllers
  comet_controller.py        COMET
  tracking_mpc.py            textbook receding-horizon NMPC baseline
  acbc_controller.py         ACBC baseline
  patient_positioning.py     table-orientation optimizer
  cfd_flow_field.py          steady CFD flow sampler
scripts/
  run_publication_benchmark.py   benchmark runner (controller variants, actuator presets)
  simulation_3d_*.py             per-controller trial loops
  summarize_replicates.py        route-bootstrap CIs and McNemar tests
  make_paper_figures.py          paper figures from results/
  import_simscale_flow.py        CFD import
  estimate_core_gain.py          far-field core gain estimate
  make_benchmark_manifest.py     benchmark and held-out route manifests
  data_prep/                     VMR geometry to vessel-segment conversion
  cfd/                           pulsatile svSolver workflow (work in progress)
  tools/                         geometry inspection helpers
run_physics_sensitivity_sweep.sh  benchmark driver
benchmark_manifests/
  five_anatomy_500um/        frozen 75-case benchmark
  tuning_heldout_seed777/    held-out tuning routes
  tuning_heldout_seed888/    held-out tuning routes
data/vessels/                vessel networks, by VMR model
data/cfd/                    CFD field metadata (arrays regenerated locally)
results/                     results reported in the paper
docs/                        modeling assumptions, CFD workflow
tests/
```

## Vessel models

| Model | VMR ID | Folder |
|---|---|---|
| Abdominal aorta (aneurysm) | 0033_H_ABAO_AAA | `data/vessels/abdominal_0033_H_ABAO_AAA` |
| Aortic root and cerebral arteries | 0248_H_AOCERE_CAS (model 40Left85Right) | `data/vessels/aortic_root_0248_H_AOCERE_CAS_40Left85Right` |
| Pulmonary arteries | 0082_H_PULM_H | `data/vessels/pulmonary_0082_H_PULM_H` |
| Thoracic aorta | 0024_H_AO_H | `data/vessels/aorta_0024_H_AO_H` |
| Aortofemoral bypass | 0049_H_ABAO_AIOD | `data/vessels/aortofemoral_0049_H_ABAO_AIOD` |
| Cerebral (held-out tuning only) | 0050_H_CERE_H | `data/vessels/cerebral_0050_H_CERE_H` |

The vessel models are derived from the Vascular Model Repository; see the
`LICENSE` files in their folders and cite N. M. Wilson, A. K. Ortiz and
A. B. Johnson, *J. Med. Devices* 7(4), 040923 (2013).

## License and citation

The code is released under the [MIT License](LICENSE). If you use it,
please cite the article above; GitHub's "Cite this repository" button reads
[CITATION.cff](CITATION.cff). Each tagged release is archived on Zenodo with
a DOI.

## Third-party code

The constrained cross-entropy optimizer in
`dynamag_simulation/core/constrained_cem_controller.py` adapts MIT-licensed
code; see `THIRD_PARTY_NOTICES/`.
