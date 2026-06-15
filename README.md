# DynaMag Simulation Suite

Physics-based simulation of multi-coil electromagnetic steering for magnetic
carrier routing research.

## Overview

DynaMag models a programmable electromagnet array steering magnetic
microspheres or larger SPION-loaded carriers through 3D workspaces and real
vessel networks. The simulation includes:

- **Regularized magnetic dipole field model** (body-scale multi-coil array, softened near sources)
- **Superparamagnetic particle** with susceptibility + saturation crossover
- **Schiller-Naumann drag** (nonlinear; reduces to Stokes at Re ≪ 1, valid to Re ≈ 800)
- **Brownian thermal noise** (Stokes-Einstein diffusion)
- **Background flow fields** (NoFlow, Uniform, Poiseuille, Pulsatile cardiac)
- **Flow-aware MPC controller** (1-step SLSQP optimisation with flow compensation)
- **Reynolds number tracking** with Stokes and finite-Re carrier-regime labels
- **Adaptive substepping** with per-substep flow evaluation

## Quick Start

```bash
# Create environment
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt

# Run the default 3-trial MPC simulation
python scripts/simulation_3d_mpc.py

# Run the paired 1 mm publication benchmark infrastructure
python scripts/run_publication_benchmark_1mm.py --smoke-test

# Run tests
python -m pytest tests/ -v
```

## Repository Structure

```
.
├── dynamag_simulation/               # Core physics and simulation engine
│   ├── core/                         # Shared physics modules
│   │   ├── constants.py              # Physical and geometric constants
│   │   ├── profiles.py               # Named particle/fluid profiles
│   │   ├── workspace.py              # 2D/3D workspace geometry
│   │   ├── magnets.py                # Electromagnet array (regularized dipole)
│   │   ├── particle.py               # Superparamagnetic particle + Schiller-Naumann drag
│   │   ├── flow.py                   # Background flow models (Uniform, Poiseuille, Pulsatile)
│   │   ├── integrator.py             # Adaptive substepping with flow evaluation
│   │   ├── controllers.py            # Flow-aware MPC controller
│   │   ├── validation.py             # Stokes / finite-Re regime checks
│   │   ├── reporting.py              # TrialResult, CSV output
│   │   └── plotting.py               # Consolidated plot generation
│   ├── simulation_2d.py              # Legacy 2D (deprecated)
│   ├── simulation_2d_final.py        # Legacy 2D final (deprecated)
│   ├── simulation_2d_model_b.py      # Legacy 2D model B (deprecated)
│   └── simulation_3d.py              # Legacy simplified 3D (deprecated)
├── scripts/                          # Executable runners and analysis tools
│   ├── simulation_3d_mpc.py          # Primary simulation entry point (MPC)
│   ├── simulation_3d_acbc.py         # ACBC supervisor over MPC allocator
│   ├── run_publication_benchmark_1mm.py # Paired 1 mm ACBC/MPC benchmark
│   ├── generate_publication_figures_1mm.py # 1 mm manuscript figures/tables
│   ├── simulation_3d_realistic.py    # Backward-compatible wrapper
│   ├── run_100_trials_flow.py        # 100-trial flow runner (500 µm microspheres)
│   ├── run_50um_100trials.py         # 100-trial flow runner (50 µm microspheres)
│   ├── run_baselines.py              # Baseline comparison runner (Random/Static control)
│   ├── run_batch.py                  # Custom batch simulation runner
│   ├── run_monte_carlo_timed.py      # Time-limited Monte Carlo data collector
│   ├── generate_poster_figures.py    # Poster figure generator (500 µm flow dataset)
│   ├── generate_poster_figures_50um.py # Poster figure generator (50 µm flow dataset)
│   ├── generate_poster_figures_50um_improved.py # Improved analysis figure generator (50 µm flow dataset)
│   └── generate_presentation_figures.py # Polished presentation figure generator (11 panels)
├── data/                             # Input/raw network datasets
│   └── real_vessel_segments.pkl      # Pickled 3D vessel network structure
├── logs/                             # Execution log files
│   ├── flow_50um_run.log
│   └── mc_run.log
├── outputs/                          # Simulation outputs, CSV data, and generated plots
├── tests/                            # Pytest test suite
│   ├── test_profiles.py
│   ├── test_velocity_consistency.py
│   ├── test_reynolds.py
│   ├── test_mpc_controller.py
│   └── test_reproducibility.py
├── docs/
│   └── modeling_assumptions.md       # Equations, units, validity limits
├── requirements.txt
└── .gitignore
```

## Physical Profiles

| Profile              | Diameter | Use Case                                  |
|----------------------|----------|-------------------------------------------|
| `microsphere_500um`  | 500 µm   | Default: benchtop phantom system          |
| `microsphere_50um`   | 50 µm    | Scaled-down runs with background flow     |
| `nanoparticle_500nm` | 500 nm   | Theoretical comparison only               |
| `spion_carrier_1mm`  | 1 mm     | Large-vessel magnetic carrier study       |

The 1 mm carrier profiles are **not nanoparticles**. Use “magnetic carrier,”
“magnetic milliparticle,” or “large-vessel carrier” for those simulations.

## Outputs

Each trial produces:

- **Plots**: 3D trajectory, distance vs time, physics (force/velocity/field),
  coil currents, z=0 field map
- **CSV**: trial summary with physics validity stats and compute timing

CSV columns include `wall_time_s`, `avg_control_step_time_s`,
`max_control_step_time_s`, `mean_reynolds`, `max_reynolds`, `low_re_valid`,
`stokes_regime`, `finite_re_drag_valid`, `drag_model`, `drag_regime_label`,
`seed`, and `profile_name`. All plots and summaries are saved under `outputs/`.

For 1 mm carrier studies, `low_re_valid=False` means the carrier is outside
the creeping-flow/Stokes regime; it is not by itself a publication failure.
The finite-Re validity field is the relevant drag-model check.

## MacBook M3 Pro (Apple Silicon) Speedup Guide

To run your simulations as fast as possible on a MacBook M3 Pro (or any M-series processor), apply the following performance enhancement strategies:

### 1. Enable Algorithmic Cost Pre-Evaluation (`fast_mode`)
We have added a `fast_mode` parameter in `continuous_mpc_controller` (enabled by default in [controllers.py](file:///Users/amrit/Projects/DynaMag-Magnetic-Nanoparticle-Steering-Simulation/dynamag_simulation/core/controllers.py)). 
- **What it does**: Instead of running a full numerical SLSQP optimization (`scipy.optimize.minimize`) on all 5 heuristic start vectors at every simulation step, it computes the scalar cost of all candidate starts first (taking < 0.1ms total) and only executes SLSQP optimization on the single best start vector. 
- **Performance Impact**: This reduces the number of SLSQP optimizations from 5 to 1 per simulation step, yielding an immediate **3x to 5x speedup** per trial with identical or superior steering performance.
- **Tune optimizer tolerances**: The optimizer `ftol` can be set to `1e-5` and `maxiter` to `50` (compared to legacy `ftol=1e-7`, `maxiter=80`) inside `continuous_mpc_controller`. This stops unnecessary iterations once the cost change is negligible.

### 2. Leverage NumPy Linked to Apple's Accelerate Framework
Apple Silicon M-series chips feature vecLib/Accelerate, which provides highly-optimized BLAS and LAPACK routines tuned for Apple's CPU architecture.
- **Ensure linking**: Starting from NumPy 1.25, the standard `pip install numpy` wheels automatically detect and link to macOS Accelerate. You can check if it is active via:
  ```bash
  python3 -c "import numpy; numpy.show_config()"
  ```
  Look for `accelerate` under the BLAS/LAPACK section. This drastically speeds up dot products, norms, and vector mathematics.

### 3. Run Independent Trials in Parallel
Since all trials in batch runs (e.g. `run_monte_carlo_timed.py`, `run_batch.py`, `run_50um_100trials.py`) are independent, they are perfect candidates for parallel execution using Python's `multiprocessing` or `concurrent.futures`.
- **How to execute**: Modify the main trial loop to run in a `ProcessPoolExecutor` with `max_workers` set to your CPU count (MacBook M3 Pro has 11 or 12 CPU cores). On M-series, utilizing 8-10 worker processes will execute trials in parallel with a **6x to 8x throughput speedup**.

## Reproducibility

All stochastic runs are seeded via `rng_seed` (default: 42). The test suite
verifies bitwise reproducibility for identical seeds.

To reproduce all results:
```bash
source .venv/bin/activate
python scripts/simulation_3d_mpc.py
```

Results appear in `outputs/simulation_outputs_3d_mpc/`.

## Documentation

See `docs/modeling_assumptions.md` for detailed equations, units, validity
limits, and a list of what is and is not modelled.
