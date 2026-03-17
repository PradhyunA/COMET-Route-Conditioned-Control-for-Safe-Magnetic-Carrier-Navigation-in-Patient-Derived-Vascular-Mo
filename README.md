# DynaMag Simulation Suite

Physics-based simulation of multi-coil electromagnetic particle steering for
magnetic drug targeting research.

## Overview

DynaMag models a programmable 8-coil electromagnet array steering a
superparamagnetic microsphere through a 3D workspace. The simulation
includes:

- **Regularized magnetic dipole field model** (8 corner electromagnets, softened at r < R)
- **Superparamagnetic particle** with susceptibility + saturation crossover
- **Schiller-Naumann drag** (nonlinear; reduces to Stokes at Re ≪ 1, valid to Re ≈ 800)
- **Brownian thermal noise** (Stokes-Einstein diffusion)
- **Background flow fields** (NoFlow, Uniform, Poiseuille, Pulsatile cardiac)
- **Flow-aware MPC controller** (1-step SLSQP optimisation with flow compensation)
- **Reynolds number tracking** and low-Re validity flagging
- **Adaptive substepping** with per-substep flow evaluation

## Quick Start

```bash
# Create environment
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt

# Run the default 3-trial MPC simulation
python simulation_3d_mpc.py

# Run tests
python -m pytest tests/ -v
```

## Repository Structure

```
.
├── simulation_3d_mpc.py              # Primary simulation entry point (MPC)
├── simulation_3d_realistic.py        # Backward-compatible wrapper
├── mpc_controller_3d.py              # Backward-compatible wrapper
├── dynamag_simulation/
│   ├── core/                         # Shared physics engine
│   │   ├── constants.py              # Physical and geometric constants
│   │   ├── profiles.py               # Named particle/fluid profiles
│   │   ├── workspace.py              # 2D/3D workspace geometry
│   │   ├── magnets.py                # Electromagnet array (regularized dipole)
│   │   ├── particle.py               # Superparamagnetic particle + Schiller-Naumann drag
│   │   ├── flow.py                   # Background flow models (Uniform, Poiseuille, Pulsatile)
│   │   ├── integrator.py             # Adaptive substepping with flow evaluation
│   │   ├── controllers.py            # Flow-aware MPC controller
│   │   ├── validation.py             # Reynolds number checks
│   │   ├── reporting.py              # TrialResult, CSV output
│   │   └── plotting.py               # Consolidated plot generation
│   ├── simulation_2d.py              # Legacy 2D (deprecated)
│   ├── simulation_2d_final.py        # Legacy 2D final (deprecated)
│   ├── simulation_2d_model_b.py      # Legacy 2D model B (deprecated)
│   └── simulation_3d.py              # Legacy simplified 3D (deprecated)
├── tests/                            # pytest test suite
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

## Outputs

Each trial produces:

- **Plots**: 3D trajectory, distance vs time, physics (force/velocity/field),
  coil currents, z=0 field map
- **CSV**: trial summary with physics validity stats and compute timing

CSV columns include `wall_time_s`, `avg_control_step_time_s`,
`max_control_step_time_s`, `mean_reynolds`, `max_reynolds`, `low_re_valid`,
`seed`, and `profile_name`.

## Reproducibility

All stochastic runs are seeded via `rng_seed` (default: 42). The test suite
verifies bitwise reproducibility for identical seeds.

To reproduce all results:
```bash
source .venv/bin/activate
python simulation_3d_mpc.py
```

Results appear in `simulation_outputs_3d_mpc/`.

## Documentation

See `docs/modeling_assumptions.md` for detailed equations, units, validity
limits, and a list of what is and is not modelled.
