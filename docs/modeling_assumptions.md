# DynaMag Modeling Assumptions

This document describes the physics, numerical methods, and validity limits
of the DynaMag simulation suite. All quantities are in **SI units** unless
noted otherwise.

---

## 1. Electromagnet Model

### Magnetic Dipole Approximation

Each of the 8 corner electromagnets is modelled as a **magnetic dipole**.
The magnetic moment vector for coil *i* carrying current *I_i* is:

```
m_i = N * I_i * pi * R^2 * hat{o}_i     (A*m^2)
```

where *N* is the number of turns, *R* is the coil radius, and `hat{o}_i` is
the coil's inward orientation (unit vector along the body diagonal toward
the workspace centre).

The field at observation point **r** from dipole *i* located at **r_i** is:

```
B_i(r) = (mu_0 / 4pi) * [3 (m_i . hat{d}) hat{d} - m_i] / |d|^3
```

where `d = r - r_i` and `hat{d} = d / |d|`.

**Validity**: The dipole approximation requires `|d| >> R`. In this system
the closest a particle can get to any coil is approximately `HALF * sqrt(3)
- HALF ≈ 0.055 m`, and the coil radius is 0.03 m, so the condition is
marginally satisfied. Near-field corrections (e.g., finite-length solenoid
model) are not included.

### Default Parameters

| Parameter       | Symbol | Default Value | Unit   |
|-----------------|--------|---------------|--------|
| Coil radius     | R      | 0.03          | m      |
| Coil length     | L      | 0.12          | m      |
| Turns           | N      | 1500          | --     |
| Max current     | I_max  | 8.0           | A      |
| Workspace side  | --     | 0.1524 (6 in) | m      |

---

## 2. Particle Model

### Superparamagnetic Microsphere

The default particle is a **500 um diameter Fe3O4-core microsphere**
(profile: `microsphere_500um`).

| Parameter                  | Symbol    | Value     | Unit    |
|----------------------------|-----------|-----------|---------|
| Diameter                   | d_p       | 500e-6    | m       |
| Effective susceptibility   | chi_eff   | 3.0       | --      |
| Saturation magnetisation   | M_sat     | 4.5e5     | A/m     |
| Fluid viscosity (water 25C)| eta       | 8.9e-4    | Pa*s    |
| Temperature                | T         | 298       | K       |

### Magnetic Gradient Force

Two regimes with smooth crossover at `B_sat = mu_0 * M_sat / chi_eff`:

**Linear (unsaturated), |B| < B_sat:**
```
F = (V_p * chi_eff) / (2 * mu_0) * grad(|B|^2)
```

**Saturated, |B| >= B_sat:**
```
F = V_p * M_sat * grad(|B|) = V_p * M_sat / (2|B|) * grad(|B|^2)
```

The gradient `grad(|B|^2)` is computed via **central finite differences**
with step size `h = 5e-5 m`.

### Stokes Drag (Overdamped Dynamics)

```
v = F / gamma    where    gamma = 6 * pi * eta * r_p
```

This assumes the particle Reynolds number is much less than 1 (creeping
flow / Stokes regime). The simulation computes and logs Re at every step.

### Brownian Thermal Noise

Stokes-Einstein diffusion coefficient:
```
D = k_B * T / gamma
```

Per-step displacement:
```
delta_x = sqrt(2 * D * dt) * N(0, 1)     (per axis)
```

---

## 3. Numerical Integration

### Adaptive Substepping

Rather than clamping displacement to a maximum step size (which distorts
the relationship between force and velocity), the simulation uses
**adaptive substepping**:

1. Compute the raw deterministic displacement `|F/gamma * dt|`.
2. If it exceeds `max_substep_disp` (default 5 mm), subdivide `dt` into
   `n = ceil(raw_disp / max_substep_disp)` equal sub-intervals.
3. Apply `n` Euler steps of duration `dt/n`, each with its own Brownian
   sample.

This preserves force-velocity consistency while preventing numerical
instability from excessively large single steps.

### Velocity Logging

The **realized velocity** is computed as:
```
v_realized = (x_{t+1} - x_t) / dt
```

This is the actual displacement divided by the control interval, and it is
what gets logged, plotted, and used for Reynolds number computation. It
correctly accounts for workspace clipping and substepping.

---

## 4. Controller (MPC)

A single-step model-predictive controller minimises a weighted cost:

```
cost = -w_progress * progress
     - w_align * step_along
     - 0.5 * alignment
     + w_dist * dist_next^2
     + w_curr * sum(I^2)
     + w_smooth * sum((I - I_prev)^2)
     + w_edge * edge_penalty
     + w_overshoot * overshoot_penalty
```

Solved by **SLSQP** with 5-6 heuristic initial guesses per step.
The optimizer convergence status is tracked; if no start converges,
the best-so-far solution is used and a warning is issued.

---

## 5. Validity Limits and What Is Not Modelled

### Included
- Dipole magnetic field (superposition of 8 coils)
- Magnetic gradient force (linear + saturated regime)
- Stokes drag (overdamped)
- Brownian thermal noise
- Workspace boundary clipping

### Not Modelled
- **Fluid flow / convection**: no background flow velocity
- **Particle-particle interactions**: single-particle simulation
- **Inertia**: overdamped assumption (valid for Re << 1 only)
- **Magnetic field gradients from finite coil geometry**: using dipole
  approximation, not Biot-Savart integration
- **Demagnetisation effects**: superparamagnetic assumption
- **Gravity / buoyancy**: assumed neutrally buoyant or negligible at
  this scale
- **Wall interactions**: hard clipping only, no hydrodynamic wall effects
- **Temperature gradients**: uniform temperature assumed

### Reynolds Number Validity

The simulation flags runs where the particle Reynolds number exceeds 0.1.
At Re > 0.1, the Stokes drag assumption breaks down and inertial effects
become significant. Published results should report the `low_re_valid`
flag and discuss any flagged runs.

For the default 500 um microsphere at typical operating velocities, Re can
approach O(1-10), indicating that the Stokes assumption may be marginal.
Users should consider this when interpreting quantitative velocity and
force predictions.
