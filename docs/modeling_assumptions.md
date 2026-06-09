# DynaMag Modeling Assumptions

This document describes the physics, numerical methods, and validity limits
of the DynaMag simulation suite. All quantities are in **SI units** unless
noted otherwise.

---

## 1. Electromagnet Model

### Regularized Magnetic Dipole

Each electromagnet is modelled as a **regularized magnetic dipole**. The
current body-scale benchmark uses a 9-magnet triangular-prism layout. The
magnetic moment vector for coil *i* carrying current *I_i* is:

```
m_i = N * I_i * pi * R^2 * hat{o}_i     (A*m^2)
```

where *N* is the number of turns, *R* is the coil radius, and `hat{o}_i` is
the coil's inward orientation (unit vector along the body diagonal toward
the workspace centre).

The field at observation point **r** from dipole *i* located at **r_i** is
computed with a **softening length** *a = R* to prevent the unphysical
1/r³ singularity at close range:

```
B_i(r) = (mu_0 / 4pi) * [3 (m_i . hat{d}) hat{d} - m_i] / (|d|^2 + a^2)^(3/2)
```

where `d = r - r_i`, `hat{d} = d / |d|`, and `a = coil_radius`.

At `|d| >> a` the regularized formula reduces to the standard point-dipole
expression. The softening reflects the fact that the dipole model is invalid
for `|d| < R` anyway, and avoids numerical overflow near the coil positions.

**Reference**: Jackson, *Classical Electrodynamics*, ch. 5 (dipole
regularization); Furlani, *Permanent Magnet and Electromechanical Devices*
(2001), ch. 4 (finite-coil corrections).

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
| Max current     | I_max  | 8.0 (default) / 13.0 (50 um runs) | A |
| Workspace side  | --     | 0.1524 (6 in) | m      |

---

## 2. Particle Model

### Superparamagnetic Microsphere

The suite includes both microsphere/nanoparticle profiles and larger
SPION-loaded carrier profiles:

| Profile              | Diameter | Key Use Case                                  |
|----------------------|----------|-----------------------------------------------|
| `microsphere_500um`  | 500 µm   | Default: benchtop phantom system              |
| `microsphere_50um`   | 50 µm    | Scaled-down runs with uniform flow            |
| `nanoparticle_500nm` | 500 nm   | Theoretical comparison (true nanoparticle)     |
| `spion_carrier_1mm`  | 1 mm     | Large-vessel magnetic-carrier steering study   |

The 1 mm carrier is **not** a nanoparticle. Manuscripts should describe it
as a magnetic carrier, magnetic milliparticle, or large-vessel carrier.

Microsphere and nanoparticle profiles share the following legacy
magnetic/fluid properties:

| Parameter                  | Symbol    | Value     | Unit    |
|----------------------------|-----------|-----------|---------| 
| Effective susceptibility   | χ_eff     | 3.0       | --      |
| Saturation magnetisation   | M_sat     | 4.5×10⁵   | A/m     |
| Fluid viscosity (water 25°C)| η        | 8.9×10⁻⁴  | Pa·s    |
| Fluid density              | ρ         | 997       | kg/m³   |
| Temperature                | T         | 298       | K       |
| Brownian motion            | --        | enabled   | --      |

### Magnetic Gradient Force

Two regimes with smooth crossover at `B_sat = μ₀ · M_sat / χ_eff`:

**Linear (unsaturated), |B| < B_sat:**
```
F = (V_p * chi_eff) / (2 * mu_0) * grad(|B|^2)
```

**Saturated, |B| ≥ B_sat:**
```
F = V_p * M_sat * grad(|B|) = V_p * M_sat / (2|B|) * grad(|B|^2)
```

For SPION-loaded carrier profiles, only the loaded magnetic volume contributes
to the magnetic force:

```
V_mag = magnetic_loading_fraction * V_carrier
```

The force prefactors use `V_mag`, while hydrodynamic drag uses the full
carrier diameter. The profile values `chi_eff` and `M_sat` are interpreted as
properties of the magnetic loading phase, not the entire carrier volume.

The gradient `grad(|B|²)` is computed via **central finite differences**
with step size `h = 5×10⁻⁵ m`.

### Drag Model — Schiller-Naumann Correlation

The simulation uses the **Schiller-Naumann** nonlinear drag correlation
(valid for Re up to ~800), **not** simple Stokes drag:

```
C_D = (24 / Re) * (1 + 0.15 * Re^0.687)
```

This corresponds to an effective drag coefficient:

```
gamma_eff = gamma_Stokes * (1 + 0.15 * Re^0.687)
```

where `gamma_Stokes = 6πηr` is the classical Stokes drag. For non-ideal
carriers, the profile field `hydrodynamic_drag_multiplier` scales
`gamma_Stokes` before the Schiller-Naumann correction is applied.

At Re ≪ 1 the correction factor → 1, recovering pure Stokes drag. At
higher Re (near coil singularities or during aggressive steering) the
increased drag prevents unphysically high velocities.

**Iterative solution**: Because Re = ρvd/η depends on velocity, which in
turn depends on drag, the effective drag is solved via **fixed-point
iteration** (up to 15 iterations, converged when ΔRe < 10⁻⁴·Re). When
Re < 0.01 the iteration is skipped and pure Stokes drag is used.

### Particle velocity (overdamped)

In the overdamped regime the particle has no inertia. The velocity is:

```
v_particle = v_flow(x, t) + F_magnetic / gamma_eff
```

where `v_flow` is the local background flow velocity and `gamma_eff` is
the Schiller-Naumann effective drag. The Reynolds number is computed from
the *magnetic* velocity component only (i.e., velocity relative to the
fluid), since the particle is entrained by the flow with no relative slip
at the flow velocity.

For 1 mm carrier studies, Reynolds numbers around `10^2` are expected. These
runs are not in the Stokes/creeping-flow regime, but remain inside the
empirical Schiller-Naumann range as long as `Re <= 800`. Reports therefore
distinguish:

- `stokes_regime`: `max_reynolds < 0.1`
- `finite_re_drag_valid`: `max_reynolds <= 800`
- `drag_regime_label`: `stokes`, `finite_re_schiller_naumann`, or
  `outside_model_range`

The legacy `low_re_valid` field is equivalent to `stokes_regime` and is kept
for backward compatibility only.

### 1 mm carrier limitations

The finite-Re carrier model is suitable for controller benchmarking, not for
medical-device validation. It neglects particle inertia, added mass, history
forces, wall contact/rolling/sliding, near-wall hydrodynamic corrections,
non-Newtonian blood behavior, and experimentally calibrated FEM magnetic
fields. These assumptions must be stated in publications.

### Brownian Thermal Noise

Stokes-Einstein diffusion coefficient:
```
D = k_B * T / gamma_Stokes
```

Per-step displacement:
```
delta_x = sqrt(2 * D * dt) * N(0, 1)     (per axis, 3D)
```

Note: the diffusion coefficient uses the *Stokes* drag (not Schiller-Naumann)
because the Einstein relation is defined for the zero-velocity equilibrium
limit.

---

## 3. Background Flow Models

The simulation supports four background flow models (`dynamag_simulation/core/flow.py`):

### NoFlow (default)
Static fluid — zero flow velocity everywhere. Backward-compatible mode.

### UniformFlow
Spatially uniform, time-constant flow vector **v_flow** (m/s). Used for the
50 µm microsphere batch runs with v = 200 µm/s in the +x direction,
simulating capillary-scale convection.

### PoiseuilleFlow — Steady Parabolic Profile
Laminar flow in a cylindrical channel of radius *R_ch* with centerline
velocity *v_max*:

```
v(r) = v_max * (1 - (r / R_ch)^2) * axis_hat
```

where *r* is the radial distance from the channel centerline. Outside the
channel radius the velocity is zero. Mean velocity = v_max / 2.

### PulsatileFlow — Cardiac-Cycle Modulation
Poiseuille profile modulated by a sinusoidal cardiac waveform:

```
v(r, t) = v_max * (1 - (r / R_ch)^2) * [1 + A * sin(2*pi*f*t)] * axis_hat
```

| Parameter               | Default | Unit |
|--------------------------|---------|------|
| Heart rate               | 60      | bpm  |
| Pulsation amplitude (A)  | 0.3     | --   |

This produces ±30% velocity oscillations at 1 Hz.

### Flow–Controller Interaction

The MPC controller's 1-step prediction **includes** the flow field:
it estimates where the particle will be after one control interval of
flow convection + magnetic drift, and optimises currents to steer
against (or with) the flow. The integrator also evaluates the flow
velocity at each sub-step for time-varying flows.

---

## 4. Numerical Integration

### Adaptive Substepping

Rather than clamping displacement to a maximum step size (which distorts
the relationship between force and velocity), the simulation uses
**adaptive substepping**:

1. Compute the total estimated displacement `|F/γ·dt| + |v_flow·dt|`.
2. If it exceeds `max_substep_disp` (default 5 mm), subdivide `dt` into
   `n = ceil(est_disp / max_substep_disp)` equal sub-intervals.
3. Apply `n` Euler steps of duration `dt/n`, each with:
   - Its own Brownian sample.
   - A fresh flow-velocity evaluation at the current particle position
     and sub-step time (important for pulsatile flows).

This preserves force-velocity consistency while preventing numerical
instability from excessively large single steps.

### Velocity Logging

The **realized velocity** is computed as:
```
v_realized = (x_{t+1} - x_t) / dt
```

This is the actual displacement divided by the control interval, and it is
what gets logged, plotted, and used for Reynolds number computation. It
correctly accounts for workspace clipping, substepping, and flow convection.

---

## 5. Controller (MPC)

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

### Default Weights

| Weight         | Symbol        | Default |
|----------------|---------------|---------|
| Progress       | w_progress    | 80.0    |
| Alignment      | w_align       | 8.0     |
| Distance²      | w_dist        | 2.0     |
| Current²       | w_curr        | 1×10⁻⁵  |
| Smoothness     | w_smooth      | 0.5     |
| Edge penalty   | w_edge        | 20.0    |
| Overshoot      | w_overshoot   | 2.0     |

### Optimizer Details

- **Method**: SLSQP (Sequential Least Squares Programming)
- **Bounds**: each current in [−I_max, +I_max]
- **Max iterations**: 80 per start
- **Tolerance**: ftol = 10⁻⁷
- **Initial guesses**: 5–6 heuristic starting points per control step:
  1. Zero currents
  2. Full-magnitude currents aligned with target direction
  3. Half-magnitude aligned
  4. Negative-full (explore repulsive configurations)
  5. Single strongest-aligned coil at max current
  6. Previous solution (warm start, when available)

The optimizer convergence status is tracked; if no start converges,
the best-so-far solution is used and a warning is issued.

### Flow-Aware Prediction

The 1-step prediction used by the cost function adds the flow velocity at
the current particle position to the magnetic drift, so that the controller
can actively compensate for fluid convection.

---

## 6. Validity Limits and What Is Not Modelled

### Included
- Regularized dipole magnetic field (coil superposition, softened near sources)
- Magnetic gradient force (linear + saturated regime with smooth crossover)
- Schiller-Naumann nonlinear drag (reduces to Stokes at Re ≪ 1, valid to Re ≈ 800)
- Brownian thermal noise (Stokes-Einstein diffusion)
- Background fluid flow (uniform, Poiseuille, pulsatile cardiac models)
- Flow-aware MPC controller prediction
- Workspace boundary clipping (hard walls)
- Adaptive substepping with per-substep flow evaluation
- Reynolds number tracking with Stokes and finite-Re validity labels

### Not Modelled
- **Particle-particle interactions**: single-particle simulation
- **Inertia**: overdamped assumption (valid for Re ≪ 1 only; Schiller-Naumann
  extends the drag correction but the position integrator is still first-order
  overdamped)
- **Magnetic field gradients from finite coil geometry**: using regularized
  dipole approximation, not full Biot-Savart integration over the solenoid
- **Demagnetisation effects**: superparamagnetic assumption (no remanence)
- **Gravity / buoyancy**: assumed neutrally buoyant or negligible at this scale
- **Hydrodynamic wall effects**: hard boundary clipping only, no lubrication
  or wall-drag corrections
- **Temperature gradients**: uniform temperature assumed
- **Non-Newtonian fluids**: fluid is treated as Newtonian (constant η)
- **Womersley effects in pulsatile flow**: the pulsatile model uses a
  quasi-steady Poiseuille profile; frequency-dependent velocity profile
  distortion (Womersley number effects) is not included

### Reynolds Number Validity

The simulation reports both creeping-flow and finite-Re drag-model regimes.
At Re > 0.1, the simple Stokes drag assumption breaks down, but the
Schiller-Naumann correction is used for drag estimates up to Re ≈ 800.
The `low_re_valid` field is retained as a legacy alias for
`stokes_regime`; 1 mm carrier publications should use
`finite_re_drag_valid` and `drag_regime_label` as the relevant validity
fields.

For the default 500 µm microsphere, Re can approach O(1–10), where the
Schiller-Naumann correction is physically necessary. For the 50 µm
microsphere under flow, Re ≈ 0.02, well within creeping flow. For 1 mm
large-vessel carriers, Re can be O(10^2), outside Stokes flow but still
inside the intended finite-Re drag-model range when `max_reynolds <= 800`.
