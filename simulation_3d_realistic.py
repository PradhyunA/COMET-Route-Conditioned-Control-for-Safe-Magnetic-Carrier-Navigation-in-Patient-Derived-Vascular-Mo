"""
simulation_3d_realistic.py
==========================
Physics-accurate 3D magnetic particle steering simulation.

Models:
  - Electromagnets as magnetic dipoles  (B in Tesla)
  - Fe₃O₄ superparamagnetic microsphere (susceptibility + saturation)
  - Stokes drag  (low-Reynolds-number regime)
  - Brownian thermal noise  (Stokes-Einstein diffusion)

All quantities in SI units.
"""

import os
import csv
import itertools
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D  # noqa: F401

# ============================================================
# PHYSICAL CONSTANTS
# ============================================================

MU_0 = 4.0 * np.pi * 1e-7          # vacuum permeability  (T·m/A)
K_B = 1.381e-23                     # Boltzmann constant   (J/K)
PI = np.pi

# ============================================================
# WORKSPACE GEOMETRY
# ============================================================

INCH_TO_METER = 0.0254
SIDE = 6.0 * INCH_TO_METER          # 0.1524 m
HALF = SIDE / 2.0                    # 0.0762 m

EPS = 1e-30                          # numerical floor

TEST_RANGE = 0.06                    # m – trial start/target extent

# Maximum displacement per timestep (stability limiter).
MAX_STEP = 5e-3                      # 5 mm


# ============================================================
# WORKSPACE
# ============================================================

class Workspace3D:
    """Cubic workspace with position clamping."""

    def __init__(self, half_side=HALF):
        self.half = float(half_side)

    def clip(self, p):
        return np.clip(np.asarray(p, dtype=float),
                       -self.half, self.half)


# ============================================================
# SOLENOID / ELECTROMAGNET ARRAY  (magnetic dipole model)
# ============================================================

class ElectromagnetArray3D_Real:
    """
    8 corner electromagnets modelled as magnetic dipoles.

    Each solenoid has:
      - position       : centre of the coil  (at a cube corner)
      - orientation     : unit vector pointing inward (along body diagonal)
      - coil radius R   : 0.01 m  (1 cm)
      - coil length L   : 0.04 m  (4 cm)
      - turns N         : 250
      - current I       : controllable,  ±I_max

    Magnetic moment:
        m_vec = N × I × π R² × orientation_hat        (A·m²)

    Dipole field at position r relative to the dipole centre:
        B(r) = (μ₀ / 4π) × [ 3(m·r̂)r̂  -  m ] / |r|³

    This is accurate when |r| >> R  (always true here since the
    closest a particle can get to a corner is ~several cm, and R=1 cm).
    """

    def __init__(
        self,
        coil_radius=0.01,
        coil_length=0.04,
        turns=250,
        max_current=8.0,
    ):
        # 8 cube corners
        signs = list(itertools.product((-1, 1), repeat=3))
        self.positions = np.array(
            [[s[0] * HALF, s[1] * HALF, s[2] * HALF] for s in signs],
            dtype=float,
        )

        # Inward orientation  (-sign / √3)
        inv_sqrt3 = 1.0 / np.sqrt(3.0)
        self.orientations = np.array(
            [[-s[0] * inv_sqrt3, -s[1] * inv_sqrt3, -s[2] * inv_sqrt3]
             for s in signs],
            dtype=float,
        )

        self.n_magnets = 8
        self.coil_radius = float(coil_radius)
        self.coil_length = float(coil_length)
        self.turns = float(turns)
        self.max_current = float(max_current)
        self.currents = np.zeros(self.n_magnets, dtype=float)

        # Moment magnitude per unit current:  N × π R²
        self.moment_per_amp = self.turns * PI * self.coil_radius ** 2

        # Pre-factor for dipole field:  μ₀ / (4π)
        self._mu_over_4pi = MU_0 / (4.0 * PI)

    def set_currents(self, currents):
        self.currents = np.clip(
            np.asarray(currents, dtype=float),
            -self.max_current, self.max_current,
        )

    def field_at(self, point):
        """
        Net magnetic field B (Tesla) at *point* from all 8 dipoles.

        B_i = (μ₀/4π) × [3(m_i · r̂)r̂ - m_i] / r³
        """
        point = np.asarray(point, dtype=float)
        B_total = np.zeros(3, dtype=float)

        for i in range(self.n_magnets):
            # Magnetic moment vector
            m_mag = self.moment_per_amp * self.currents[i]
            m_vec = m_mag * self.orientations[i]

            # Vector from dipole to observation point
            r_vec = point - self.positions[i]
            r = np.linalg.norm(r_vec)
            if r < 1e-10:
                continue  # skip singularity (particle AT magnet)
            r_hat = r_vec / r

            m_dot_rhat = np.dot(m_vec, r_hat)

            B_i = self._mu_over_4pi * (
                3.0 * m_dot_rhat * r_hat - m_vec
            ) / (r ** 3)

            B_total += B_i

        return B_total

    def grad_B_squared(self, point, h=5e-5):
        """
        Gradient of |B|² via central finite differences.

        Returns a 3-vector  ∇(|B|²)  in  T²/m.
        """
        point = np.asarray(point, dtype=float)

        def B_sq(p):
            B = self.field_at(p)
            return np.dot(B, B)

        grad = np.zeros(3, dtype=float)
        for axis in range(3):
            dp = np.zeros(3)
            dp[axis] = h
            grad[axis] = (B_sq(point + dp) - B_sq(point - dp)) / (2.0 * h)

        return grad


# ============================================================
# MAGNETIC PARTICLE  (superparamagnetic microsphere)
# ============================================================

class MagneticParticle3D:
    """
    Superparamagnetic Fe₃O₄-core microsphere in viscous fluid.

    Force model
    -----------
    Linear (unsaturated) regime  (|B| < B_sat):
        F = (V_p × χ_eff) / (2 μ₀) × ∇(|B|²)

    Saturated regime  (|B| ≥ B_sat):
        F = V_p × M_s × ∇|B|     (= V_p × M_s / (2|B|) × ∇|B|²)

    We use a smooth cross-over between the two.

    Dynamics
    --------
    Overdamped (Stokes regime,  Re ≪ 1):
        v = F / γ       where  γ = 6πηr_p

    Brownian displacement per step:
        Δx = √(2 D Δt) × N(0,1)      D = k_B T / γ
    """

    def __init__(
        self,
        position,
        diameter=500e-6,
        chi_eff=3.0,
        M_sat=4.5e5,
        fluid_viscosity=8.9e-4,
        temperature=298.0,
        enable_brownian=True,
        rng_seed=42,
    ):
        self.pos = np.asarray(position, dtype=float)
        self.vel = np.zeros(3, dtype=float)

        # Particle geometry
        self.radius = diameter / 2.0
        self.volume = (4.0 / 3.0) * PI * self.radius ** 3

        # Magnetic properties
        self.chi_eff = float(chi_eff)
        self.M_sat = float(M_sat)
        self.B_sat = MU_0 * self.M_sat / self.chi_eff   # cross-over field

        # Force pre-factor (linear regime):  V χ / (2 μ₀)
        self.force_prefactor_linear = (
            self.volume * self.chi_eff / (2.0 * MU_0)
        )

        # Force pre-factor (saturated regime):  V M_s
        self.force_prefactor_sat = self.volume * self.M_sat

        # Stokes drag coefficient:  γ = 6πηr
        self.eta = float(fluid_viscosity)
        self.drag = 6.0 * PI * self.eta * self.radius

        # Brownian diffusion
        self.temperature = float(temperature)
        self.D = K_B * self.temperature / self.drag
        self.enable_brownian = enable_brownian
        self.rng = np.random.default_rng(rng_seed)

    def magnetic_force(self, B_at_pos, grad_B_sq):
        """
        Compute the magnetic gradient force (Newtons).

        Parameters
        ----------
        B_at_pos   : 3-vector, field at particle position  (T)
        grad_B_sq  : 3-vector, gradient of |B|² at particle  (T²/m)
        """
        B_mag = np.linalg.norm(B_at_pos)

        if B_mag < self.B_sat:
            # Linear regime:  F = (V χ / 2μ₀) ∇|B|²
            F = self.force_prefactor_linear * grad_B_sq
        else:
            # Saturated regime:  F = V M_s ∇|B|
            # ∇|B| = ∇(|B|²) / (2|B|)
            F = self.force_prefactor_sat * grad_B_sq / (2.0 * B_mag)

        return F

    def update(self, force, dt, max_step=MAX_STEP):
        """
        Overdamped step with optional Brownian noise.
        """
        # Deterministic step:  v = F / drag
        self.vel = force / self.drag
        det_step = self.vel * dt

        # Brownian step
        if self.enable_brownian:
            sigma = np.sqrt(2.0 * self.D * dt)
            brownian_step = sigma * self.rng.standard_normal(3)
        else:
            brownian_step = np.zeros(3)

        total_step = det_step + brownian_step

        # Clamp total displacement
        step_mag = np.linalg.norm(total_step)
        if step_mag > max_step:
            total_step *= max_step / step_mag

        self.pos = self.pos + total_step


# ============================================================
# CANDIDATE SET HELPERS
# ============================================================

def precompute_combos(candidates):
    return np.array(
        list(itertools.product(candidates, repeat=8)), dtype=float,
    )


def choose_candidates(dist):
    """Coarse far, finer near.  Kept small for 8-magnet combos."""
    if dist > 0.04:
        return (-8.0, 0.0, 8.0)            # 6 561
    elif dist > 0.015:
        return (-8.0, -4.0, 0.0, 4.0, 8.0) # 390 625
    else:
        return (-4.0, 0.0, 4.0)             # 6 561


def adaptive_lambda(base, dist):
    if dist > 0.05:
        return 0.15 * base
    if dist > 0.02:
        return 0.35 * base
    if dist > 0.008:
        return 0.70 * base
    return base


# ============================================================
# CONTROLLERS
# ============================================================

def lookahead_controller(
    particle,
    target_pos,
    magnet_array,
    workspace,
    dt,
    all_combos,
    lambda_current=5e-5,
    edge_buffer=0.006,
    max_step=MAX_STEP,
):
    """
    Brute-force lookahead controller adapted for real physics.

    For every candidate current combination:
      1. Compute B and ∇|B|² at particle position.
      2. Compute magnetic force F.
      3. Simulate one overdamped step (with step clamping).
      4. Score the result.
    """
    p_pos = particle.pos.copy()
    target_pos = np.asarray(target_pos, dtype=float)

    desired = target_pos - p_pos
    dist = np.linalg.norm(desired)

    if dist < EPS:
        return np.zeros(8), np.zeros(3), 0.0, 0.0

    desired_hat = desired / dist
    lam = adaptive_lambda(lambda_current, dist)

    best_score = -np.inf
    best_currents = None
    best_force = None
    best_step_along = -np.inf

    for combo in all_combos:
        magnet_array.set_currents(combo)

        B = magnet_array.field_at(p_pos)
        grad_Bsq = magnet_array.grad_B_squared(p_pos)
        F = particle.magnetic_force(B, grad_Bsq)

        # Simulated deterministic step  (no Brownian in the controller)
        raw_step = (F / particle.drag) * dt
        raw_mag = np.linalg.norm(raw_step)
        step_vec = raw_step * (max_step / raw_mag) if raw_mag > max_step else raw_step

        step_along = np.dot(step_vec, desired_hat)

        x_raw = p_pos + step_vec
        x_next = workspace.clip(x_raw)

        clipped = np.linalg.norm(x_raw - x_next)
        dist_next = np.linalg.norm(x_next - target_pos)
        progress = dist - dist_next

        F_norm = np.linalg.norm(F)
        alignment = np.dot(desired_hat, F / (F_norm + EPS))

        margins = workspace.half - np.abs(x_next)
        edge_pen = max(0.0, edge_buffer - np.min(margins))

        overshoot_pen = max(0.0, np.linalg.norm(step_vec) - 0.65 * dist)

        nonzero_bonus = 0.002 if (np.any(np.abs(combo) > 1e-12)
                                  and step_along > 0) else 0.0

        score = (
            10.0 * progress
            + 3.0 * step_along
            - 1.2 * dist_next
            + 0.10 * alignment
            + nonzero_bonus
            - lam * np.sum(combo ** 2)
            - 30.0 * clipped
            - 5.0 * edge_pen
            - 1.2 * overshoot_pen
        )

        if score > best_score:
            best_score = score
            best_currents = combo.copy()
            best_force = F.copy()
            best_step_along = step_along

    return best_currents, best_force, best_score, best_step_along


def directional_fallback(
    particle,
    target_pos,
    magnet_array,
    all_combos,
    dt,
    lambda_current=1e-5,
    max_step=MAX_STEP,
):
    """Fallback: maximise motion toward target."""
    p_pos = particle.pos.copy()
    target_pos = np.asarray(target_pos, dtype=float)

    desired = target_pos - p_pos
    dist = np.linalg.norm(desired)

    if dist < EPS:
        return np.zeros(8), np.zeros(3), 0.0

    desired_hat = desired / dist

    best_score = -np.inf
    best_currents = None
    best_force = None

    for combo in all_combos:
        magnet_array.set_currents(combo)

        B = magnet_array.field_at(p_pos)
        grad_Bsq = magnet_array.grad_B_squared(p_pos)
        F = particle.magnetic_force(B, grad_Bsq)

        raw_step = (F / particle.drag) * dt
        raw_mag = np.linalg.norm(raw_step)
        step_vec = raw_step * (max_step / raw_mag) if raw_mag > max_step else raw_step

        step_along = np.dot(step_vec, desired_hat)
        perp = step_vec - step_along * desired_hat

        score = (
            5.0 * step_along
            - 1.5 * np.linalg.norm(perp)
            - lambda_current * np.sum(combo ** 2)
        )

        if score > best_score:
            best_score = score
            best_currents = combo.copy()
            best_force = F.copy()

    return best_currents, best_force, best_score


# ============================================================
# SIMULATION
# ============================================================

def simulate_trial_realistic(
    start,
    target,
    dt=0.02,
    steps=3000,
    tolerance=0.003,
    coil_radius=0.03,
    coil_length=0.12,
    turns=1500,
    max_current=8.0,
    particle_diameter=500e-6,
    chi_eff=3.0,
    M_sat=4.5e5,
    fluid_viscosity=8.9e-4,
    temperature=298.0,
    enable_brownian=True,
    lambda_current=5e-5,
    max_step=MAX_STEP,
    rng_seed=42,
):
    workspace = Workspace3D()
    magnets = ElectromagnetArray3D_Real(
        coil_radius=coil_radius,
        coil_length=coil_length,
        turns=turns,
        max_current=max_current,
    )
    particle = MagneticParticle3D(
        position=start,
        diameter=particle_diameter,
        chi_eff=chi_eff,
        M_sat=M_sat,
        fluid_viscosity=fluid_viscosity,
        temperature=temperature,
        enable_brownian=enable_brownian,
        rng_seed=rng_seed,
    )

    start = np.asarray(start, dtype=float)
    target = np.asarray(target, dtype=float)

    trajectory = [particle.pos.copy()]
    distance_log = []
    current_log = []
    field_log = []       # |B| in Tesla
    force_log = []       # |F| in Newtons
    velocity_log = []    # |v| in m/s
    score_log = []

    reached = False
    combo_cache = {}

    for step in range(steps):
        cur_dist = np.linalg.norm(particle.pos - target)
        cand = choose_candidates(cur_dist)
        if cand not in combo_cache:
            combo_cache[cand] = precompute_combos(cand)
        all_combos = combo_cache[cand]

        best_currents, best_force, best_score, best_step_along = (
            lookahead_controller(
                particle=particle,
                target_pos=target,
                magnet_array=magnets,
                workspace=workspace,
                dt=dt,
                all_combos=all_combos,
                lambda_current=lambda_current,
                edge_buffer=0.006,
                max_step=max_step,
            )
        )

        # Fallback
        if cur_dist > tolerance:
            if (best_currents is None
                    or np.all(np.abs(best_currents) < 1e-12)
                    or best_step_along < 1e-12):
                fb_c, fb_f, fb_s = directional_fallback(
                    particle=particle,
                    target_pos=target,
                    magnet_array=magnets,
                    all_combos=all_combos,
                    dt=dt,
                    lambda_current=1e-5,
                    max_step=max_step,
                )
                best_currents = fb_c
                best_force = fb_f
                best_score = fb_s

        magnets.set_currents(best_currents)
        B = magnets.field_at(particle.pos)

        particle.update(best_force, dt, max_step=max_step)
        particle.pos = workspace.clip(particle.pos)

        trajectory.append(particle.pos.copy())
        distance = np.linalg.norm(particle.pos - target)

        distance_log.append(distance)
        current_log.append(magnets.currents.copy())
        field_log.append(np.linalg.norm(B))
        force_log.append(np.linalg.norm(best_force))
        velocity_log.append(np.linalg.norm(particle.vel))
        score_log.append(best_score)

        if distance < tolerance:
            reached = True
            break

        if (step + 1) % 200 == 0:
            v = np.linalg.norm(particle.vel)
            print(f"  step {step+1:>5d}/{steps}  "
                  f"dist={distance:.5f} m  "
                  f"|B|={np.linalg.norm(B):.3e} T  "
                  f"|F|={np.linalg.norm(best_force):.3e} N  "
                  f"|v|={v:.3e} m/s")

    trajectory = np.array(trajectory)
    distance_log = np.array(distance_log)
    current_log = np.array(current_log)
    field_log = np.array(field_log)
    force_log = np.array(force_log)
    velocity_log = np.array(velocity_log)
    score_log = np.array(score_log)

    final_error = np.linalg.norm(trajectory[-1] - target)
    max_current_used = (np.max(np.abs(current_log))
                        if len(current_log) > 0 else 0.0)

    return {
        "start": start,
        "target": target,
        "trajectory": trajectory,
        "distance_log": distance_log,
        "current_log": current_log,
        "field_log": field_log,
        "force_log": force_log,
        "velocity_log": velocity_log,
        "score_log": score_log,
        "magnets": magnets.positions.copy(),
        "orientations": magnets.orientations.copy(),
        "reached": reached,
        "steps_taken": len(trajectory) - 1,
        "final_position": trajectory[-1],
        "final_error": final_error,
        "max_current_used": max_current_used,
        "workspace_half": HALF,
        "params": {
            "dt": dt,
            "steps": steps,
            "tolerance": tolerance,
            "coil_radius": coil_radius,
            "coil_length": coil_length,
            "turns": turns,
            "max_current": max_current,
            "particle_diameter": particle_diameter,
            "chi_eff": chi_eff,
            "M_sat": M_sat,
            "fluid_viscosity": fluid_viscosity,
            "temperature": temperature,
            "enable_brownian": enable_brownian,
            "lambda_current": lambda_current,
            "max_step": max_step,
        },
    }


# ============================================================
# REPORTING & VISUALISATION
# ============================================================

def print_summary(result, trial_name):
    print(f"\n{'='*60}")
    print(f"  {trial_name}")
    print(f"{'='*60}")
    print(f"  Start           : {result['start']}")
    print(f"  Target          : {result['target']}")
    print(f"  Final position  : {result['final_position']}")
    print(f"  Steps taken     : {result['steps_taken']}")
    print(f"  Final error (m) : {result['final_error']:.6f}")
    print(f"  Final error (mm): {result['final_error']*1000:.3f}")
    print(f"  Reached target  : {result['reached']}")
    print(f"  Max current (A) : {result['max_current_used']:.1f}")
    if len(result['force_log']) > 0:
        print(f"  Peak |F|  (N)   : {np.max(result['force_log']):.3e}")
        print(f"  Peak |v|  (m/s) : {np.max(result['velocity_log']):.3e}")
        print(f"  Peak |B|  (T)   : {np.max(result['field_log']):.3e}")


def make_output_dir(path="simulation_outputs_3d_realistic"):
    os.makedirs(path, exist_ok=True)
    return path


def save_trajectory_plot_3d(result, trial_name, output_dir):
    traj = result["trajectory"]
    target = result["target"]
    magnets = result["magnets"]
    half = result["workspace_half"]

    fig = plt.figure(figsize=(10, 10))
    ax = fig.add_subplot(111, projection="3d")

    ax.plot(traj[:, 0], traj[:, 1], traj[:, 2], lw=2, label="Path")
    ax.scatter(*traj[0], s=120, marker="o", label="Start")
    ax.scatter(*target, s=120, marker="*", label="Target")
    ax.scatter(magnets[:, 0], magnets[:, 1], magnets[:, 2],
               s=80, marker="^", label="Magnets")

    ax.set_xlim(-half, half)
    ax.set_ylim(-half, half)
    ax.set_zlim(-half, half)
    ax.set_xlabel("X (m)")
    ax.set_ylabel("Y (m)")
    ax.set_zlabel("Z (m)")
    ax.set_title(f"{trial_name} – 3D Trajectory (realistic)")
    ax.legend(fontsize=8)

    fn = os.path.join(output_dir, f"{trial_name}_trajectory_3d.png")
    plt.savefig(fn, dpi=300, bbox_inches="tight")
    plt.close()


def save_distance_plot(result, trial_name, output_dir):
    fig, ax1 = plt.subplots(figsize=(10, 5))

    steps = np.arange(len(result["distance_log"]))
    t_sec = steps * result["params"]["dt"]

    ax1.plot(t_sec, result["distance_log"] * 1000, lw=2, color="tab:blue")
    ax1.set_xlabel("Time (s)")
    ax1.set_ylabel("Distance to Target (mm)", color="tab:blue")
    ax1.grid(True, alpha=0.3)
    ax1.set_title(f"{trial_name} – Distance to Target")

    fn = os.path.join(output_dir, f"{trial_name}_distance.png")
    plt.savefig(fn, dpi=300, bbox_inches="tight")
    plt.close()


def save_physics_plot(result, trial_name, output_dir):
    """Combined plot: force, velocity, field magnitude over time."""
    if len(result["force_log"]) == 0:
        return

    steps = np.arange(len(result["force_log"]))
    t_sec = steps * result["params"]["dt"]

    fig, axes = plt.subplots(3, 1, figsize=(10, 10), sharex=True)

    axes[0].plot(t_sec, result["force_log"], lw=1.5, color="tab:red")
    axes[0].set_ylabel("|F| (N)")
    axes[0].set_title(f"{trial_name} – Physics Over Time")
    axes[0].grid(True, alpha=0.3)
    axes[0].ticklabel_format(axis="y", style="scientific", scilimits=(0, 0))

    axes[1].plot(t_sec, np.array(result["velocity_log"]) * 1e6,
                 lw=1.5, color="tab:green")
    axes[1].set_ylabel("|v| (μm/s)")
    axes[1].grid(True, alpha=0.3)

    axes[2].plot(t_sec, np.array(result["field_log"]) * 1e3,
                 lw=1.5, color="tab:purple")
    axes[2].set_ylabel("|B| (mT)")
    axes[2].set_xlabel("Time (s)")
    axes[2].grid(True, alpha=0.3)

    plt.tight_layout()
    fn = os.path.join(output_dir, f"{trial_name}_physics.png")
    plt.savefig(fn, dpi=300, bbox_inches="tight")
    plt.close()


def save_field_map_3d(result, trial_name, output_dir, grid_points=15):
    """z=0 midplane field map using first-step currents."""
    if len(result["current_log"]) == 0:
        return

    half = result["workspace_half"]
    params = result["params"]

    array = ElectromagnetArray3D_Real(
        coil_radius=params["coil_radius"],
        coil_length=params["coil_length"],
        turns=params["turns"],
        max_current=params["max_current"],
    )
    array.set_currents(result["current_log"][0])

    xs = np.linspace(-half, half, grid_points)
    ys = np.linspace(-half, half, grid_points)
    X, Y = np.meshgrid(xs, ys)

    U = np.zeros_like(X)
    V = np.zeros_like(Y)
    M = np.zeros_like(X)

    for i in range(grid_points):
        for j in range(grid_points):
            B = array.field_at(np.array([X[i, j], Y[i, j], 0.0]))
            U[i, j] = B[0]
            V[i, j] = B[1]
            M[i, j] = np.linalg.norm(B)

    if np.all(M < 1e-20):
        return

    mask = M > 1e-20
    U_n = np.zeros_like(U)
    V_n = np.zeros_like(V)
    U_n[mask] = U[mask] / M[mask]
    V_n[mask] = V[mask] / M[mask]

    plt.figure(figsize=(8, 8))
    plt.quiver(X, Y, U_n, V_n, M * 1e3)   # colour = mT
    plt.colorbar(label="|B| (mT)")

    magnets = result["magnets"]
    plt.scatter(magnets[:, 0], magnets[:, 1], s=80, marker="^",
                label="Magnets (proj.)")
    plt.scatter(result["start"][0], result["start"][1], s=100,
                label="Start (proj.)")
    plt.scatter(result["target"][0], result["target"][1], s=100,
                label="Target (proj.)")

    plt.xlim(-half, half)
    plt.ylim(-half, half)
    plt.gca().set_aspect("equal")
    plt.grid(True, alpha=0.3)
    plt.legend()
    plt.xlabel("X (m)")
    plt.ylabel("Y (m)")
    plt.title(f"{trial_name} – Field Map z=0 (mT)")

    fn = os.path.join(output_dir, f"{trial_name}_fieldmap_z0.png")
    plt.savefig(fn, dpi=300, bbox_inches="tight")
    plt.close()


def save_results_csv(results, output_dir, filename="trial_summary.csv"):
    path = os.path.join(output_dir, filename)
    with open(path, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow([
            "trial_name",
            "start_x", "start_y", "start_z",
            "target_x", "target_y", "target_z",
            "final_x", "final_y", "final_z",
            "steps_taken", "sim_time_s",
            "final_error_m", "final_error_mm",
            "reached_target", "max_current_A",
            "peak_force_N", "peak_velocity_m_s", "peak_field_T",
        ])

        for name, r in results.items():
            w.writerow([
                name,
                r["start"][0], r["start"][1], r["start"][2],
                r["target"][0], r["target"][1], r["target"][2],
                r["final_position"][0], r["final_position"][1],
                r["final_position"][2],
                r["steps_taken"],
                r["steps_taken"] * r["params"]["dt"],
                r["final_error"],
                r["final_error"] * 1000.0,
                r["reached"],
                r["max_current_used"],
                np.max(r["force_log"]) if len(r["force_log"]) > 0 else 0,
                np.max(r["velocity_log"]) if len(r["velocity_log"]) > 0 else 0,
                np.max(r["field_log"]) if len(r["field_log"]) > 0 else 0,
            ])


# ============================================================
# TRIAL RUNNER
# ============================================================

def run_all_trials():
    EDGE = TEST_RANGE

    trials = {
        "trial_1_corner_to_corner": {
            "start": (-EDGE, -EDGE, -EDGE),
            "target": (EDGE, EDGE, EDGE),
        },
        "trial_2_opposite_edges": {
            "start": (-EDGE, EDGE, -EDGE),
            "target": (EDGE, -EDGE, EDGE),
        },
        "trial_3_center_to_corner": {
            "start": (0.0, 0.0, 0.0),
            "target": (EDGE, EDGE, EDGE),
        },
    }

    output_dir = make_output_dir("simulation_outputs_3d_realistic")
    results = {}

    sim_kwargs = {
        "dt": 0.02,              # 20 ms per step
        "steps": 3000,           # max 60 s
        "tolerance": 0.003,      # 3 mm
        "coil_radius": 0.03,     # 3 cm
        "coil_length": 0.12,     # 12 cm
        "turns": 1500,
        "max_current": 8.0,
        "particle_diameter": 500e-6,    # 500 μm Fe₃O₄ microsphere
        "chi_eff": 3.0,
        "M_sat": 4.5e5,
        "fluid_viscosity": 8.9e-4,      # water @ 25 °C
        "temperature": 298.0,
        "enable_brownian": True,
        "lambda_current": 5e-5,
        "max_step": MAX_STEP,
        "rng_seed": 42,
    }

    for trial_name, cfg in trials.items():
        print(f"\n>>> Running {trial_name} …")
        result = simulate_trial_realistic(
            start=cfg["start"],
            target=cfg["target"],
            **sim_kwargs,
        )
        results[trial_name] = result

        print_summary(result, trial_name)
        save_trajectory_plot_3d(result, trial_name, output_dir)
        save_distance_plot(result, trial_name, output_dir)
        save_physics_plot(result, trial_name, output_dir)
        save_field_map_3d(result, trial_name, output_dir)

    save_results_csv(results, output_dir)
    print(f"\nSaved plots and CSV to: {output_dir}")
    return results


if __name__ == "__main__":
    run_all_trials()
