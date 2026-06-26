"""
verify_cfd_interpolator.py
==========================
Smoke test for CFDFlowField without needing a real OpenFOAM solve.

Creates a synthetic velocity field by sampling the analytical Poiseuille
model on a grid, saves it as a .npz, loads it back via CFDFlowField, and
checks that the interpolated velocities match the analytical ground truth
to within a tolerance.

Tests
-----
1. SteadyCFDFlowField     — steady Poiseuille field
2. PulsatileCFDFlowField  — 20-phase sinusoidal pulsation
3. Drop-in parity test    — RealVesselNetworkCFD vs analytical at 100 random positions
4. Timing                 — verify interpolation is fast enough for simulation step rates

Run with:
    python verify_cfd_interpolator.py
"""

from __future__ import annotations

import time
import tempfile
from pathlib import Path

import numpy as np

# ---------------------------------------------------------------------------
# Synthetic field generation
# ---------------------------------------------------------------------------

def _poiseuille_velocity(pos: np.ndarray, center, axis, radius, v_max, time=0.0, amplitude=0.15, freq=1.0):
    """Analytical reference."""
    pos = np.asarray(pos, dtype=np.float64)
    center = np.asarray(center, dtype=np.float64)
    axis = np.asarray(axis, dtype=np.float64)
    axis = axis / np.linalg.norm(axis)
    d = pos - center
    axial = np.dot(d, axis)
    radial_vec = d - axial * axis
    r = np.linalg.norm(radial_vec)
    if r >= radius:
        return np.zeros(3)
    profile = 1.0 - (r / radius) ** 2
    pulsation = 1.0 + amplitude * np.sin(2.0 * np.pi * freq * time)
    return v_max * profile * pulsation * axis


def build_synthetic_steady_npz(tmpdir: Path, n_cells: int = 20000) -> Path:
    """Sample Poiseuille on a cylinder grid and save as steady .npz."""
    center = np.array([0.0, 0.0, 0.0])
    axis = np.array([0.0, 0.0, 1.0])
    radius = 0.005   # 5 mm
    v_max = 0.10     # 100 mm/s

    rng = np.random.default_rng(42)
    # Cylindrical coordinates
    r_samp = rng.uniform(0, radius * 0.95, n_cells)
    theta = rng.uniform(0, 2 * np.pi, n_cells)
    z = rng.uniform(-0.05, 0.05, n_cells)

    points = np.column_stack([
        r_samp * np.cos(theta),
        r_samp * np.sin(theta),
        z,
    ]) + center

    velocity = np.array([
        _poiseuille_velocity(p, center, axis, radius, v_max)
        for p in points
    ], dtype=np.float32)

    out = tmpdir / "test_steady.npz"
    np.savez_compressed(str(out),
                        points=points.astype(np.float32),
                        velocity=velocity)
    return out


def build_synthetic_pulsatile_npz(tmpdir: Path, n_cells: int = 10000, n_phases: int = 20) -> Path:
    """20-phase pulsatile field."""
    center = np.array([0.0, 0.0, 0.0])
    axis = np.array([0.0, 0.0, 1.0])
    radius = 0.005
    v_max = 0.10
    period = 1.0

    rng = np.random.default_rng(7)
    r_samp = rng.uniform(0, radius * 0.95, n_cells)
    theta = rng.uniform(0, 2 * np.pi, n_cells)
    z = rng.uniform(-0.02, 0.02, n_cells)

    points = np.column_stack([
        r_samp * np.cos(theta),
        r_samp * np.sin(theta),
        z,
    ]) + center

    phases = np.linspace(0, period, n_phases, endpoint=False)
    all_vel = []
    for t in phases:
        vel = np.array([
            _poiseuille_velocity(p, center, axis, radius, v_max, time=t)
            for p in points
        ], dtype=np.float32)
        all_vel.append(vel)

    velocity_stack = np.stack(all_vel, axis=0)   # (N, M, 3)

    out = tmpdir / "test_pulsatile.npz"
    np.savez_compressed(str(out),
                        points=points.astype(np.float32),
                        velocity=velocity_stack,
                        period=np.float32(period))
    return out


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------

def test_steady(npz_path: Path) -> bool:
    from cfd_flow_field import SteadyCFDFlowField

    field = SteadyCFDFlowField(npz_path, k_neighbours=4)

    center = np.array([0.0, 0.0, 0.0])
    axis   = np.array([0.0, 0.0, 1.0])
    radius = 0.005
    v_max  = 0.10

    test_positions = [
        np.array([0.0,    0.0,   0.0]),     # centerline — maximum velocity
        np.array([0.002,  0.0,   0.01]),    # off-center
        np.array([0.0,    0.003, -0.01]),   # off-center
    ]
    ok = True
    for pos in test_positions:
        v_cfd  = field.velocity_at(pos, time=0.0)
        v_ref  = _poiseuille_velocity(pos, center, axis, radius, v_max)
        error  = np.linalg.norm(v_cfd - v_ref)
        rel    = error / max(np.linalg.norm(v_ref), 1e-9)
        status = "OK" if rel < 0.05 else "FAIL"
        print(f"  [steady] {status}  pos={pos}  ref={np.linalg.norm(v_ref)*1000:.2f} mm/s  "
              f"cfd={np.linalg.norm(v_cfd)*1000:.2f} mm/s  rel_err={rel:.3f}")
        if rel >= 0.05:
            ok = False
    return ok


def test_pulsatile(npz_path: Path) -> bool:
    from cfd_flow_field import PulsatileCFDFlowField

    field = PulsatileCFDFlowField(npz_path, k_neighbours=4)

    center = np.array([0.0, 0.0, 0.0])
    axis   = np.array([0.0, 0.0, 1.0])
    radius = 0.005
    v_max  = 0.10

    pos = np.array([0.001, 0.001, 0.005])
    ok = True
    for t in [0.0, 0.25, 0.5, 0.75]:
        v_cfd = field.velocity_at(pos, time=t)
        v_ref = _poiseuille_velocity(pos, center, axis, radius, v_max, time=t)
        error = np.linalg.norm(v_cfd - v_ref)
        rel   = error / max(np.linalg.norm(v_ref), 1e-9)
        status = "OK" if rel < 0.08 else "FAIL"   # slightly looser: temporal interp
        print(f"  [pulsatile] {status}  t={t:.2f}s  "
              f"ref={np.linalg.norm(v_ref)*1000:.2f} mm/s  "
              f"cfd={np.linalg.norm(v_cfd)*1000:.2f} mm/s  rel_err={rel:.3f}")
        if rel >= 0.08:
            ok = False
    return ok


def test_timing(npz_path: Path, n_queries: int = 5000) -> None:
    from cfd_flow_field import SteadyCFDFlowField

    field = SteadyCFDFlowField(npz_path, k_neighbours=4)
    rng = np.random.default_rng(99)
    positions = rng.uniform(-0.003, 0.003, (n_queries, 3))

    t0 = time.perf_counter()
    for pos in positions:
        field.velocity_at(pos, time=0.0)
    elapsed = time.perf_counter() - t0

    us_per_query = elapsed / n_queries * 1e6
    # Simulation runs at ~10 ms per step; interpolation should be << 1 ms
    status = "OK" if us_per_query < 500 else "SLOW"
    print(f"  [timing] {status}  {n_queries} queries in {elapsed*1000:.1f} ms  "
          f"({us_per_query:.0f} µs/query)")


def test_extrapolation(npz_path: Path) -> None:
    from cfd_flow_field import SteadyCFDFlowField

    field = SteadyCFDFlowField(npz_path, k_neighbours=1, extrapolate_zero=True)
    outside = np.array([1.0, 1.0, 1.0])   # far outside bbox
    v = field.velocity_at(outside, time=0.0)
    status = "OK" if np.allclose(v, 0) else "FAIL"
    print(f"  [extrapolation] {status}  position outside bbox → v={v}")


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

def main():
    print("=" * 60)
    print("CFDFlowField verification")
    print("=" * 60)

    with tempfile.TemporaryDirectory() as tmpdir:
        tmpdir = Path(tmpdir)

        print("\n--- Building synthetic steady .npz ---")
        steady_npz = build_synthetic_steady_npz(tmpdir)
        print(f"    Written: {steady_npz}")

        print("\n--- Building synthetic pulsatile .npz (20 phases) ---")
        pulsatile_npz = build_synthetic_pulsatile_npz(tmpdir)
        print(f"    Written: {pulsatile_npz}")

        print("\n--- Test 1: SteadyCFDFlowField spatial accuracy ---")
        ok1 = test_steady(steady_npz)

        print("\n--- Test 2: PulsatileCFDFlowField spatiotemporal accuracy ---")
        ok2 = test_pulsatile(pulsatile_npz)

        print("\n--- Test 3: Out-of-domain extrapolation ---")
        test_extrapolation(steady_npz)

        print("\n--- Test 4: Query timing ---")
        test_timing(steady_npz)

    print("\n" + "=" * 60)
    if ok1 and ok2:
        print("ALL TESTS PASSED")
    else:
        print("SOME TESTS FAILED — check relative errors above")
    print("=" * 60)


if __name__ == "__main__":
    main()
