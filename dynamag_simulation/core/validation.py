"""
Physics-regime validation utilities.

The legacy low-Re check is retained for Stokes-regime microsphere and
nanoparticle simulations. Larger magnetic carriers can operate outside
creeping flow, so publication-facing 1 mm carrier studies should report
the finite-Re drag-model regime instead of treating ``Re >= 0.1`` as a
failure by itself.
"""

import numpy as np


def compute_reynolds(velocity_mag: float, diameter: float,
                     fluid_density: float, fluid_viscosity: float) -> float:
    """Particle Reynolds number: Re = rho * v * d / eta."""
    return fluid_density * velocity_mag * diameter / fluid_viscosity


def check_low_re_validity(reynolds_values, threshold: float = 0.1):
    """
    Evaluate whether a run stayed within the low-Re regime.

    Returns
    -------
    dict with keys:
        mean_reynolds  : float
        max_reynolds   : float
        low_re_valid   : bool   (True if max Re < threshold)
    """
    stats = classify_reynolds_regime(
        reynolds_values,
        stokes_threshold=threshold,
    )

    # Backward-compatible alias used by older scripts and CSV readers.
    stats["low_re_valid"] = stats["stokes_regime"]

    return stats


def classify_reynolds_regime(
    reynolds_values,
    stokes_threshold: float = 0.1,
    finite_re_limit: float = 800.0,
    drag_model: str = "schiller_naumann",
):
    """
    Classify Reynolds values for both Stokes and finite-Re carrier studies.

    Returns keys suitable for ``TrialResult`` and CSV export:
        mean_reynolds
        max_reynolds
        stokes_regime
        finite_re_drag_valid
        drag_model
        drag_regime_label
    """
    arr = np.asarray(reynolds_values, dtype=float)
    arr = arr[np.isfinite(arr)]

    if len(arr) == 0:
        max_re = 0.0
        mean_re = 0.0
    else:
        max_re = float(np.max(arr))
        mean_re = float(np.mean(arr))

    stokes_regime = bool(max_re < stokes_threshold)
    finite_re_drag_valid = bool(max_re <= finite_re_limit)

    if stokes_regime:
        label = "stokes"
    elif finite_re_drag_valid:
        label = "finite_re_schiller_naumann"
    else:
        label = "outside_model_range"

    return {
        "mean_reynolds": mean_re,
        "max_reynolds": max_re,
        "stokes_regime": stokes_regime,
        "finite_re_drag_valid": finite_re_drag_valid,
        "drag_model": drag_model,
        "drag_regime_label": label,
    }
