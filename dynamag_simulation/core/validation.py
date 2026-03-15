"""
Physics-regime validation utilities.

Primary check: Reynolds number must stay in the creeping-flow
(Stokes) regime for the overdamped drag model to be valid.
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
    arr = np.asarray(reynolds_values, dtype=float)
    if len(arr) == 0:
        return {"mean_reynolds": 0.0, "max_reynolds": 0.0, "low_re_valid": True}

    return {
        "mean_reynolds": float(np.mean(arr)),
        "max_reynolds": float(np.max(arr)),
        "low_re_valid": bool(np.max(arr) < threshold),
    }
