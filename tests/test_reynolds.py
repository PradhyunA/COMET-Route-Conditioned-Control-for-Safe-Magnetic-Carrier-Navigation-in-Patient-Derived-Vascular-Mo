"""Tests for Reynolds number computation and low-Re flagging."""

import numpy as np
import pytest

from dynamag_simulation.core.validation import (
    compute_reynolds,
    check_low_re_validity,
    classify_reynolds_regime,
)
from dynamag_simulation.core.profiles import MICROSPHERE_500UM


def test_reynolds_formula():
    re = compute_reynolds(
        velocity_mag=1e-3,
        diameter=500e-6,
        fluid_density=997.0,
        fluid_viscosity=8.9e-4,
    )
    expected = 997.0 * 1e-3 * 500e-6 / 8.9e-4
    assert re == pytest.approx(expected, rel=1e-10)


def test_zero_velocity_gives_zero_re():
    re = compute_reynolds(0.0, 500e-6, 997.0, 8.9e-4)
    assert re == 0.0


def test_low_re_valid_below_threshold():
    stats = check_low_re_validity([0.001, 0.05, 0.09])
    assert stats["low_re_valid"] is True
    assert stats["max_reynolds"] < 0.1


def test_low_re_invalid_above_threshold():
    stats = check_low_re_validity([0.001, 0.05, 0.15])
    assert stats["low_re_valid"] is False
    assert stats["max_reynolds"] == pytest.approx(0.15)


def test_empty_log_is_valid():
    stats = check_low_re_validity([])
    assert stats["low_re_valid"] is True
    assert stats["max_reynolds"] == 0.0


def test_high_force_triggers_invalid_re():
    """Synthetic scenario: artificially large velocity should flag Re > 0.1."""
    profile = MICROSPHERE_500UM
    vel_large = 1.0  # 1 m/s -- absurdly fast for Stokes
    re = compute_reynolds(vel_large, profile.diameter,
                          profile.fluid_density, profile.fluid_viscosity)
    assert re > 0.1
    stats = check_low_re_validity([re])
    assert stats["low_re_valid"] is False


def test_finite_re_carrier_regime_is_valid_but_not_stokes():
    stats = classify_reynolds_regime([140.0, 190.0])
    assert stats["stokes_regime"] is False
    assert stats["finite_re_drag_valid"] is True
    assert stats["drag_regime_label"] == "finite_re_schiller_naumann"
    assert stats["drag_model"] == "schiller_naumann"


def test_outside_drag_model_range_flags_invalid():
    stats = classify_reynolds_regime([801.0])
    assert stats["stokes_regime"] is False
    assert stats["finite_re_drag_valid"] is False
    assert stats["drag_regime_label"] == "outside_model_range"
