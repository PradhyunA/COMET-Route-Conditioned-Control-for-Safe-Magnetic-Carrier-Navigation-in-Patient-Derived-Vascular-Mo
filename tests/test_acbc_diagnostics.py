"""Tests for ACBC publication diagnostics schema."""

from dataclasses import fields

from dynamag_simulation.core.acbc_controller import ACBCStepInfo


def test_acbc_step_info_contains_publication_diagnostics():
    names = {field.name for field in fields(ACBCStepInfo)}

    assert "state" in names
    assert "transition" in names
    assert "selected_authority_margin" in names
    assert "stall_escape_target" in names
    assert "best_route_idx" in names
    assert "stagnant_steps" in names
    assert "wall_clearance_m" in names
    assert "wall_fraction" in names
