"""Tests for ACBC state-machine stabilization helpers."""

from types import SimpleNamespace

import numpy as np

from dynamag_simulation.core.acbc_controller import (
    ACBC_ASSESS,
    ACBC_COMMIT,
    ACBC_FREE,
    ACBC_RECOVER,
    ACBCController,
    BranchCandidate,
)


def _controller_stub():
    c = ACBCController.__new__(ACBCController)
    c.state = ACBC_FREE
    c.prev_state = ACBC_FREE
    c.step_count = 0
    c.transition_count = 0
    c.last_transition_step = None
    c.transition_cooldown_steps = 4
    c.commit_start_step = None
    c.min_commitment_steps = 10
    c.last_recovery_step = None
    c.min_recovery_steps = 8
    c.recovery_count = 0
    c.surface_clearance_target_m = 0.0005
    c.wall_clearance_threshold_m = 0.0007
    c.recovery_enter_wall_clearance = 0.0010
    c.recovery_exit_wall_clearance = 0.0015
    c.min_wall_safety_fraction = 0.015
    c.near_wall_edge_multiplier = 6.0
    c.selected_branch = None
    c.selected_branch_id = None
    c.commit_target_idx = None
    c.escape_start_step = None
    c.escape_start_route_idx = None
    c.best_route_idx = 0
    return c


def _candidate(branch_seg_id=7):
    return BranchCandidate(
        branch_seg_id=branch_seg_id,
        bifurcation_point=np.array([0.0, 0.0, 0.0]),
        direction=np.array([1.0, 0.0, 0.0]),
        target_point=np.array([0.003, 0.0, 0.0]),
        flow_projection_m_s=0.0,
        max_magnetic_projection_m_s=0.1,
        authority_margin=1.2,
        wall_safety_margin_m=0.001,
        wall_safety_fraction=0.2,
        route_alignment=1.0,
        recoverability=1.0,
        score=5.0,
        capturable=True,
    )


def test_transition_cooldown_blocks_non_safety_transition():
    c = _controller_stub()
    c.step_count = 12
    c.last_transition_step = 10

    transitioned = c._transition(ACBC_ASSESS, reason="test")

    assert transitioned is False
    assert c.state == ACBC_FREE
    assert c.transition_count == 0


def test_hard_safety_transition_bypasses_cooldown():
    c = _controller_stub()
    c.step_count = 12
    c.last_transition_step = 10

    transitioned = c._transition(
        ACBC_RECOVER,
        hard_override=True,
        reason="hard_wall_safety",
    )

    assert transitioned is True
    assert c.state == ACBC_RECOVER
    assert c.transition_count == 1
    assert c.last_recovery_step == 12


def test_commitment_dwell_blocks_early_exit():
    c = _controller_stub()
    c.state = ACBC_COMMIT
    c.commit_start_step = 20
    c.step_count = 25

    transitioned = c._transition(ACBC_FREE, reason="commitment_complete")

    assert transitioned is False
    assert c.state == ACBC_COMMIT
    assert c.transition_count == 0


def test_commitment_exit_allowed_after_dwell():
    c = _controller_stub()
    c.state = ACBC_COMMIT
    c.commit_start_step = 20
    c.step_count = 31

    transitioned = c._transition(ACBC_FREE, reason="commitment_complete")

    assert transitioned is True
    assert c.state == ACBC_FREE
    assert c.transition_count == 1


def test_branch_lock_stores_commitment_metadata():
    c = _controller_stub()
    c.step_count = 42
    navigator = SimpleNamespace(
        waypoints=[
            np.array([0.0, 0.0, 0.0]),
            np.array([0.002, 0.0, 0.0]),
            np.array([0.003, 0.0, 0.0]),
            np.array([0.004, 0.0, 0.0]),
        ]
    )

    c._acquire_branch_lock(_candidate(branch_seg_id=11), navigator)

    assert c.selected_branch_id == 11
    assert c.commit_start_step == 42
    assert c.commit_target_idx == 2


def test_required_clearance_uses_common_absolute_threshold():
    c = _controller_stub()
    particle = SimpleNamespace(radius=0.00025)

    assert np.isclose(c._required_wall_clearance(particle), 0.0007)


def test_hard_wall_safety_uses_required_clearance():
    c = _controller_stub()
    particle = SimpleNamespace(radius=0.00025)

    assert c._hard_wall_unsafe(0.00069, 0.5, particle) is True
    assert c._hard_wall_unsafe(0.00071, -1.0, particle) is False


def test_hard_wall_safety_releases_branch_lock_and_enters_recovery():
    c = _controller_stub()
    c.state = ACBC_COMMIT
    c.step_count = 30
    c.last_transition_step = 29
    c.selected_branch = _candidate(branch_seg_id=12)
    c.selected_branch_id = 12
    c.commit_start_step = 30
    c.commit_target_idx = 5

    transitioned = c._enter_recovery_for_wall_safety("hard_wall_safety")

    assert transitioned is True
    assert c.state == ACBC_RECOVER
    assert c.selected_branch is None
    assert c.selected_branch_id is None
    assert c.commit_start_step is None
    assert c.commit_target_idx is None
    assert c.recovery_count == 1


def test_near_wall_edge_multiplier_increases_effective_edge_weight():
    c = _controller_stub()
    particle = SimpleNamespace(radius=0.00025)

    boosted = c._edge_weight_with_near_wall_boost(
        base_w_edge=50.0,
        current_w_edge=50.0,
        wall_clearance=0.0012,
        particle=particle,
    )
    unchanged = c._edge_weight_with_near_wall_boost(
        base_w_edge=50.0,
        current_w_edge=50.0,
        wall_clearance=0.0020,
        particle=particle,
    )

    assert boosted == 300.0
    assert unchanged == 50.0
