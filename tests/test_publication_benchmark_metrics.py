"""Tests for paired publication benchmark bookkeeping."""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))

from scripts.run_publication_benchmark import (
    CONTROLLER_VARIANTS,
    _paired_difference_rows,
)


def _row(controller, reached, steps):
    return {
        "flow_scale": 1.0,
        "pair_id": "pair_1",
        "trial_kind": "random",
        "controller_key": controller,
        "reached_target": reached,
        "steps_taken": steps,
        "sim_time_s": steps * 0.02,
        "wall_time_s": float(steps),
        "avg_control_step_time_s": 1.0,
        "path_efficiency": 0.9,
        "current_rms_A": 2.0,
        "energy_proxy_A2s": 3.0,
        "min_wall_clearance_mm": 0.7,
        "min_surface_clearance_mm": 0.45,
    }


def test_primary_controller_variants_include_published_cem_mpc():
    assert {
        "icem_mpc_default_shared",
        "icem_mpc_tuned_shared",
        "pets_cem_mpc",
        "naive_mpc",
        "mpc_default",
        "mpc_tuned",
        "acbc_same_weights",
        "acbc_tuned",
        "smc_dob_inspired",
    }.issubset(CONTROLLER_VARIANTS)

    published = CONTROLLER_VARIANTS["pets_cem_mpc"]
    assert published.kind == "cem_mpc"
    assert published.kwargs["optimizer"] == "pets_cem"
    assert published.kwargs["cem_population_size"] == 400
    assert published.kwargs["cem_num_elites"] == 40

    smc_dob = CONTROLLER_VARIANTS["smc_dob_inspired"]
    assert smc_dob.kind == "smc_dob_inspired"
    assert "not an exact reproduction" in smc_dob.label
    assert smc_dob.kwargs["sliding_gamma"] == 2.0
    assert smc_dob.kwargs["sliding_lambda"] == 1.0


def test_icem_shared_objective_variants_only_change_task_cost_weights():
    default = CONTROLLER_VARIANTS["icem_mpc_default_shared"]
    tuned = CONTROLLER_VARIANTS["icem_mpc_tuned_shared"]

    assert default.kind == tuned.kind == "cem_mpc"
    assert default.kwargs["optimizer"] == tuned.kwargs["optimizer"] == "icem"
    for key in (
        "cem_horizon",
        "cem_population_size",
        "cem_num_elites",
        "cem_max_iterations",
        "cem_alpha",
        "icem_init_std",
        "icem_population_decay_factor",
        "icem_fraction_elites_reused",
        "icem_noise_beta",
        "icem_use_mean_action",
        "icem_keep_previous_elites",
        "icem_shift_elites_over_time",
    ):
        assert default.kwargs[key] == tuned.kwargs[key]

    assert default.kwargs["w_progress"] == 120.0
    assert default.kwargs["w_align"] == 8.0
    assert default.kwargs["w_route"] == 0.0
    assert tuned.kwargs["w_progress"] == 450.0
    assert tuned.kwargs["w_align"] == 35.0
    assert tuned.kwargs["w_route"] == 15.0


def test_five_comet_ablation_programs_are_available():
    assert {
        "comet",
        "comet_no_authority_reserve",
        "comet_no_commitment_memory",
        "comet_no_downstream_value",
        "comet_no_terminal_precision",
    }.issubset(CONTROLLER_VARIANTS)


def test_paired_differences_match_trials_by_flow_and_pair():
    rows = [
        _row("mpc_default", True, 100),
        _row("acbc_same_weights", True, 90),
    ]
    differences = _paired_difference_rows(rows)
    step_difference = next(
        row for row in differences if row["metric"] == "steps_taken"
    )

    assert step_difference["baseline"] == "mpc_default"
    assert step_difference["comparator"] == "acbc_same_weights"
    assert step_difference["paired_difference"] == -10.0
