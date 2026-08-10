"""
Paired publication benchmark for the 1 mm magnetic-carrier ACBC study.

This runner is intentionally controller-comparison focused:
    - identical vessel networks, seeds, start/target pairs, hardware, and flow
    - separated MPC-default, MPC-tuned, ACBC-same-weights, ACBC-tuned variants
    - optional ACBC ablations
    - paired CSV and JSON summaries suitable for manuscript figures/tables

Default settings are publication-scale and can be slow. Use --smoke-test for
quick validation.
"""

from __future__ import annotations

import argparse
import csv
import json
import os
import platform
import subprocess
import sys
import tempfile
import time
from concurrent.futures import ProcessPoolExecutor, ThreadPoolExecutor, as_completed
from collections import defaultdict
from dataclasses import dataclass
from typing import Callable

import numpy as np

os.environ.setdefault(
    "MPLCONFIGDIR",
    os.path.join(tempfile.gettempdir(), "dynamag_matplotlib"),
)
os.environ.setdefault("MPLBACKEND", "Agg")

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from dynamag_simulation.core.real_vessel_network import (
    RealVesselNetwork,
    RoutePlanningOptions,
)
from dynamag_simulation.core.profiles import SPION_CARRIER_1MM, SPION_CARRIER_500UM
from dynamag_simulation.core.constants import DEFAULT_TARGET_TOLERANCE
from dynamag_simulation.core.reporting import save_results_csv

from simulation_3d_pid import (
    DEFAULT_VESSEL_FILE,
    DEVICE_MODE,
    FIXED_COIL_LENGTH,
    FIXED_COIL_RADIUS,
    FIXED_CORE_MU_R_EFFECTIVE,
    FIXED_CORE_RADIUS,
    FIXED_CORE_SATURATION_T,
    FIXED_MAX_CURRENT,
    FIXED_PLACEMENT_RADIUS,
    FIXED_TURNS,
    save_pid_diagnostics,
    simulate_trial_pid,
)
from simulation_3d_comet import save_comet_diagnostics, simulate_trial_comet

BENCHMARK_EARLY_STOP_WINDOW_STEPS = 200
BENCHMARK_EARLY_STOP_IMPROVEMENT_THRESHOLD_M = 0.002


MPC_DEFAULT_WEIGHTS = {
    "w_progress": 120.0,
    "w_align": 8.0,
    "w_dist": 1.0,
    "w_curr": 1e-5,
    "w_smooth": 1e-4,
    "w_edge": 50.0,
    "w_overshoot": 2.0,
    "w_route": 0.0,
}

ACBC_TUNED_WEIGHTS = {
    "w_progress": 450.0,
    "w_align": 35.0,
    "w_dist": 2.0,
    "w_curr": 1e-7,
    "w_smooth": 1e-6,
    "w_edge": 250.0,
    "w_overshoot": 8.0,
    "w_route": 15.0,
}

PROFILE_CHOICES = {
    "spion_carrier_1mm": SPION_CARRIER_1MM,
    "spion_carrier_500um": SPION_CARRIER_500UM,
}


@dataclass(frozen=True)
class ControllerVariant:
    label: str
    kind: str
    kwargs: dict


CONTROLLER_VARIANTS = {
    "naive_mpc": ControllerVariant(
        label="Naive-MPC-final-target",
        kind="naive_mpc",
        kwargs={**MPC_DEFAULT_WEIGHTS, "target_mode": "final_target"},
    ),
    "mpc_default": ControllerVariant(
        label="MPC-default",
        kind="mpc",
        kwargs=dict(MPC_DEFAULT_WEIGHTS),
    ),
    "mpc_tuned": ControllerVariant(
        label="MPC-tuned",
        kind="mpc",
        kwargs=dict(ACBC_TUNED_WEIGHTS),
    ),
    "acbc_same_weights": ControllerVariant(
        label="ACBC-same-weights",
        kind="acbc",
        kwargs=dict(MPC_DEFAULT_WEIGHTS),
    ),
    "acbc_tuned": ControllerVariant(
        label="ACBC-tuned",
        kind="acbc",
        kwargs=dict(ACBC_TUNED_WEIGHTS),
    ),
    "comet": ControllerVariant(
        label="COMET",
        kind="comet",
        kwargs={},
    ),
    "comet_no_authority_reserve": ControllerVariant(
        label="COMET-no-authority-reserve",
        kind="comet",
        kwargs={
            "authority_reserve_weight": 0.0,
            "capture_reserve_gain": 0.0,
            "capture_urgency_gain": 0.0,
            "hard_commit_reserve_threshold": -1e9,
            "rescue_reserve_threshold": -1e9,
            "rescue_exit_reserve": -1e9,
        },
    ),
    "comet_no_commitment_memory": ControllerVariant(
        label="COMET-no-commitment-memory",
        kind="comet",
        kwargs={
            "intent_smoothing": 0.0,
            "min_commitment_dwell_steps": 0,
            "branch_switch_penalty": 0.0,
        },
    ),
    "comet_no_downstream_value": ControllerVariant(
        label="COMET-no-downstream-value",
        kind="comet",
        kwargs={
            "downstream_value_weight": 0.0,
        },
    ),
    "comet_no_terminal_precision": ControllerVariant(
        label="COMET-no-terminal-precision-servo",
        kind="comet",
        kwargs={
            "terminal_precision_distance": 0.0,
            "terminal_precision_kp": 0.0,
            "terminal_precision_kv": 0.0,
        },
    ),
    "comet_no_flow_scaled_reserve": ControllerVariant(
        label="COMET-no-flow-scaled-reserve",
        kind="comet",
        kwargs={
            "wrong_branch_flow_penalty_gain": 0.0,
            "flow_commitment_reaction_time": 0.0,
            "max_flow_commitment_extension": 0.0,
            "flow_branch_adaptation_min_scale": 1e9,
            "terminal_feasibility_min_flow_scale": 1e9,
            "rescue_min_flow_scale": 1e9,
        },
    ),
    "comet_no_flow_compensation": ControllerVariant(
        label="COMET-no-flow-compensation",
        kind="comet",
        kwargs={
            "flow_compensation_gain": 0.0,
        },
    ),
    "pid": ControllerVariant(
        label="PID-route-aware",
        kind="pid",
        kwargs={},
    ),
}

ACBC_ABLATION_VARIANTS = {
    "acbc_no_branch_commitment": ControllerVariant(
        label="ACBC-no-branch-commitment",
        kind="acbc",
        kwargs={**ACBC_TUNED_WEIGHTS, "enable_branch_commitment": False},
    ),
    "acbc_no_authority_assessment": ControllerVariant(
        label="ACBC-no-authority-assessment",
        kind="acbc",
        kwargs={**ACBC_TUNED_WEIGHTS, "enable_authority_assessment": False},
    ),
    "acbc_no_recovery": ControllerVariant(
        label="ACBC-no-recovery",
        kind="acbc",
        kwargs={**ACBC_TUNED_WEIGHTS, "enable_recovery": False},
    ),
    "acbc_no_stall_escape": ControllerVariant(
        label="ACBC-no-stall-escape",
        kind="acbc",
        kwargs={**ACBC_TUNED_WEIGHTS, "enable_stall_escape": False},
    ),
    "acbc_no_route_term": ControllerVariant(
        label="ACBC-no-route-term",
        kind="acbc",
        kwargs={**ACBC_TUNED_WEIGHTS, "w_route": 0.0},
    ),
}


PAIRED_HEADER = [
    "run_id",
    "flow_scale",
    "pair_id",
    "trial_kind",
    "start_x_m",
    "start_y_m",
    "start_z_m",
    "target_x_m",
    "target_y_m",
    "target_z_m",
    "controller_key",
    "controller_label",
    "controller_kind",
    "reached_target",
    "steps_taken",
    "sim_time_s",
    "final_error_m",
    "final_error_mm",
    "early_stopped",
    "early_stop_reason",
    "early_stop_window_steps",
    "early_stop_improvement_threshold_m",
    "early_stop_improvement_threshold_mm",
    "early_stop_measured_improvement_m",
    "early_stop_measured_improvement_mm",
    "path_length_m",
    "planned_route_length_m",
    "route_length_ratio",
    "path_efficiency",
    "euclidean_distance_m",
    "euclidean_path_efficiency",
    "min_wall_clearance_m",
    "min_wall_clearance_mm",
    "min_surface_clearance_m",
    "min_surface_clearance_mm",
    "clearance_target_m",
    "clearance_target_mm",
    "clearance_target_met",
    "current_rms_A",
    "max_current_A",
    "energy_proxy_A2s",
    "wall_time_s",
    "total_control_time_s",
    "avg_control_step_time_s",
    "p95_control_step_time_s",
    "max_control_step_time_s",
    "mean_reynolds",
    "max_reynolds",
    "stokes_regime",
    "finite_re_drag_valid",
    "drag_model",
    "drag_regime_label",
    "peak_force_N",
    "peak_velocity_m_s",
    "peak_field_T",
    "acbc_transition_count",
    "acbc_free_fraction",
    "acbc_assess_fraction",
    "acbc_commit_fraction",
    "acbc_recovery_fraction",
    "acbc_escape_fraction",
    "max_power_W",
    "max_voltage_V",
    "max_coil_temp_C",
]


def _git_metadata():
    def run_git(args):
        try:
            return subprocess.check_output(
                ["git", *args],
                text=True,
                stderr=subprocess.DEVNULL,
            ).strip()
        except Exception:
            return None

    return {
        "commit": run_git(["rev-parse", "HEAD"]),
        "branch": run_git(["rev-parse", "--abbrev-ref", "HEAD"]),
        "status_short": run_git(["status", "--short"]),
    }


def _software_metadata():
    return {
        "python": sys.version,
        "platform": platform.platform(),
        "machine": platform.machine(),
        "processor": platform.processor(),
        "logical_cpu_count": os.cpu_count(),
        "numpy": np.__version__,
    }


def _trial_pairs(
    network,
    rng,
    n_random,
    vessel_file="",
    profile=SPION_CARRIER_500UM,
    wall_clearance_threshold_m=0.0007,
    magnetic_authority_m_s=0.180,
):
    """
    Generate paired start/target trials.

    Network-specific canonical_001 routes are used for the publication
    easy/medium/hard benchmark so we do not accidentally choose bad generic
    outlet_end() targets on the 40Left85Right or 0050 cerebral networks.
    """
    vessel_file_str = str(vessel_file)

    def mm_to_m(values):
        return np.asarray(values, dtype=float) / 1000.0

    inlet = network.inlet_start()
    outlet = network.outlet_end()

    pairs = []

    if "40Left85Right" in vessel_file_str:
        canonical_start = mm_to_m([-104.88859564, 39.14727271, -376.76280737])
        canonical_target = mm_to_m([-76.45376772, 123.07464331, -166.75929725])
        canonical_id = "canonical_001_medium_40Left85Right_endpoint03"
    elif "0050_H_CERE_H" in vessel_file_str:
        canonical_start = mm_to_m([-30.80946207, 2.76482804, -47.19573259])
        canonical_target = mm_to_m([-21.23204619, -38.94729167, 48.83672297])
        canonical_id = "canonical_001_hard_0050_endpoint08"
    else:
        canonical_start = inlet
        canonical_target = outlet
        canonical_id = "canonical_001_inlet_to_outlet"

    pairs.append(
        {
            "pair_id": canonical_id,
            "trial_kind": "canonical",
            "start": canonical_start,
            "target": canonical_target,
        }
    )

    # Keep three canonical routes total, but avoid bad outlet_end() on the
    # medium/hard networks by anchoring canonical_002/003 around the good route.
    pairs.append(
        {
            "pair_id": "canonical_002_random_to_publication_target",
            "trial_kind": "canonical",
            "start": network.random_point_in_vessel(rng),
            "target": canonical_target,
        }
    )
    pairs.append(
        {
            "pair_id": "canonical_003_publication_start_to_random",
            "trial_kind": "canonical",
            "start": canonical_start,
            "target": network.random_point_in_vessel(rng),
        }
    )

    for i in range(int(n_random)):
        pairs.append(
            {
                "pair_id": f"random_{i + 1:04d}",
                "trial_kind": "random",
                "start": network.random_point_in_vessel(rng),
                "target": network.random_point_in_vessel(rng),
            }
        )

    particle_radius = 0.5 * profile.diameter
    planning_options = RoutePlanningOptions(
        wall_clearance_m=wall_clearance_threshold_m,
        particle_radius_m=particle_radius,
        minimum_surface_clearance_m=max(
            wall_clearance_threshold_m - particle_radius, 0.0,
        ),
        magnetic_authority_m_s=magnetic_authority_m_s,
        minimum_authority_reserve_m_s=0.0,
    )

    feasible_pairs = []
    for pair in pairs:
        plan = network.plan_route(
            pair["start"], pair["target"], planning_options,
        )
        if not plan.feasible:
            print(
                f"Skipping infeasible pair {pair['pair_id']}: {plan.reason}",
                flush=True,
            )
            continue
        pair["route_length_m"] = plan.route_length_m
        pair["route_min_clearance_m"] = plan.min_center_clearance_m
        pair["route_min_authority_reserve_m_s"] = plan.min_authority_reserve_m_s
        feasible_pairs.append(pair)

    random_count = sum(
        pair["trial_kind"] == "random" for pair in feasible_pairs
    )
    attempts = 0
    while random_count < int(n_random) and attempts < max(100, 50 * int(n_random)):
        attempts += 1
        start = network.random_point_in_vessel(
            rng, min_radius=planning_options.required_center_clearance_m,
        )
        target = network.random_point_in_vessel(
            rng, min_radius=planning_options.required_center_clearance_m,
        )
        plan = network.plan_route(start, target, planning_options)
        if not plan.feasible:
            continue
        random_count += 1
        feasible_pairs.append({
            "pair_id": f"random_{random_count:04d}",
            "trial_kind": "random",
            "start": start,
            "target": target,
            "route_length_m": plan.route_length_m,
            "route_min_clearance_m": plan.min_center_clearance_m,
            "route_min_authority_reserve_m_s": plan.min_authority_reserve_m_s,
        })

    if random_count < int(n_random):
        raise RuntimeError(
            f"Could only generate {random_count}/{n_random} feasible random pairs"
        )

    for index, pair in enumerate(
        (p for p in feasible_pairs if p["trial_kind"] == "random"),
        start=1,
    ):
        pair["pair_id"] = f"random_{index:04d}"

    print("Generated route-feasible trial pairs:", flush=True)
    for pair in feasible_pairs:
        s_mm = np.asarray(pair["start"], dtype=float) * 1000.0
        t_mm = np.asarray(pair["target"], dtype=float) * 1000.0
        dist_mm = float(np.linalg.norm(t_mm - s_mm))
        print(
            f"  {pair['pair_id']} kind={pair['trial_kind']} "
            f"start_mm={s_mm.tolist()} target_mm={t_mm.tolist()} "
            f"straight_mm={dist_mm:.3f}",
            flush=True,
        )

    return feasible_pairs


def _requested_values(value, label):
    values = [item.strip() for item in str(value or "").split(",") if item.strip()]
    if not values:
        raise ValueError(f"{label} must contain at least one value")
    return values


def _format_flow_pair_combos(combos, limit=30):
    formatted = [
        f"(flow_scale={flow:g}, pair_id={pair_id})"
        for flow, pair_id in sorted(combos)
    ]
    if len(formatted) > limit:
        return formatted[:limit] + [f"... {len(formatted) - limit} more"]
    return formatted


def _trial_pairs_from_csv(
    csv_path,
    requested_pair_ids,
    requested_flow_scales=None,
    default_trial_kind="rerun",
    network_factory=None,
    profile=SPION_CARRIER_500UM,
    wall_clearance_threshold_m=0.0007,
    magnetic_authority_m_s=0.180,
):
    """
    Load exact saved start/target pairs from a previous paired_results.csv.

    One paired_results.csv row exists per controller. This helper deduplicates
    to one route pair for each unique (flow_scale, pair_id), then validates the
    route is still feasible in the current network before returning it.
    """
    requested_pair_ids = set(requested_pair_ids)
    requested_flow_scale_set = (
        {float(flow) for flow in requested_flow_scales}
        if requested_flow_scales is not None
        else None
    )

    coordinate_columns = {
        "start_x_m",
        "start_y_m",
        "start_z_m",
        "target_x_m",
        "target_y_m",
        "target_z_m",
    }
    required_non_coordinate_columns = {"flow_scale", "pair_id"}

    with open(csv_path, newline="") as f:
        reader = csv.DictReader(f)
        available_columns = list(reader.fieldnames or [])
        missing_non_coordinate_columns = sorted(
            required_non_coordinate_columns - set(available_columns)
        )
        if missing_non_coordinate_columns:
            raise ValueError(
                "Cannot use --rerun-from-csv because the CSV is missing "
                f"required columns: {missing_non_coordinate_columns}. "
                f"Available columns: {available_columns}."
            )
        rows = list(reader)

    found_pair_ids = {row["pair_id"] for row in rows if row["pair_id"] in requested_pair_ids}
    missing_pair_ids = sorted(requested_pair_ids - found_pair_ids)
    if missing_pair_ids:
        raise ValueError(
            f"Requested pair_id values not found in {csv_path}: {missing_pair_ids}"
        )

    filtered_rows = [row for row in rows if row["pair_id"] in requested_pair_ids]
    if requested_flow_scale_set is not None:
        filtered_rows = [
            row
            for row in filtered_rows
            if float(row["flow_scale"]) in requested_flow_scale_set
        ]
        found_flow_scales = {float(row["flow_scale"]) for row in filtered_rows}
        missing_flow_scales = sorted(requested_flow_scale_set - found_flow_scales)
        if missing_flow_scales:
            raise ValueError(
                "Requested flow scale values not found for the requested "
                f"pair IDs in {csv_path}: {missing_flow_scales}"
            )

    if not filtered_rows:
        raise ValueError(
            f"No rerun rows matched pair IDs {sorted(requested_pair_ids)}"
        )

    coordinate_source = (
        "paired_results_csv"
        if coordinate_columns.issubset(set(available_columns))
        else "benchmark_summary_seed_records"
    )

    def vector_from_values(values, pair_id, flow_scale, prefix):
        try:
            return np.asarray(
                [
                    float(values[0]),
                    float(values[1]),
                    float(values[2]),
                ],
                dtype=float,
            )
        except Exception as exc:
            raise ValueError(
                f"Invalid {prefix} coordinate values for pair_id="
                f"{pair_id} flow_scale={flow_scale}"
            ) from exc

    def vector_from_row(row, prefix):
        return vector_from_values(
            [
                row[f"{prefix}_x_m"],
                row[f"{prefix}_y_m"],
                row[f"{prefix}_z_m"],
            ],
            row.get("pair_id"),
            row.get("flow_scale"),
            prefix,
        )

    selected_rows_by_key = {}
    for row in filtered_rows:
        key = (float(row["flow_scale"]), row["pair_id"])
        selected_rows_by_key.setdefault(key, row)

    seed_records_by_key = {}
    if coordinate_source == "benchmark_summary_seed_records":
        summary_path = os.path.join(
            os.path.dirname(os.path.abspath(csv_path)),
            "benchmark_summary.json",
        )
        if not os.path.exists(summary_path):
            missing_columns = sorted(coordinate_columns - set(available_columns))
            raise ValueError(
                "Cannot use --rerun-from-csv with this paired_results.csv "
                f"because it lacks coordinate columns {missing_columns}, and "
                f"no benchmark_summary.json fallback was found at {summary_path}. "
                "Targeted rerun requires either coordinate columns in "
                "paired_results.csv or benchmark_summary.json seed_records."
            )

        with open(summary_path) as f:
            summary = json.load(f)
        seed_records = summary.get("seed_records")
        if not isinstance(seed_records, list) or not seed_records:
            raise ValueError(
                f"benchmark_summary.json fallback at {summary_path} does not "
                "contain usable seed_records. Targeted rerun requires either "
                "coordinate columns in paired_results.csv or benchmark_summary.json "
                "seed_records with flow_scale, pair_id, start, and target."
            )

        for record in seed_records:
            try:
                flow_scale = float(record["flow_scale"])
                pair_id = record["pair_id"]
            except Exception:
                continue
            seed_records_by_key[(flow_scale, pair_id)] = record

        available_seed_combos = set(seed_records_by_key)
        missing_seed_combos = sorted(
            set(selected_rows_by_key) - available_seed_combos
        )
        if missing_seed_combos:
            raise ValueError(
                "Requested pair/flow combinations were present in "
                "paired_results.csv but could not be found in "
                f"benchmark_summary.json seed_records: "
                f"{_format_flow_pair_combos(missing_seed_combos)}. "
                "Available seed_records combinations: "
                f"{_format_flow_pair_combos(available_seed_combos)}"
            )

    unique_pairs = {}
    for row in filtered_rows:
        flow_scale = float(row["flow_scale"])
        pair_id = row["pair_id"]
        key = (flow_scale, pair_id)
        trial_kind = row.get("trial_kind") or default_trial_kind
        route_metadata = {}

        if coordinate_source == "paired_results_csv":
            start = vector_from_row(row, "start")
            target = vector_from_row(row, "target")
        else:
            record = seed_records_by_key[key]
            start = vector_from_values(
                record.get("start"),
                pair_id,
                flow_scale,
                "start",
            )
            target = vector_from_values(
                record.get("target"),
                pair_id,
                flow_scale,
                "target",
            )
            for field in (
                "route_length_m",
                "route_min_clearance_m",
                "route_min_authority_reserve_m_s",
            ):
                if field in record and record[field] is not None:
                    route_metadata[field] = record[field]

        if key in unique_pairs:
            existing = unique_pairs[key]
            if (
                not np.allclose(existing["start"], start, rtol=0.0, atol=1e-12)
                or not np.allclose(existing["target"], target, rtol=0.0, atol=1e-12)
            ):
                raise ValueError(
                    f"Conflicting coordinates for pair_id={pair_id} "
                    f"flow_scale={flow_scale:g} in {csv_path}"
                )
            if existing["trial_kind"] != trial_kind and not existing["trial_kind"]:
                existing["trial_kind"] = trial_kind
            continue

        unique_pairs[key] = {
            "flow_scale": flow_scale,
            "pair_id": pair_id,
            "trial_kind": trial_kind,
            "start": start,
            "target": target,
            **route_metadata,
        }

    particle_radius = 0.5 * profile.diameter
    planning_options = RoutePlanningOptions(
        wall_clearance_m=wall_clearance_threshold_m,
        particle_radius_m=particle_radius,
        minimum_surface_clearance_m=max(
            wall_clearance_threshold_m - particle_radius, 0.0,
        ),
        magnetic_authority_m_s=magnetic_authority_m_s,
        minimum_authority_reserve_m_s=0.0,
    )

    pairs_by_flow = defaultdict(list)
    for (flow_scale, pair_id), pair in sorted(unique_pairs.items()):
        if network_factory is None:
            raise ValueError("network_factory is required for rerun route validation")
        network = network_factory(flow_scale)
        plan = network.plan_route(pair["start"], pair["target"], planning_options)
        if not plan.feasible:
            raise RuntimeError(
                f"Rerun pair is infeasible and cannot be reproduced safely: "
                f"flow_scale={flow_scale:g} pair_id={pair_id}: {plan.reason}"
            )
        pair["route_length_m"] = pair.get("route_length_m", plan.route_length_m)
        pair["route_min_clearance_m"] = pair.get(
            "route_min_clearance_m",
            plan.min_center_clearance_m,
        )
        pair["route_min_authority_reserve_m_s"] = pair.get(
            "route_min_authority_reserve_m_s",
            plan.min_authority_reserve_m_s,
        )
        pairs_by_flow[flow_scale].append(pair)

    print(
        "Loaded route-feasible rerun pairs from "
        f"{coordinate_source}:",
        flush=True,
    )
    for flow_scale in sorted(pairs_by_flow):
        for pair in pairs_by_flow[flow_scale]:
            s_mm = np.asarray(pair["start"], dtype=float) * 1000.0
            t_mm = np.asarray(pair["target"], dtype=float) * 1000.0
            dist_mm = float(np.linalg.norm(t_mm - s_mm))
            print(
                f"  flow={flow_scale:g} pair={pair['pair_id']} "
                f"kind={pair['trial_kind']} start_mm={s_mm.tolist()} "
                f"target_mm={t_mm.tolist()} straight_mm={dist_mm:.3f}",
                flush=True,
            )

    return dict(pairs_by_flow), coordinate_source

def _path_efficiency_metrics(network, start, target, trajectory):
    trajectory = np.asarray(trajectory, dtype=float)
    segments = np.diff(trajectory, axis=0)
    path_length = float(np.sum(np.linalg.norm(segments, axis=1)))

    start = np.asarray(start, dtype=float)
    target = np.asarray(target, dtype=float)
    euclidean_distance = float(np.linalg.norm(target - start))
    euclidean_efficiency = (
        euclidean_distance / path_length if path_length > 0 else 0.0
    )

    planned_waypoints = network.plan_waypoints(start, target)
    planned_path = np.asarray(
        [start, *[np.asarray(point, dtype=float) for point in planned_waypoints]],
        dtype=float,
    )
    planned_segments = np.diff(planned_path, axis=0)
    planned_route_length = float(
        np.sum(np.linalg.norm(planned_segments, axis=1))
    )
    route_length_ratio = (
        planned_route_length / path_length if path_length > 0 else 0.0
    )

    # A trajectory can geometrically cut a curved centerline corner while still
    # remaining in the vessel. Treat that as fully efficient rather than >100%.
    path_efficiency = float(np.clip(route_length_ratio, 0.0, 1.0))

    return {
        "path_length_m": path_length,
        "planned_route_length_m": planned_route_length,
        "route_length_ratio": route_length_ratio,
        "path_efficiency": path_efficiency,
        "euclidean_distance_m": euclidean_distance,
        "euclidean_path_efficiency": euclidean_efficiency,
    }


def _state_fractions(result):
    counts = defaultdict(int)
    log = getattr(result, "acbc_log", [])
    for row in log:
        counts[row.get("state", "")] += 1

    total = max(1, len(log))
    return {
        "acbc_transition_count": sum(
            1 for row in log if bool(row.get("transition", False))
        ),
        "acbc_free_fraction": counts["FREE_NAVIGATION"] / total,
        "acbc_assess_fraction": counts["VIABILITY_ASSESSMENT"] / total,
        "acbc_commit_fraction": counts["BRANCH_COMMITMENT"] / total,
        "acbc_recovery_fraction": counts["RECOVERY_MODE"] / total,
        "acbc_escape_fraction": counts["STALL_ESCAPE"] / total,
    }




def _safe_float(value, default=float("nan")):
    """
    Convert a value to float safely for benchmark metrics.

    Handles None, arrays/lists, NaN, inf, and nonnumeric values.
    """
    try:
        if value is None:
            return default

        if isinstance(value, (list, tuple)):
            if len(value) == 0:
                return default
            value = value[0]

        # Handles numpy arrays without requiring this helper to import numpy locally.
        if hasattr(value, "size") and hasattr(value, "ravel"):
            if value.size == 0:
                return default
            value = value.ravel()[0]

        out = float(value)

        # np is already imported in this benchmark file.
        if not np.isfinite(out):
            return default

        return out
    except Exception:
        return default

def _wall_clearance_stats(network, trajectory):
    """
    Return minimum vessel wall clearance over a trajectory.

    Clearance = nearest centerline distance - local vessel radius.
    Positive means inside the vessel with remaining wall clearance.
    Negative means outside/through wall by that amount.

    This helper is intentionally self-contained so publication metrics do
    not fail if the network class does not expose a dedicated clearance API.
    """
    traj = np.asarray(trajectory, dtype=float)

    if traj.ndim != 2 or traj.shape[0] == 0:
        return float("nan")

    if not hasattr(network, "find_segment"):
        return float("nan")

    min_clearance = float("inf")
    for p in traj:
        seg, radial = network.find_segment(p)
        if seg is None:
            return float("-inf")
        min_clearance = min(min_clearance, float(seg.radius - radial))

    return min_clearance

def _result_metrics(run_id, flow_scale, pair, controller_key, variant, result, network):
    traj = np.asarray(result.trajectory, dtype=float)
    pair_start = np.asarray(pair["start"], dtype=float)
    pair_target = np.asarray(pair["target"], dtype=float)
    path_metrics = _path_efficiency_metrics(
        network=network,
        start=result.start,
        target=result.target,
        trajectory=traj,
    )

    current_log = np.asarray(result.current_log, dtype=float)
    if current_log.size:
        current_rms = float(np.sqrt(np.mean(current_log ** 2)))
        energy_proxy = float(np.sum(current_log ** 2) * result.dt)
    else:
        current_rms = 0.0
        energy_proxy = 0.0

    min_clearance = _wall_clearance_stats(network, traj)
    state = _state_fractions(result)
    params = getattr(result, "params", {})
    particle_diameter_m = _safe_float(params.get("particle_diameter_m"), 0.0)
    particle_radius_m = 0.5 * particle_diameter_m
    clearance_target_m = _safe_float(
        params.get("wall_clearance_threshold_m"),
        params.get("acbc_wall_clearance_threshold_m", 0.0007),
    )
    min_surface_clearance_m = min_clearance - particle_radius_m
    clearance_target_met = bool(min_clearance >= clearance_target_m)
    step_times = np.asarray(result.step_time_log, dtype=float)
    total_control_time_s = float(np.sum(step_times)) if step_times.size else 0.0
    p95_control_step_time_s = (
        float(np.percentile(step_times, 95)) if step_times.size else 0.0
    )

    return {
        "run_id": run_id,
        "flow_scale": flow_scale,
        "pair_id": pair["pair_id"],
        "trial_kind": pair["trial_kind"],
        "start_x_m": float(pair_start[0]),
        "start_y_m": float(pair_start[1]),
        "start_z_m": float(pair_start[2]),
        "target_x_m": float(pair_target[0]),
        "target_y_m": float(pair_target[1]),
        "target_z_m": float(pair_target[2]),
        "controller_key": controller_key,
        "controller_label": variant.label,
        "controller_kind": variant.kind,
        "reached_target": bool(result.reached),
        "steps_taken": int(result.steps_taken),
        "sim_time_s": float(result.steps_taken * result.dt),
        "final_error_m": float(result.final_error),
        "final_error_mm": float(result.final_error * 1000.0),
        "early_stopped": bool(getattr(result, "early_stopped", False)),
        "early_stop_reason": getattr(result, "early_stop_reason", ""),
        "early_stop_window_steps": int(
            getattr(result, "early_stop_window_steps", 0) or 0
        ),
        "early_stop_improvement_threshold_m": float(
            getattr(result, "early_stop_improvement_threshold_m", np.nan)
        ),
        "early_stop_improvement_threshold_mm": float(
            getattr(result, "early_stop_improvement_threshold_m", np.nan) * 1000.0
        ),
        "early_stop_measured_improvement_m": float(
            getattr(result, "early_stop_measured_improvement_m", np.nan)
        ),
        "early_stop_measured_improvement_mm": float(
            getattr(result, "early_stop_measured_improvement_m", np.nan) * 1000.0
        ),
        **path_metrics,
        "min_wall_clearance_m": min_clearance,
        "min_wall_clearance_mm": min_clearance * 1000.0,
        "min_surface_clearance_m": min_surface_clearance_m,
        "min_surface_clearance_mm": min_surface_clearance_m * 1000.0,
        "clearance_target_m": clearance_target_m,
        "clearance_target_mm": clearance_target_m * 1000.0,
        "clearance_target_met": clearance_target_met,
        "current_rms_A": current_rms,
        "max_current_A": float(result.max_current_used),
        "energy_proxy_A2s": energy_proxy,
        "wall_time_s": float(result.wall_time_s),
        "total_control_time_s": total_control_time_s,
        "avg_control_step_time_s": float(result.avg_control_step_time_s),
        "p95_control_step_time_s": p95_control_step_time_s,
        "max_control_step_time_s": float(result.max_control_step_time_s),
        "mean_reynolds": float(result.mean_reynolds),
        "max_reynolds": float(result.max_reynolds),
        "stokes_regime": bool(result.stokes_regime),
        "finite_re_drag_valid": bool(result.finite_re_drag_valid),
        "drag_model": result.drag_model,
        "drag_regime_label": result.drag_regime_label,
        "peak_force_N": float(np.max(result.force_log)) if len(result.force_log) else 0.0,
        "peak_velocity_m_s": (
            float(np.max(result.velocity_log)) if len(result.velocity_log) else 0.0
        ),
        "peak_field_T": float(np.max(result.field_log)) if len(result.field_log) else 0.0,
        **state,
        "max_power_W": _safe_float(params.get("max_power_W")),
        "max_voltage_V": _safe_float(params.get("max_voltage_V")),
        "max_coil_temp_C": _safe_float(params.get("max_coil_temp_C")),
    }


def _save_trial_npz(result, metrics, output_dir):
    os.makedirs(output_dir, exist_ok=True)
    np.savez(
        os.path.join(output_dir, f"{result.trial_name}_data.npz"),
        start=result.start,
        target=result.target,
        final_position=result.final_position,
        trajectory=result.trajectory,
        distance_log=result.distance_log,
        current_log=result.current_log,
        field_log=result.field_log,
        force_log=result.force_log,
        velocity_log=result.velocity_log,
        reynolds_log=result.reynolds_log,
        cost_log=result.cost_log,
        step_time_log=result.step_time_log,
        reached=result.reached,
        steps_taken=result.steps_taken,
        final_error=result.final_error,
        early_stopped=getattr(result, "early_stopped", False),
        early_stop_reason=getattr(result, "early_stop_reason", ""),
        early_stop_window_steps=getattr(result, "early_stop_window_steps", 0),
        early_stop_improvement_threshold_m=getattr(
            result,
            "early_stop_improvement_threshold_m",
            np.nan,
        ),
        early_stop_measured_improvement_m=getattr(
            result,
            "early_stop_measured_improvement_m",
            np.nan,
        ),
        dt=result.dt,
        metrics=json.dumps(metrics),
    )


def _write_paired_csv(rows, path):
    with open(path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=PAIRED_HEADER)
        writer.writeheader()
        for row in rows:
            writer.writerow({key: row.get(key, "") for key in PAIRED_HEADER})


def _aggregate(rows):
    grouped = defaultdict(list)
    for row in rows:
        grouped[(row["flow_scale"], row["controller_key"])].append(row)

    out = []
    for (flow_scale, controller), group in sorted(grouped.items()):
        success = np.array([bool(r["reached_target"]) for r in group], dtype=bool)
        final_error = np.array([r["final_error_mm"] for r in group], dtype=float)
        sim_time = np.array([r["sim_time_s"] for r in group], dtype=float)
        finite_re = np.array([bool(r["finite_re_drag_valid"]) for r in group], dtype=bool)
        steps = np.array([r["steps_taken"] for r in group], dtype=float)
        early_stopped = np.array(
            [bool(r.get("early_stopped", False)) for r in group],
            dtype=bool,
        )
        early_stop_improvements = np.array(
            [
                r["early_stop_measured_improvement_mm"]
                for r in group
                if bool(r.get("early_stopped", False))
            ],
            dtype=float,
        )
        wall_time = np.array([r["wall_time_s"] for r in group], dtype=float)
        control_step = np.array(
            [r["avg_control_step_time_s"] for r in group], dtype=float
        )
        path_efficiency = np.array(
            [r["path_efficiency"] for r in group], dtype=float
        )
        current_rms = np.array([r["current_rms_A"] for r in group], dtype=float)
        energy = np.array([r["energy_proxy_A2s"] for r in group], dtype=float)
        clearance = np.array(
            [r["min_wall_clearance_mm"] for r in group], dtype=float
        )
        out.append({
            "flow_scale": flow_scale,
            "controller_key": controller,
            "n": len(group),
            "success_rate": float(np.mean(success)) if len(success) else 0.0,
            "median_final_error_mm": float(np.median(final_error)) if len(group) else np.nan,
            "median_sim_time_s": float(np.median(sim_time)) if len(group) else np.nan,
            "median_steps_to_target": float(np.median(steps)) if len(group) else np.nan,
            "early_stop_count": int(np.sum(early_stopped)) if len(group) else 0,
            "early_stop_rate": float(np.mean(early_stopped)) if len(group) else 0.0,
            "early_stop_window_steps": int(
                group[0].get("early_stop_window_steps", 0)
            ) if len(group) else 0,
            "early_stop_improvement_threshold_mm": float(
                group[0].get("early_stop_improvement_threshold_mm", np.nan)
            ) if len(group) else np.nan,
            "median_early_stop_measured_improvement_mm": (
                float(np.median(early_stop_improvements))
                if early_stop_improvements.size else np.nan
            ),
            "median_wall_time_s": float(np.median(wall_time)) if len(group) else np.nan,
            "median_avg_control_step_time_s": (
                float(np.median(control_step)) if len(group) else np.nan
            ),
            "median_path_efficiency": (
                float(np.median(path_efficiency)) if len(group) else np.nan
            ),
            "median_current_rms_A": (
                float(np.median(current_rms)) if len(group) else np.nan
            ),
            "median_energy_proxy_A2s": (
                float(np.median(energy)) if len(group) else np.nan
            ),
            "median_min_wall_clearance_mm": (
                float(np.median(clearance)) if len(group) else np.nan
            ),
            "clearance_target_met_rate": float(np.mean([
                bool(r["clearance_target_met"]) for r in group
            ])) if len(group) else 0.0,
            "finite_re_drag_valid_rate": (
                float(np.mean(finite_re)) if len(finite_re) else 0.0
            ),
            "mean_transitions": float(np.mean([
                r["acbc_transition_count"] for r in group
            ])) if len(group) else 0.0,
        })

    return out


PRIMARY_COMPARISONS = (
    ("naive_mpc", "acbc_same_weights"),
    ("mpc_default", "acbc_same_weights"),
    ("mpc_tuned", "acbc_tuned"),
)

PAIRED_METRICS = (
    "reached_target",
    "steps_taken",
    "sim_time_s",
    "wall_time_s",
    "avg_control_step_time_s",
    "path_efficiency",
    "current_rms_A",
    "energy_proxy_A2s",
    "min_wall_clearance_mm",
    "min_surface_clearance_mm",
)


def _paired_difference_rows(rows):
    by_flow_pair = defaultdict(dict)
    for row in rows:
        by_flow_pair[(row["flow_scale"], row["pair_id"])][
            row["controller_key"]
        ] = row

    differences = []
    for (flow_scale, pair_id), controllers in sorted(by_flow_pair.items()):
        for baseline, comparator in PRIMARY_COMPARISONS:
            if baseline not in controllers or comparator not in controllers:
                continue
            base = controllers[baseline]
            comp = controllers[comparator]
            for metric in PAIRED_METRICS:
                base_value = float(base[metric])
                comparator_value = float(comp[metric])
                differences.append({
                    "flow_scale": flow_scale,
                    "pair_id": pair_id,
                    "trial_kind": base["trial_kind"],
                    "baseline": baseline,
                    "comparator": comparator,
                    "metric": metric,
                    "baseline_value": base_value,
                    "comparator_value": comparator_value,
                    "paired_difference": comparator_value - base_value,
                })
    return differences


def _write_paired_differences(rows, path):
    fields = [
        "flow_scale", "pair_id", "trial_kind", "baseline", "comparator",
        "metric", "baseline_value", "comparator_value", "paired_difference",
    ]
    with open(path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()
        writer.writerows(_paired_difference_rows(rows))


def _paired_stats(rows):
    try:
        from scipy.stats import wilcoxon
    except Exception:
        wilcoxon = None

    by_flow_pair = defaultdict(dict)
    for row in rows:
        key = (row["flow_scale"], row["pair_id"])
        by_flow_pair[key][row["controller_key"]] = row

    stats = []
    for baseline, comparator in PRIMARY_COMPARISONS:
        for flow_scale in sorted({r["flow_scale"] for r in rows}):
            pairs = [
                controllers
                for (flow, _), controllers in by_flow_pair.items()
                if flow == flow_scale
                and baseline in controllers
                and comparator in controllers
            ]
            if not pairs:
                continue

            for metric in PAIRED_METRICS:
                base = np.array([p[baseline][metric] for p in pairs], dtype=float)
                comp = np.array([p[comparator][metric] for p in pairs], dtype=float)
                diff = comp - base
                p_value = None
                if wilcoxon is not None and len(diff) > 0 and np.any(diff != 0):
                    try:
                        p_value = float(wilcoxon(comp, base).pvalue)
                    except Exception:
                        p_value = None

                stats.append({
                    "flow_scale": flow_scale,
                    "baseline": baseline,
                    "comparator": comparator,
                    "metric": metric,
                    "n_pairs": len(pairs),
                    "median_paired_difference": float(np.median(diff)),
                    "mean_paired_difference": float(np.mean(diff)),
                    "wilcoxon_p_value": p_value,
                })

    return stats


def _write_benchmark_summary(root, run_id, seed_records, paired_rows):
    summary = {
        "run_id": run_id,
        "config_path": "run_config.json",
        "paired_results_path": "paired_results.csv",
        "paired_differences_path": "paired_differences.csv",
        "n_rows": len(paired_rows),
        "n_pairs": len(seed_records),
        "seed_records": seed_records,
        "aggregate": _aggregate(paired_rows),
        "paired_statistics": _paired_stats(paired_rows),
    }
    with open(os.path.join(root, "benchmark_summary.json"), "w") as f:
        json.dump(summary, f, indent=2)
    return summary


def _run_variant(variant, pair, network, trial_seed, common_kwargs):
    if variant.kind == "mpc":
        from simulation_3d_mpc import simulate_trial_mpc

        kwargs = {
            **common_kwargs,
            **variant.kwargs,
            "early_stop_enabled": True,
            "early_stop_window_steps": BENCHMARK_EARLY_STOP_WINDOW_STEPS,
            "early_stop_improvement_threshold_m": (
                BENCHMARK_EARLY_STOP_IMPROVEMENT_THRESHOLD_M
            ),
        }
        return simulate_trial_mpc(
            start=pair["start"],
            target=pair["target"],
            network=network,
            trial_name=pair["pair_id"],
            rng_seed=trial_seed,
            **kwargs,
        )

    if variant.kind == "naive_mpc":
        from simulation_3d_naive_mpc import simulate_trial_naive_mpc

        return simulate_trial_naive_mpc(
            start=pair["start"],
            target=pair["target"],
            network=network,
            trial_name=pair["pair_id"],
            rng_seed=trial_seed,
            **common_kwargs,
            **variant.kwargs,
        )

    if variant.kind == "acbc":
        from simulation_3d_acbc import simulate_trial_acbc

        kwargs = {
            **common_kwargs,
            **variant.kwargs,
            "early_stop_enabled": True,
            "early_stop_window_steps": BENCHMARK_EARLY_STOP_WINDOW_STEPS,
            "early_stop_improvement_threshold_m": (
                BENCHMARK_EARLY_STOP_IMPROVEMENT_THRESHOLD_M
            ),
        }
        return simulate_trial_acbc(
            start=pair["start"],
            target=pair["target"],
            network=network,
            trial_name=pair["pair_id"],
            rng_seed=trial_seed,
            **kwargs,
        )

    if variant.kind == "comet":
        kwargs = {
            **common_kwargs,
            **variant.kwargs,
            "early_stop_enabled": True,
            "early_stop_window_steps": BENCHMARK_EARLY_STOP_WINDOW_STEPS,
            "early_stop_improvement_threshold_m": (
                BENCHMARK_EARLY_STOP_IMPROVEMENT_THRESHOLD_M
            ),
        }
        return simulate_trial_comet(
            start=pair["start"],
            target=pair["target"],
            network=network,
            trial_name=pair["pair_id"],
            rng_seed=trial_seed,
            **kwargs,
        )

    if variant.kind == "pid":
        return simulate_trial_pid(
            start=pair["start"],
            target=pair["target"],
            network=network,
            trial_name=pair["pair_id"],
            rng_seed=trial_seed,
            **common_kwargs,
            **variant.kwargs,
        )

    raise ValueError(f"Unknown controller kind: {variant.kind}")


def _run_variant_task(task):
    variant = task["variant"]
    pair = task["pair"]
    network = RealVesselNetwork(task["vessel_file"], flow_scale=task["flow_scale"])
    result = _run_variant(
        variant=variant,
        pair=pair,
        network=network,
        trial_seed=task["trial_seed"],
        common_kwargs=task["common_kwargs"],
    )
    return {
        "flow_scale": task["flow_scale"],
        "pair": pair,
        "pair_idx": task["pair_idx"],
        "controller_key": task["controller_key"],
        "variant": variant,
        "result": result,
    }


def run_benchmark(
    output_dir,
    run_id,
    vessel_file,
    flow_scales,
    n_random_trials,
    seed,
    controller_keys,
    include_ablations,
    dt,
    steps,
    tolerance,
    device_mode,
    profile_key,
    wall_clearance_threshold_m=0.0007,
    workers=1,
    rerun_from_csv=None,
    rerun_pair_ids=None,
    rerun_flow_scales=None,
    rerun_trial_kind="rerun",
):
    if include_ablations:
        variants = {**CONTROLLER_VARIANTS, **ACBC_ABLATION_VARIANTS}
    else:
        variants = dict(CONTROLLER_VARIANTS)

    selected = {key: variants[key] for key in controller_keys}
    profile = PROFILE_CHOICES[profile_key]

    root = os.path.abspath(os.path.join(output_dir, run_id))
    os.makedirs(root, exist_ok=True)

    common_kwargs = {
        "dt": dt,
        "steps": steps,
        "tolerance": tolerance,
        "coil_radius": FIXED_COIL_RADIUS,
        "coil_length": FIXED_COIL_LENGTH,
        "core_radius": FIXED_CORE_RADIUS,
        "turns": FIXED_TURNS,
        "max_current": FIXED_MAX_CURRENT,
        "placement_radius": FIXED_PLACEMENT_RADIUS,
        "core_relative_permeability": FIXED_CORE_MU_R_EFFECTIVE,
        "core_saturation_tesla": FIXED_CORE_SATURATION_T,
        "device_mode": device_mode,
        "profile": profile,
        "wall_clearance_threshold_m": wall_clearance_threshold_m,
    }

    rerun_mode_enabled = rerun_from_csv is not None
    rerun_pairs_by_flow = None
    rerun_coordinate_source = None
    active_flow_scales = list(flow_scales)
    if rerun_mode_enabled:
        if not rerun_pair_ids:
            raise ValueError("--rerun-pair-ids is required with --rerun-from-csv")
        rerun_pairs_by_flow, rerun_coordinate_source = _trial_pairs_from_csv(
            csv_path=rerun_from_csv,
            requested_pair_ids=rerun_pair_ids,
            requested_flow_scales=rerun_flow_scales,
            default_trial_kind=rerun_trial_kind,
            network_factory=lambda flow: RealVesselNetwork(
                vessel_file, flow_scale=flow,
            ),
            profile=profile,
            wall_clearance_threshold_m=wall_clearance_threshold_m,
        )
        active_flow_scales = sorted(rerun_pairs_by_flow)

    config = {
        "run_id": run_id,
        "vessel_file": vessel_file,
        "flow_scales": active_flow_scales,
        "n_random_trials": n_random_trials,
        "seed": seed,
        "controller_keys": list(selected),
        "include_ablations": include_ablations,
        "profile_key": profile_key,
        "wall_clearance_threshold_m": wall_clearance_threshold_m,
        "workers": int(max(1, workers)),
        "rerun_mode_enabled": rerun_mode_enabled,
        "rerun_from_csv": rerun_from_csv,
        "rerun_pair_ids": list(rerun_pair_ids or []),
        "rerun_flow_scales": (
            list(rerun_flow_scales) if rerun_flow_scales is not None else None
        ),
        "rerun_coordinate_source": rerun_coordinate_source,
        "early_stopping": {
            "enabled_controller_kinds": ["mpc", "acbc", "comet"],
            "excluded_controller_kinds": [],
            "window_steps": BENCHMARK_EARLY_STOP_WINDOW_STEPS,
            "improvement_threshold_m": BENCHMARK_EARLY_STOP_IMPROVEMENT_THRESHOLD_M,
            "improvement_threshold_mm": (
                BENCHMARK_EARLY_STOP_IMPROVEMENT_THRESHOLD_M * 1000.0
            ),
        },
        "software": _software_metadata(),
        "profile": {
            "name": profile.name,
            "diameter_m": profile.diameter,
            "diameter_um": profile.diameter * 1e6,
            "magnetic_loading_fraction": profile.magnetic_loading_fraction,
            "hydrodynamic_drag_multiplier": profile.hydrodynamic_drag_multiplier,
        },
        "common_kwargs": {
            key: value for key, value in common_kwargs.items() if key != "profile"
        },
        "variants": {
            key: {
                "label": variant.label,
                "kind": variant.kind,
                "kwargs": variant.kwargs,
            }
            for key, variant in selected.items()
        },
        "git": _git_metadata(),
    }
    with open(os.path.join(root, "run_config.json"), "w") as f:
        json.dump(config, f, indent=2)

    paired_rows = []
    all_seed_records = []
    workers = int(max(1, workers))

    for flow_scale in active_flow_scales:
        network = RealVesselNetwork(vessel_file, flow_scale=flow_scale)
        if rerun_mode_enabled:
            pairs = rerun_pairs_by_flow[flow_scale]
        else:
            rng = np.random.default_rng(seed + int(round(flow_scale * 1000)))
            pairs = _trial_pairs(
                network,
                rng,
                n_random_trials,
                vessel_file=vessel_file,
                profile=profile,
                wall_clearance_threshold_m=wall_clearance_threshold_m,
            )
        results_by_controller = defaultdict(list)

        tasks = []
        for pair_idx, pair in enumerate(pairs):
            trial_seed = seed + 100000 * int(round(flow_scale * 10)) + pair_idx
            all_seed_records.append({
                "flow_scale": flow_scale,
                "pair_id": pair["pair_id"],
                "seed": trial_seed,
                "start": np.asarray(pair["start"], dtype=float).tolist(),
                "target": np.asarray(pair["target"], dtype=float).tolist(),
                "route_length_m": pair.get("route_length_m"),
                "route_min_clearance_m": pair.get("route_min_clearance_m"),
                "route_min_authority_reserve_m_s": pair.get(
                    "route_min_authority_reserve_m_s"
                ),
            })

            for controller_key, variant in selected.items():
                tasks.append({
                    "flow_scale": flow_scale,
                    "pair": pair,
                    "pair_idx": pair_idx,
                    "controller_key": controller_key,
                    "variant": variant,
                    "trial_seed": trial_seed,
                    "common_kwargs": common_kwargs,
                    "vessel_file": vessel_file,
                })

        def handle_completed(item):
            pair = item["pair"]
            controller_key = item["controller_key"]
            variant = item["variant"]
            result = item["result"]

            out_dir = os.path.join(root, f"flow_{flow_scale:g}x", controller_key)
            os.makedirs(out_dir, exist_ok=True)
            metrics = _result_metrics(
                run_id=run_id,
                flow_scale=flow_scale,
                pair=pair,
                controller_key=controller_key,
                variant=variant,
                result=result,
                network=network,
            )
            paired_rows.append(metrics)
            _save_trial_npz(result, metrics, out_dir)
            _write_paired_csv(paired_rows, os.path.join(root, "paired_results.csv"))
            _write_paired_differences(
                paired_rows,
                os.path.join(root, "paired_differences.csv"),
            )
            _write_benchmark_summary(root, run_id, all_seed_records, paired_rows)
            results_by_controller[controller_key].append(result)

            result._benchmark_controller_key = controller_key
            result._benchmark_flow_scale = flow_scale

        if workers == 1:
            for task in tasks:
                print(
                    f"[{run_id}] flow={flow_scale:g} "
                    f"pair={task['pair']['pair_id']} "
                    f"controller={task['controller_key']}",
                    flush=True,
                )
                result = _run_variant(
                    variant=task["variant"],
                    pair=task["pair"],
                    network=network,
                    trial_seed=task["trial_seed"],
                    common_kwargs=common_kwargs,
                )
                handle_completed({**task, "result": result})
        else:
            executor_cls = ProcessPoolExecutor
            try:
                executor = executor_cls(max_workers=workers)
            except PermissionError:
                print(
                    "ProcessPoolExecutor unavailable; falling back to threads.",
                    flush=True,
                )
                executor = ThreadPoolExecutor(max_workers=workers)

            with executor:
                futures = []
                for task in tasks:
                    print(
                        f"[{run_id}] submit flow={flow_scale:g} "
                        f"pair={task['pair']['pair_id']} "
                        f"controller={task['controller_key']}",
                        flush=True,
                    )
                    futures.append(executor.submit(_run_variant_task, task))

                for future in as_completed(futures):
                    item = future.result()
                    print(
                        f"[{run_id}] done flow={item['flow_scale']:g} "
                        f"pair={item['pair']['pair_id']} "
                        f"controller={item['controller_key']}",
                        flush=True,
                    )
                    handle_completed(item)

        # Write per-controller summaries for this flow after completing all pairs.
        for controller_key, variant in selected.items():
            out_dir = os.path.join(root, f"flow_{flow_scale:g}x", controller_key)
            save_results_csv(results_by_controller[controller_key], out_dir)
            if variant.kind == "acbc":
                from simulation_3d_acbc import save_acbc_diagnostics

                save_acbc_diagnostics(results_by_controller[controller_key], out_dir)
            if variant.kind == "comet":
                save_comet_diagnostics(results_by_controller[controller_key], out_dir)
            if variant.kind == "pid":
                save_pid_diagnostics(results_by_controller[controller_key], out_dir)

    paired_csv = os.path.join(root, "paired_results.csv")
    _write_paired_csv(paired_rows, paired_csv)
    _write_paired_differences(
        paired_rows,
        os.path.join(root, "paired_differences.csv"),
    )

    _write_benchmark_summary(root, run_id, all_seed_records, paired_rows)

    print(f"\nBenchmark outputs saved to: {root}")
    print(f"Paired CSV: {paired_csv}")

    return root


def _parse_flow_scales(value):
    return [float(v.strip()) for v in value.split(",") if v.strip()]


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--output-dir", default="outputs/publication_benchmark_1mm")
    parser.add_argument("--run-id", default=time.strftime("%Y%m%d_%H%M%S"))
    parser.add_argument("--vessel-file", default=DEFAULT_VESSEL_FILE)
    parser.add_argument("--flow-scales", default="1,3,5")
    parser.add_argument("--n-random-trials", type=int, default=100)
    parser.add_argument("--seed", type=int, default=20260609)
    parser.add_argument(
        "--rerun-from-csv",
        default=None,
        help=(
            "Existing paired_results.csv to load exact start/target coordinates "
            "from for targeted reruns."
        ),
    )
    parser.add_argument(
        "--rerun-pair-ids",
        default=None,
        help="Comma-separated pair_id values to rerun from --rerun-from-csv.",
    )
    parser.add_argument(
        "--rerun-flow-scales",
        default=None,
        help=(
            "Optional comma-separated flow scales to rerun from "
            "--rerun-from-csv."
        ),
    )
    parser.add_argument(
        "--rerun-trial-kind",
        default="rerun",
        help="Fallback trial_kind for rerun CSVs that do not include trial_kind.",
    )
    parser.add_argument(
        "--controllers",
        default="mpc_default,mpc_tuned,acbc_same_weights,acbc_tuned",
        help="Comma-separated controller keys. Use --list-controllers to inspect.",
    )
    parser.add_argument("--include-ablations", action="store_true")
    parser.add_argument("--list-controllers", action="store_true")
    parser.add_argument("--dt", type=float, default=0.02)
    parser.add_argument("--steps", type=int, default=4000)
    parser.add_argument("--tolerance", type=float, default=DEFAULT_TARGET_TOLERANCE)
    parser.add_argument(
        "--wall-clearance-threshold-mm",
        type=float,
        default=0.7,
        help="Common center-to-wall clearance criterion for every controller.",
    )
    parser.add_argument(
        "--workers",
        type=int,
        default=1,
        help="Number of parallel trial/controller workers to run.",
    )
    parser.add_argument("--device-mode", default=DEVICE_MODE)
    parser.add_argument(
        "--profile",
        choices=sorted(PROFILE_CHOICES),
        default="spion_carrier_1mm",
        help="Carrier physical profile to use for all paired trials.",
    )
    parser.add_argument(
        "--smoke-test",
        action="store_true",
        help="Run one random pair at flow=1 with the four primary controllers.",
    )
    args = parser.parse_args()

    variants = {**CONTROLLER_VARIANTS, **ACBC_ABLATION_VARIANTS}
    if args.list_controllers:
        for key, variant in variants.items():
            print(f"{key}: {variant.label} ({variant.kind})")
        return

    flow_scales = _parse_flow_scales(args.flow_scales)
    n_random_trials = args.n_random_trials
    controllers = [c.strip() for c in args.controllers.split(",") if c.strip()]
    rerun_pair_ids = None
    rerun_flow_scales = None

    if args.rerun_from_csv:
        if not args.rerun_pair_ids:
            raise ValueError("--rerun-pair-ids is required with --rerun-from-csv")
        rerun_pair_ids = _requested_values(args.rerun_pair_ids, "--rerun-pair-ids")
        if args.rerun_flow_scales:
            rerun_flow_scales = _parse_flow_scales(args.rerun_flow_scales)

    if args.smoke_test:
        flow_scales = [1.0]
        n_random_trials = 1

    if args.include_ablations:
        allowed = {**CONTROLLER_VARIANTS, **ACBC_ABLATION_VARIANTS}
    else:
        allowed = CONTROLLER_VARIANTS

    unknown = sorted(set(controllers) - set(allowed))
    if unknown:
        raise ValueError(f"Unknown controller keys: {unknown}")

    run_benchmark(
        output_dir=args.output_dir,
        run_id=args.run_id,
        vessel_file=args.vessel_file,
        flow_scales=flow_scales,
        n_random_trials=n_random_trials,
        seed=args.seed,
        controller_keys=controllers,
        include_ablations=args.include_ablations,
        dt=args.dt,
        steps=args.steps,
        tolerance=args.tolerance,
        device_mode=args.device_mode,
        profile_key=args.profile,
        wall_clearance_threshold_m=args.wall_clearance_threshold_mm / 1000.0,
        workers=args.workers,
        rerun_from_csv=args.rerun_from_csv,
        rerun_pair_ids=rerun_pair_ids,
        rerun_flow_scales=rerun_flow_scales,
        rerun_trial_kind=args.rerun_trial_kind,
    )


if __name__ == "__main__":
    main()
