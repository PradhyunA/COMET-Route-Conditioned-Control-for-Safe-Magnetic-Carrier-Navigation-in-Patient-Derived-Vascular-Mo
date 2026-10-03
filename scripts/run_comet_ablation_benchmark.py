"""
COMET mechanism/stress ablation benchmark.

This runner is intentionally narrower than the full publication benchmark:
  - it runs COMET and COMET ablations only;
  - it keeps geometry, start/target pairs, profile, hardware, and route planning paired;
  - it adds stress modes that expose mechanism-level differences;
  - it writes ablation-specific diagnostics needed for the manuscript tables.

Recommended examples
--------------------
Standard ablation:
    python -u scripts/run_comet_ablation_benchmark.py \
      --run-id comet_ablation_standard \
      --flow-scales 1,3,5 \
      --n-random-trials 4 \
      --profile spion_carrier_500um \
      --workers 8

High-flow stress ablation:
    python -u scripts/run_comet_ablation_benchmark.py \
      --run-id comet_ablation_high_flow \
      --stress-mode high_flow \
      --flow-scales 7,10 \
      --n-random-trials 2 \
      --profile spion_carrier_500um \
      --workers 8

Low-authority stress ablation:
    python -u scripts/run_comet_ablation_benchmark.py \
      --run-id comet_ablation_low_authority \
      --stress-mode low_authority \
      --flow-scales 5 \
      --n-random-trials 4 \
      --profile spion_carrier_500um \
      --low-authority-current 8 \
      --workers 8
"""

from __future__ import annotations

import argparse
import csv
import json
import os
import platform
import sys
import tempfile
import time
from collections import defaultdict
from concurrent.futures import ProcessPoolExecutor, ThreadPoolExecutor, as_completed
from dataclasses import dataclass
from typing import Any

import numpy as np

os.environ.setdefault(
    "MPLCONFIGDIR",
    os.path.join(tempfile.gettempdir(), "dynamag_matplotlib"),
)
os.environ.setdefault("MPLBACKEND", "Agg")

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from dynamag_simulation.core.constants import DEFAULT_TARGET_TOLERANCE
from dynamag_simulation.core.profiles import SPION_CARRIER_1MM, SPION_CARRIER_500UM
from dynamag_simulation.core.real_vessel_network import RealVesselNetwork, RoutePlanningOptions
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
)
from simulation_3d_comet import save_comet_diagnostics, simulate_trial_comet


PROFILE_CHOICES = {
    "spion_carrier_1mm": SPION_CARRIER_1MM,
    "spion_carrier_500um": SPION_CARRIER_500UM,
}


@dataclass(frozen=True)
class AblationVariant:
    key: str
    label: str
    kwargs: dict[str, Any]


COMET_ABLATION_VARIANTS: dict[str, AblationVariant] = {
    "comet": AblationVariant("comet", "COMET-full", {}),

    "comet_no_terminal_precision": AblationVariant(
        "comet_no_terminal_precision",
        "COMET-no-terminal-precision",
        {"ablate_terminal_precision": True},
    ),
    "comet_no_flow_scaled_reserve": AblationVariant(
        "comet_no_flow_scaled_reserve",
        "COMET-no-flow-scaled-reserve",
        {"ablate_flow_scaled_reserve": True},
    ),
    "comet_no_occupancy_measure": AblationVariant(
        "comet_no_occupancy_measure",
        "COMET-no-occupancy-measure",
        {"ablate_occupancy_measure": True},
    ),
    "comet_no_branch_shaping": AblationVariant(
        "comet_no_branch_shaping",
        "COMET-no-branch-shaping",
        {"ablate_branch_conditioned_motion_shaping": True},
    ),
    "comet_no_polyline_tracking": AblationVariant(
        "comet_no_polyline_tracking",
        "COMET-no-polyline-tracking",
        {"ablate_polyline_tracking": True},
    ),

    "comet_no_flow_compensation": AblationVariant(
        "comet_no_flow_compensation",
        "COMET-no-flow-compensation",
        {"ablate_flow_compensation": True},
    ),
    "comet_slsqp_allocator": AblationVariant(
        "comet_slsqp_allocator",
        "COMET-SLSQP-allocator",
        {"allocator": "slsqp"},
    ),

    "comet_core_only": AblationVariant(
        "comet_core_only",
        "COMET-core-only",
        {
            "ablate_authority_reserve": True,
            "ablate_commitment_memory": True,
            "ablate_downstream_value": True,
            "ablate_flow_scaled_reserve": True,
            "ablate_terminal_precision": True,
            "ablate_occupancy_measure": True,
        },
    ),
}

ROW_FIELDS = [
    "run_id",
    "stress_mode",
    "vessel_key",
    "vessel_label",
    "flow_scale",
    "pair_id",
    "trial_kind",
    "controller_key",
    "controller_label",
    "start_x_m",
    "start_y_m",
    "start_z_m",
    "target_x_m",
    "target_y_m",
    "target_z_m",
    "reached_target",
    "steps_taken",
    "sim_time_s",
    "final_error_m",
    "final_error_mm",
    "path_length_m",
    "planned_route_length_m",
    "path_efficiency",
    "min_wall_clearance_m",
    "min_wall_clearance_mm",
    "clearance_target_met",
    "current_rms_A",
    "max_current_A",
    "energy_proxy_A2s",
    "wall_time_s",
    "avg_control_step_time_s",
    "p95_control_step_time_s",
    "max_control_step_time_s",
    "mean_reynolds",
    "max_reynolds",
    "finite_re_drag_valid",
    "drag_regime_label",
    "peak_force_N",
    "peak_velocity_m_s",
    "peak_field_T",
    "max_power_W",
    "max_voltage_V",
    "max_coil_temp_C",
    "comet_branch_switch_count",
    "comet_low_authority_fraction",
    "comet_mean_intent_entropy",
    "comet_max_intent_entropy",
    "comet_safety_filter_fraction",
    "comet_terminal_precision_fraction",
    "comet_current_saturation_fraction",
    "comet_route_regression_count",
    "comet_terminal_overshoot_count",
    "comet_mean_capture_probability",
    "comet_min_selected_reserve_m",
    "comet_min_selected_reserve_mm",
]


SUMMARY_FIELDS = [
    "run_id",
    "stress_mode",
    "flow_scale",
    "controller_key",
    "controller_label",
    "n",
    "success_rate",
    "median_final_error_mm",
    "max_final_error_mm",
    "mean_steps_effective",
    "median_steps_taken",
    "median_path_efficiency",
    "median_current_rms_A",
    "median_energy_proxy_A2s",
    "median_min_wall_clearance_mm",
    "clearance_target_met_rate",
    "finite_re_drag_valid_rate",
    "median_branch_switch_count",
    "median_low_authority_fraction",
    "median_mean_intent_entropy",
    "median_safety_filter_fraction",
    "median_terminal_precision_fraction",
    "median_current_saturation_fraction",
    "median_route_regression_count",
    "median_terminal_overshoot_count",
    "median_min_selected_reserve_mm",
]


def _parse_csv_list(value: str) -> list[str]:
    return [item.strip() for item in str(value).split(",") if item.strip()]


def _parse_flow_scales(value: str) -> list[float]:
    return [float(item) for item in _parse_csv_list(value)]


def _safe_float(value: Any, default: float = float("nan")) -> float:
    try:
        if value is None:
            return default
        out = float(value)
        if not np.isfinite(out):
            return default
        return out
    except Exception:
        return default


def _vector_mm_to_m(values: list[float]) -> np.ndarray:
    return np.asarray(values, dtype=float) / 1000.0


MANUSCRIPT_VESSEL_FILES = {
    "cm_corrected": "data/real_vessel_segments_cm_corrected.pkl",
    "40Left85Right": (
        "data/vessels/aortic_root_cerebral_40Left85Right/"
        "real_vessel_segments.pkl"
    ),
    "pulmonary_0082_H_PULM_H": (
        "data/vessels/pulmonary_0082_H_PULM_H/"
        "real_vessel_segments_cm_corrected.pkl"
    ),
}


def _load_selected_cases(path: str, flow_scales: list[float]) -> list[dict[str, Any]]:
    """Load the 45 matched manuscript cases from full-COMET rows."""
    allowed_flows = {float(value) for value in flow_scales}
    cases = []
    seen = set()
    with open(path, newline="") as f:
        for row in csv.DictReader(f):
            if row.get("controller_key") != "comet":
                continue
            flow_scale = float(row["flow_scale"])
            if flow_scale not in allowed_flows:
                continue
            vessel_key = row["vessel_key"]
            if vessel_key not in MANUSCRIPT_VESSEL_FILES:
                raise ValueError(f"No vessel-file mapping for selected vessel {vessel_key!r}")
            case_key = (vessel_key, flow_scale, row["pair_id"])
            if case_key in seen:
                raise ValueError(f"Duplicate full-COMET selected case: {case_key}")
            seen.add(case_key)
            route_seed = _safe_float(row.get("route_seed"))
            if not np.isfinite(route_seed):
                route_seed = _safe_float(row.get("replacement_seed"))
            cases.append({
                "vessel_key": vessel_key,
                "vessel_label": row.get("vessel_label", vessel_key),
                "vessel_file": MANUSCRIPT_VESSEL_FILES[vessel_key],
                "flow_scale": flow_scale,
                "pair": {
                    "pair_id": row["pair_id"],
                    "trial_kind": row.get("trial_kind", "selected"),
                    "start": np.asarray([
                        float(row["start_x_m"]), float(row["start_y_m"]),
                        float(row["start_z_m"]),
                    ]),
                    "target": np.asarray([
                        float(row["target_x_m"]), float(row["target_y_m"]),
                        float(row["target_z_m"]),
                    ]),
                    "route_length_m": _safe_float(
                        row.get("route_length_m_from_seed_record")
                    ),
                    "route_min_clearance_m": _safe_float(
                        row.get("route_min_clearance_m_from_seed_record")
                    ),
                    "route_min_authority_reserve_m_s": _safe_float(
                        row.get("route_min_authority_reserve_m_s_from_seed_record")
                    ),
                },
                "trial_seed": int(route_seed) if np.isfinite(route_seed) else None,
            })

    cases.sort(key=lambda item: (
        item["vessel_key"], item["flow_scale"], item["pair"]["pair_id"]
    ))
    return cases


def _trial_pairs(
    network: RealVesselNetwork,
    rng: np.random.Generator,
    n_random: int,
    vessel_file: str,
    profile,
    wall_clearance_threshold_m: float,
    magnetic_authority_m_s: float = 0.180,
) -> list[dict[str, Any]]:
    """Generate route-feasible canonical + random start/target pairs."""
    vessel_file_str = str(vessel_file)

    inlet = network.inlet_start()
    outlet = network.outlet_end()

    if "40Left85Right" in vessel_file_str:
        canonical_start = _vector_mm_to_m([-104.88859564, 39.14727271, -376.76280737])
        canonical_target = _vector_mm_to_m([-76.45376772, 123.07464331, -166.75929725])
        canonical_id = "canonical_001_medium_40Left85Right_endpoint03"
    elif "0050_H_CERE_H" in vessel_file_str:
        canonical_start = _vector_mm_to_m([-30.80946207, 2.76482804, -47.19573259])
        canonical_target = _vector_mm_to_m([-21.23204619, -38.94729167, 48.83672297])
        canonical_id = "canonical_001_hard_0050_endpoint08"
    else:
        canonical_start = inlet
        canonical_target = outlet
        canonical_id = "canonical_001_inlet_to_outlet"

    requested = [
        {
            "pair_id": canonical_id,
            "trial_kind": "canonical",
            "start": canonical_start,
            "target": canonical_target,
        },
        {
            "pair_id": "canonical_002_random_to_publication_target",
            "trial_kind": "canonical",
            "start": network.random_point_in_vessel(rng),
            "target": canonical_target,
        },
        {
            "pair_id": "canonical_003_publication_start_to_random",
            "trial_kind": "canonical",
            "start": canonical_start,
            "target": network.random_point_in_vessel(rng),
        },
    ]

    particle_radius = 0.5 * profile.diameter
    planning_options = RoutePlanningOptions(
        wall_clearance_m=wall_clearance_threshold_m,
        particle_radius_m=particle_radius,
        minimum_surface_clearance_m=max(wall_clearance_threshold_m - particle_radius, 0.0),
        magnetic_authority_m_s=magnetic_authority_m_s,
        minimum_authority_reserve_m_s=0.0,
    )

    feasible = []
    for pair in requested:
        plan = network.plan_route(pair["start"], pair["target"], planning_options)
        if plan.feasible:
            pair["route_length_m"] = plan.route_length_m
            pair["route_min_clearance_m"] = plan.min_center_clearance_m
            pair["route_min_authority_reserve_m_s"] = plan.min_authority_reserve_m_s
            feasible.append(pair)
        else:
            print(f"Skipping infeasible canonical pair {pair['pair_id']}: {plan.reason}", flush=True)

    random_count = 0
    attempts = 0
    max_attempts = max(100, 50 * int(n_random))
    while random_count < int(n_random) and attempts < max_attempts:
        attempts += 1
        start = network.random_point_in_vessel(
            rng,
            min_radius=planning_options.required_center_clearance_m,
        )
        target = network.random_point_in_vessel(
            rng,
            min_radius=planning_options.required_center_clearance_m,
        )
        plan = network.plan_route(start, target, planning_options)
        if not plan.feasible:
            continue
        random_count += 1
        feasible.append(
            {
                "pair_id": f"random_{random_count:04d}",
                "trial_kind": "random",
                "start": start,
                "target": target,
                "route_length_m": plan.route_length_m,
                "route_min_clearance_m": plan.min_center_clearance_m,
                "route_min_authority_reserve_m_s": plan.min_authority_reserve_m_s,
            }
        )

    if random_count < int(n_random):
        raise RuntimeError(f"Could only generate {random_count}/{n_random} feasible random pairs")

    print("Generated route-feasible COMET ablation pairs:", flush=True)
    for pair in feasible:
        s_mm = np.asarray(pair["start"]) * 1000.0
        t_mm = np.asarray(pair["target"]) * 1000.0
        print(
            f"  {pair['pair_id']} kind={pair['trial_kind']} "
            f"start_mm={s_mm.tolist()} target_mm={t_mm.tolist()} "
            f"straight_mm={np.linalg.norm(t_mm - s_mm):.3f}",
            flush=True,
        )

    return feasible


def _path_metrics(network, start, target, trajectory) -> dict[str, float]:
    trajectory = np.asarray(trajectory, dtype=float)
    if len(trajectory) >= 2:
        path_length = float(np.sum(np.linalg.norm(np.diff(trajectory, axis=0), axis=1)))
    else:
        path_length = 0.0

    start = np.asarray(start, dtype=float)
    target = np.asarray(target, dtype=float)
    planned_waypoints = network.plan_waypoints(start, target)
    planned_path = np.asarray([start, *[np.asarray(w) for w in planned_waypoints]], dtype=float)
    if len(planned_path) >= 2:
        planned_route_length = float(np.sum(np.linalg.norm(np.diff(planned_path, axis=0), axis=1)))
    else:
        planned_route_length = 0.0

    return {
        "path_length_m": path_length,
        "planned_route_length_m": planned_route_length,
        "path_efficiency": float(np.clip(planned_route_length / path_length, 0.0, 1.0)) if path_length > 0 else 0.0,
    }


def _min_wall_clearance(network, trajectory) -> float:
    min_clearance = float("inf")
    for p in np.asarray(trajectory, dtype=float):
        seg, radial = network.find_segment(p)
        if seg is None:
            return float("-inf")
        min_clearance = min(min_clearance, float(seg.radius - radial))
    return min_clearance


def _comet_diagnostic_metrics(result) -> dict[str, float | int]:
    log = getattr(result, "comet_log", [])
    if not log:
        return {
            "comet_branch_switch_count": 0,
            "comet_low_authority_fraction": float("nan"),
            "comet_mean_intent_entropy": float("nan"),
            "comet_max_intent_entropy": float("nan"),
            "comet_safety_filter_fraction": float("nan"),
            "comet_terminal_precision_fraction": float("nan"),
            "comet_current_saturation_fraction": float("nan"),
            "comet_route_regression_count": 0,
            "comet_terminal_overshoot_count": 0,
            "comet_mean_capture_probability": float("nan"),
            "comet_min_selected_reserve_m": float("nan"),
            "comet_min_selected_reserve_mm": float("nan"),
        }

    selected = []
    reserves = []
    captures = []
    entropies = []
    safeties = []
    terminal_precision = []
    saturations = []
    arcs = []
    overshoots = []

    for row in log:
        selected.append(row[6])
        reserves.append(_safe_float(row[7]))
        captures.append(_safe_float(row[8]))
        entropies.append(_safe_float(row[10]))
        arcs.append(_safe_float(row[14]))
        safeties.append(bool(row[17]))
        terminal_precision.append(str(row[20]) == "precision")
        saturations.append(bool(row[28]))
        try:
            overshoots.append(int(row[30]))
        except Exception:
            overshoots.append(0)

    branch_switches = 0
    prev = None
    for sid in selected:
        if sid is None or sid == "" or str(sid).lower() == "none":
            continue
        if prev is not None and sid != prev:
            branch_switches += 1
        prev = sid

    route_regressions = 0
    for a, b in zip(arcs, arcs[1:]):
        if np.isfinite(a) and np.isfinite(b) and b < a - 1e-5:
            route_regressions += 1

    reserves_arr = np.asarray(reserves, dtype=float)
    captures_arr = np.asarray(captures, dtype=float)
    entropies_arr = np.asarray(entropies, dtype=float)

    finite_reserves = reserves_arr[np.isfinite(reserves_arr)]
    finite_captures = captures_arr[np.isfinite(captures_arr)]
    finite_entropies = entropies_arr[np.isfinite(entropies_arr)]

    return {
        "comet_branch_switch_count": int(branch_switches),
        "comet_low_authority_fraction": float(np.mean(finite_reserves < 0.0)) if finite_reserves.size else float("nan"),
        "comet_mean_intent_entropy": float(np.mean(finite_entropies)) if finite_entropies.size else float("nan"),
        "comet_max_intent_entropy": float(np.max(finite_entropies)) if finite_entropies.size else float("nan"),
        "comet_safety_filter_fraction": float(np.mean(safeties)) if safeties else float("nan"),
        "comet_terminal_precision_fraction": float(np.mean(terminal_precision)) if terminal_precision else float("nan"),
        "comet_current_saturation_fraction": float(np.mean(saturations)) if saturations else float("nan"),
        "comet_route_regression_count": int(route_regressions),
        "comet_terminal_overshoot_count": int(max(overshoots)) if overshoots else 0,
        "comet_mean_capture_probability": float(np.mean(finite_captures)) if finite_captures.size else float("nan"),
        "comet_min_selected_reserve_m": float(np.min(finite_reserves)) if finite_reserves.size else float("nan"),
        "comet_min_selected_reserve_mm": float(np.min(finite_reserves) * 1000.0) if finite_reserves.size else float("nan"),
    }


def _result_row(run_id, stress_mode, flow_scale, pair, variant, result, network, clearance_target_m):
    current_log = np.asarray(result.current_log, dtype=float)
    current_rms = float(np.sqrt(np.mean(current_log ** 2))) if current_log.size else 0.0
    energy_proxy = float(np.sum(current_log ** 2) * result.dt) if current_log.size else 0.0
    step_times = np.asarray(result.step_time_log, dtype=float)
    p95_step = float(np.percentile(step_times, 95)) if step_times.size else 0.0
    min_clearance = _min_wall_clearance(network, result.trajectory)
    params = getattr(result, "params", {})

    row = {
        "run_id": run_id,
        "stress_mode": stress_mode,
        "vessel_key": pair.get("vessel_key", ""),
        "vessel_label": pair.get("vessel_label", ""),
        "flow_scale": float(flow_scale),
        "pair_id": pair["pair_id"],
        "trial_kind": pair["trial_kind"],
        "controller_key": variant.key,
        "controller_label": variant.label,
        "start_x_m": float(pair["start"][0]),
        "start_y_m": float(pair["start"][1]),
        "start_z_m": float(pair["start"][2]),
        "target_x_m": float(pair["target"][0]),
        "target_y_m": float(pair["target"][1]),
        "target_z_m": float(pair["target"][2]),
        "reached_target": bool(result.reached),
        "steps_taken": int(result.steps_taken),
        "sim_time_s": float(result.steps_taken * result.dt),
        "final_error_m": float(result.final_error),
        "final_error_mm": float(result.final_error * 1000.0),
        **_path_metrics(network, result.start, result.target, result.trajectory),
        "min_wall_clearance_m": min_clearance,
        "min_wall_clearance_mm": min_clearance * 1000.0,
        "clearance_target_met": bool(min_clearance >= clearance_target_m),
        "current_rms_A": current_rms,
        "max_current_A": float(result.max_current_used),
        "energy_proxy_A2s": energy_proxy,
        "wall_time_s": float(result.wall_time_s),
        "avg_control_step_time_s": float(result.avg_control_step_time_s),
        "p95_control_step_time_s": p95_step,
        "max_control_step_time_s": float(result.max_control_step_time_s),
        "mean_reynolds": float(result.mean_reynolds),
        "max_reynolds": float(result.max_reynolds),
        "finite_re_drag_valid": bool(result.finite_re_drag_valid),
        "drag_regime_label": result.drag_regime_label,
        "peak_force_N": float(np.max(result.force_log)) if len(result.force_log) else 0.0,
        "peak_velocity_m_s": float(np.max(result.velocity_log)) if len(result.velocity_log) else 0.0,
        "peak_field_T": float(np.max(result.field_log)) if len(result.field_log) else 0.0,
        "max_power_W": _safe_float(params.get("max_power_W")),
        "max_voltage_V": _safe_float(params.get("max_voltage_V")),
        "max_coil_temp_C": _safe_float(params.get("max_coil_temp_C")),
        **_comet_diagnostic_metrics(result),
    }
    return row


def _write_csv(rows: list[dict[str, Any]], path: str, fields: list[str]) -> None:
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()
        for row in rows:
            writer.writerow({field: row.get(field, "") for field in fields})


def _aggregate(rows: list[dict[str, Any]], run_id: str, stress_mode: str, step_budget: int) -> list[dict[str, Any]]:
    groups = defaultdict(list)
    for row in rows:
        groups[(row["flow_scale"], row["controller_key"])].append(row)

    summary = []
    for (flow_scale, controller_key), group in sorted(groups.items()):
        label = group[0]["controller_label"]
        success = np.asarray([bool(r["reached_target"]) for r in group], dtype=bool)
        final_error = np.asarray([float(r["final_error_mm"]) for r in group], dtype=float)
        steps_effective = np.asarray(
            [float(r["steps_taken"] if r["reached_target"] else step_budget) for r in group],
            dtype=float,
        )

        def med(field):
            values = np.asarray([_safe_float(r.get(field)) for r in group], dtype=float)
            values = values[np.isfinite(values)]
            return float(np.median(values)) if values.size else float("nan")

        summary.append(
            {
                "run_id": run_id,
                "stress_mode": stress_mode,
                "flow_scale": flow_scale,
                "controller_key": controller_key,
                "controller_label": label,
                "n": len(group),
                "success_rate": float(np.mean(success)) if success.size else 0.0,
                "median_final_error_mm": float(np.median(final_error)) if final_error.size else float("nan"),
                "max_final_error_mm": float(np.max(final_error)) if final_error.size else float("nan"),
                "mean_steps_effective": float(np.mean(steps_effective)) if steps_effective.size else float("nan"),
                "median_steps_taken": med("steps_taken"),
                "median_path_efficiency": med("path_efficiency"),
                "median_current_rms_A": med("current_rms_A"),
                "median_energy_proxy_A2s": med("energy_proxy_A2s"),
                "median_min_wall_clearance_mm": med("min_wall_clearance_mm"),
                "clearance_target_met_rate": float(np.mean([bool(r["clearance_target_met"]) for r in group])),
                "finite_re_drag_valid_rate": float(np.mean([bool(r["finite_re_drag_valid"]) for r in group])),
                "median_branch_switch_count": med("comet_branch_switch_count"),
                "median_low_authority_fraction": med("comet_low_authority_fraction"),
                "median_mean_intent_entropy": med("comet_mean_intent_entropy"),
                "median_safety_filter_fraction": med("comet_safety_filter_fraction"),
                "median_terminal_precision_fraction": med("comet_terminal_precision_fraction"),
                "median_current_saturation_fraction": med("comet_current_saturation_fraction"),
                "median_route_regression_count": med("comet_route_regression_count"),
                "median_terminal_overshoot_count": med("comet_terminal_overshoot_count"),
                "median_min_selected_reserve_mm": med("comet_min_selected_reserve_mm"),
            }
        )

    return summary


def _software_metadata() -> dict[str, Any]:
    return {
        "python": sys.version,
        "platform": platform.platform(),
        "machine": platform.machine(),
        "processor": platform.processor(),
        "logical_cpu_count": os.cpu_count(),
        "numpy": np.__version__,
    }


def _common_kwargs(args, profile, max_current: float) -> dict[str, Any]:
    return {
        "dt": args.dt,
        "steps": args.steps,
        "tolerance": args.tolerance,
        "coil_radius": FIXED_COIL_RADIUS,
        "coil_length": FIXED_COIL_LENGTH,
        "core_radius": FIXED_CORE_RADIUS,
        "turns": FIXED_TURNS,
        "max_current": max_current,
        "placement_radius": FIXED_PLACEMENT_RADIUS,
        "core_relative_permeability": FIXED_CORE_MU_R_EFFECTIVE,
        "core_saturation_tesla": FIXED_CORE_SATURATION_T,
        "device_mode": args.device_mode,
        "profile": profile,
        "wall_clearance_threshold_m": args.wall_clearance_threshold_mm / 1000.0,
        "early_stop_enabled": not bool(args.selected_cases_csv),
        "early_stop_window_steps": args.early_stop_window_steps,
        "early_stop_improvement_threshold_m": args.early_stop_improvement_threshold_m,
        "verbose_every": args.verbose_every,
    }


def _run_task(task: dict[str, Any]) -> dict[str, Any]:
    network = RealVesselNetwork(task["vessel_file"], flow_scale=task["flow_scale"])
    result = simulate_trial_comet(
        start=task["pair"]["start"],
        target=task["pair"]["target"],
        network=network,
        trial_name=task["pair"]["pair_id"],
        rng_seed=task["trial_seed"],
        **task["common_kwargs"],
        **task["variant"].kwargs,
    )
    return {
        "vessel_key": task["vessel_key"],
        "vessel_label": task["vessel_label"],
        "vessel_file": task["vessel_file"],
        "flow_scale": task["flow_scale"],
        "pair": task["pair"],
        "variant": task["variant"],
        "result": result,
    }


def run_benchmark(args) -> str:
    profile = PROFILE_CHOICES[args.profile]
    selected_keys = _parse_csv_list(args.controllers)
    unknown = sorted(set(selected_keys) - set(COMET_ABLATION_VARIANTS))
    if unknown:
        raise ValueError(f"Unknown COMET ablation variant(s): {unknown}")
    selected = [COMET_ABLATION_VARIANTS[key] for key in selected_keys]

    flow_scales = _parse_flow_scales(args.flow_scales)
    if args.stress_mode == "high_flow" and args.flow_scales == "1,3,5":
        flow_scales = [7.0, 10.0]
    elif args.stress_mode == "low_authority" and args.flow_scales == "1,3,5":
        flow_scales = [5.0]

    effective_max_current = FIXED_MAX_CURRENT
    if args.stress_mode == "low_authority":
        effective_max_current = float(args.low_authority_current)

    run_id = args.run_id or time.strftime("%Y%m%d_%H%M%S")
    root = os.path.abspath(os.path.join(args.output_dir, run_id))
    os.makedirs(root, exist_ok=True)

    config = {
        "run_id": run_id,
        "stress_mode": args.stress_mode,
        "vessel_file": args.vessel_file if not args.selected_cases_csv else None,
        "selected_cases_csv": args.selected_cases_csv,
        "flow_scales": flow_scales,
        "n_random_trials": args.n_random_trials,
        "seed": args.seed,
        "workers": max(1, int(args.workers)),
        "steps": int(args.steps),
        "dt": float(args.dt),
        "early_stop_enabled": not bool(args.selected_cases_csv),
        "controllers": selected_keys,
        "profile": {
            "name": profile.name,
            "diameter_m": profile.diameter,
            "diameter_um": profile.diameter * 1e6,
        },
        "max_current_A": effective_max_current,
        "wall_clearance_threshold_m": args.wall_clearance_threshold_mm / 1000.0,
        "software": _software_metadata(),
        "variants": {key: COMET_ABLATION_VARIANTS[key].kwargs for key in selected_keys},
    }
    with open(os.path.join(root, "run_config.json"), "w") as f:
        json.dump(config, f, indent=2)

    rows = []
    seed_records = []
    workers = max(1, int(args.workers))
    common_kwargs = _common_kwargs(args, profile, effective_max_current)

    if args.selected_cases_csv:
        cases = _load_selected_cases(args.selected_cases_csv, flow_scales)
        expected = len(MANUSCRIPT_VESSEL_FILES) * 5 * len(flow_scales)
        if (not args.allow_partial_selected_cases) and (len(cases) != expected):
            raise ValueError(
                f"Selected-case CSV yielded {len(cases)} cases; expected {expected} "
                f"({len(MANUSCRIPT_VESSEL_FILES)} anatomies x 5 routes x "
                f"{len(flow_scales)} flows)"
            )
    else:
        cases = []
        for flow_scale in flow_scales:
            network = RealVesselNetwork(args.vessel_file, flow_scale=flow_scale)
            print(network.segment_summary(), flush=True)
            rng = np.random.default_rng(args.seed + int(round(flow_scale * 1000)))
            pairs = _trial_pairs(
                network=network,
                rng=rng,
                n_random=args.n_random_trials,
                vessel_file=args.vessel_file,
                profile=profile,
                wall_clearance_threshold_m=args.wall_clearance_threshold_mm / 1000.0,
            )
            for pair in pairs:
                cases.append({
                    "vessel_key": "single_vessel",
                    "vessel_label": "Single vessel",
                    "vessel_file": args.vessel_file,
                    "flow_scale": flow_scale,
                    "pair": pair,
                    "trial_seed": None,
                })

    tasks = []
    for case_idx, case in enumerate(cases):
        pair = dict(case["pair"])
        pair["vessel_key"] = case["vessel_key"]
        pair["vessel_label"] = case["vessel_label"]
        trial_seed = case["trial_seed"]
        if trial_seed is None:
            trial_seed = args.seed + case_idx
        seed_records.append({
            "vessel_key": case["vessel_key"],
            "vessel_file": case["vessel_file"],
            "flow_scale": case["flow_scale"],
            "pair_id": pair["pair_id"],
            "seed": trial_seed,
            "start": np.asarray(pair["start"], dtype=float).tolist(),
            "target": np.asarray(pair["target"], dtype=float).tolist(),
            "route_length_m": pair.get("route_length_m"),
            "route_min_clearance_m": pair.get("route_min_clearance_m"),
            "route_min_authority_reserve_m_s": pair.get("route_min_authority_reserve_m_s"),
        })
        for variant in selected:
            tasks.append({
                "vessel_key": case["vessel_key"],
                "vessel_label": case["vessel_label"],
                "vessel_file": case["vessel_file"],
                "flow_scale": case["flow_scale"],
                "pair": pair,
                "variant": variant,
                "trial_seed": trial_seed,
                "common_kwargs": common_kwargs,
            })

    network_cache = {}
    results_by_bucket = defaultdict(list)

    def handle(item):
        variant = item["variant"]
        pair = item["pair"]
        flow_scale = item["flow_scale"]
        bucket = (item["vessel_key"], flow_scale, variant.key)
        network_key = (item["vessel_file"], flow_scale)
        if network_key not in network_cache:
            network_cache[network_key] = RealVesselNetwork(
                item["vessel_file"], flow_scale=flow_scale
            )
        row = _result_row(
            run_id=run_id,
            stress_mode=args.stress_mode,
            flow_scale=flow_scale,
            pair=pair,
            variant=variant,
            result=item["result"],
            network=network_cache[network_key],
            clearance_target_m=args.wall_clearance_threshold_mm / 1000.0,
        )
        rows.append(row)
        results_by_bucket[bucket].append(item["result"])
        _write_csv(rows, os.path.join(root, "ablation_results.csv"), ROW_FIELDS)
        _write_csv(
            _aggregate(rows, run_id, args.stress_mode, args.steps),
            os.path.join(root, "ablation_summary.csv"), SUMMARY_FIELDS,
        )

    def task_label(task):
        return (
            f"anatomy={task['vessel_key']} flow={task['flow_scale']:g} "
            f"pair={task['pair']['pair_id']} variant={task['variant'].key}"
        )

    if workers == 1:
        for task in tasks:
            print(f"[{run_id}] {task_label(task)}", flush=True)
            handle(_run_task(task))
    else:
        try:
            executor = ProcessPoolExecutor(max_workers=workers)
        except PermissionError:
            executor = ThreadPoolExecutor(max_workers=workers)
        with executor:
            futures = []
            for task in tasks:
                print(f"[{run_id}] submit {task_label(task)}", flush=True)
                futures.append(executor.submit(_run_task, task))
            for future in as_completed(futures):
                item = future.result()
                print(
                    f"[{run_id}] done anatomy={item['vessel_key']} "
                    f"flow={item['flow_scale']:g} pair={item['pair']['pair_id']} "
                    f"variant={item['variant'].key}", flush=True,
                )
                handle(item)

    for (vessel_key, flow_scale, variant_key), results in results_by_bucket.items():
        out_dir = os.path.join(
            root, vessel_key, f"flow_{flow_scale:g}x", variant_key
        )
        os.makedirs(out_dir, exist_ok=True)
        save_results_csv(results, out_dir)
        save_comet_diagnostics(results, out_dir)

    summary = {
        **config,
        "n_rows": len(rows),
        "n_pairs": len(seed_records),
        "seed_records": seed_records,
        "results_csv": "ablation_results.csv",
        "summary_csv": "ablation_summary.csv",
    }
    with open(os.path.join(root, "benchmark_summary.json"), "w") as f:
        json.dump(summary, f, indent=2)

    _write_csv(rows, os.path.join(root, "ablation_results.csv"), ROW_FIELDS)
    _write_csv(
        _aggregate(rows, run_id, args.stress_mode, args.steps),
        os.path.join(root, "ablation_summary.csv"),
        SUMMARY_FIELDS,
    )
    print(f"\nCOMET ablation outputs saved to: {root}", flush=True)
    return root


def main():
    parser = argparse.ArgumentParser(description="Run COMET mechanism/stress ablations.")
    parser.add_argument("--output-dir", default="outputs/comet_ablation_benchmark")
    parser.add_argument("--run-id", default=None)
    parser.add_argument("--vessel-file", default=DEFAULT_VESSEL_FILE)
    parser.add_argument(
        "--selected-cases-csv",
        default=None,
        help=(
            "Use the full-COMET rows in a manuscript selected-results CSV "
            "instead of generating routes for one vessel."
        ),
    )
    parser.add_argument(
        "--allow-partial-selected-cases",
        action="store_true",
        help="Allow fewer than the frozen 45 selected cases for implementation sanity checks only.",
    )
    parser.add_argument("--flow-scales", default="1,3,5")
    parser.add_argument("--n-random-trials", type=int, default=4)
    parser.add_argument("--seed", type=int, default=20260609)
    parser.add_argument(
        "--controllers",
        default=",".join(COMET_ABLATION_VARIANTS.keys()),
        help="Comma-separated COMET variant keys. Use --list-controllers to inspect.",
    )
    parser.add_argument(
        "--stress-mode",
        choices=["standard", "high_flow", "low_authority", "branch_ambiguity"],
        default="standard",
    )
    parser.add_argument("--low-authority-current", type=float, default=8.0)
    parser.add_argument("--dt", type=float, default=0.02)
    parser.add_argument("--steps", type=int, default=4000)
    parser.add_argument("--tolerance", type=float, default=DEFAULT_TARGET_TOLERANCE)
    parser.add_argument("--wall-clearance-threshold-mm", type=float, default=0.7)
    parser.add_argument("--profile", choices=sorted(PROFILE_CHOICES), default="spion_carrier_500um")
    parser.add_argument("--device-mode", default=DEVICE_MODE)
    parser.add_argument("--workers", type=int, default=1)
    parser.add_argument("--early-stop-window-steps", type=int, default=200)
    parser.add_argument("--early-stop-improvement-threshold-m", type=float, default=0.002)
    parser.add_argument("--verbose-every", type=int, default=0)
    parser.add_argument("--list-controllers", action="store_true")
    parser.add_argument("--smoke-test", action="store_true")
    args = parser.parse_args()
    controllers_explicit = any(
        arg == "--controllers" or arg.startswith("--controllers=")
        for arg in sys.argv[1:]
    )
    steps_explicit = any(
        arg == "--steps" or arg.startswith("--steps=") for arg in sys.argv[1:]
    )

    if args.list_controllers:
        for key, variant in COMET_ABLATION_VARIANTS.items():
            print(f"{key}: {variant.label} kwargs={variant.kwargs}")
        return

    if args.smoke_test:
        args.flow_scales = "1"
        args.n_random_trials = 1
        if not controllers_explicit:
            args.controllers = "comet,comet_no_terminal_precision"
        args.steps = min(args.steps, 300)
        args.verbose_every = args.verbose_every or 50
    elif args.selected_cases_csv and not steps_explicit:
        # The source manuscript benchmark configs used a 2,000-step budget.
        args.steps = 2000

    run_benchmark(args)


if __name__ == "__main__":
    main()
