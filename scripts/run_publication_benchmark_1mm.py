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
import subprocess
import sys
import tempfile
import time
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

from dynamag_simulation.core.real_vessel_network import RealVesselNetwork
from dynamag_simulation.core.profiles import SPION_CARRIER_1MM, SPION_CARRIER_500UM
from dynamag_simulation.core.constants import DEFAULT_TARGET_TOLERANCE
from dynamag_simulation.core.reporting import save_results_csv

from simulation_3d_acbc import (
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
    save_acbc_diagnostics,
    simulate_trial_acbc,
)
from simulation_3d_mpc import simulate_trial_mpc


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
    "controller_key",
    "controller_label",
    "controller_kind",
    "reached_target",
    "steps_taken",
    "sim_time_s",
    "final_error_m",
    "final_error_mm",
    "path_length_m",
    "path_efficiency",
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
    "avg_control_step_time_s",
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


def _trial_pairs(network, rng, n_random):
    inlet = network.inlet_start()
    outlet = network.outlet_end()

    pairs = [
        {
            "pair_id": "canonical_001_inlet_to_outlet",
            "trial_kind": "canonical",
            "start": inlet,
            "target": outlet,
        },
        {
            "pair_id": "canonical_002_mid_to_outlet",
            "trial_kind": "canonical",
            "start": network.random_point_in_vessel(rng),
            "target": outlet,
        },
        {
            "pair_id": "canonical_003_inlet_to_mid",
            "trial_kind": "canonical",
            "start": inlet,
            "target": network.random_point_in_vessel(rng),
        },
    ]

    for i in range(int(n_random)):
        pairs.append({
            "pair_id": f"random_{i + 1:04d}",
            "trial_kind": "random",
            "start": network.random_point_in_vessel(rng),
            "target": network.random_point_in_vessel(rng),
        })

    return pairs


def _safe_float(value, default=np.nan):
    try:
        return float(value)
    except Exception:
        return float(default)


def _wall_clearance_stats(network, trajectory):
    clearances = []
    for pos in np.asarray(trajectory, dtype=float):
        seg, radial_dist = network.find_segment(pos)
        if seg is None:
            continue
        clearances.append(float(seg.radius - radial_dist))

    if not clearances:
        return np.nan

    return float(np.min(clearances))


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


def _result_metrics(run_id, flow_scale, pair, controller_key, variant, result, network):
    traj = np.asarray(result.trajectory, dtype=float)
    segments = np.diff(traj, axis=0)
    path_length = float(np.sum(np.linalg.norm(segments, axis=1)))
    straight_dist = float(np.linalg.norm(result.target - result.start))
    path_eff = straight_dist / path_length if path_length > 0 else 0.0

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
    surface_clearance_target_m = _safe_float(
        params.get("acbc_surface_clearance_target_m"),
        0.0005,
    )
    clearance_target_m = particle_radius_m + surface_clearance_target_m
    min_surface_clearance_m = min_clearance - particle_radius_m
    clearance_target_met = bool(min_clearance >= clearance_target_m)

    return {
        "run_id": run_id,
        "flow_scale": flow_scale,
        "pair_id": pair["pair_id"],
        "trial_kind": pair["trial_kind"],
        "controller_key": controller_key,
        "controller_label": variant.label,
        "controller_kind": variant.kind,
        "reached_target": bool(result.reached),
        "steps_taken": int(result.steps_taken),
        "sim_time_s": float(result.steps_taken * result.dt),
        "final_error_m": float(result.final_error),
        "final_error_mm": float(result.final_error * 1000.0),
        "path_length_m": path_length,
        "path_efficiency": path_eff,
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
        "avg_control_step_time_s": float(result.avg_control_step_time_s),
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
        out.append({
            "flow_scale": flow_scale,
            "controller_key": controller,
            "n": len(group),
            "success_rate": float(np.mean(success)) if len(success) else 0.0,
            "median_final_error_mm": float(np.median(final_error)) if len(group) else np.nan,
            "median_sim_time_s": float(np.median(sim_time)) if len(group) else np.nan,
            "finite_re_drag_valid_rate": (
                float(np.mean(finite_re)) if len(finite_re) else 0.0
            ),
            "mean_transitions": float(np.mean([
                r["acbc_transition_count"] for r in group
            ])) if len(group) else 0.0,
        })

    return out


def _paired_stats(rows, baseline="mpc_tuned", comparator="acbc_tuned"):
    try:
        from scipy.stats import wilcoxon
    except Exception:
        wilcoxon = None

    by_flow_pair = defaultdict(dict)
    for row in rows:
        key = (row["flow_scale"], row["pair_id"])
        by_flow_pair[key][row["controller_key"]] = row

    stats = []
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

        for metric in ("final_error_mm", "sim_time_s", "path_efficiency"):
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
        return simulate_trial_mpc(
            start=pair["start"],
            target=pair["target"],
            network=network,
            trial_name=pair["pair_id"],
            rng_seed=trial_seed,
            **common_kwargs,
            **variant.kwargs,
        )

    if variant.kind == "acbc":
        return simulate_trial_acbc(
            start=pair["start"],
            target=pair["target"],
            network=network,
            trial_name=pair["pair_id"],
            rng_seed=trial_seed,
            **common_kwargs,
            **variant.kwargs,
        )

    raise ValueError(f"Unknown controller kind: {variant.kind}")


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
    }

    config = {
        "run_id": run_id,
        "vessel_file": vessel_file,
        "flow_scales": flow_scales,
        "n_random_trials": n_random_trials,
        "seed": seed,
        "controller_keys": list(selected),
        "include_ablations": include_ablations,
        "profile_key": profile_key,
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

    for flow_scale in flow_scales:
        rng = np.random.default_rng(seed + int(round(flow_scale * 1000)))
        network = RealVesselNetwork(vessel_file, flow_scale=flow_scale)
        pairs = _trial_pairs(network, rng, n_random_trials)
        results_by_controller = defaultdict(list)

        for pair_idx, pair in enumerate(pairs):
            trial_seed = seed + 100000 * int(round(flow_scale * 10)) + pair_idx
            all_seed_records.append({
                "flow_scale": flow_scale,
                "pair_id": pair["pair_id"],
                "seed": trial_seed,
                "start": np.asarray(pair["start"], dtype=float).tolist(),
                "target": np.asarray(pair["target"], dtype=float).tolist(),
            })

            for controller_key, variant in selected.items():
                print(
                    f"[{run_id}] flow={flow_scale:g} "
                    f"pair={pair['pair_id']} controller={controller_key}",
                    flush=True,
                )
                result = _run_variant(
                    variant=variant,
                    pair=pair,
                    network=network,
                    trial_seed=trial_seed,
                    common_kwargs=common_kwargs,
                )

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
                _write_benchmark_summary(root, run_id, all_seed_records, paired_rows)
                results_by_controller[controller_key].append(result)

                result._benchmark_controller_key = controller_key
                result._benchmark_flow_scale = flow_scale

        # Write per-controller summaries for this flow after completing all pairs.
        for controller_key, variant in selected.items():
            out_dir = os.path.join(root, f"flow_{flow_scale:g}x", controller_key)
            save_results_csv(results_by_controller[controller_key], out_dir)
            if variant.kind == "acbc":
                save_acbc_diagnostics(results_by_controller[controller_key], out_dir)

    paired_csv = os.path.join(root, "paired_results.csv")
    _write_paired_csv(paired_rows, paired_csv)

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
        "--controllers",
        default="mpc_default,mpc_tuned,acbc_same_weights,acbc_tuned",
        help="Comma-separated controller keys. Use --list-controllers to inspect.",
    )
    parser.add_argument("--include-ablations", action="store_true")
    parser.add_argument("--list-controllers", action="store_true")
    parser.add_argument("--dt", type=float, default=0.02)
    parser.add_argument("--steps", type=int, default=4000)
    parser.add_argument("--tolerance", type=float, default=DEFAULT_TARGET_TOLERANCE)
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
    )


if __name__ == "__main__":
    main()
