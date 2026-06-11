"""
Parallel Candidate 8 smoke test on the mapped 0050_H_CERE_H vessel.

Runs four controllers in parallel:
    - mpc_default
    - mpc_tuned
    - acbc_same_weights
    - acbc_tuned

Settings:
    - Candidate 8 adversarial branch-trap route
    - 0050_H_CERE_H mapped vessel
    - 3x flow
    - 500 um SPION carrier
    - fixed_enclosing_stress_test device mode
    - 1000 max steps
    - 4 workers
"""

from __future__ import annotations

import csv
import json
import os
import sys
import tempfile
import time
from concurrent.futures import ProcessPoolExecutor, as_completed
from collections import defaultdict

import numpy as np


# Keep each worker single-threaded so four controller workers do not oversubscribe
# NumPy/BLAS CPU pools.
for variable in (
    "OMP_NUM_THREADS",
    "OPENBLAS_NUM_THREADS",
    "MKL_NUM_THREADS",
    "VECLIB_MAXIMUM_THREADS",
    "NUMEXPR_NUM_THREADS",
):
    os.environ.setdefault(variable, "1")


os.environ.setdefault(
    "MPLCONFIGDIR",
    os.path.join(tempfile.gettempdir(), "dynamag_matplotlib"),
)
os.environ.setdefault("MPLBACKEND", "Agg")


REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
SCRIPTS_DIR = os.path.abspath(os.path.dirname(__file__))

sys.path.insert(0, REPO_ROOT)
sys.path.insert(0, SCRIPTS_DIR)


from dynamag_simulation.core.real_vessel_network import RealVesselNetwork
from dynamag_simulation.core.profiles import SPION_CARRIER_500UM
from dynamag_simulation.core.constants import DEFAULT_TARGET_TOLERANCE
from dynamag_simulation.core.reporting import save_results_csv

from run_publication_benchmark import (
    CONTROLLER_VARIANTS,
    _result_metrics,
    _save_trial_npz,
    _write_paired_csv,
    _write_paired_differences,
    _write_benchmark_summary,
)

from simulation_3d_mpc import simulate_trial_mpc
from simulation_3d_naive_mpc import simulate_trial_naive_mpc
from simulation_3d_acbc import (
    simulate_trial_acbc,
    save_acbc_diagnostics,
    FIXED_COIL_RADIUS,
    FIXED_COIL_LENGTH,
    FIXED_CORE_RADIUS,
    FIXED_TURNS,
    FIXED_MAX_CURRENT,
    FIXED_PLACEMENT_RADIUS,
    FIXED_CORE_MU_R_EFFECTIVE,
    FIXED_CORE_SATURATION_T,
)


VESSEL_FILE = "data/0050_H_CERE_H/real_vessel_segments.pkl"

OUTPUT_DIR = "outputs/candidate8_0050_smoke"
RUN_ID = "candidate8_0050_500um_flow3_fixed_1000step_parallel"

FLOW_SCALE = 3.0
STEPS = 1000
DT = 0.02
TOLERANCE = DEFAULT_TARGET_TOLERANCE
DEVICE_MODE = "fixed_enclosing_stress_test"
PROFILE = SPION_CARRIER_500UM
WALL_CLEARANCE_THRESHOLD_M = 0.0007
SEED = 20260610
WORKERS = 4

# Candidate 8 from route finder output, converted from mm to meters.
START = np.array([-0.01678,  0.01367, -0.01668], dtype=float)
TARGET = np.array([-0.01040, -0.01594,  0.06980], dtype=float)

CONTROLLERS = [
    "mpc_default",
    "mpc_tuned",
    "acbc_same_weights",
    "acbc_tuned",
]


def _run_controller_task(task: dict) -> dict:
    controller_key = task["controller_key"]
    variant = task["variant"]
    pair = task["pair"]
    common_kwargs = task["common_kwargs"]

    network = RealVesselNetwork(task["vessel_file"], flow_scale=task["flow_scale"])

    print(
        f"[worker] starting controller={controller_key} "
        f"pair={pair['pair_id']} flow={task['flow_scale']}x",
        flush=True,
    )

    if variant.kind == "mpc":
        result = simulate_trial_mpc(
            start=pair["start"],
            target=pair["target"],
            network=network,
            trial_name=pair["pair_id"],
            rng_seed=task["seed"],
            **common_kwargs,
            **variant.kwargs,
        )

    elif variant.kind == "naive_mpc":
        result = simulate_trial_naive_mpc(
            start=pair["start"],
            target=pair["target"],
            network=network,
            trial_name=pair["pair_id"],
            rng_seed=task["seed"],
            **common_kwargs,
            **variant.kwargs,
        )

    elif variant.kind == "acbc":
        result = simulate_trial_acbc(
            start=pair["start"],
            target=pair["target"],
            network=network,
            trial_name=pair["pair_id"],
            rng_seed=task["seed"],
            **common_kwargs,
            **variant.kwargs,
        )

    else:
        raise ValueError(f"Unknown controller kind: {variant.kind}")

    print(
        f"[worker] finished controller={controller_key} "
        f"reached={result.reached} "
        f"steps={result.steps_taken} "
        f"final_error_mm={result.final_error * 1000.0:.3f}",
        flush=True,
    )

    return {
        "controller_key": controller_key,
        "variant": variant,
        "pair": pair,
        "result": result,
        "flow_scale": task["flow_scale"],
    }


def _print_final_table(rows: list[dict]) -> None:
    if not rows:
        print("No rows to summarize.")
        return

    rows = sorted(rows, key=lambda r: CONTROLLERS.index(r["controller_key"]))

    print()
    print("=" * 120)
    print("Candidate 8 smoke-test summary")
    print("=" * 120)
    print(
        f"{'controller':22s} "
        f"{'reached':8s} "
        f"{'steps':>8s} "
        f"{'final_mm':>12s} "
        f"{'path_eff':>10s} "
        f"{'energy_A2s':>12s} "
        f"{'wall_time':>10s} "
        f"{'mean_Re':>10s} "
        f"{'max_Re':>10s}"
    )
    print("-" * 120)

    for row in rows:
        print(
            f"{row['controller_key']:22s} "
            f"{str(row['reached_target']):8s} "
            f"{int(row['steps_taken']):8d} "
            f"{float(row['final_error_mm']):12.3f} "
            f"{float(row['path_efficiency']):10.3f} "
            f"{float(row['energy_proxy_A2s']):12.3f} "
            f"{float(row['wall_time_s']):10.2f} "
            f"{float(row['mean_reynolds']):10.3f} "
            f"{float(row['max_reynolds']):10.3f}"
        )

    print("=" * 120)


def main() -> None:
    started = time.perf_counter()

    if not os.path.exists(VESSEL_FILE):
        raise FileNotFoundError(
            f"Missing vessel file: {VESSEL_FILE}\n"
            "Run the VMTK centerline conversion first."
        )

    root = os.path.abspath(os.path.join(OUTPUT_DIR, RUN_ID))
    os.makedirs(root, exist_ok=True)

    print("=" * 80, flush=True)
    print("Parallel Candidate 8 single-route smoke test", flush=True)
    print("=" * 80, flush=True)
    print(f"Run ID: {RUN_ID}", flush=True)
    print(f"Output root: {root}", flush=True)
    print(f"Vessel file: {VESSEL_FILE}", flush=True)
    print(f"Flow scale: {FLOW_SCALE}x", flush=True)
    print(f"Device mode: {DEVICE_MODE}", flush=True)
    print(f"Profile: {PROFILE.name}", flush=True)
    print(f"Steps: {STEPS}", flush=True)
    print(f"dt: {DT}", flush=True)
    print(f"Workers: {WORKERS}", flush=True)
    print(f"Start m: {START}", flush=True)
    print(f"Target m: {TARGET}", flush=True)
    print(f"Start mm: {START * 1000.0}", flush=True)
    print(f"Target mm: {TARGET * 1000.0}", flush=True)
    print(f"Chord distance cm: {np.linalg.norm(TARGET - START) * 100.0:.2f}", flush=True)
    print("=" * 80, flush=True)

    network = RealVesselNetwork(VESSEL_FILE, flow_scale=FLOW_SCALE)
    print(network.segment_summary(), flush=True)

    pair = {
        "pair_id": "candidate8_adversarial_branch_trap",
        "trial_kind": "adversarial_candidate8",
        "start": START,
        "target": TARGET,
    }

    common_kwargs = {
        "dt": DT,
        "steps": STEPS,
        "tolerance": TOLERANCE,
        "coil_radius": FIXED_COIL_RADIUS,
        "coil_length": FIXED_COIL_LENGTH,
        "core_radius": FIXED_CORE_RADIUS,
        "turns": FIXED_TURNS,
        "max_current": FIXED_MAX_CURRENT,
        "placement_radius": FIXED_PLACEMENT_RADIUS,
        "core_relative_permeability": FIXED_CORE_MU_R_EFFECTIVE,
        "core_saturation_tesla": FIXED_CORE_SATURATION_T,
        "device_mode": DEVICE_MODE,
        "profile": PROFILE,
        "wall_clearance_threshold_m": WALL_CLEARANCE_THRESHOLD_M,
    }

    config = {
        "run_id": RUN_ID,
        "vessel_file": VESSEL_FILE,
        "flow_scale": FLOW_SCALE,
        "steps": STEPS,
        "dt": DT,
        "device_mode": DEVICE_MODE,
        "profile": PROFILE.name,
        "start_m": START.tolist(),
        "target_m": TARGET.tolist(),
        "start_mm": (START * 1000.0).tolist(),
        "target_mm": (TARGET * 1000.0).tolist(),
        "controllers": CONTROLLERS,
        "workers": WORKERS,
        "route_name": "candidate8_adversarial_branch_trap",
        "wall_clearance_threshold_m": WALL_CLEARANCE_THRESHOLD_M,
    }

    with open(os.path.join(root, "run_config.json"), "w") as f:
        json.dump(config, f, indent=2)

    selected = {key: CONTROLLER_VARIANTS[key] for key in CONTROLLERS}

    tasks = []
    for idx, (controller_key, variant) in enumerate(selected.items()):
        tasks.append(
            {
                "controller_key": controller_key,
                "variant": variant,
                "pair": pair,
                "common_kwargs": common_kwargs,
                "vessel_file": VESSEL_FILE,
                "flow_scale": FLOW_SCALE,
                "seed": SEED + idx,
            }
        )

    paired_rows = []
    results_by_controller = defaultdict(list)

    print()
    print(f"Submitting {len(tasks)} controller tasks on {WORKERS} workers...", flush=True)

    with ProcessPoolExecutor(max_workers=WORKERS) as executor:
        futures = [executor.submit(_run_controller_task, task) for task in tasks]

        for future in as_completed(futures):
            item = future.result()

            controller_key = item["controller_key"]
            variant = item["variant"]
            result = item["result"]

            result.trial_name = f"{pair['pair_id']}_{controller_key}"
            result._benchmark_controller_key = controller_key
            result._benchmark_flow_scale = FLOW_SCALE

            metrics = _result_metrics(
                run_id=RUN_ID,
                flow_scale=FLOW_SCALE,
                pair=pair,
                controller_key=controller_key,
                variant=variant,
                result=result,
                network=network,
            )

            paired_rows.append(metrics)
            results_by_controller[controller_key].append(result)

            out_dir = os.path.join(root, f"flow_{FLOW_SCALE:g}x", controller_key)
            os.makedirs(out_dir, exist_ok=True)

            _save_trial_npz(result, metrics, out_dir)

            if variant.kind == "acbc":
                save_acbc_diagnostics([result], out_dir)

            _write_paired_csv(paired_rows, os.path.join(root, "paired_results.csv"))
            _write_paired_differences(
                paired_rows,
                os.path.join(root, "paired_differences.csv"),
            )

            print(
                f"[main] saved controller={controller_key} "
                f"reached={result.reached} "
                f"steps={result.steps_taken} "
                f"final_error_mm={result.final_error * 1000.0:.3f}",
                flush=True,
            )

    for controller_key, results in results_by_controller.items():
        out_dir = os.path.join(root, f"flow_{FLOW_SCALE:g}x", controller_key)
        save_results_csv(results, out_dir)

    seed_records = [
        {
            "flow_scale": FLOW_SCALE,
            "pair_id": pair["pair_id"],
            "seed": SEED,
            "start": START.tolist(),
            "target": TARGET.tolist(),
        }
    ]

    _write_paired_csv(paired_rows, os.path.join(root, "paired_results.csv"))
    _write_paired_differences(
        paired_rows,
        os.path.join(root, "paired_differences.csv"),
    )
    _write_benchmark_summary(root, RUN_ID, seed_records, paired_rows)

    _print_final_table(paired_rows)

    elapsed = time.perf_counter() - started

    print()
    print("=" * 80)
    print("Done.")
    print(f"Elapsed wall time: {elapsed:.2f} s")
    print(f"Outputs saved to: {root}")
    print(f"Paired results: {os.path.join(root, 'paired_results.csv')}")
    print("=" * 80)


if __name__ == "__main__":
    main()
