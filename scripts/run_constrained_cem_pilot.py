"""Five frozen manuscript cases across three anatomies, with optional paired objective."""
import argparse
import csv
import json
from pathlib import Path
import statistics
import time

from run_publication_benchmark import run_benchmark


ROOT = Path(__file__).resolve().parents[1]
CASES = (
    ("cm_corrected", "data/real_vessel_segments_cm_corrected.pkl",
     "canonical_001_inlet_to_outlet", [1.0, 5.0]),
    ("40Left85Right", "data/vessels/aortic_root_cerebral_40Left85Right/real_vessel_segments.pkl",
     "canonical_001_medium_40Left85Right_endpoint03", [1.0, 5.0]),
    ("pulmonary_0082_H_PULM_H", "data/vessels/pulmonary_0082_H_PULM_H/real_vessel_segments_cm_corrected.pkl",
     "canonical_001_inlet_to_outlet", [3.0]),
)


def summarize(rows, diagnostics):
    errors = [float(r["final_error_mm"]) for r in rows]
    steps = [int(r["steps_taken"]) for r in rows]
    return {
        "n_cases": len(rows),
        "success_count": sum(r["reached_target"].lower() == "true" for r in rows),
        "median_final_error_mm": statistics.median(errors),
        "max_final_error_mm": max(errors),
        "median_steps": statistics.median(steps),
        "mean_effective_steps": statistics.fmean(steps),
        "wall_exit_cases": sum(r["wall_exit"].lower() == "true" for r in rows),
        "no_feasible_plan_steps": sum(r["zero_current_fallback"] for r in diagnostics),
        "best_effort_plan_steps": sum(r.get("best_effort_fallback", False) for r in diagnostics),
        "feasible_plan_steps": sum(r["feasible"] for r in diagnostics),
        "controller_calls": len(diagnostics),
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--objective",
        choices=["default", "tuned", "recovery", "both", "compare-recovery"],
        default="default",
    )
    parser.add_argument("--steps", type=int, default=2000)
    parser.add_argument("--workers", type=int, default=1)
    parser.add_argument("--seed", type=int, default=20260611)
    parser.add_argument("--smoke", action="store_true", help="Two control steps per case; not a success benchmark")
    parser.add_argument("--output-root", type=Path)
    args = parser.parse_args()
    controllers = {
        "default": ["constrained_cem_mpc"],
        "tuned": ["constrained_cem_mpc_tuned_shared"],
        "recovery": ["constrained_cem_mpc_tuned_recovery"],
        "both": ["constrained_cem_mpc", "constrained_cem_mpc_tuned_shared"],
        "compare-recovery": [
            "constrained_cem_mpc", "constrained_cem_mpc_tuned_recovery"
        ],
    }[args.objective]
    steps = 2 if args.smoke else args.steps
    if steps < 1 or args.workers < 1:
        parser.error("steps and workers must be positive")
    mode = "smoke" if args.smoke else "pilot"
    output = args.output_root or ROOT / "outputs" / "constrained_cem" / f"{mode}_{time.strftime('%Y%m%d_%H%M%S')}"
    output = output.resolve()
    if output.exists():
        parser.error(f"Output already exists; choose a fresh --output-root: {output}")
    # Validate every frozen selection before starting an expensive run.
    for anatomy, vessel, pair, flows in CASES:
        if not (ROOT / vessel).is_file():
            parser.error(f"Missing vessel file: {vessel}")
        with (ROOT / "benchmark_manifests/icem_45case_500um" / f"{anatomy}.csv").open() as f:
            source = list(csv.DictReader(f))
        for flow in flows:
            assert sum(r["pair_id"] == pair and float(r["flow_scale"]) == flow for r in source) == 1
    print(f"{mode}: 5 frozen cases per controller; {len(controllers) * 5} rollouts; output={output}", flush=True)
    for anatomy, vessel, pair, flows in CASES:
        run_benchmark(
            output_dir=str(output), run_id=anatomy, vessel_file=str(ROOT / vessel),
            flow_scales=flows, n_random_trials=0, seed=args.seed,
            controller_keys=controllers, include_ablations=False, dt=0.02,
            steps=steps, tolerance=0.0005, device_mode="moving_adaptive_enclosing_gantry",
            profile_key="spion_carrier_500um", wall_clearance_threshold_m=0.0007,
            workers=args.workers,
            rerun_from_csv=str(ROOT / "benchmark_manifests/icem_45case_500um" / f"{anatomy}.csv"),
            rerun_pair_ids=[pair], rerun_flow_scales=flows, early_stop_enabled=False,
        )
    rows = []
    for path in sorted(output.glob("*/paired_results.csv")):
        with path.open() as f:
            rows.extend(csv.DictReader(f))
    report = {"mode": mode, "max_steps": steps, "controllers": {}}
    for controller in controllers:
        selected = [r for r in rows if r["controller_key"] == controller]
        assert len(selected) == 5, f"Expected 5 completed cases for {controller}"
        assert all(r["early_stopped"].lower() == "false" for r in selected)
        diagnostics = []
        files = list(output.glob(f"*/flow_*x/{controller}/*_constrained_cem.json"))
        assert len(files) == 5
        for path in files:
            diagnostics.extend(json.loads(path.read_text())["steps"])
        assert len(diagnostics) == sum(int(r["steps_taken"]) for r in selected)
        report["controllers"][controller] = summarize(selected, diagnostics)
    (output / "pilot_summary.json").write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps(report, indent=2), flush=True)
    print(f"Results: {output}", flush=True)


if __name__ == "__main__":
    main()
