"""Continue the efficiency benchmark at 3x and 5x flow without naive MPC."""

from __future__ import annotations

import argparse
import os
import time


# Avoid nested numerical thread pools inside the six benchmark workers.
for variable in (
    "OMP_NUM_THREADS",
    "OPENBLAS_NUM_THREADS",
    "MKL_NUM_THREADS",
    "VECLIB_MAXIMUM_THREADS",
    "NUMEXPR_NUM_THREADS",
):
    os.environ.setdefault(variable, "1")

from run_publication_benchmark import (  # noqa: E402
    DEFAULT_TARGET_TOLERANCE,
    DEFAULT_VESSEL_FILE,
    DEVICE_MODE,
    run_benchmark,
)


CONTROLLERS = [
    "mpc_default",
    "mpc_tuned",
    "acbc_same_weights",
    "acbc_tuned",
]


def main():
    parser = argparse.ArgumentParser(
        description="Run the paired efficiency benchmark at 3x and 5x flow."
    )
    parser.add_argument(
        "--output-dir",
        default="outputs/publication_benchmark_1mm",
    )
    parser.add_argument(
        "--run-id",
        default=time.strftime("efficiency_flow3_flow5_%Y%m%d_%H%M%S"),
    )
    parser.add_argument("--vessel-file", default=DEFAULT_VESSEL_FILE)
    parser.add_argument(
        "--n-random-trials",
        type=int,
        default=25,
        help="Random matched pairs per flow, in addition to 3 canonical pairs.",
    )
    parser.add_argument("--seed", type=int, default=20260610)
    parser.add_argument("--profile", default="spion_carrier_500um")
    parser.add_argument("--dt", type=float, default=0.02)
    parser.add_argument("--steps", type=int, default=4000)
    parser.add_argument("--tolerance", type=float, default=DEFAULT_TARGET_TOLERANCE)
    parser.add_argument("--wall-clearance-threshold-mm", type=float, default=0.7)
    parser.add_argument("--workers", type=int, default=6)
    parser.add_argument("--device-mode", default=DEVICE_MODE)
    args = parser.parse_args()

    flow_scales = [3.0, 5.0]
    pairs_per_flow = args.n_random_trials + 3
    total_trials = pairs_per_flow * len(flow_scales) * len(CONTROLLERS)
    print(
        f"Starting {total_trials} controller trials: "
        f"{pairs_per_flow} matched pairs/flow x 2 flows x "
        f"{len(CONTROLLERS)} controllers on {args.workers} workers.",
        flush=True,
    )

    run_benchmark(
        output_dir=args.output_dir,
        run_id=args.run_id,
        vessel_file=args.vessel_file,
        flow_scales=flow_scales,
        n_random_trials=args.n_random_trials,
        seed=args.seed,
        controller_keys=CONTROLLERS,
        include_ablations=False,
        dt=args.dt,
        steps=args.steps,
        tolerance=args.tolerance,
        device_mode=args.device_mode,
        profile_key=args.profile,
        wall_clearance_threshold_m=args.wall_clearance_threshold_mm / 1000.0,
        workers=args.workers,
    )


if __name__ == "__main__":
    main()
