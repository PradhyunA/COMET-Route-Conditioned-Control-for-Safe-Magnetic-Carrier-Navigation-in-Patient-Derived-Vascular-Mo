# """Continue the paired efficiency study at 3x and 5x flow on four cores."""

# from __future__ import annotations

# import argparse
# import os
# import time


# # Keep each worker single-threaded so four trial workers do not oversubscribe
# # NumPy/BLAS CPU pools.
# for variable in (
#     "OMP_NUM_THREADS",
#     "OPENBLAS_NUM_THREADS",
#     "MKL_NUM_THREADS",
#     "VECLIB_MAXIMUM_THREADS",
#     "NUMEXPR_NUM_THREADS",
# ):
#     os.environ.setdefault(variable, "1")

# from run_publication_benchmark import (  # noqa: E402
#     DEFAULT_TARGET_TOLERANCE,
#     DEFAULT_VESSEL_FILE,
#     DEVICE_MODE,
#     run_benchmark,
# )


# CONTROLLERS = [
#     "mpc_default",
#     "mpc_tuned",
#     "acbc_same_weights",
#     "acbc_tuned",
# ]


# def main():
#     parser = argparse.ArgumentParser(
#         description="Paired 3x/5x ACBC and route-aware MPC continuation run."
#     )
#     parser.add_argument(
#         "--output-dir",
#         default="outputs/publication_benchmark_1mm",
#     )
#     parser.add_argument(
#         "--run-id",
#         default=time.strftime("efficiency_flow3_5_%Y%m%d_%H%M%S"),
#     )
#     parser.add_argument("--vessel-file", default=DEFAULT_VESSEL_FILE)
#     parser.add_argument("--flow-scales", default="3,5")
#     parser.add_argument(
#         "--n-random-trials",
#         type=int,
#         default=25,
#         help="Random matched pairs per flow, plus the 3 canonical pairs.",
#     )
#     parser.add_argument(
#         "--seed",
#         type=int,
#         default=20260610,
#         help="Matches the completed 1x benchmark configuration.",
#     )
#     parser.add_argument("--profile", default="spion_carrier_500um")
#     parser.add_argument("--dt", type=float, default=0.02)
#     parser.add_argument("--steps", type=int, default=4000)
#     parser.add_argument("--tolerance", type=float, default=DEFAULT_TARGET_TOLERANCE)
#     parser.add_argument("--wall-clearance-threshold-mm", type=float, default=0.7)
#     parser.add_argument("--workers", type=int, default=4)
#     parser.add_argument("--device-mode", default=DEVICE_MODE)
#     args = parser.parse_args()

#     flow_scales = [
#         float(value.strip())
#         for value in args.flow_scales.split(",")
#         if value.strip()
#     ]
#     pairs_per_flow = args.n_random_trials + 3
#     total_trials = pairs_per_flow * len(flow_scales) * len(CONTROLLERS)
#     print(
#         f"Starting {total_trials} controller trials: "
#         f"{pairs_per_flow} matched pairs/flow x {len(flow_scales)} flows x "
#         f"{len(CONTROLLERS)} controllers on {args.workers} workers.",
#         flush=True,
#     )

#     run_benchmark(
#         output_dir=args.output_dir,
#         run_id=args.run_id,
#         vessel_file=args.vessel_file,
#         flow_scales=flow_scales,
#         n_random_trials=args.n_random_trials,
#         seed=args.seed,
#         controller_keys=CONTROLLERS,
#         include_ablations=False,
#         dt=args.dt,
#         steps=args.steps,
#         tolerance=args.tolerance,
#         device_mode=args.device_mode,
#         profile_key=args.profile,
#         wall_clearance_threshold_m=args.wall_clearance_threshold_mm / 1000.0,
#         workers=args.workers,
#     )


# if __name__ == "__main__":
#     main()

"""Continue the paired efficiency study at 3x flow on four cores using the mapped 0050_H_CERE_H vessel."""

from __future__ import annotations

import argparse
import os
import time


# Keep each worker single-threaded so four trial workers do not oversubscribe
# NumPy/BLAS CPU pools.
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
    DEVICE_MODE,
    run_benchmark,
)


# ---------------------------------------------------------------------
# New mapped vessel geometry
# ---------------------------------------------------------------------
# This is the converted VMTK centerline network generated from:
#   data/0050_H_CERE_H/centerlines_all.vtp
#
# It was converted with raw units interpreted as centimeters:
#   raw coordinate unit -> 0.01 m
#
# Converted output:
#   data/0050_H_CERE_H/real_vessel_segments.pkl
# ---------------------------------------------------------------------
DEFAULT_MAPPED_VESSEL_FILE = "data/0050_H_CERE_H/real_vessel_segments.pkl"


CONTROLLERS = [
    "mpc_default",
    "mpc_tuned",
    "acbc_same_weights",
    "acbc_tuned",
]


def parse_flow_scales(flow_scales_text: str) -> list[float]:
    """Parse comma-separated flow scales such as '3' or '3,5'."""
    flow_scales = [
        float(value.strip())
        for value in flow_scales_text.split(",")
        if value.strip()
    ]

    if not flow_scales:
        raise ValueError(
            "No valid flow scales were provided. Example: --flow-scales 3"
        )

    return flow_scales


def main() -> None:
    parser = argparse.ArgumentParser(
        description=(
            "Paired 3x ACBC and route-aware MPC efficiency benchmark "
            "on the mapped 0050_H_CERE_H branching vessel."
        )
    )

    parser.add_argument(
        "--output-dir",
        default="outputs/publication_benchmark_0050_flow3",
        help="Directory where benchmark outputs will be saved.",
    )

    parser.add_argument(
        "--run-id",
        default=time.strftime("efficiency_0050_500um_flow3_%Y%m%d_%H%M%S"),
        help="Run identifier used in output filenames/folders.",
    )

    parser.add_argument(
        "--vessel-file",
        default=DEFAULT_MAPPED_VESSEL_FILE,
        help=(
            "Path to the vessel segment PKL file. Defaults to the newly mapped "
            "0050_H_CERE_H vessel."
        ),
    )

    parser.add_argument(
        "--flow-scales",
        default="3",
        help=(
            "Comma-separated flow scale factors. Default is '3' only. "
            "Use '3,5' if you later want both 3x and 5x."
        ),
    )

    parser.add_argument(
        "--n-random-trials",
        type=int,
        default=25,
        help="Random matched pairs per flow, plus the 3 canonical pairs.",
    )

    parser.add_argument(
        "--seed",
        type=int,
        default=20260610,
        help="Random seed for matched route generation.",
    )

    parser.add_argument(
        "--profile",
        default="spion_carrier_500um",
        help="Particle/carrier profile key.",
    )

    parser.add_argument(
        "--dt",
        type=float,
        default=0.02,
        help="Simulation timestep in seconds.",
    )

    parser.add_argument(
        "--steps",
        type=int,
        default=4000,
        help="Maximum simulation steps per trial.",
    )

    parser.add_argument(
        "--tolerance",
        type=float,
        default=DEFAULT_TARGET_TOLERANCE,
        help="Target-reaching tolerance in meters.",
    )

    parser.add_argument(
        "--wall-clearance-threshold-mm",
        type=float,
        default=0.7,
        help="Wall-clearance threshold in millimeters for safety metrics.",
    )

    parser.add_argument(
        "--workers",
        type=int,
        default=4,
        help="Number of parallel worker processes.",
    )

    parser.add_argument(
        "--device-mode",
        default=DEVICE_MODE,
        help="Magnetic device mode passed into the benchmark.",
    )

    args = parser.parse_args()

    flow_scales = parse_flow_scales(args.flow_scales)

    if not os.path.exists(args.vessel_file):
        raise FileNotFoundError(
            f"Vessel file not found: {args.vessel_file}\n"
            "Expected the mapped vessel file at:\n"
            f"  {DEFAULT_MAPPED_VESSEL_FILE}\n"
            "Make sure you already ran the VMTK conversion step."
        )

    pairs_per_flow = args.n_random_trials + 3
    total_trials = pairs_per_flow * len(flow_scales) * len(CONTROLLERS)

    print("=" * 80, flush=True)
    print("Starting paired efficiency benchmark", flush=True)
    print("=" * 80, flush=True)
    print(f"Run ID: {args.run_id}", flush=True)
    print(f"Output dir: {args.output_dir}", flush=True)
    print(f"Vessel file: {args.vessel_file}", flush=True)
    print(f"Flow scales: {flow_scales}", flush=True)
    print(f"Particle profile: {args.profile}", flush=True)
    print(f"Controllers: {CONTROLLERS}", flush=True)
    print(f"Random trials per flow: {args.n_random_trials}", flush=True)
    print(f"Canonical pairs per flow: 3", flush=True)
    print(f"Matched pairs per flow: {pairs_per_flow}", flush=True)
    print(f"Workers: {args.workers}", flush=True)
    print(f"Steps: {args.steps}", flush=True)
    print(f"dt: {args.dt}", flush=True)
    print(f"Total controller trials: {total_trials}", flush=True)
    print("=" * 80, flush=True)

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



