"""
Freeze a 15-case benchmark manifest (5 routes x flow 1/3/5) for one anatomy.

Pairs are drawn exactly as run_publication_benchmark draws them
(rng seed + 1000 * flow scale, three canonical routes and two random ones,
infeasible routes redrawn), so a new anatomy gets the same protocol as the
frozen icem_45case_500um manifests.

    python scripts/make_benchmark_manifest.py VESSEL_PKL OUT_CSV
"""

import argparse
import csv
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))

import run_publication_benchmark as rpb  # noqa: E402
from dynamag_simulation.core.profiles import SPION_CARRIER_500UM  # noqa: E402

FIELDS = ["flow_scale", "pair_id", "trial_kind", "start_x_m", "start_y_m", "start_z_m",
          "target_x_m", "target_y_m", "target_z_m"]


def main():
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("vessel_file")
    parser.add_argument("out_csv")
    parser.add_argument("--seed", type=int, default=20260611)
    parser.add_argument("--flow-scales", default="1,3,5")
    parser.add_argument("--n-random", type=int, default=2)
    parser.add_argument("--wall-clearance-threshold-m", type=float, default=0.0007)
    parser.add_argument("--route-clearance-m", type=float, default=None,
                        help="Centre clearance routes must offer (default: the wall threshold). "
                             "Set above it to leave a tracking margin.")
    parser.add_argument("--allow-missing-canonical", action="store_true",
                        help="Keep going when a canonical route is infeasible (tuning sets).")
    args = parser.parse_args()

    rows = []
    for flow_scale in (float(v) for v in args.flow_scales.split(",")):
        network = rpb._make_benchmark_network(args.vessel_file, flow_scale)
        rng = np.random.default_rng(args.seed + int(round(flow_scale * 1000)))
        pairs = rpb._trial_pairs(
            network, rng, args.n_random, vessel_file=args.vessel_file,
            profile=SPION_CARRIER_500UM,
            wall_clearance_threshold_m=args.route_clearance_m or args.wall_clearance_threshold_m,
        )
        if sum(p["trial_kind"] == "canonical" for p in pairs) != 3 and not args.allow_missing_canonical:
            raise RuntimeError(f"A canonical route is infeasible at flow {flow_scale}")
        for pair in pairs:
            rows.append({
                "flow_scale": flow_scale, "pair_id": pair["pair_id"], "trial_kind": pair["trial_kind"],
                **{f"start_{a}_m": float(v) for a, v in zip("xyz", pair["start"])},
                **{f"target_{a}_m": float(v) for a, v in zip("xyz", pair["target"])},
            })

    Path(args.out_csv).parent.mkdir(parents=True, exist_ok=True)
    with open(args.out_csv, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=FIELDS)
        writer.writeheader()
        writer.writerows(rows)
    print(f"Wrote {len(rows)} cases to {args.out_csv}")


if __name__ == "__main__":
    main()
