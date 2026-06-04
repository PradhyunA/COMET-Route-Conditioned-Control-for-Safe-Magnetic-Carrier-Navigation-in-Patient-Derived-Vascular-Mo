"""
Rescale pickled real-vessel segment geometry.

Use this when a SimVascular model stored in centimeters was converted as if
coordinates were millimeters. Geometry fields are scaled; flow velocity is
left unchanged by default because it is already stored in m/s.
"""

import argparse
import pickle
from pathlib import Path

import numpy as np


GEOMETRY_KEYS = ("start", "end", "radius")


def _stats(segments):
    starts = np.array([s["start"] for s in segments], dtype=float)
    ends = np.array([s["end"] for s in segments], dtype=float)
    radii = np.array([s["radius"] for s in segments], dtype=float)
    pts = np.vstack([starts, ends])

    lo = np.minimum(starts, ends) - radii[:, None]
    hi = np.maximum(starts, ends) + radii[:, None]
    tube_span = hi.max(axis=0) - lo.min(axis=0)

    return {
        "segments": len(segments),
        "centerline_span_mm": (pts.ptp(axis=0) * 1000.0).tolist(),
        "tube_span_mm": (tube_span * 1000.0).tolist(),
        "radius_mm_min_median_max": np.percentile(
            radii * 1000.0,
            [0, 50, 100],
        ).tolist(),
        "diameter_mm_min_median_max": np.percentile(
            2.0 * radii * 1000.0,
            [0, 50, 100],
        ).tolist(),
    }


def rescale_segments(segments, geometry_scale, velocity_scale=1.0):
    corrected = []

    for segment in segments:
        item = dict(segment)
        item["start"] = np.asarray(item["start"], dtype=np.float32) * geometry_scale
        item["end"] = np.asarray(item["end"], dtype=np.float32) * geometry_scale
        item["radius"] = float(item["radius"]) * geometry_scale

        if "v_mean" in item:
            item["v_mean"] = float(item["v_mean"]) * velocity_scale

        corrected.append(item)

    return corrected


def main():
    parser = argparse.ArgumentParser(
        description="Rescale geometry fields in a real_vessel_segments.pkl file.",
    )
    parser.add_argument(
        "--input",
        default="data/real_vessel_segments.pkl",
        help="Input pickle path.",
    )
    parser.add_argument(
        "--output",
        default="data/real_vessel_segments_cm_corrected.pkl",
        help="Output pickle path.",
    )
    parser.add_argument(
        "--geometry-scale",
        type=float,
        default=10.0,
        help="Scale for start/end/radius. Use 10 for cm-vs-mm correction.",
    )
    parser.add_argument(
        "--velocity-scale",
        type=float,
        default=1.0,
        help="Optional scale for v_mean. Leave at 1 when v_mean is already m/s.",
    )
    args = parser.parse_args()

    input_path = Path(args.input)
    output_path = Path(args.output)

    with input_path.open("rb") as f:
        segments = pickle.load(f)

    corrected = rescale_segments(
        segments,
        geometry_scale=args.geometry_scale,
        velocity_scale=args.velocity_scale,
    )

    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("wb") as f:
        pickle.dump(corrected, f, protocol=pickle.HIGHEST_PROTOCOL)

    print("Before:")
    for key, value in _stats(segments).items():
        print(f"  {key}: {value}")

    print("\nAfter:")
    for key, value in _stats(corrected).items():
        print(f"  {key}: {value}")

    print(f"\nWrote: {output_path}")


if __name__ == "__main__":
    main()
