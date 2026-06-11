import argparse
import json
import os
import pickle
import numpy as np


def main():
    parser = argparse.ArgumentParser(
        description="Export DynaMag real_vessel_segments.pkl to JSON for HTML visualization."
    )
    parser.add_argument(
        "--input",
        required=True,
        help="Path to real_vessel_segments.pkl",
    )
    parser.add_argument(
        "--output",
        required=True,
        help="Path to output JSON file",
    )
    args = parser.parse_args()

    with open(args.input, "rb") as f:
        segments = pickle.load(f)

    if not isinstance(segments, list) or len(segments) == 0:
        raise ValueError("Expected a non-empty list of segment dicts.")

    exported_segments = []
    all_points = []

    for seg in segments:
        start_m = np.array(seg["start"], dtype=float)
        end_m = np.array(seg["end"], dtype=float)
        radius_m = float(seg["radius"])

        start_mm = (start_m * 1000.0).tolist()
        end_mm = (end_m * 1000.0).tolist()
        radius_mm = radius_m * 1000.0

        exported_segments.append(
            {
                "seg_id": int(seg.get("seg_id", -1)),
                "start_mm": start_mm,
                "end_mm": end_mm,
                "radius_mm": radius_mm,
                "v_mean_mps": float(seg.get("v_mean", 0.0)),
                "level": int(seg.get("level", 0)),
            }
        )

        all_points.append(start_m * 1000.0)
        all_points.append(end_m * 1000.0)

    all_points = np.array(all_points, dtype=float)
    bbox_min_mm = all_points.min(axis=0)
    bbox_max_mm = all_points.max(axis=0)
    bbox_size_mm = bbox_max_mm - bbox_min_mm
    center_mm = 0.5 * (bbox_min_mm + bbox_max_mm)

    data = {
        "metadata": {
            "source_pkl": args.input,
            "n_segments": len(exported_segments),
            "units": "mm",
        },
        "bbox_mm": {
            "min": bbox_min_mm.tolist(),
            "max": bbox_max_mm.tolist(),
            "size": bbox_size_mm.tolist(),
            "center": center_mm.tolist(),
        },
        "segments": exported_segments,
    }

    os.makedirs(os.path.dirname(args.output), exist_ok=True)
    with open(args.output, "w") as f:
        json.dump(data, f)

    print(f"Saved JSON to: {args.output}")
    print(f"Segments: {len(exported_segments)}")
    print(f"BBox min mm: {bbox_min_mm}")
    print(f"BBox max mm: {bbox_max_mm}")
    print(f"BBox size mm: {bbox_size_mm}")
    print(f"Center mm: {center_mm}")


if __name__ == "__main__":
    main()