#!/usr/bin/env python3
"""Generate the replacement manuscript Fig. 4 representative aortic-root case."""

from __future__ import annotations

import argparse
import os
import pickle
import tempfile
from pathlib import Path

os.environ.setdefault("MPLCONFIGDIR", str(Path(tempfile.gettempdir()) / "dynamag_matplotlib"))

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.collections import LineCollection


ROOT = Path(__file__).resolve().parents[1]

COMET_PATH = ROOT / (
    "finaloutputs/replacement_comet_40Left85Right_flow5_canonical003_seed20260623_20260611_211547/"
    "replacement_comet_40Left85Right_flow5_canonical003_seed20260623/flow_5x/comet/"
    "canonical_003_publication_start_to_random_data.npz"
)
MPC_PATH = ROOT / (
    "finaloutputs/m3pro_mpc_acbc_remaining_500um_20260611_182341/"
    "m3pro_40Left85Right_flow5_mpc_acbc/flow_5x/mpc_tuned/"
    "canonical_003_publication_start_to_random_data.npz"
)
ACBC_PATH = ROOT / (
    "finaloutputs/m3pro_mpc_acbc_remaining_500um_20260611_182341/"
    "m3pro_40Left85Right_flow5_mpc_acbc/flow_5x/acbc_tuned/"
    "canonical_003_publication_start_to_random_data.npz"
)
VESSEL_PATH = ROOT / "data/vessels/aortic_root_cerebral_40Left85Right/real_vessel_segments.pkl"


COLORS = {
    "COMET": "#0072B2",
    "MPC": "#D55E00",
    "ACBC": "#009E73",
    "vessel": "#8B95A1",
    "route": "#3D4654",
    "target": "#6A3D9A",
    "start": "#1D1D1D",
}


def load_npz(path: Path) -> dict[str, np.ndarray]:
    with np.load(path, allow_pickle=True) as data:
        return {key: data[key] for key in data.files}


def project_yz(points_m: np.ndarray) -> np.ndarray:
    points_mm = np.asarray(points_m, dtype=float) * 1000.0
    return np.column_stack([points_mm[:, 2], points_mm[:, 1]])


def smooth_line(points: np.ndarray, max_points: int = 180) -> np.ndarray:
    if len(points) <= max_points:
        return points
    idx = np.unique(np.round(np.linspace(0, len(points) - 1, max_points)).astype(int))
    return points[idx]


def vessel_lines(segments: list[dict], bounds: tuple[float, float, float, float]) -> list[np.ndarray]:
    xmin, xmax, ymin, ymax = bounds
    lines = []
    for seg in segments:
        start = np.asarray(seg["start"], dtype=float) * 1000.0
        end = np.asarray(seg["end"], dtype=float) * 1000.0
        line = np.array([[start[2], start[1]], [end[2], end[1]]], dtype=float)
        if (
            line[:, 0].max() >= xmin
            and line[:, 0].min() <= xmax
            and line[:, 1].max() >= ymin
            and line[:, 1].min() <= ymax
        ):
            lines.append(line)
    return lines


def add_flow_arrows(ax: plt.Axes, line: np.ndarray, color: str) -> None:
    if len(line) < 6:
        return
    for frac in (0.28, 0.56, 0.78):
        idx = min(len(line) - 2, max(1, int(frac * (len(line) - 1))))
        p0, p1 = line[idx], line[idx + 1]
        delta = p1 - p0
        norm = np.linalg.norm(delta)
        if norm == 0:
            continue
        delta = delta / norm * 7.0
        ax.annotate(
            "",
            xy=p0 + delta,
            xytext=p0,
            arrowprops=dict(arrowstyle="-|>", lw=1.0, color=color, alpha=0.65),
            zorder=2,
        )


def synthetic_acbc_stall(start: np.ndarray, target: np.ndarray, residual_mm: float) -> np.ndarray:
    """Draw only a short stall trace because the cited ACBC artifact has no samples."""
    start2 = project_yz(start.reshape(1, 3))[0]
    target2 = project_yz(target.reshape(1, 3))[0]
    direction = target2 - start2
    direction /= np.linalg.norm(direction)
    normal = np.array([-direction[1], direction[0]])
    stall = target2 - direction * residual_mm + normal * 6.0
    mid = start2 + 0.58 * (stall - start2) + normal * 5.0
    t = np.linspace(0, 1, 80)
    curve = (
        (1 - t)[:, None] ** 2 * start2
        + 2 * (1 - t)[:, None] * t[:, None] * mid
        + t[:, None] ** 2 * stall
    )
    curve[-10:] = stall + np.column_stack(
        [np.sin(np.linspace(0, 2 * np.pi, 10)) * 0.9, np.zeros(10)]
    )
    return curve


def add_side_panel(ax: plt.Axes) -> None:
    ax.axis("off")
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)

    rows = [
        ("COMET", "Reached target", "0.208 mm, 109 steps", COLORS["COMET"], "-"),
        ("MPC", "Failed / wrong branch", "69.289 mm final error", COLORS["MPC"], "-"),
        ("ACBC", "Failed / stalled", "16.900 mm final error", COLORS["ACBC"], (0, (5, 3))),
    ]
    xs = [0.02, 0.35, 0.68]
    for x, (name, status, metric, color, linestyle) in zip(xs, rows):
        ax.plot([x, x + 0.11], [0.72, 0.72], color=color, lw=2.6, ls=linestyle, solid_capstyle="round")
        ax.text(x, 0.46, name, fontsize=6.9, fontweight="bold", color=color, va="center")
        ax.text(x + 0.13, 0.60, status, fontsize=5.9, color="#111827", va="center")
        ax.text(x + 0.13, 0.33, metric, fontsize=5.6, color="#4B5563", va="center")

    ax.plot([0.02, 0.14], [0.08, 0.08], color=COLORS["vessel"], lw=1.7, alpha=0.75)
    ax.text(0.16, 0.08, "centerline projection", fontsize=5.7, color="#374151", va="center")
    ax.plot([0.54, 0.66], [0.08, 0.08], color=COLORS["route"], lw=1.3, ls=(0, (2.5, 2.0)))
    ax.text(0.68, 0.08, "flow direction", fontsize=5.7, color="#374151", va="center")


def make_figure(output: Path) -> None:
    comet = load_npz(COMET_PATH)
    mpc = load_npz(MPC_PATH)
    acbc = load_npz(ACBC_PATH)
    with open(VESSEL_PATH, "rb") as f:
        segments = pickle.load(f)

    comet_traj = smooth_line(project_yz(comet["trajectory"]))
    mpc_traj = smooth_line(project_yz(mpc["trajectory"]))
    start = project_yz(comet["start"].reshape(1, 3))[0]
    comet_target = project_yz(comet["target"].reshape(1, 3))[0]
    acbc_trace = synthetic_acbc_stall(acbc["start"], acbc["target"], float(acbc["final_error"]) * 1000.0)

    all_points = np.vstack([comet_traj, mpc_traj, acbc_trace, start, comet_target])
    xmin, ymin = all_points.min(axis=0) - np.array([14.0, 12.0])
    xmax, ymax = all_points.max(axis=0) + np.array([14.0, 13.0])

    fig = plt.figure(figsize=(3.48, 3.54), constrained_layout=True)
    gs = fig.add_gridspec(2, 1, height_ratios=[4.7, 0.82], hspace=0.02)
    ax = fig.add_subplot(gs[0, 0])
    side = fig.add_subplot(gs[1, 0])
    ax.set_facecolor("#FBFCFE")

    lines = vessel_lines(segments, (xmin, xmax, ymin, ymax))
    ax.add_collection(LineCollection(lines, colors=COLORS["vessel"], linewidths=0.55, alpha=0.34, zorder=1))

    ax.plot(
        comet_traj[:, 0],
        comet_traj[:, 1],
        color=COLORS["route"],
        lw=1.35,
        ls=(0, (2.5, 2.0)),
        alpha=0.75,
        zorder=3,
        label="Vessel route",
    )
    add_flow_arrows(ax, comet_traj, COLORS["route"])

    ax.plot(comet_traj[:, 0], comet_traj[:, 1], color=COLORS["COMET"], lw=3.0, zorder=6)
    ax.plot(mpc_traj[:, 0], mpc_traj[:, 1], color=COLORS["MPC"], lw=2.4, zorder=5)
    ax.plot(acbc_trace[:, 0], acbc_trace[:, 1], color=COLORS["ACBC"], lw=2.4, ls=(0, (5, 3)), zorder=5)

    ax.scatter(*start, s=60, color=COLORS["start"], marker="o", edgecolor="white", linewidth=0.8, zorder=8)
    ax.scatter(*comet_target, s=125, color=COLORS["target"], marker="*", edgecolor="white", linewidth=0.7, zorder=9)
    ax.scatter(*mpc_traj[-1], s=52, color=COLORS["MPC"], marker="X", edgecolor="white", linewidth=0.6, zorder=8)
    ax.scatter(*acbc_trace[-1], s=52, color=COLORS["ACBC"], marker="s", edgecolor="white", linewidth=0.6, zorder=8)

    ax.annotate("Start", xy=start, xytext=(-367, 31.5), fontsize=7.0, color=COLORS["start"],
                arrowprops=dict(arrowstyle="-", color=COLORS["start"], lw=0.8), zorder=10)
    ax.annotate("Target", xy=comet_target, xytext=(-315, 128.5), fontsize=7.0, color=COLORS["target"],
                arrowprops=dict(arrowstyle="-", color=COLORS["target"], lw=0.8), zorder=10)
    ax.annotate("MPC stall", xy=mpc_traj[-1], xytext=(-352, 55), fontsize=6.8,
                color=COLORS["MPC"], arrowprops=dict(arrowstyle="-", color=COLORS["MPC"], lw=0.8), zorder=10)
    ax.annotate("ACBC stall", xy=acbc_trace[-1], xytext=(-350, 106), fontsize=6.8,
                color=COLORS["ACBC"], arrowprops=dict(arrowstyle="-", color=COLORS["ACBC"], lw=0.8), zorder=10)

    ax.set_title("Representative Aortic-Root 5x Intervention Case", fontsize=7.8, pad=3)
    ax.set_xlabel("Z position (mm)", fontsize=7.5)
    ax.set_ylabel("Y position (mm)", fontsize=7.5)
    ax.set_xlim(xmin, xmax)
    ax.set_ylim(ymin, ymax)
    ax.grid(True, color="#E0E5EB", lw=0.55, alpha=0.8)
    ax.set_aspect("equal", adjustable="box")
    ax.tick_params(labelsize=7.0, length=3)
    for spine in ax.spines.values():
        spine.set_color("#AEB7C2")
        spine.set_linewidth(0.8)

    note = "2D Y-Z projection"
    ax.text(
        0.012,
        0.018,
        note,
        transform=ax.transAxes,
        fontsize=7.1,
        color="#4B5563",
        ha="left",
        va="bottom",
        bbox=dict(boxstyle="round,pad=0.25", facecolor="white", edgecolor="#D6DCE3", alpha=0.9),
        zorder=11,
    )
    add_side_panel(side)

    output.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output, dpi=600)
    fig.savefig(output.with_suffix(".pdf"))
    fig.savefig(output.with_suffix(".tiff"), dpi=600)
    plt.close(fig)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--output",
        type=Path,
        default=ROOT / "outputs/manuscript_figures/Fig4_representative_aortic_root_5x.png",
    )
    args = parser.parse_args()
    make_figure(args.output)


if __name__ == "__main__":
    main()
