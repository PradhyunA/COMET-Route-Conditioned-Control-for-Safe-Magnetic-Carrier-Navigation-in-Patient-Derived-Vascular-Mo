#!/usr/bin/env python3

from pathlib import Path
import argparse
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

plt.rcParams.update({
    "figure.dpi": 160,
    "savefig.dpi": 400,
    "font.size": 11,
    "axes.titlesize": 14,
    "axes.labelsize": 12,
    "xtick.labelsize": 11,
    "ytick.labelsize": 10,
    "legend.fontsize": 10,
    "axes.spines.top": False,
    "axes.spines.right": False,
})

CONTROLLER_MAP = {
    "comet": "COMET",
    "mpc_tuned": "MPC",
    "acbc_tuned": "ACBC",
}

COLOR_MAP = {
    "COMET": "#1f77b4",
    "MPC": "#ff7f0e",
    "ACBC": "#2ca02c",
}

ORDER = ["COMET", "MPC", "ACBC"]


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--csv",
        default="dynamag_final_selected_results_500um_360_ideal.csv",
        help="Final 360 row CSV file.",
    )
    parser.add_argument(
        "--outdir",
        default="finaloutputs/paper_results_figures",
        help="Output directory.",
    )
    args = parser.parse_args()

    csv_path = Path(args.csv)
    outdir = Path(args.outdir)
    outdir.mkdir(parents=True, exist_ok=True)

    df = pd.read_csv(csv_path)

    required = ["controller_key", "flow_scale", "final_error_mm"]
    missing = [c for c in required if c not in df.columns]
    if missing:
        raise ValueError(f"Missing required columns: {missing}")

    df = df[df["controller_key"].isin(CONTROLLER_MAP.keys())].copy()
    df["controller_label"] = df["controller_key"].map(CONTROLLER_MAP)
    df["flow_scale"] = pd.to_numeric(df["flow_scale"], errors="coerce")
    df["final_error_mm"] = pd.to_numeric(df["final_error_mm"], errors="coerce")

    panels = [
        ("All 45 cases per controller", df),
        ("5x flow only", df[np.isclose(df["flow_scale"], 5.0)].copy()),
    ]

    fig, axes = plt.subplots(1, 2, figsize=(14.5, 6.2), constrained_layout=True)
    fig.suptitle(
        "Final target error distribution by controller",
        fontsize=18,
        y=1.03,
        fontweight="bold",
    )

    for panel_idx, (ax, (title, subdf)) in enumerate(zip(axes, panels)):
        data = [
            subdf.loc[subdf["controller_label"] == label, "final_error_mm"]
            .dropna()
            .to_numpy()
            for label in ORDER
        ]

        box = ax.boxplot(
            data,
            positions=np.arange(1, len(ORDER) + 1),
            widths=0.55,
            patch_artist=True,
            showfliers=False,
            medianprops=dict(color="black", linewidth=1.6),
            whiskerprops=dict(color="black", linewidth=1.1),
            capprops=dict(color="black", linewidth=1.1),
            boxprops=dict(linewidth=1.1),
        )

        for patch, label in zip(box["boxes"], ORDER):
            patch.set_facecolor(COLOR_MAP[label])
            patch.set_alpha(0.32)
            patch.set_edgecolor("black")

        # Raw points with light jitter. No median/max labels.
        for i, label in enumerate(ORDER, start=1):
            vals = subdf.loc[subdf["controller_label"] == label, "final_error_mm"].dropna().to_numpy()
            rng = np.random.default_rng(500 + panel_idx * 10 + i)
            x = rng.normal(i, 0.045, size=len(vals))

            ax.scatter(
                x,
                vals,
                s=24,
                alpha=0.62,
                color=COLOR_MAP[label],
                edgecolors="white",
                linewidths=0.35,
                zorder=3,
            )

        ax.set_title(title, fontsize=15, pad=10)
        ax.set_xticks(np.arange(1, len(ORDER) + 1))
        ax.set_xticklabels(ORDER)
        ax.set_yscale("log")
        ax.set_ylabel("Final target error (mm, log scale)")
        ax.grid(True, axis="y", which="both", alpha=0.22, linewidth=0.8)

        # Success tolerance reference line.
        ax.axhline(
            1.5,
            color="gray",
            linestyle="--",
            linewidth=1.2,
            alpha=0.7,
        )
        ax.text(
            0.02,
            0.92,
            "1.5 mm success tolerance",
            transform=ax.transAxes,
            fontsize=9,
            color="dimgray",
            ha="left",
            va="center",
            bbox=dict(facecolor="white", edgecolor="none", alpha=0.75, pad=2),
        )

        positive = subdf.loc[subdf["final_error_mm"] > 0, "final_error_mm"]
        if len(positive) > 0:
            ax.set_ylim(max(positive.min() * 0.45, 1e-5), positive.max() * 2.4)

    png = outdir / "fig02_final_error_distribution_clean.png"
    pdf = outdir / "fig02_final_error_distribution_clean.pdf"

    fig.savefig(png, bbox_inches="tight")
    fig.savefig(pdf, bbox_inches="tight")
    plt.close(fig)

    print("Saved:")
    print(" -", png)
    print(" -", pdf)


if __name__ == "__main__":
    main()
