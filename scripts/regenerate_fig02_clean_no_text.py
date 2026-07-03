#!/usr/bin/env python3

from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

CSV = Path("dynamag_final_selected_results_500um_360_ideal.csv")
OUTDIR = Path("finaloutputs/paper_results_figures")
OUTDIR.mkdir(parents=True, exist_ok=True)

df = pd.read_csv(CSV)

controller_map = {
    "comet": "COMET",
    "mpc_tuned": "MPC",
    "acbc_tuned": "ACBC",
}

colors = {
    "COMET": "#1f77b4",
    "MPC": "#ff7f0e",
    "ACBC": "#2ca02c",
}

order = ["COMET", "MPC", "ACBC"]

df = df[df["controller_key"].isin(controller_map.keys())].copy()
df["controller"] = df["controller_key"].map(controller_map)
df["flow_scale"] = pd.to_numeric(df["flow_scale"], errors="coerce")
df["final_error_mm"] = pd.to_numeric(df["final_error_mm"], errors="coerce")

fig, axes = plt.subplots(1, 2, figsize=(14, 5.6), constrained_layout=True)

panels = [
    ("All 45 cases per controller", df),
    ("5x flow only", df[df["flow_scale"] == 5.0].copy()),
]

for ax, (title, sub) in zip(axes, panels):
    data = [
        sub.loc[sub["controller"] == c, "final_error_mm"].dropna().values
        for c in order
    ]

    bp = ax.boxplot(
        data,
        patch_artist=True,
        widths=0.55,
        showfliers=False,
        medianprops=dict(color="black", linewidth=1.5),
        whiskerprops=dict(color="black", linewidth=1.0),
        capprops=dict(color="black", linewidth=1.0),
        boxprops=dict(color="black", linewidth=1.0),
    )

    for patch, c in zip(bp["boxes"], order):
        patch.set_facecolor(colors[c])
        patch.set_alpha(0.32)

    rng = np.random.default_rng(42)
    for i, c in enumerate(order, start=1):
        vals = sub.loc[sub["controller"] == c, "final_error_mm"].dropna().values
        x = rng.normal(i, 0.045, len(vals))
        ax.scatter(
            x,
            vals,
            s=22,
            color=colors[c],
            alpha=0.60,
            edgecolors="none",
            zorder=3,
        )

    ax.set_title(title, fontsize=15, pad=10)
    ax.set_xticks([1, 2, 3])
    ax.set_xticklabels(order, fontsize=12)
    ax.set_yscale("log")
    ax.set_ylabel("Final target error (mm, log scale)", fontsize=12)

    # Clean grid: major lines only, no dense minor log stripes
    ax.grid(True, axis="y", which="major", alpha=0.22)
    ax.grid(False, axis="y", which="minor")

    vals = sub["final_error_mm"].dropna()
    vals = vals[vals > 0]
    ax.set_ylim(max(vals.min() * 0.5, 1e-5), vals.max() * 2.0)

fig.suptitle("Final target error distribution by controller", fontsize=18, y=1.03)

fig.savefig(OUTDIR / "fig02_final_error_distribution.png", dpi=400, bbox_inches="tight")
fig.savefig(OUTDIR / "fig02_final_error_distribution.pdf", bbox_inches="tight")

plt.close(fig)

print("Saved clean Figure 2:")
print(OUTDIR / "fig02_final_error_distribution.png")
print(OUTDIR / "fig02_final_error_distribution.pdf")
