#!/usr/bin/env python3
"""
Generate final paper figures from the cleaned DynaMag paper CSV.

Input:
    dynamag_paper_discussion_results_500um.csv

Outputs:
    finaloutputs/final_paper_figures/
        fig_success_rate_by_algorithm_vessel_flow.png/pdf
        fig_final_error_by_algorithm.png/pdf
        fig_steps_by_algorithm.png/pdf
        table_success_rate_by_algorithm_vessel_flow.csv
        table_summary_by_algorithm.csv
"""

from pathlib import Path
import argparse
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


# -----------------------------
# Display labels
# -----------------------------

CONTROLLER_LABELS = {
    "comet": "COMET",
    "comet_no_authority_reserve": "COMET - no authority reserve",
    "comet_no_commitment_memory": "COMET - no commitment memory",
    "comet_no_downstream_value": "COMET - no downstream value",
    "comet_no_terminal_precision": "COMET - no terminal precision",
    "comet_no_flow_scaled_reserve": "COMET - no flow-scaled reserve",
    "mpc_tuned": "MPC",
    "acbc_tuned": "aCBC",
}

MAIN_ALGORITHMS = ["comet", "mpc_tuned", "acbc_tuned"]
ABLATION_ALGORITHMS = [
    "comet",
    "comet_no_authority_reserve",
    "comet_no_commitment_memory",
    "comet_no_downstream_value",
    "comet_no_terminal_precision",
    "comet_no_flow_scaled_reserve",
]

VESSEL_ORDER = [
    "cm_corrected",
    "40Left85Right",
    "pulmonary_0082_H_PULM_H",
]

VESSEL_LABELS = {
    "cm_corrected": "Abdominal aortic",
    "40Left85Right": "Aortic-root branching",
    "pulmonary_0082_H_PULM_H": "Pulmonary arterial",
}


def normalize_bool(series: pd.Series) -> pd.Series:
    """Convert mixed True/False strings/bools to bool."""
    if series.dtype == bool:
        return series
    return series.astype(str).str.lower().isin(["true", "1", "yes"])


def clean_df(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()

    if "paper_include" in df.columns:
        df = df[normalize_bool(df["paper_include"])].copy()

    df["reached_target"] = normalize_bool(df["reached_target"])
    df["flow_scale"] = df["flow_scale"].astype(float)

    df["algorithm"] = df["controller_key"].map(CONTROLLER_LABELS).fillna(df["controller_key"])
    df["vascular_model"] = df["vessel_key"].map(VESSEL_LABELS).fillna(df.get("vessel_label", df["vessel_key"]))
    df["flow_label"] = df["flow_scale"].map(lambda x: f"{int(x)}x" if float(x).is_integer() else f"{x:g}x")

    # Route type for possible filtering/summary later.
    df["route_type"] = np.where(df["pair_id"].astype(str).str.startswith("canonical"), "Canonical", "Random")

    return df


def savefig(fig, outdir: Path, name: str):
    outdir.mkdir(parents=True, exist_ok=True)
    fig.savefig(outdir / f"{name}.png", dpi=600, bbox_inches="tight")
    fig.savefig(outdir / f"{name}.pdf", bbox_inches="tight")
    plt.close(fig)


def fig_success_rate(df: pd.DataFrame, outdir: Path):
    """
    Figure 1: success rate by vessel/flow for COMET, MPC, and aCBC.
    Uses only main algorithms.
    """
    sub = df[df["controller_key"].isin(MAIN_ALGORITHMS)].copy()

    summary = (
        sub.groupby(["vessel_key", "vascular_model", "flow_scale", "flow_label", "controller_key", "algorithm"])
        .agg(
            trials=("reached_target", "size"),
            successes=("reached_target", "sum"),
            success_rate=("reached_target", "mean"),
        )
        .reset_index()
    )
    summary["success_rate_percent"] = 100.0 * summary["success_rate"]
    summary.to_csv(outdir / "table_success_rate_by_algorithm_vessel_flow.csv", index=False)

    # Ordered categories: vessel | flow
    categories = []
    for vessel in VESSEL_ORDER:
        for flow in [1.0, 3.0, 5.0]:
            rows = summary[(summary["vessel_key"] == vessel) & (summary["flow_scale"] == flow)]
            if not rows.empty:
                label = f"{VESSEL_LABELS.get(vessel, vessel)}\n{int(flow)}x flow"
                categories.append((vessel, flow, label))

    x = np.arange(len(categories))
    width = 0.25

    fig, ax = plt.subplots(figsize=(11.0, 4.8))

    for i, ctrl in enumerate(MAIN_ALGORITHMS):
        vals = []
        for vessel, flow, _ in categories:
            row = summary[
                (summary["vessel_key"] == vessel)
                & (summary["flow_scale"] == flow)
                & (summary["controller_key"] == ctrl)
            ]
            vals.append(float(row["success_rate_percent"].iloc[0]) if not row.empty else np.nan)

        ax.bar(x + (i - 1) * width, vals, width, label=CONTROLLER_LABELS[ctrl])

    ax.set_ylabel("Target success rate (%)")
    ax.set_ylim(0, 105)
    ax.set_xticks(x)
    ax.set_xticklabels([c[2] for c in categories], rotation=35, ha="right")
    ax.set_title("Target convergence across vascular models and flow conditions")
    ax.legend(frameon=False, ncol=3, loc="upper center", bbox_to_anchor=(0.5, 1.14))
    ax.grid(axis="y", alpha=0.25)

    savefig(fig, outdir, "fig_success_rate_by_algorithm_vessel_flow")


def fig_final_error(df: pd.DataFrame, outdir: Path):
    """
    Figure 2: final distance to target by algorithm.
    Uses main algorithms and COMET ablations.
    Log scale makes MPC/aCBC failures visible without hiding COMET precision.
    """
    sub = df[df["controller_key"].isin(MAIN_ALGORITHMS + ABLATION_ALGORITHMS)].copy()

    order = [
        "comet",
        "comet_no_authority_reserve",
        "comet_no_commitment_memory",
        "comet_no_downstream_value",
        "comet_no_flow_scaled_reserve",
        "comet_no_terminal_precision",
        "mpc_tuned",
        "acbc_tuned",
    ]
    order = [x for x in order if x in set(sub["controller_key"])]

    data = []
    labels = []
    for ctrl in order:
        vals = sub.loc[sub["controller_key"] == ctrl, "final_error_mm"].dropna().astype(float)
        # Avoid log issues if any zero appears.
        vals = np.maximum(vals, 1e-4)
        data.append(vals.values)
        labels.append(CONTROLLER_LABELS.get(ctrl, ctrl))

    table = (
        sub.groupby(["controller_key", "algorithm"])
        .agg(
            trials=("final_error_mm", "size"),
            success_rate=("reached_target", "mean"),
            median_final_error_mm=("final_error_mm", "median"),
            mean_final_error_mm=("final_error_mm", "mean"),
            max_final_error_mm=("final_error_mm", "max"),
        )
        .reset_index()
    )
    table["success_rate_percent"] = 100.0 * table["success_rate"]
    table.to_csv(outdir / "table_summary_by_algorithm.csv", index=False)

    fig, ax = plt.subplots(figsize=(10.5, 4.8))

    bp = ax.boxplot(data, labels=labels, showfliers=True, patch_artist=True)
    for box in bp["boxes"]:
        box.set_alpha(0.65)

    ax.set_yscale("log")
    ax.set_ylabel("Final distance to target (mm, log scale)")
    ax.set_title("Final targeting accuracy by controller")
    ax.grid(axis="y", which="both", alpha=0.25)
    ax.tick_params(axis="x", labelrotation=35)

    # Optional reference line for ~1.5 mm success threshold.
    ax.axhline(1.5, linestyle="--", linewidth=1.0, alpha=0.6)
    ax.text(len(labels) + 0.05, 1.5, "1.5 mm tolerance", va="center", fontsize=8)

    savefig(fig, outdir, "fig_final_error_by_algorithm")


def fig_steps(df: pd.DataFrame, outdir: Path):
    """
    Figure 3: steps to target / efficiency by algorithm.
    Uses main algorithms only because this is primarily a controller comparison.
    """
    sub = df[df["controller_key"].isin(MAIN_ALGORITHMS)].copy()

    order = [x for x in MAIN_ALGORITHMS if x in set(sub["controller_key"])]

    data = []
    labels = []
    for ctrl in order:
        vals = sub.loc[sub["controller_key"] == ctrl, "steps_taken"].dropna().astype(float)
        data.append(vals.values)
        labels.append(CONTROLLER_LABELS.get(ctrl, ctrl))

    table = (
        sub.groupby(["controller_key", "algorithm"])
        .agg(
            trials=("steps_taken", "size"),
            median_steps=("steps_taken", "median"),
            mean_steps=("steps_taken", "mean"),
            max_steps=("steps_taken", "max"),
            success_rate=("reached_target", "mean"),
        )
        .reset_index()
    )
    table["success_rate_percent"] = 100.0 * table["success_rate"]
    table.to_csv(outdir / "table_steps_by_algorithm.csv", index=False)

    fig, ax = plt.subplots(figsize=(7.5, 4.8))

    bp = ax.boxplot(data, labels=labels, showfliers=True, patch_artist=True)
    for box in bp["boxes"]:
        box.set_alpha(0.65)

    ax.set_ylabel("Control steps")
    ax.set_title("Controller efficiency measured by steps taken")
    ax.grid(axis="y", alpha=0.25)

    savefig(fig, outdir, "fig_steps_by_algorithm")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--csv",
        default="finaloutputs/dynamag_paper_discussion_results_500um.csv",
        help="Path to cleaned paper discussion CSV.",
    )
    parser.add_argument(
        "--outdir",
        default="finaloutputs/final_paper_figures",
        help="Directory to save figures.",
    )
    args = parser.parse_args()

    csv_path = Path(args.csv)
    outdir = Path(args.outdir)
    outdir.mkdir(parents=True, exist_ok=True)

    df = pd.read_csv(csv_path)
    df = clean_df(df)

    fig_success_rate(df, outdir)
    fig_final_error(df, outdir)
    fig_steps(df, outdir)

    print(f"Saved publication figures to: {outdir.resolve()}")
    print("Generated:")
    for p in sorted(outdir.iterdir()):
        print(" -", p.name)


if __name__ == "__main__":
    main()
