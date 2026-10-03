#!/usr/bin/env python3

import argparse
from pathlib import Path
import math
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.ticker import PercentFormatter
from matplotlib.gridspec import GridSpec

# -----------------------------
# Global style
# -----------------------------
plt.rcParams.update({
    "figure.dpi": 160,
    "savefig.dpi": 400,
    "font.size": 11,
    "axes.titlesize": 14,
    "axes.labelsize": 12,
    "xtick.labelsize": 10,
    "ytick.labelsize": 10,
    "legend.fontsize": 10,
    "axes.spines.top": False,
    "axes.spines.right": False,
})

TRIAL_BUDGET_STEPS = 2000

CONTROLLER_MAP = {
    "comet": "COMET",
    "mpc_tuned": "MPC",
    "acbc_tuned": "ACBC",
    "comet_no_authority_reserve": "No authority reserve",
    "comet_no_commitment_memory": "No commitment memory",
    "comet_no_downstream_value": "No downstream value",
    "comet_no_terminal_precision": "No terminal precision",
    "comet_no_flow_scaled_reserve": "No flow scaled reserve",
}

MAIN_CONTROLLER_ORDER = ["COMET", "MPC", "ACBC"]
ABLATION_ORDER = [
    "COMET",
    "No authority reserve",
    "No commitment memory",
    "No downstream value",
    "No flow scaled reserve",
    "No terminal precision",
]

# Publication-friendly colors
COLOR_MAP = {
    "COMET": "#1f77b4",
    "MPC": "#ff7f0e",
    "ACBC": "#2ca02c",
    "No authority reserve": "#7f7f7f",
    "No commitment memory": "#9467bd",
    "No downstream value": "#8c564b",
    "No flow scaled reserve": "#bcbd22",
    "No terminal precision": "#d62728",
}

VESSEL_MAP = {
    "cm_corrected": "Abdominal aortic",
    "aorta": "Abdominal aortic",
    "m3pro_aorta": "Abdominal aortic",
    "40Left85Right": "Aortic root branching",
    "m3pro_40Left85Right": "Aortic root branching",
    "pulmonary_0082_H_PULM_H": "Pulmonary arterial",
    "pulmonary": "Pulmonary arterial",
}


def find_col(df, candidates, required=True):
    for c in candidates:
        if c in df.columns:
            return c
    if required:
        raise ValueError(f"Could not find any of these columns: {candidates}")
    return None


def clean_vessel_name(value):
    s = str(value)
    for key, label in VESSEL_MAP.items():
        if key in s:
            return label
    return s


def clean_controller_name(value):
    s = str(value)
    return CONTROLLER_MAP.get(s, s)


def clean_df(df):
    controller_col = find_col(df, ["controller_key", "controller", "algorithm"])
    flow_col = find_col(df, ["flow_scale", "flow"])
    success_col = find_col(df, ["reached_target", "success"])
    error_col = find_col(df, ["final_error_mm", "final_target_error_mm", "distance_mm"])
    steps_col = find_col(df, ["steps_taken", "steps", "num_steps"])

    vessel_col = find_col(df, ["vessel_key", "vascular_model", "vessel_model", "network", "run_id"], required=False)
    pair_col = find_col(df, ["pair_id", "route_id", "trial_id"], required=False)

    out = df.copy()
    out["controller_key_clean"] = out[controller_col].astype(str)
    out["controller_label"] = out["controller_key_clean"].map(clean_controller_name)
    out["flow_scale"] = pd.to_numeric(out[flow_col], errors="coerce")
    out["reached_target"] = out[success_col].astype(bool)
    out["final_error_mm"] = pd.to_numeric(out[error_col], errors="coerce")
    out["steps_taken"] = pd.to_numeric(out[steps_col], errors="coerce")

    if vessel_col is not None:
        out["vessel_label"] = out[vessel_col].map(clean_vessel_name)
    else:
        out["vessel_label"] = "Unknown"

    if pair_col is not None:
        out["pair_id"] = out[pair_col].astype(str)
    else:
        out["pair_id"] = ""

    out["reported_steps"] = np.where(out["reached_target"], out["steps_taken"], TRIAL_BUDGET_STEPS)

    # Optional physical validity columns
    optional_numeric = [
        "max_current_A",
        "max_power_W",
        "max_voltage_V",
        "max_coil_temp_C",
        "max_reynolds",
    ]
    for c in optional_numeric:
        if c in out.columns:
            out[c] = pd.to_numeric(out[c], errors="coerce")

    return out


def summarize_controller(df):
    g = df.groupby("controller_label", dropna=False)
    summary = g.agg(
        Trials=("controller_label", "size"),
        Successes=("reached_target", "sum"),
        Success_Rate=("reached_target", "mean"),
        Median_Final_Target_Error_mm=("final_error_mm", "median"),
        Max_Final_Target_Error_mm=("final_error_mm", "max"),
        Mean_Steps=("reported_steps", "mean"),
    ).reset_index()

    summary["Success"] = summary["Successes"].astype(int).astype(str) + "/" + summary["Trials"].astype(int).astype(str)
    summary["Success_Rate"] = 100.0 * summary["Success_Rate"]

    return summary[[
        "controller_label",
        "Trials",
        "Success",
        "Success_Rate",
        "Median_Final_Target_Error_mm",
        "Max_Final_Target_Error_mm",
        "Mean_Steps",
    ]]


def save_table_csv(df, path):
    path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(path, index=False)


def style_axes(ax, ygrid=True):
    if ygrid:
        ax.grid(True, axis="y", alpha=0.25, linewidth=0.8)
    else:
        ax.grid(False)


def fig01_main_controller_summary(df, outdir):
    main = df[df["controller_label"].isin(MAIN_CONTROLLER_ORDER)].copy()
    summary = summarize_controller(main)
    summary["controller_label"] = pd.Categorical(summary["controller_label"], MAIN_CONTROLLER_ORDER, ordered=True)
    summary = summary.sort_values("controller_label")

    fig = plt.figure(figsize=(13.5, 5.3))
    gs = GridSpec(1, 2, figure=fig, width_ratios=[1.0, 1.15])

    ax1 = fig.add_subplot(gs[0, 0])
    ax2 = fig.add_subplot(gs[0, 1])

    labels = summary["controller_label"].tolist()
    rates = summary["Success_Rate"].to_numpy()
    means = summary["Mean_Steps"].to_numpy()
    xs = np.arange(len(labels))
    colors = [COLOR_MAP[l] for l in labels]

    # Left: success rates
    bars1 = ax1.bar(xs, rates, color=colors, width=0.65)
    ax1.set_ylim(0, 105)
    ax1.set_ylabel("Success rate (%)")
    ax1.set_title("Overall controller success rate")
    ax1.set_xticks(xs)
    ax1.set_xticklabels(labels)
    ax1.yaxis.set_major_formatter(PercentFormatter())
    style_axes(ax1, ygrid=True)
    for bar, success_text, rate in zip(bars1, summary["Success"], rates):
        ax1.text(bar.get_x() + bar.get_width()/2, rate + 2.0, f"{success_text}\n{rate:.1f}%",
                 ha="center", va="bottom", fontsize=10)

    # Right: mean steps with failed trials counted as 2000
    bars2 = ax2.bar(xs, means, color=colors, width=0.65)
    ax2.set_ylabel("Mean steps")
    ax2.set_title("Mean steps (failed trials counted as 2000)")
    ax2.set_xticks(xs)
    ax2.set_xticklabels(labels)
    ax2.set_ylim(0, max(means) * 1.18)
    style_axes(ax2, ygrid=True)
    for bar, val in zip(bars2, means):
        ax2.text(bar.get_x() + bar.get_width()/2, val + max(means)*0.02, f"{val:.1f}",
                 ha="center", va="bottom", fontsize=10)

    fig.suptitle("Figure 1. Main controller comparison across the full benchmark", y=1.02, fontsize=15)
    fig.tight_layout()
    fig.savefig(outdir / "fig01_main_controller_summary.png", bbox_inches="tight")
    fig.savefig(outdir / "fig01_main_controller_summary.pdf", bbox_inches="tight")
    plt.close(fig)


def jitter(n, scale=0.07, seed=0):
    rng = np.random.default_rng(seed)
    return rng.uniform(-scale, scale, n)


def fig02_final_error_distribution(df, outdir):
    main = df[df["controller_label"].isin(MAIN_CONTROLLER_ORDER)].copy()

    fig = plt.figure(figsize=(13.5, 5.5))
    gs = GridSpec(1, 2, figure=fig, width_ratios=[1.0, 1.0])

    subsets = [
        ("All 45 cases per controller", main),
        ("5x flow only", main[np.isclose(main["flow_scale"], 5.0)]),
    ]

    for idx, (title, sub) in enumerate(subsets):
        ax = fig.add_subplot(gs[0, idx])

        data = [sub.loc[sub["controller_label"] == c, "final_error_mm"].dropna().to_numpy()
                for c in MAIN_CONTROLLER_ORDER]

        bp = ax.boxplot(
            data,
            patch_artist=True,
            widths=0.55,
            showfliers=False,
            medianprops=dict(color="black", linewidth=1.5),
            whiskerprops=dict(linewidth=1.0),
            capprops=dict(linewidth=1.0),
            boxprops=dict(linewidth=1.0),
        )

        for patch, c in zip(bp["boxes"], MAIN_CONTROLLER_ORDER):
            patch.set_facecolor(COLOR_MAP[c])
            patch.set_alpha(0.35)

        for i, c in enumerate(MAIN_CONTROLLER_ORDER, start=1):
            vals = sub.loc[sub["controller_label"] == c, "final_error_mm"].dropna().to_numpy()
            if len(vals) > 0:
                ax.scatter(
                    np.full(len(vals), i) + jitter(len(vals), seed=i),
                    vals,
                    s=18,
                    alpha=0.55,
                    color=COLOR_MAP[c],
                    edgecolors="none",
                )
                med = np.median(vals)
                mx = np.max(vals)
                ax.text(i + 0.22, med, f"median {med:.3f} mm", fontsize=9, va="center")
                ax.text(i + 0.22, mx, f"max {mx:.3f} mm", fontsize=9, va="center")

        ax.set_yscale("log")
        ax.set_ylabel("Final target error (mm, log scale)")
        ax.set_xticks([1, 2, 3])
        ax.set_xticklabels(MAIN_CONTROLLER_ORDER)
        ax.set_title(title)
        style_axes(ax, ygrid=True)

    fig.suptitle("Figure 2. Final target error distribution by controller", y=1.02, fontsize=15)
    fig.tight_layout()
    fig.savefig(outdir / "fig02_final_error_distribution.png", bbox_inches="tight")
    fig.savefig(outdir / "fig02_final_error_distribution.pdf", bbox_inches="tight")
    plt.close(fig)


def fig03_flow5_by_vessel(df, outdir):
    main = df[df["controller_label"].isin(MAIN_CONTROLLER_ORDER)].copy()
    flow5 = main[np.isclose(main["flow_scale"], 5.0)].copy()

    vessel_order = ["Abdominal aortic", "Aortic root branching", "Pulmonary arterial"]

    success = flow5.pivot_table(
        index="vessel_label", columns="controller_label", values="reached_target", aggfunc="mean"
    ).reindex(index=vessel_order, columns=MAIN_CONTROLLER_ORDER)

    median_error = flow5.pivot_table(
        index="vessel_label", columns="controller_label", values="final_error_mm", aggfunc="median"
    ).reindex(index=vessel_order, columns=MAIN_CONTROLLER_ORDER)

    mean_steps = flow5.pivot_table(
        index="vessel_label", columns="controller_label", values="reported_steps", aggfunc="mean"
    ).reindex(index=vessel_order, columns=MAIN_CONTROLLER_ORDER)

    fig = plt.figure(figsize=(15.5, 5.8))
    gs = GridSpec(1, 3, figure=fig, width_ratios=[1.0, 1.0, 1.05])

    panels = [
        ("Success rate at 5x flow", success * 100.0, "Success rate (%)", "YlGnBu", "{:.1f}%"),
        ("Median final error at 5x flow", median_error, "Median error (mm)", "magma_r", "{:.3f}"),
        ("Mean steps at 5x flow", mean_steps, "Mean steps", "viridis", "{:.1f}"),
    ]

    for i, (title, mat, cbar_label, cmap, fmt) in enumerate(panels):
        ax = fig.add_subplot(gs[0, i])
        arr = mat.to_numpy(dtype=float)
        im = ax.imshow(arr, aspect="auto", cmap=cmap)

        ax.set_xticks(np.arange(len(MAIN_CONTROLLER_ORDER)))
        ax.set_xticklabels(MAIN_CONTROLLER_ORDER, rotation=0)
        ax.set_yticks(np.arange(len(vessel_order)))
        ax.set_yticklabels(vessel_order)
        ax.set_title(title, pad=12)

        for r in range(arr.shape[0]):
            for c in range(arr.shape[1]):
                val = arr[r, c]
                if np.isfinite(val):
                    ax.text(c, r, fmt.format(val), ha="center", va="center", color="black", fontsize=10)

        cbar = fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
        cbar.set_label(cbar_label)

    fig.suptitle("Figure 3. Five times flow performance by vascular model", y=1.02, fontsize=15)
    fig.tight_layout()
    fig.savefig(outdir / "fig03_flow5_by_vessel.png", bbox_inches="tight")
    fig.savefig(outdir / "fig03_flow5_by_vessel.pdf", bbox_inches="tight")
    plt.close(fig)


def fig04_representative_case(df, outdir):
    main = df[df["controller_label"].isin(MAIN_CONTROLLER_ORDER)].copy()

    subset = main[
        (main["vessel_label"] == "Aortic root branching") &
        (np.isclose(main["flow_scale"], 5.0)) &
        (main["pair_id"] == "canonical_003_publication_start_to_random")
    ].copy()

    if subset.empty:
        print("Skipping fig04_representative_case: representative canonical route not found.")
        return

    subset["controller_label"] = pd.Categorical(subset["controller_label"], MAIN_CONTROLLER_ORDER, ordered=True)
    subset = subset.sort_values("controller_label")

    labels = subset["controller_label"].tolist()
    errors = subset["final_error_mm"].to_numpy()
    steps = subset["reported_steps"].to_numpy()
    success = subset["reached_target"].to_numpy()

    fig = plt.figure(figsize=(12.8, 5.2))
    gs = GridSpec(1, 2, figure=fig, width_ratios=[1.0, 1.0])

    ax1 = fig.add_subplot(gs[0, 0])
    ax2 = fig.add_subplot(gs[0, 1])

    xs = np.arange(len(labels))
    colors = [COLOR_MAP[l] for l in labels]

    bars1 = ax1.bar(xs, errors, color=colors, width=0.65)
    ax1.set_ylabel("Final target error (mm)")
    ax1.set_title("Representative canonical route\nFinal target error")
    ax1.set_xticks(xs)
    ax1.set_xticklabels(labels)
    style_axes(ax1, ygrid=True)
    for bar, val, succ in zip(bars1, errors, success):
        tag = "success" if succ else "failed"
        ax1.text(bar.get_x() + bar.get_width()/2, val + max(errors)*0.03, f"{val:.3f}\n{tag}",
                 ha="center", va="bottom", fontsize=10)

    bars2 = ax2.bar(xs, steps, color=colors, width=0.65)
    ax2.set_ylabel("Steps")
    ax2.set_title("Representative canonical route\nSteps")
    ax2.set_xticks(xs)
    ax2.set_xticklabels(labels)
    ax2.set_ylim(0, max(steps) * 1.12)
    style_axes(ax2, ygrid=True)
    for bar, val in zip(bars2, steps):
        ax2.text(bar.get_x() + bar.get_width()/2, val + max(steps)*0.02, f"{val:.0f}",
                 ha="center", va="bottom", fontsize=10)

    fig.suptitle(
        "Figure 4. Representative aortic root branching 5x canonical route outcome",
        y=1.03,
        fontsize=15
    )
    fig.tight_layout()
    fig.savefig(outdir / "fig04_representative_route_case.png", bbox_inches="tight")
    fig.savefig(outdir / "fig04_representative_route_case.pdf", bbox_inches="tight")
    plt.close(fig)


def fig05_ablation(df, outdir):
    ablation_keys = [
        "comet",
        "comet_no_authority_reserve",
        "comet_no_commitment_memory",
        "comet_no_downstream_value",
        "comet_no_flow_scaled_reserve",
        "comet_no_terminal_precision",
    ]
    sub = df[df["controller_key_clean"].isin(ablation_keys)].copy()

    summary = summarize_controller(sub)
    summary["controller_label"] = pd.Categorical(summary["controller_label"], ABLATION_ORDER, ordered=True)
    summary = summary.sort_values("controller_label")

    labels = summary["controller_label"].tolist()
    medians = summary["Median_Final_Target_Error_mm"].to_numpy()
    maxes = summary["Max_Final_Target_Error_mm"].to_numpy()
    successes = summary["Success"].tolist()

    fig = plt.figure(figsize=(12.5, 6.2))
    ax = fig.add_subplot(111)

    ys = np.arange(len(labels))
    colors = [COLOR_MAP.get(l, "#7f7f7f") for l in labels]
    bars = ax.barh(ys, medians, color=colors, alpha=0.85)

    ax.scatter(maxes, ys, marker="D", s=55, color="black", label="Maximum error")
    ax.set_yticks(ys)
    ax.set_yticklabels(labels)
    ax.invert_yaxis()
    ax.set_xlabel("Final target error (mm)")
    ax.set_title("Figure 5. COMET ablation comparison")
    style_axes(ax, ygrid=False)
    ax.grid(True, axis="x", alpha=0.25)

    for y, med, mx, succ in zip(ys, medians, maxes, successes):
        ax.text(med + max(medians) * 0.08 + 0.003, y, f"median {med:.3f} mm | max {mx:.3f} mm | {succ}",
                va="center", fontsize=9)

    ax.legend(loc="lower right")
    fig.tight_layout()
    fig.savefig(outdir / "fig05_comet_ablation.png", bbox_inches="tight")
    fig.savefig(outdir / "fig05_comet_ablation.pdf", bbox_inches="tight")
    plt.close(fig)


def fig06_physical_validity(df, outdir):
    needed = ["max_current_A", "max_power_W", "max_voltage_V", "max_coil_temp_C"]
    if not all(c in df.columns for c in needed):
        print("Skipping fig06_physical_validity: required hardware columns not present.")
        return

    metrics = [
        ("Current (A)", "max_current_A", 13.104),
        ("Power (W)", "max_power_W", 250.0),
        ("Voltage (V)", "max_voltage_V", 19.2),
        ("Coil temp (°C)", "max_coil_temp_C", 25.5),
    ]

    medians = [df[col].median() for _, col, _ in metrics]
    maxima = [df[col].max() for _, col, _ in metrics]
    limits = [lim for _, _, lim in metrics]
    labels = [lab for lab, _, _ in metrics]

    fig = plt.figure(figsize=(13.5, 5.5))
    gs = GridSpec(1, 2, figure=fig, width_ratios=[1.1, 0.9])

    ax1 = fig.add_subplot(gs[0, 0])
    x = np.arange(len(labels))
    w = 0.26

    ax1.bar(x - w, medians, width=w, label="Median observed", alpha=0.8)
    ax1.bar(x, maxima, width=w, label="Maximum observed", alpha=0.8)
    ax1.bar(x + w, limits, width=w, label="Constraint / cap", alpha=0.8)
    ax1.set_xticks(x)
    ax1.set_xticklabels(labels, rotation=15)
    ax1.set_title("Hardware validity metrics")
    style_axes(ax1, ygrid=True)
    ax1.legend()

    # Reynolds panel
    ax2 = fig.add_subplot(gs[0, 1])
    if "max_reynolds" in df.columns:
        vals = df["max_reynolds"].dropna().to_numpy()
        ax2.hist(vals, bins=14, alpha=0.8)
        ax2.axvline(np.median(vals), linestyle="--", linewidth=1.5, label=f"Median = {np.median(vals):.3f}")
        ax2.axvline(np.max(vals), linestyle=":", linewidth=1.8, label=f"Max = {np.max(vals):.3f}")
        ax2.set_title("Maximum Reynolds number distribution")
        ax2.set_xlabel("Maximum Reynolds number")
        ax2.set_ylabel("Trial count")
        style_axes(ax2, ygrid=True)
        ax2.legend()
    else:
        ax2.axis("off")
        ax2.text(0.5, 0.5, "max_reynolds column not available", ha="center", va="center")

    fig.suptitle("Figure 6. Physical validity and hardware constraint summary", y=1.02, fontsize=15)
    fig.tight_layout()
    fig.savefig(outdir / "fig06_physical_validity.png", bbox_inches="tight")
    fig.savefig(outdir / "fig06_physical_validity.pdf", bbox_inches="tight")
    plt.close(fig)


def write_tables(df, outdir):
    table_dir = outdir / "tables"
    table_dir.mkdir(parents=True, exist_ok=True)

    main = df[df["controller_label"].isin(MAIN_CONTROLLER_ORDER)].copy()

    # Table I
    t1 = summarize_controller(main)
    t1["controller_label"] = pd.Categorical(t1["controller_label"], MAIN_CONTROLLER_ORDER, ordered=True)
    t1 = t1.sort_values("controller_label")
    save_table_csv(t1, table_dir / "table01_main_controller_summary.csv")

    # Table II
    t2 = summarize_controller(main[np.isclose(main["flow_scale"], 5.0)])
    t2["controller_label"] = pd.Categorical(t2["controller_label"], MAIN_CONTROLLER_ORDER, ordered=True)
    t2 = t2.sort_values("controller_label")
    save_table_csv(t2, table_dir / "table02_controller_performance_at_5x.csv")

    # Table III
    t3 = summarize_controller(
        main[(main["vessel_label"] == "Aortic root branching") & (np.isclose(main["flow_scale"], 5.0))]
    )
    t3["controller_label"] = pd.Categorical(t3["controller_label"], MAIN_CONTROLLER_ORDER, ordered=True)
    t3 = t3.sort_values("controller_label")
    save_table_csv(t3, table_dir / "table03_aortic_root_5x.csv")

    # Table IV
    ablation_keys = [
        "comet",
        "comet_no_authority_reserve",
        "comet_no_commitment_memory",
        "comet_no_downstream_value",
        "comet_no_flow_scaled_reserve",
        "comet_no_terminal_precision",
    ]
    t4 = summarize_controller(df[df["controller_key_clean"].isin(ablation_keys)].copy())
    t4["controller_label"] = pd.Categorical(t4["controller_label"], ABLATION_ORDER, ordered=True)
    t4 = t4.sort_values("controller_label")
    save_table_csv(t4, table_dir / "table04_comet_ablation_summary.csv")

    # Table V optional physical validity
    phys_cols = ["max_current_A", "max_power_W", "max_voltage_V", "max_coil_temp_C", "max_reynolds"]
    have_phys = [c for c in phys_cols if c in df.columns]
    if have_phys:
        rows = []
        for c in have_phys:
            rows.append({
                "Metric": c,
                "Median": float(pd.to_numeric(df[c], errors="coerce").median()),
                "Maximum": float(pd.to_numeric(df[c], errors="coerce").max()),
            })
        t5 = pd.DataFrame(rows)
        save_table_csv(t5, table_dir / "table05_physical_validity_summary.csv")


def main():
    parser = argparse.ArgumentParser(description="Generate publication figures from the final DynaMag CSV.")
    parser.add_argument(
        "--csv",
        type=str,
        default="dynamag_final_selected_results_500um_360_ideal.csv",
        help="Path to the final selected CSV. Default assumes the file is in the repo root.",
    )
    parser.add_argument(
        "--outdir",
        type=str,
        default="paper_results_figures",
        help="Directory to save figures and tables.",
    )
    args = parser.parse_args()

    csv_path = Path(args.csv)
    outdir = Path(args.outdir)
    outdir.mkdir(parents=True, exist_ok=True)

    if not csv_path.exists():
        raise FileNotFoundError(f"CSV not found: {csv_path}")

    raw = pd.read_csv(csv_path)
    df = clean_df(raw)

    # Generate figures
    fig01_main_controller_summary(df, outdir)
    fig02_final_error_distribution(df, outdir)
    fig03_flow5_by_vessel(df, outdir)
    fig04_representative_case(df, outdir)
    fig05_ablation(df, outdir)
    fig06_physical_validity(df, outdir)

    # Write tables
    write_tables(df, outdir)

    print(f"\nSaved figures and tables to: {outdir.resolve()}")
    print("\nFiles generated:")
    for p in sorted(outdir.rglob("*")):
        if p.is_file():
            print(" -", p)


if __name__ == "__main__":
    main()
