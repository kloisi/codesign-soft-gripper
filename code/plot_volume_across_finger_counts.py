# plot_volume_across_finger_counts.py
#
# run like: python plot_volume_across_finger_counts.py logs/sweep_vol_fingers_YYYYMMDD_HHMMSS/summary.csv

import os
import argparse

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from matplotlib.patches import Patch


FULL_W = 6.8
HALF_W = 3.3
DPI = 300


def setup_style():
    plt.rcParams.update({
        "figure.facecolor": "white",
        "axes.facecolor": "white",
        "savefig.facecolor": "white",
        "savefig.bbox": "tight",
        "font.family": "serif",
        "font.serif": [
            "Computer Modern Roman",
            "CMU Serif",
            "Latin Modern Roman",
            "DejaVu Serif",
        ],
        "mathtext.fontset": "cm",
        "axes.unicode_minus": False,
        "font.size": 10,
        "axes.labelsize": 10,
        "axes.titlesize": 11,
        "xtick.labelsize": 9,
        "ytick.labelsize": 9,
        "legend.fontsize": 9,
        "axes.spines.top": False,
        "axes.spines.right": False,
        "axes.grid": False,
        "lines.linewidth": 1.8,
    })


def save(fig, out_dir, stem):
    os.makedirs(out_dir, exist_ok=True)
    fig.savefig(os.path.join(out_dir, f"{stem}.png"), dpi=DPI)
    plt.close(fig)


def read_data(csv_path):
    df = pd.read_csv(csv_path)

    if "status" in df.columns:
        df["status"] = df["status"].fillna("").astype(str).str.strip()
        df = df[df["status"] == "ok"].copy()

    numeric_cols = [
        "finger_num",
        "seed",
        "radius_mean",
        "final_loss",
        "avg_force",
        "init_vol_vox",
        "final_vol_vox",
        "delta_vol_vox",
        "reduction_pct",
        "final_over_init",
        "runtime_s",
    ]
    for col in numeric_cols:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce")

    df = df.dropna(subset=["object", "finger_num", "init_vol_vox", "final_vol_vox"]).copy()
    df["finger_num"] = df["finger_num"].astype(int)

    if "reduction_pct" not in df.columns or df["reduction_pct"].isna().all():
        df["reduction_pct"] = 100.0 * (df["init_vol_vox"] - df["final_vol_vox"]) / df["init_vol_vox"]

    if "final_over_init" not in df.columns or df["final_over_init"].isna().all():
        df["final_over_init"] = df["final_vol_vox"] / df["init_vol_vox"]

    return df


def aggregate_by_finger(df):
    rows = []

    for nf, g in df.groupby("finger_num"):
        init_vals = g["init_vol_vox"].dropna().to_numpy()
        final_vals = g["final_vol_vox"].dropna().to_numpy()
        red_vals = g["reduction_pct"].dropna().to_numpy()
        ratio_vals = g["final_over_init"].dropna().to_numpy()
        radius_vals = g["radius_mean"].dropna().to_numpy()
        runtime_vals = g["runtime_s"].dropna().to_numpy()

        row = {
            "finger_num": int(nf),
            "count": int(len(g)),

            "init_mean": np.mean(init_vals),
            "init_std": np.std(init_vals, ddof=1) if len(init_vals) > 1 else np.nan,
            "init_median": np.median(init_vals),
            "init_q25": np.percentile(init_vals, 25),
            "init_q75": np.percentile(init_vals, 75),

            "final_mean": np.mean(final_vals),
            "final_std": np.std(final_vals, ddof=1) if len(final_vals) > 1 else np.nan,
            "final_median": np.median(final_vals),
            "final_q25": np.percentile(final_vals, 25),
            "final_q75": np.percentile(final_vals, 75),

            "reduction_mean": np.mean(red_vals),
            "reduction_std": np.std(red_vals, ddof=1) if len(red_vals) > 1 else np.nan,
            "reduction_median": np.median(red_vals),
            "reduction_q25": np.percentile(red_vals, 25),
            "reduction_q75": np.percentile(red_vals, 75),

            "ratio_mean": np.mean(ratio_vals),
            "ratio_std": np.std(ratio_vals, ddof=1) if len(ratio_vals) > 1 else np.nan,
            "ratio_median": np.median(ratio_vals),
            "ratio_q25": np.percentile(ratio_vals, 25),
            "ratio_q75": np.percentile(ratio_vals, 75),

            "radius_mean_mean": np.mean(radius_vals) if len(radius_vals) > 0 else np.nan,
            "radius_mean_median": np.median(radius_vals) if len(radius_vals) > 0 else np.nan,

            "runtime_mean": np.mean(runtime_vals) if len(runtime_vals) > 0 else np.nan,
            "runtime_median": np.median(runtime_vals) if len(runtime_vals) > 0 else np.nan,
        }
        rows.append(row)

    summary = pd.DataFrame(rows).sort_values("finger_num").reset_index(drop=True)

    summary["init_gain_abs_vs_prev"] = np.nan
    summary["init_gain_pct_vs_prev"] = np.nan
    summary["final_gain_abs_vs_prev"] = np.nan
    summary["final_gain_pct_vs_prev"] = np.nan
    summary["reduction_gain_abs_vs_prev"] = np.nan
    summary["reduction_gain_pct_vs_prev"] = np.nan

    for i in range(1, len(summary)):
        prev_init = summary.loc[i - 1, "init_median"]
        curr_init = summary.loc[i, "init_median"]

        prev_final = summary.loc[i - 1, "final_median"]
        curr_final = summary.loc[i, "final_median"]

        prev_red = summary.loc[i - 1, "reduction_median"]
        curr_red = summary.loc[i, "reduction_median"]

        summary.loc[i, "init_gain_abs_vs_prev"] = prev_init - curr_init
        summary.loc[i, "final_gain_abs_vs_prev"] = prev_final - curr_final
        summary.loc[i, "reduction_gain_abs_vs_prev"] = curr_red - prev_red

        if np.isfinite(prev_init) and abs(prev_init) > 1e-12:
            summary.loc[i, "init_gain_pct_vs_prev"] = 100.0 * (prev_init - curr_init) / prev_init

        if np.isfinite(prev_final) and abs(prev_final) > 1e-12:
            summary.loc[i, "final_gain_pct_vs_prev"] = 100.0 * (prev_final - curr_final) / prev_final

        if np.isfinite(prev_red) and abs(prev_red) > 1e-12:
            summary.loc[i, "reduction_gain_pct_vs_prev"] = 100.0 * (curr_red - prev_red) / prev_red

    best_init = summary["init_median"].min()
    best_final = summary["final_median"].min()
    best_reduction = summary["reduction_median"].max()

    summary["init_distance_to_best_pct"] = 100.0 * (summary["init_median"] - best_init) / best_init
    summary["final_distance_to_best_pct"] = 100.0 * (summary["final_median"] - best_final) / best_final
    summary["reduction_distance_to_best_pct"] = 100.0 * (best_reduction - summary["reduction_median"]) / best_reduction

    summary["within_5pct_of_best_init"] = summary["init_median"] <= 1.05 * best_init
    summary["within_5pct_of_best_final"] = summary["final_median"] <= 1.05 * best_final
    summary["within_5pct_of_best_reduction"] = summary["reduction_median"] >= 0.95 * best_reduction

    return summary


def per_object_best(df):
    rows = []

    for obj, g in df.groupby("object"):
        g = g.sort_values("finger_num").copy()

        i_best_init = g["init_vol_vox"].idxmin()
        i_best_final = g["final_vol_vox"].idxmin()
        i_best_reduction = g["reduction_pct"].idxmax()

        rows.append({
            "object": obj,
            "best_init_finger_count": int(g.loc[i_best_init, "finger_num"]),
            "best_init_voxvol": float(g.loc[i_best_init, "init_vol_vox"]),
            "best_final_finger_count": int(g.loc[i_best_final, "finger_num"]),
            "best_final_voxvol": float(g.loc[i_best_final, "final_vol_vox"]),
            "best_reduction_finger_count": int(g.loc[i_best_reduction, "finger_num"]),
            "best_reduction_pct": float(g.loc[i_best_reduction, "reduction_pct"]),
        })

    return pd.DataFrame(rows).sort_values("object")


def plot_band(summary, y_med, y_q25, y_q75, ylabel, out_dir, stem):
    x = summary["finger_num"].to_numpy()

    fig, ax = plt.subplots(figsize=(4.8, 4.15))

    band_color = "#AFC4DA"
    line_color = "#1F4E79"

    ax.fill_between(
        x,
        summary[y_q25],
        summary[y_q75],
        color=band_color,
        alpha=0.45,
        linewidth=0,
        zorder=1,
    )
    ax.plot(
        x,
        summary[y_med],
        color=line_color,
        linewidth=2.3,
        marker="o",
        markersize=5.2,
        zorder=2,
    )

    ax.set_xlabel("Number of fingers")
    ax.set_ylabel(ylabel)
    ax.set_xticks(x)
    ax.set_xlim(x.min() - 0.35, x.max() + 0.35)
    ax.grid(axis="y", color="#E6E6E6", linewidth=0.8)

    handles = [
        Line2D([0], [0], color=line_color, lw=2.3, label="Median"),
        Patch(facecolor=band_color, edgecolor="none", alpha=0.45, label="Interquartile range"),
    ]
    ax.legend(handles=handles, frameon=False, loc="best")

    save(fig, out_dir, stem)


def plot_final_spaghetti(df, summary, out_dir):
    fig, ax = plt.subplots(figsize=(FULL_W, 4.0))

    indiv_color = "#D0D5DB"
    med_color = "#1F4E79"

    for _, g in df.groupby("object", sort=True):
        g = g.sort_values("finger_num")
        ax.plot(
            g["finger_num"],
            g["final_vol_vox"],
            color=indiv_color,
            linewidth=1.0,
            alpha=0.9,
            zorder=1,
        )

    ax.plot(
        summary["finger_num"],
        summary["final_median"],
        color=med_color,
        linewidth=2.2,
        marker="o",
        markersize=4.8,
        zorder=2,
    )

    ax.set_xlabel("Number of fingers")
    ax.set_ylabel("Final enclosed cavity volume")
    ax.set_xticks(summary["finger_num"])
    ax.grid(axis="y", color="#E6E6E6", linewidth=0.8)

    handles = [
        Line2D([0], [0], color=indiv_color, lw=1.2, label="Individual corals"),
        Line2D([0], [0], color=med_color, lw=2.2, label="Median"),
    ]
    ax.legend(handles=handles, frameon=False, loc="best")

    save(fig, out_dir, "final_volume_spaghetti")


def plot_incremental_gain(summary, col, ylabel, out_dir, stem):
    d = summary.dropna(subset=[col]).copy()
    if d.empty:
        return

    fig, ax = plt.subplots(figsize=(HALF_W + 1.1, 3.7))

    ax.bar(
        d["finger_num"].astype(str),
        d[col],
        color="#4C78A8",
        width=0.72,
    )

    ax.axhline(0.0, color="#444444", linewidth=0.9)
    ax.set_xlabel("Number of fingers")
    ax.set_ylabel(ylabel)
    ax.grid(axis="y", color="#E6E6E6", linewidth=0.8)

    save(fig, out_dir, stem)


def plot_distance_to_best(summary, col, ylabel, out_dir, stem):
    fig, ax = plt.subplots(figsize=(HALF_W + 1.1, 3.7))

    ax.plot(
        summary["finger_num"],
        summary[col],
        color="#1F4E79",
        marker="o",
        linewidth=2.2,
        markersize=4.8,
    )

    ax.axhline(5.0, color="#888888", linewidth=0.9, linestyle="--")
    ax.set_xlabel("Number of fingers")
    ax.set_ylabel(ylabel)
    ax.set_xticks(summary["finger_num"])
    ax.grid(axis="y", color="#E6E6E6", linewidth=0.8)

    save(fig, out_dir, stem)


def plot_best_count_bar(best_df, col, xlabel, out_dir, stem):
    vals, counts = np.unique(best_df[col].to_numpy(dtype=int), return_counts=True)

    fig, ax = plt.subplots(figsize=(HALF_W + 0.9, 3.6))
    ax.bar(vals.astype(str), counts, color="#4C78A8", width=0.72)
    ax.set_xlabel(xlabel)
    ax.set_ylabel("Number of corals")
    ax.grid(axis="y", color="#E6E6E6", linewidth=0.8)

    save(fig, out_dir, stem)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("csv_path", help="Path to the finger-count sweep CSV.")
    args = parser.parse_args()

    setup_style()

    df = read_data(args.csv_path)
    out_dir = os.path.join(os.path.dirname(args.csv_path), "plots_report")
    os.makedirs(out_dir, exist_ok=True)

    summary = aggregate_by_finger(df)
    best_df = per_object_best(df)

    summary.to_csv(os.path.join(out_dir, "finger_count_summary.csv"), index=False)
    best_df.to_csv(os.path.join(out_dir, "best_finger_count_per_object.csv"), index=False)

    plot_band(
        summary,
        y_med="init_median",
        y_q25="init_q25",
        y_q75="init_q75",
        ylabel="Initial enclosed cavity volume",
        out_dir=out_dir,
        stem="initial_volume_median_iqr",
    )

    plot_band(
        summary,
        y_med="final_median",
        y_q25="final_q25",
        y_q75="final_q75",
        ylabel="Final enclosed cavity volume",
        out_dir=out_dir,
        stem="final_volume_median_iqr",
    )

    plot_band(
        summary,
        y_med="reduction_median",
        y_q25="reduction_q25",
        y_q75="reduction_q75",
        ylabel="Reduction in enclosed cavity volume (%)",
        out_dir=out_dir,
        stem="reduction_percent_median_iqr",
    )

    plot_band(
        summary,
        y_med="radius_mean_median",
        y_q25="radius_mean_median",
        y_q75="radius_mean_median",
        ylabel="Median initialized radius",
        out_dir=out_dir,
        stem="radius_mean_median",
    )

    plot_final_spaghetti(df, summary, out_dir)

    plot_incremental_gain(
        summary,
        col="init_gain_pct_vs_prev",
        ylabel="Initial-volume improvement\nvs previous finger count (%)",
        out_dir=out_dir,
        stem="initial_volume_incremental_gain",
    )

    plot_incremental_gain(
        summary,
        col="final_gain_pct_vs_prev",
        ylabel="Final-volume improvement\nvs previous finger count (%)",
        out_dir=out_dir,
        stem="final_volume_incremental_gain",
    )

    plot_incremental_gain(
        summary,
        col="reduction_gain_pct_vs_prev",
        ylabel="Reduction improvement\nvs previous finger count (%)",
        out_dir=out_dir,
        stem="reduction_percent_incremental_gain",
    )

    plot_distance_to_best(
        summary,
        col="final_distance_to_best_pct",
        ylabel="Distance to best final volume (%)",
        out_dir=out_dir,
        stem="final_volume_distance_to_best",
    )

    plot_best_count_bar(
        best_df,
        col="best_final_finger_count",
        xlabel="Best final finger count",
        out_dir=out_dir,
        stem="best_final_count_per_object",
    )

    plot_best_count_bar(
        best_df,
        col="best_init_finger_count",
        xlabel="Best initial finger count",
        out_dir=out_dir,
        stem="best_init_count_per_object",
    )

    print(f"Saved plots to: {out_dir}")
    print("Saved tables:")
    print(" ", os.path.join(out_dir, "finger_count_summary.csv"))
    print(" ", os.path.join(out_dir, "best_finger_count_per_object.csv"))


if __name__ == "__main__":
    main()