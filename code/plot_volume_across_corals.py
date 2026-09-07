# plot_volume_across_corals.py

# run like: python plot_volume_across_corals.py logs/sweep_vol_corals_20260316_010702_f6  

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
        "font.size": 9,
        "axes.labelsize": 9,
        "axes.titlesize": 10,
        "xtick.labelsize": 8,
        "ytick.labelsize": 8,
        "legend.fontsize": 8,
        "axes.spines.top": False,
        "axes.spines.right": False,
        "axes.grid": False,
        "lines.linewidth": 1.6,
    })


def save(fig, out_dir, stem):
    os.makedirs(out_dir, exist_ok=True)
    fig.savefig(os.path.join(out_dir, f"{stem}.png"), dpi=DPI)
    plt.close(fig)


def pretty_name(name):
    return name.replace("_", " ")


def read_data(run_dir):
    summary = pd.read_csv(os.path.join(run_dir, "summary.csv"))
    timeseries = pd.read_csv(os.path.join(run_dir, "timeseries.csv"))

    if "error" not in summary.columns:
        summary["error"] = ""

    summary["error"] = summary["error"].fillna("").astype(str)
    summary = summary[summary["error"].str.strip() == ""].copy()

    for col in [
        "finger_num",
        "seed",
        "init_pose_ok",
        "force_opt_ok",
        "init_vol_vox_m3",
        "final_vol_vox_m3",
        "delta_vol_vox_m3",
        "pct_change",
        "pct_reduction",
    ]:
        if col in summary.columns:
            summary[col] = pd.to_numeric(summary[col], errors="coerce")

    for col in ["frame", "t", "vol_vox", "vol_norm_init"]:
        if col in timeseries.columns:
            timeseries[col] = pd.to_numeric(timeseries[col], errors="coerce")

    timeseries = timeseries[timeseries["object"].isin(summary["object"])].copy()

    if "pct_reduction" not in summary.columns:
        summary["pct_reduction"] = (
            100.0
            * (summary["init_vol_vox_m3"] - summary["final_vol_vox_m3"])
            / summary["init_vol_vox_m3"]
        )

    return summary, timeseries


def plot_normalized_trajectories(timeseries, out_dir):
    df = timeseries.copy().sort_values(["object", "t", "frame"])

    if "vol_norm_init" not in df.columns or df["vol_norm_init"].isna().all():
        initial = df.groupby("object")["vol_vox"].transform("first")
        df["vol_norm_init"] = df["vol_vox"] / initial

    # keep only valid rows
    df = df[np.isfinite(df["t"]) & np.isfinite(df["vol_norm_init"])].copy()

    # statistics over corals at each saved time
    agg = (
        df.groupby("t", as_index=False)["vol_norm_init"]
        .agg(
            median="median",
            q25=lambda x: np.percentile(x, 25),
            q75=lambda x: np.percentile(x, 75),
        )
    )

    fig, ax = plt.subplots(figsize=(FULL_W, 4.2))

    indiv_color = "#B9C2CC"
    band_color = "#AFC4DA"
    center_color = "#123E6D"

    for _, g in df.groupby("object", sort=True):
        ax.plot(
            g["t"],
            g["vol_norm_init"],
            color=indiv_color,
            linewidth=0.9,
            alpha=0.75,
            zorder=1,
        )

    ax.fill_between(
        agg["t"],
        agg["q25"],
        agg["q75"],
        color=band_color,
        alpha=0.45,
        linewidth=0,
        zorder=2,
    )

    ax.plot(
        agg["t"],
        agg["median"],
        color=center_color,
        linewidth=2.2,
        zorder=3,
    )

    y_min = max(0.0, float(df["vol_norm_init"].min()) - 0.04)
    y_max = max(1.02, float(df["vol_norm_init"].max()) + 0.02)

    ax.set_xlabel("Time (s)")
    ax.set_ylabel(r"Normalized enclosed cavity volume $V(t)/V_0$")
    ax.set_xlim(left=0.0)
    ax.set_ylim(y_min, y_max)
    ax.grid(axis="y", color="#E6E6E6", linewidth=0.8)

    handles = [
        Line2D([0], [0], color=indiv_color, lw=1.2, label="Individual corals"),
        Line2D([0], [0], color=center_color, lw=2.2, label="Median"),
        Patch(facecolor=band_color, edgecolor="none", alpha=0.45, label="Interquartile range"),
    ]
    ax.legend(handles=handles, frameon=False, loc="upper right")

    save(fig, out_dir, "timeseries_normalized_median_iqr")


def plot_reduction_bars(summary, out_dir):
    d = summary[["object", "pct_reduction"]].copy()
    d = d.sort_values("pct_reduction", ascending=True)

    labels = [pretty_name(x) for x in d["object"]]

    fig_h = max(4.2, 0.32 * len(d))
    fig, ax = plt.subplots(figsize=(FULL_W, fig_h))

    bar_color = "#4C78A8"

    ax.barh(labels, d["pct_reduction"], color=bar_color, height=0.72)

    for y, val in enumerate(d["pct_reduction"]):
        ax.text(val + 0.7, y, f"{val:.1f}", va="center", ha="left", fontsize=8)

    ax.set_xlabel("Reduction in enclosed cavity volume (%)")
    ax.set_ylabel("Coral")
    ax.set_xlim(0, max(5, float(d["pct_reduction"].max()) * 1.12))
    ax.grid(axis="x", color="#E6E6E6", linewidth=0.8)

    save(fig, out_dir, "reduction_percent_sorted_bar")


def plot_reduction_boxplot(summary, out_dir):
    values = pd.to_numeric(summary["pct_reduction"], errors="coerce").dropna().to_numpy()
    if len(values) == 0:
        return

    fig, ax = plt.subplots(figsize=(HALF_W + 0.9, 4.0))

    bp = ax.boxplot(
        [values],
        widths=0.45,
        patch_artist=True,
        medianprops={"color": "#222222", "linewidth": 1.4},
        boxprops={"linewidth": 1.0, "edgecolor": "#4F4F4F"},
        whiskerprops={"linewidth": 1.0, "color": "#4F4F4F"},
        capprops={"linewidth": 1.0, "color": "#4F4F4F"},
    )

    bp["boxes"][0].set_facecolor("#AFC4DA")

    rng = np.random.default_rng(7)
    x = 1.0 + rng.uniform(-0.08, 0.08, size=len(values))
    ax.scatter(
        x,
        values,
        s=22,
        color="#1F4E79",
        alpha=0.85,
        zorder=3,
    )

    ax.set_xticks([1], ["All corals"])
    ax.set_ylabel("Reduction in enclosed cavity volume (%)")
    ax.grid(axis="y", color="#E6E6E6", linewidth=0.8)

    save(fig, out_dir, "reduction_percent_boxplot")


def main():
    parser = argparse.ArgumentParser(
        description="Create report plots for enclosure-volume sweep across corals."
    )
    parser.add_argument(
        "run_dir",
        help="Path to the sweep output folder.",
    )
    args = parser.parse_args()

    run_dir = args.run_dir

    setup_style()
    summary, timeseries = read_data(run_dir)

    out_dir = os.path.join(run_dir, "plots_report")

    plot_normalized_trajectories(timeseries, out_dir)
    plot_reduction_bars(summary, out_dir)
    plot_reduction_boxplot(summary, out_dir)

    print(f"Saved report plots to: {out_dir}")


if __name__ == "__main__":
    main()