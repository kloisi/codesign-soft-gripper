# plot_dt_sweep.py
#
# plotting for time-step sweep results from sweep_dt.py

# run like: python plot_dt_sweep.py --input_dir logs/sweep_dt_YYYYMMDD_HHMMSS

# Saves to <input_dir>/plots_report/ :
# - volume_trajectory_overlay.png : enclosed-volume trajectories for one repeat
# - volume_trajectory_overlay_normalized.png : same trajectories normalized by initial volume
# - final_volume_vs_dt.png : final enclosed volume versus physics time step
# - reduction_percent_vs_dt.png : enclosure improvement versus physics time step
# - runtime_vs_dt.png : rollout runtime versus physics time step
# - aggregate_stats.csv : mean/std/median/IQR over repeats for each substep setting

import os
import argparse

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from matplotlib.patches import Patch


FULL_W = 6.8
HALF_W = 3.3
TRAJ_W = 6.5
TRAJ_H = 4.1
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
        "font.size": 11,
        "axes.labelsize": 11,
        "axes.titlesize": 12,
        "xtick.labelsize": 10,
        "ytick.labelsize": 10,
        "legend.fontsize": 11,
        "axes.spines.top": False,
        "axes.spines.right": False,
        "axes.grid": False,
        "lines.linewidth": 2.2,
        "lines.markersize": 5,
    })


def save(fig, out_dir, stem):
    os.makedirs(out_dir, exist_ok=True)
    fig.savefig(os.path.join(out_dir, f"{stem}.png"), dpi=DPI)
    plt.close(fig)


def iqr(values):
    values = pd.to_numeric(pd.Series(values), errors="coerce").dropna().to_numpy()
    if len(values) == 0:
        return np.nan
    return float(np.percentile(values, 75) - np.percentile(values, 25))


def load_summary(input_dir):
    path = os.path.join(input_dir, "summary.csv")
    if not os.path.exists(path):
        raise FileNotFoundError(f"Could not find {path}")

    df = pd.read_csv(path)

    numeric_cols = [
        "rep",
        "fps",
        "substeps",
        "dt",
        "rollout_time_s",
        "runtime_s",
        "runtime_per_step_ms",
        "initial_volume",
        "final_volume",
        "min_volume",
        "reduction_percent",
    ]
    for col in numeric_cols:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce")

    if "status" in df.columns:
        df["status"] = df["status"].fillna("").astype(str).str.strip()
        df = df[df["status"] == "ok"].copy()

    df = df.sort_values(["substeps", "rep"]).reset_index(drop=True)
    return df


def aggregate_summary(df):
    rows = []

    for substeps, group in df.groupby("substeps"):
        init_vals = group["initial_volume"].dropna().to_numpy()
        final_vals = group["final_volume"].dropna().to_numpy()
        reduction_vals = group["reduction_percent"].dropna().to_numpy()
        runtime_vals = group["runtime_s"].dropna().to_numpy()

        rows.append({
            "substeps": int(substeps),
            "dt": float(group["dt"].iloc[0]),
            "n_runs": int(len(group)),

            "initial_mean": np.mean(init_vals),
            "initial_std": np.std(init_vals, ddof=1) if len(init_vals) > 1 else np.nan,
            "initial_median": np.median(init_vals),
            "initial_q25": np.percentile(init_vals, 25),
            "initial_q75": np.percentile(init_vals, 75),
            "initial_iqr": iqr(init_vals),

            "final_mean": np.mean(final_vals),
            "final_std": np.std(final_vals, ddof=1) if len(final_vals) > 1 else np.nan,
            "final_median": np.median(final_vals),
            "final_q25": np.percentile(final_vals, 25),
            "final_q75": np.percentile(final_vals, 75),
            "final_iqr": iqr(final_vals),

            "reduction_mean": np.mean(reduction_vals),
            "reduction_std": np.std(reduction_vals, ddof=1) if len(reduction_vals) > 1 else np.nan,
            "reduction_median": np.median(reduction_vals),
            "reduction_q25": np.percentile(reduction_vals, 25),
            "reduction_q75": np.percentile(reduction_vals, 75),
            "reduction_iqr": iqr(reduction_vals),

            "runtime_mean": np.mean(runtime_vals),
            "runtime_std": np.std(runtime_vals, ddof=1) if len(runtime_vals) > 1 else np.nan,
            "runtime_median": np.median(runtime_vals),
            "runtime_q25": np.percentile(runtime_vals, 25),
            "runtime_q75": np.percentile(runtime_vals, 75),
            "runtime_iqr": iqr(runtime_vals),
        })

    return pd.DataFrame(rows).sort_values("dt").reset_index(drop=True)


def choose_rep_for_trajectories(df, rep):
    if rep is not None:
        out = df[df["rep"] == rep].copy()
        if len(out) == 0:
            raise ValueError(f"No successful runs found for rep={rep}")
        return out

    first_rep = int(df["rep"].min())
    return df[df["rep"] == first_rep].copy()


def load_timeseries(input_dir, filename):
    path = os.path.join(input_dir, filename)
    if not os.path.exists(path):
        return None

    ts = pd.read_csv(path)
    ts["time_s"] = pd.to_numeric(ts["time_s"], errors="coerce")
    ts["volume"] = pd.to_numeric(ts["volume"], errors="coerce")
    ts = ts.dropna(subset=["time_s", "volume"]).sort_values("time_s")
    return ts


def plot_trajectory_overlay(df_rep, input_dir, out_dir):
    fig, ax = plt.subplots(figsize=(TRAJ_W, TRAJ_H))

    for _, row in df_rep.sort_values("dt").iterrows():
        ts = load_timeseries(input_dir, row["timeseries_csv"])
        if ts is None or len(ts) == 0:
            continue

        label = rf"$N_{{\mathrm{{sub}}}}={int(row['substeps'])}$, $\Delta t={row['dt']:.2e}\,\mathrm{{s}}$"
        ax.plot(ts["time_s"], ts["volume"], label=label)

    ax.set_xlabel("Time [s]")
    ax.set_ylabel("Enclosed cavity volume")
    ax.grid(axis="y", color="#E6E6E6", linewidth=0.8)
    ax.legend(
        frameon=False,
        loc="best",
        ncol=1,
        handlelength=3.0,
        labelspacing=0.45,
        borderaxespad=0.4,
    )

    save(fig, out_dir, "volume_trajectory_overlay")


def plot_trajectory_overlay_normalized(df_rep, input_dir, out_dir):
    fig, ax = plt.subplots(figsize=(TRAJ_W, TRAJ_H))

    for _, row in df_rep.sort_values("dt").iterrows():
        ts = load_timeseries(input_dir, row["timeseries_csv"])
        if ts is None or len(ts) == 0:
            continue

        v0 = row["initial_volume"]
        if not np.isfinite(v0) or abs(v0) < 1e-12:
            continue

        label = rf"$N_{{\mathrm{{sub}}}}={int(row['substeps'])}$, $\Delta t={row['dt']:.2e}\,\mathrm{{s}}$"
        ax.plot(ts["time_s"], ts["volume"] / v0, label=label)

    ax.set_xlabel("Time [s]")
    ax.set_ylabel("Normalized enclosed volume")
    ax.grid(axis="y", color="#E6E6E6", linewidth=0.8)
    ax.legend(
        frameon=False,
        loc="best",
        ncol=1,
        handlelength=3.0,
        labelspacing=0.45,
        borderaxespad=0.4,
    )

    save(fig, out_dir, "volume_trajectory_overlay_normalized")


def plot_band_vs_dt(stats, y_med, y_q25, y_q75, ylabel, out_dir, stem):
    x = stats["dt"].to_numpy()

    fig, ax = plt.subplots(figsize=(4.8, 4.15))

    band_color = "#AFC4DA"
    line_color = "#1F4E79"

    ax.fill_between(
        x,
        stats[y_q25],
        stats[y_q75],
        color=band_color,
        alpha=0.45,
        linewidth=0,
        zorder=1,
    )
    ax.plot(
        x,
        stats[y_med],
        color=line_color,
        linewidth=2.3,
        marker="o",
        markersize=5.2,
        zorder=2,
    )

    for _, row in stats.iterrows():
        ax.annotate(
            f"{int(row['substeps'])}",
            (row["dt"], row[y_med]),
            textcoords="offset points",
            xytext=(0, 6),
            ha="center",
        )

    ax.set_xscale("log")
    ax.set_xlabel("Physics time step $\\Delta t$ [s]")
    ax.set_ylabel(ylabel)
    ax.grid(axis="y", color="#E6E6E6", linewidth=0.8)

    handles = [
        Line2D([0], [0], color=line_color, lw=2.3, label="Median"),
        Patch(facecolor=band_color, edgecolor="none", alpha=0.45, label="Interquartile range"),
    ]
    ax.legend(handles=handles, frameon=False, loc="best")

    save(fig, out_dir, stem)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input_dir", type=str, required=True)
    parser.add_argument("--rep", type=int, default=None, help="repeat index used for trajectory overlays")
    args = parser.parse_args()

    setup_style()

    input_dir = os.path.abspath(args.input_dir)
    out_dir = os.path.join(input_dir, "plots_report")
    os.makedirs(out_dir, exist_ok=True)

    summary = load_summary(input_dir)
    stats = aggregate_summary(summary)
    stats.to_csv(os.path.join(out_dir, "aggregate_stats.csv"), index=False)

    df_rep = choose_rep_for_trajectories(summary, args.rep)

    plot_trajectory_overlay(df_rep, input_dir, out_dir)
    plot_trajectory_overlay_normalized(df_rep, input_dir, out_dir)

    plot_band_vs_dt(
        stats=stats,
        y_med="final_median",
        y_q25="final_q25",
        y_q75="final_q75",
        ylabel="Final enclosed cavity volume",
        out_dir=out_dir,
        stem="final_volume_vs_dt",
    )

    plot_band_vs_dt(
        stats=stats,
        y_med="reduction_median",
        y_q25="reduction_q25",
        y_q75="reduction_q75",
        ylabel="Reduction in enclosed cavity volume (%)",
        out_dir=out_dir,
        stem="reduction_percent_vs_dt",
    )

    plot_band_vs_dt(
        stats=stats,
        y_med="runtime_median",
        y_q25="runtime_q25",
        y_q75="runtime_q75",
        ylabel="Rollout runtime [s]",
        out_dir=out_dir,
        stem="runtime_vs_dt",
    )

    print(f"Saved plots to: {out_dir}")
    print("Saved table:")
    print(" ", os.path.join(out_dir, "aggregate_stats.csv"))


if __name__ == "__main__":
    main()