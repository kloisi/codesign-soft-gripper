# plot_cloth_params.py
#
# run like:
#   python plot_cloth_params.py logs/sweep_cloth_params_YYYYMMDD_HHMMSS_f6
#
# Saves to <run_dir>/plots_report/ :
# - final_volume_heatmap_mass_*.png
# - reduction_percent_heatmap_mass_*.png
# - rebound_ratio_heatmap_mass_*.png
# - final_volume_vs_stiffness_mass_*.png
# - reduction_percent_vs_stiffness_mass_*.png
# - rebound_ratio_vs_stiffness_mass_*.png
# - normalized_volume_timeseries_mass_*.png
# - cloth_param_summary_for_plots.csv

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


def read_data(run_dir):
    summary_path = os.path.join(run_dir, "summary.csv")
    timeseries_path = os.path.join(run_dir, "timeseries.csv")

    if not os.path.exists(summary_path):
        raise FileNotFoundError(f"Could not find {summary_path}")
    if not os.path.exists(timeseries_path):
        raise FileNotFoundError(f"Could not find {timeseries_path}")

    summary = pd.read_csv(summary_path)
    timeseries = pd.read_csv(timeseries_path)

    if "error" in summary.columns:
        summary["error"] = summary["error"].fillna("").astype(str).str.strip()
        summary = summary[summary["error"] == ""].copy()

    summary_cols = [
        "alpha_k",
        "alpha_m",
        "alpha_d",
        "rep",
        "seed",
        "init_vol_vox_m3",
        "final_vol_vox_m3",
        "min_vol_vox_m3",
        "pct_reduction",
        "rebound_ratio",
        "rollout_runtime_s",
    ]
    for col in summary_cols:
        if col in summary.columns:
            summary[col] = pd.to_numeric(summary[col], errors="coerce")

    timeseries_cols = [
        "alpha_k",
        "alpha_m",
        "alpha_d",
        "rep",
        "frame",
        "t",
        "vol_vox",
        "vol_norm_init",
    ]
    for col in timeseries_cols:
        if col in timeseries.columns:
            timeseries[col] = pd.to_numeric(timeseries[col], errors="coerce")

    summary = summary.dropna(subset=["alpha_k", "alpha_m", "alpha_d", "final_vol_vox_m3"]).copy()
    timeseries = timeseries.dropna(subset=["alpha_k", "alpha_m", "alpha_d", "t", "vol_norm_init"]).copy()

    return summary, timeseries


def aggregate_summary(summary):
    rows = []

    for (alpha_k, alpha_m, alpha_d), g in summary.groupby(["alpha_k", "alpha_m", "alpha_d"]):
        final_vals = g["final_vol_vox_m3"].dropna().to_numpy()
        reduction_vals = g["pct_reduction"].dropna().to_numpy()
        rebound_vals = g["rebound_ratio"].dropna().to_numpy()
        runtime_vals = g["rollout_runtime_s"].dropna().to_numpy()

        rows.append({
            "alpha_k": float(alpha_k),
            "alpha_m": float(alpha_m),
            "alpha_d": float(alpha_d),
            "n_runs": int(len(g)),

            "final_mean": np.mean(final_vals) if len(final_vals) > 0 else np.nan,
            "final_std": np.std(final_vals, ddof=1) if len(final_vals) > 1 else np.nan,
            "final_median": np.median(final_vals) if len(final_vals) > 0 else np.nan,
            "final_q25": np.percentile(final_vals, 25) if len(final_vals) > 0 else np.nan,
            "final_q75": np.percentile(final_vals, 75) if len(final_vals) > 0 else np.nan,

            "reduction_mean": np.mean(reduction_vals) if len(reduction_vals) > 0 else np.nan,
            "reduction_std": np.std(reduction_vals, ddof=1) if len(reduction_vals) > 1 else np.nan,
            "reduction_median": np.median(reduction_vals) if len(reduction_vals) > 0 else np.nan,
            "reduction_q25": np.percentile(reduction_vals, 25) if len(reduction_vals) > 0 else np.nan,
            "reduction_q75": np.percentile(reduction_vals, 75) if len(reduction_vals) > 0 else np.nan,

            "rebound_mean": np.mean(rebound_vals) if len(rebound_vals) > 0 else np.nan,
            "rebound_std": np.std(rebound_vals, ddof=1) if len(rebound_vals) > 1 else np.nan,
            "rebound_median": np.median(rebound_vals) if len(rebound_vals) > 0 else np.nan,
            "rebound_q25": np.percentile(rebound_vals, 25) if len(rebound_vals) > 0 else np.nan,
            "rebound_q75": np.percentile(rebound_vals, 75) if len(rebound_vals) > 0 else np.nan,

            "runtime_mean": np.mean(runtime_vals) if len(runtime_vals) > 0 else np.nan,
            "runtime_std": np.std(runtime_vals, ddof=1) if len(runtime_vals) > 1 else np.nan,
            "runtime_median": np.median(runtime_vals) if len(runtime_vals) > 0 else np.nan,
            "runtime_q25": np.percentile(runtime_vals, 25) if len(runtime_vals) > 0 else np.nan,
            "runtime_q75": np.percentile(runtime_vals, 75) if len(runtime_vals) > 0 else np.nan,
        })

    return pd.DataFrame(rows).sort_values(["alpha_m", "alpha_k", "alpha_d"]).reset_index(drop=True)


def make_heatmap(piv, title, xlabel, ylabel, cbar_label, out_dir, stem, fmt="{:.3g}"):
    x = list(piv.columns)
    y = list(piv.index)
    z = piv.values.astype(float)

    fig, ax = plt.subplots(figsize=(4.9, 4.2))
    im = ax.imshow(z, aspect="auto", interpolation="nearest")

    ax.set_xticks(range(len(x)), [str(v) for v in x])
    ax.set_yticks(range(len(y)), [str(v) for v in y])
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    ax.set_title(title)

    cb = fig.colorbar(im, ax=ax)
    cb.set_label(cbar_label)

    for i in range(len(y)):
        for j in range(len(x)):
            val = z[i, j]
            if np.isfinite(val):
                ax.text(j, i, fmt.format(val), ha="center", va="center", fontsize=8)

    save(fig, out_dir, stem)


def plot_heatmaps(stats, out_dir):
    mass_vals = sorted(stats["alpha_m"].unique())

    for alpha_m in mass_vals:
        d = stats[np.isclose(stats["alpha_m"], alpha_m)].copy()

        piv_final = d.pivot(index="alpha_k", columns="alpha_d", values="final_median").sort_index().sort_index(axis=1)
        make_heatmap(
            piv=piv_final,
            title=rf"Final enclosed volume, $\alpha_m={alpha_m}$",
            xlabel=rf"Damping scale $\alpha_d$",
            ylabel=rf"Stiffness scale $\alpha_k$",
            cbar_label="Final enclosed volume",
            out_dir=out_dir,
            stem=f"final_volume_heatmap_mass_{str(alpha_m).replace('.', 'p')}",
            fmt="{:.4f}",
        )

        piv_reduction = d.pivot(index="alpha_k", columns="alpha_d", values="reduction_median").sort_index().sort_index(axis=1)
        make_heatmap(
            piv=piv_reduction,
            title=rf"Reduction in enclosed volume, $\alpha_m={alpha_m}$",
            xlabel=rf"Damping scale $\alpha_d$",
            ylabel=rf"Stiffness scale $\alpha_k$",
            cbar_label="Reduction (%)",
            out_dir=out_dir,
            stem=f"reduction_percent_heatmap_mass_{str(alpha_m).replace('.', 'p')}",
            fmt="{:.2f}",
        )

        piv_rebound = d.pivot(index="alpha_k", columns="alpha_d", values="rebound_median").sort_index().sort_index(axis=1)
        make_heatmap(
            piv=piv_rebound,
            title=rf"Rebound ratio, $\alpha_m={alpha_m}$",
            xlabel=rf"Damping scale $\alpha_d$",
            ylabel=rf"Stiffness scale $\alpha_k$",
            cbar_label=r"$V_{\mathrm{final}} / V_{\mathrm{min}}$",
            out_dir=out_dir,
            stem=f"rebound_ratio_heatmap_mass_{str(alpha_m).replace('.', 'p')}",
            fmt="{:.4f}",
        )


def plot_metric_vs_stiffness(stats, alpha_m, y_med, y_q25, y_q75, ylabel, out_dir, stem):
    d = stats[np.isclose(stats["alpha_m"], alpha_m)].copy()
    damp_vals = sorted(d["alpha_d"].unique())

    fig, ax = plt.subplots(figsize=(FULL_W, 4.0))

    for alpha_d in damp_vals:
        g = d[np.isclose(d["alpha_d"], alpha_d)].sort_values("alpha_k")
        if len(g) == 0:
            continue

        x = g["alpha_k"].to_numpy()
        y = g[y_med].to_numpy()
        q25 = g[y_q25].to_numpy()
        q75 = g[y_q75].to_numpy()

        ax.fill_between(x, q25, q75, alpha=0.18, linewidth=0)
        ax.plot(x, y, marker="o", label=rf"$\alpha_d={alpha_d}$")

    ax.set_xscale("log")
    ax.set_xlabel(r"Stiffness scale $\alpha_k$")
    ax.set_ylabel(ylabel)
    ax.grid(axis="y", color="#E6E6E6", linewidth=0.8)
    ax.legend(frameon=False, loc="best", ncol=1)

    save(fig, out_dir, stem)


def plot_trends(stats, out_dir):
    mass_vals = sorted(stats["alpha_m"].unique())

    for alpha_m in mass_vals:
        mtag = str(alpha_m).replace(".", "p")

        plot_metric_vs_stiffness(
            stats=stats,
            alpha_m=alpha_m,
            y_med="final_median",
            y_q25="final_q25",
            y_q75="final_q75",
            ylabel="Final enclosed volume",
            out_dir=out_dir,
            stem=f"final_volume_vs_stiffness_mass_{mtag}",
        )

        plot_metric_vs_stiffness(
            stats=stats,
            alpha_m=alpha_m,
            y_med="reduction_median",
            y_q25="reduction_q25",
            y_q75="reduction_q75",
            ylabel="Reduction in enclosed volume (%)",
            out_dir=out_dir,
            stem=f"reduction_percent_vs_stiffness_mass_{mtag}",
        )

        plot_metric_vs_stiffness(
            stats=stats,
            alpha_m=alpha_m,
            y_med="rebound_median",
            y_q25="rebound_q25",
            y_q75="rebound_q75",
            ylabel=r"Rebound ratio $V_{\mathrm{final}}/V_{\mathrm{min}}$",
            out_dir=out_dir,
            stem=f"rebound_ratio_vs_stiffness_mass_{mtag}",
        )


def choose_rep(timeseries, rep=None):
    if rep is not None:
        out = timeseries[timeseries["rep"] == rep].copy()
        if len(out) == 0:
            raise ValueError(f"No timeseries found for rep={rep}")
        return out

    first_rep = int(timeseries["rep"].min())
    return timeseries[timeseries["rep"] == first_rep].copy()


def plot_timeseries(timeseries, out_dir, rep=None):
    df = choose_rep(timeseries, rep=rep)
    mass_vals = sorted(df["alpha_m"].unique())
    damp_vals = sorted(df["alpha_d"].unique())

    for alpha_m in mass_vals:
        d = df[np.isclose(df["alpha_m"], alpha_m)].copy()
        if len(d) == 0:
            continue

        fig, axes = plt.subplots(
            len(damp_vals),
            1,
            figsize=(FULL_W, 2.3 * len(damp_vals)),
            squeeze=False,
        )

        any_line = False

        for row_idx, alpha_d in enumerate(damp_vals):
            ax = axes[row_idx, 0]
            g_d = d[np.isclose(d["alpha_d"], alpha_d)].copy()

            for alpha_k, g_k in g_d.groupby("alpha_k"):
                g_k = g_k.sort_values(["t", "frame"])
                ax.plot(
                    g_k["t"],
                    g_k["vol_norm_init"],
                    label=rf"$\alpha_k={alpha_k}$",
                )
                any_line = True

            ax.set_ylabel(r"$V(t)/V_0$")
            ax.set_title(rf"$\alpha_m={alpha_m}$, $\alpha_d={alpha_d}$")
            ax.grid(axis="y", color="#E6E6E6", linewidth=0.8)
            ax.legend(frameon=False, loc="best", ncol=2)

        axes[-1, 0].set_xlabel("Time (s)")

        if any_line:
            save(fig, out_dir, f"normalized_volume_timeseries_mass_{str(alpha_m).replace('.', 'p')}")
        else:
            plt.close(fig)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("run_dir", help="Path to the cloth-parameter sweep output folder.")
    parser.add_argument("--rep", type=int, default=None, help="Repeat index used for the timeseries overlays.")
    args = parser.parse_args()

    setup_style()

    run_dir = os.path.abspath(args.run_dir)
    out_dir = os.path.join(run_dir, "plots_report")
    os.makedirs(out_dir, exist_ok=True)

    summary, timeseries = read_data(run_dir)
    stats = aggregate_summary(summary)

    stats.to_csv(os.path.join(out_dir, "cloth_param_summary_for_plots.csv"), index=False)

    plot_heatmaps(stats, out_dir)
    plot_trends(stats, out_dir)
    plot_timeseries(timeseries, out_dir, rep=args.rep)

    print(f"Saved plots to: {out_dir}")
    print("Saved table:")
    print(" ", os.path.join(out_dir, "cloth_param_summary_for_plots.csv"))


if __name__ == "__main__":
    main()