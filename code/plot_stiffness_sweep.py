import os
import argparse
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


def to_num(df, cols):
    for c in cols:
        if c in df.columns:
            df[c] = pd.to_numeric(df[c], errors="coerce")


def savefig(out_dir, name):
    path = os.path.join(out_dir, name)
    plt.tight_layout()
    plt.savefig(path, dpi=220)
    plt.close()
    print("[OK] wrote", path)


def _marker_for_mass(m):
    # stable mapping for common mass scales
    if np.isclose(m, 0.5):
        return "o"
    if np.isclose(m, 1.0):
        return "s"
    if np.isclose(m, 2.0):
        return "^"
    return "o"


def make_heatmap(
    piv: pd.DataFrame,
    title: str,
    xlabel: str,
    ylabel: str,
    cbar_label: str,
    out_dir: str,
    filename: str,
    annotate=True,
):
    # piv index = stiff, columns = damp
    x = list(piv.columns)
    y = list(piv.index)
    Z = piv.values.astype(float)

    plt.figure(figsize=(7.6, 5.6))
    im = plt.imshow(Z, aspect="auto", interpolation="nearest")

    plt.xticks(range(len(x)), [str(v) for v in x])
    plt.yticks(range(len(y)), [str(v) for v in y])
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    plt.title(title)
    cb = plt.colorbar(im)
    cb.set_label(cbar_label)

    if annotate:
        for i in range(len(y)):
            for j in range(len(x)):
                val = Z[i, j]
                if np.isfinite(val):
                    plt.text(j, i, f"{val:.3g}", ha="center", va="center", fontsize=8)

    savefig(out_dir, filename)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--csv", type=str, default="stiffness_sweep_forwardlike_v1.csv")
    parser.add_argument("--out", type=str, default="stiffness_sweep_plots")
    parser.add_argument("--no_annot", action="store_true", help="disable heatmap annotations")
    args = parser.parse_args()

    out_dir = args.out
    os.makedirs(out_dir, exist_ok=True)

    df = pd.read_csv(args.csv)

    # numeric columns (robust to blanks)
    num_cols = [
        "finger_num",
        "cloth_stiff_scale", "cloth_mass_scale", "cloth_damp_scale",
        "fps", "sim_substeps", "dt", "num_frames", "total_steps",
        "real_time_per_step_ms",
        "num_ok", "phys_ok", "diff_norm",
        "max_abs_pos", "max_speed",
        "vol_final", "t95", "tail_cv", "overshoot_rel",
        "kernel_seed",
    ]
    to_num(df, num_cols)

    # drop rows without key params
    df = df[np.isfinite(df["cloth_stiff_scale"]) & np.isfinite(df["cloth_mass_scale"]) & np.isfinite(df["cloth_damp_scale"])].copy()

    # helpful derived
    df["tail_cv_safe"] = np.maximum(df["tail_cv"].astype(float), 1e-12)
    df["log10_tail_cv"] = np.log10(df["tail_cv_safe"])

    # aggregate duplicates (you have repeats for some combos)
    keys = ["cloth_stiff_scale", "cloth_mass_scale", "cloth_damp_scale"]
    agg = {
        "vol_final": ["mean", "std"],
        "overshoot_rel": ["mean", "std"],
        "tail_cv": ["mean", "std"],
        "log10_tail_cv": ["mean", "std"],
        "t95": ["mean", "std"],
        "max_speed": ["mean", "std"],
        "real_time_per_step_ms": ["mean", "std"],
        "num_ok": "mean",
        "phys_ok": "mean",
        "diff_norm": ["mean", "std"],
        "max_abs_pos": ["mean", "std"],
        "kernel_seed": "first",
    }

    g = df.groupby(keys, as_index=False).agg(agg)

    # flatten multiindex columns
    g.columns = [
        "_".join([c for c in col if c]) if isinstance(col, tuple) else col
        for col in g.columns
    ]

    # rename common
    rename = {
        "num_ok_mean": "num_ok_rate",
        "phys_ok_mean": "phys_ok_rate",
    }
    g = g.rename(columns=rename)

    # fill std nan
    for c in g.columns:
        if c.endswith("_std"):
            g[c] = g[c].fillna(0.0)

    # sort for nicer plots
    g = g.sort_values(["cloth_mass_scale", "cloth_damp_scale", "cloth_stiff_scale"])

    # write grouped summary
    summary_csv = os.path.join(out_dir, "grouped_summary.csv")
    g.to_csv(summary_csv, index=False)
    print("[OK] wrote", summary_csv)

    # quick print: how metrics move with stiffness (averaged over mass/damp)
    g_stiff = g.groupby("cloth_stiff_scale", as_index=False).agg(
        vol_final_mean=("vol_final_mean", "mean"),
        overshoot_rel_mean=("overshoot_rel_mean", "mean"),
        tail_cv_mean=("tail_cv_mean", "mean"),
        log10_tail_cv_mean=("log10_tail_cv_mean", "mean"),
        max_speed_mean=("max_speed_mean", "mean"),
        runtime_ms_mean=("real_time_per_step_ms_mean", "mean"),
    )
    print("\nAverages vs stiffness (over mass,damp):")
    print(g_stiff.to_string(index=False))

    annotate = (not args.no_annot)

    # ---------------------------------------
    # 1) Heatmaps per mass scale (stiff x damp)
    # ---------------------------------------
    mass_vals = sorted(g["cloth_mass_scale"].unique().tolist())
    damp_vals = sorted(g["cloth_damp_scale"].unique().tolist())
    stiff_vals = sorted(g["cloth_stiff_scale"].unique().tolist())

    for m in mass_vals:
        gm = g[np.isclose(g["cloth_mass_scale"], m)].copy()

        # vol_final
        piv = gm.pivot(index="cloth_stiff_scale", columns="cloth_damp_scale", values="vol_final_mean").reindex(index=stiff_vals, columns=damp_vals)
        make_heatmap(
            piv,
            title=f"vol_final (mean) heatmap  mass_scale={m}",
            xlabel="cloth_damp_scale",
            ylabel="cloth_stiff_scale",
            cbar_label="vol_final mean",
            out_dir=out_dir,
            filename=f"heat_vol_final_mass{str(m).replace('.','p')}.png",
            annotate=annotate,
        )

        # overshoot
        piv = gm.pivot(index="cloth_stiff_scale", columns="cloth_damp_scale", values="overshoot_rel_mean").reindex(index=stiff_vals, columns=damp_vals)
        make_heatmap(
            piv,
            title=f"overshoot_rel (mean) heatmap  mass_scale={m}",
            xlabel="cloth_damp_scale",
            ylabel="cloth_stiff_scale",
            cbar_label="overshoot_rel mean",
            out_dir=out_dir,
            filename=f"heat_overshoot_mass{str(m).replace('.','p')}.png",
            annotate=annotate,
        )

        # tail cv (log10)
        piv = gm.pivot(index="cloth_stiff_scale", columns="cloth_damp_scale", values="log10_tail_cv_mean").reindex(index=stiff_vals, columns=damp_vals)
        make_heatmap(
            piv,
            title=f"log10(tail_cv) (mean) heatmap  mass_scale={m}",
            xlabel="cloth_damp_scale",
            ylabel="cloth_stiff_scale",
            cbar_label="log10(tail_cv) mean",
            out_dir=out_dir,
            filename=f"heat_logtailcv_mass{str(m).replace('.','p')}.png",
            annotate=annotate,
        )

        # max_speed
        piv = gm.pivot(index="cloth_stiff_scale", columns="cloth_damp_scale", values="max_speed_mean").reindex(index=stiff_vals, columns=damp_vals)
        make_heatmap(
            piv,
            title=f"max_speed (mean) heatmap  mass_scale={m}",
            xlabel="cloth_damp_scale",
            ylabel="cloth_stiff_scale",
            cbar_label="max_speed mean",
            out_dir=out_dir,
            filename=f"heat_maxspeed_mass{str(m).replace('.','p')}.png",
            annotate=annotate,
        )

    # ---------------------------------------
    # 2) Trend plots: metric vs stiffness
    #    for each mass, lines per damp
    # ---------------------------------------
    def plot_trends(metric_mean_col, ylabel, fname, logy=False):
        plt.figure(figsize=(7.6, 5.2))
        for m in mass_vals:
            gm = g[np.isclose(g["cloth_mass_scale"], m)]
            for d in damp_vals:
                gmd = gm[np.isclose(gm["cloth_damp_scale"], d)].sort_values("cloth_stiff_scale")
                if gmd.empty:
                    continue
                plt.plot(
                    gmd["cloth_stiff_scale"],
                    gmd[metric_mean_col],
                    marker=_marker_for_mass(m),
                    linestyle="-",
                    label=f"mass={m}, damp={d}",
                    alpha=0.9,
                )
        plt.xlabel("cloth_stiff_scale (ke/ka)")
        plt.ylabel(ylabel)
        plt.title(f"{ylabel} vs stiffness (lines: damp, marker: mass)")
        plt.grid(True, alpha=0.3)
        plt.xscale("log")
        if logy:
            plt.yscale("log")
        plt.legend(fontsize=8, ncol=2)
        savefig(out_dir, fname)

    plot_trends("vol_final_mean", "vol_final mean", "trend_vol_final_vs_stiff.png", logy=False)
    plot_trends("overshoot_rel_mean", "overshoot_rel mean", "trend_overshoot_vs_stiff.png", logy=False)
    plot_trends("tail_cv_mean", "tail_cv mean", "trend_tailcv_vs_stiff.png", logy=True)
    plot_trends("max_speed_mean", "max_speed mean", "trend_maxspeed_vs_stiff.png", logy=False)
    plot_trends("real_time_per_step_ms_mean", "real_time_per_step_ms mean", "trend_runtime_per_step_vs_stiff.png", logy=False)

    # ---------------------------------------
    # 3) Pareto style scatter plots
    # ---------------------------------------
    # 3a) vol_final vs overshoot_rel
    plt.figure(figsize=(7.6, 5.2))
    for _, r in g.iterrows():
        s = r["cloth_stiff_scale"]
        m = r["cloth_mass_scale"]
        d = r["cloth_damp_scale"]
        x = r["overshoot_rel_mean"]
        y = r["vol_final_mean"]
        if not (np.isfinite(x) and np.isfinite(y)):
            continue
        plt.scatter(x, y, marker=_marker_for_mass(m), alpha=0.9)
        # light annotation (keeps readable)
        plt.annotate(f"s={s},d={d}", (x, y), textcoords="offset points", xytext=(4, 3), fontsize=7, alpha=0.8)
    plt.xlabel("overshoot_rel mean (lower is better)")
    plt.ylabel("vol_final mean (higher is better)")
    plt.title("Pareto view: vol_final vs overshoot_rel\n(marker encodes mass, label shows (stiff,damp))")
    plt.grid(True, alpha=0.3)
    savefig(out_dir, "pareto_vol_vs_overshoot.png")

    # 3b) vol_final vs tail_cv (log x)
    plt.figure(figsize=(7.6, 5.2))
    for _, r in g.iterrows():
        s = r["cloth_stiff_scale"]
        m = r["cloth_mass_scale"]
        d = r["cloth_damp_scale"]
        x = max(float(r["tail_cv_mean"]), 1e-12)
        y = r["vol_final_mean"]
        if not (np.isfinite(x) and np.isfinite(y)):
            continue
        plt.scatter(x, y, marker=_marker_for_mass(m), alpha=0.9)
        plt.annotate(f"s={s},d={d}", (x, y), textcoords="offset points", xytext=(4, 3), fontsize=7, alpha=0.8)
    plt.xscale("log")
    plt.xlabel("tail_cv mean (log scale, lower is better)")
    plt.ylabel("vol_final mean (higher is better)")
    plt.title("Pareto view: vol_final vs tail_cv\n(marker encodes mass, label shows (stiff,damp))")
    plt.grid(True, alpha=0.3)
    savefig(out_dir, "pareto_vol_vs_tailcv.png")

    # ---------------------------------------
    # 4) Small diagnostic: is t95 informative?
    # ---------------------------------------
    if "t95_mean" in g.columns:
        plt.figure(figsize=(7.6, 4.8))
        plt.scatter(g["cloth_stiff_scale"], g["t95_mean"], alpha=0.9)
        plt.xscale("log")
        plt.xlabel("cloth_stiff_scale")
        plt.ylabel("t95 mean [s]")
        plt.title("t95 vs stiffness (if flat, metric is saturated)")
        plt.grid(True, alpha=0.3)
        savefig(out_dir, "diag_t95_vs_stiff.png")

    print(f"\n[OK] All plots saved to: {out_dir}")


if __name__ == "__main__":
    main()
