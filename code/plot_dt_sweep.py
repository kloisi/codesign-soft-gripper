# plot_dt_sweep.py

import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


CSV_PATH = "dt_sweep_results_forwardlike.csv"  # change if needed
OUT_DIR = "dt_sweep_plots_forwardlike"
os.makedirs(OUT_DIR, exist_ok=True)


def to_num(df, cols):
    for c in cols:
        if c in df.columns:
            df[c] = pd.to_numeric(df[c], errors="coerce")


def savefig(path):
    plt.tight_layout()
    plt.savefig(path, dpi=200)
    plt.close()


def plot_metric_vs_dt(df, group, metric, ylabel, filename, logy=False):
    """
    Per-rep scatter + mean±std errorbar vs dt.
    """
    if metric not in df.columns:
        print(f"[WARN] Missing column: {metric}, skipping {filename}")
        return

    # scatter per run
    plt.figure(figsize=(7.5, 5))
    plt.scatter(df["dt_us"], df[metric], alpha=0.7)

    # mean ± std per setting
    plt.errorbar(
        group["dt_us"],
        group[f"{metric}_mean"],
        yerr=group[f"{metric}_std"],
        marker="o",
        linestyle="-",
        capsize=3,
    )

    plt.xlabel("dt [µs]")
    plt.ylabel(ylabel)
    plt.title(f"{metric} vs dt (scatter=rep, line=mean±std)")
    plt.grid(True, alpha=0.3)
    if logy:
        plt.yscale("log")
    # dt is usually nicer on log-x when you add more points later
    plt.xscale("log")

    savefig(os.path.join(OUT_DIR, filename))


def plot_two_metrics_vs_dt(group, m1, m2, y1, y2, filename, logy=False):
    if (m1 not in group.columns) or (m2 not in group.columns):
        print(f"[WARN] Missing grouped cols for {m1}/{m2}, skipping {filename}")
        return

    plt.figure(figsize=(7.5, 5))
    plt.plot(group["dt_us"], group[m1], marker="o", linestyle="-", label=y1)
    plt.plot(group["dt_us"], group[m2], marker="o", linestyle="-", label=y2)
    plt.xlabel("dt [µs]")
    plt.title(f"{y1} and {y2} vs dt")
    plt.grid(True, alpha=0.3)
    plt.xscale("log")
    if logy:
        plt.yscale("log")
    plt.legend()
    savefig(os.path.join(OUT_DIR, filename))


def main():
    df = pd.read_csv(CSV_PATH)

    # numeric columns (robust to blanks)
    num_cols = [
        "fps", "sim_substeps", "rep",
        "dt", "num_frames", "total_steps",
        "t_pose_s", "t_forceopt_s", "t_forward_s", "real_time_per_step_ms",
        "num_ok", "phys_ok", "diff_norm", "max_abs_pos", "max_speed",
        "forceopt_loss0", "forceopt_lossT", "forces_mean", "forces_max",
        "vol_final", "vol_final_ref", "vol_abs_err", "vol_rel_err",
        "vol_traj_rmse", "q_final_rms_err",
    ]
    to_num(df, num_cols)

    # derived
    df["dt_us"] = df["dt"] * 1e6
    df["dt_ns"] = df["dt"] * 1e9

    # guard: only keep rows with dt present
    df = df[np.isfinite(df["dt"])].copy()

    # group: one row per (fps, sim_substeps)
    keys = ["fps", "sim_substeps"]
    agg = {
        "dt": "first",
        "dt_us": "first",
        "dt_ns": "first",
        "num_frames": "first",
        "total_steps": "first",
        "num_ok": "mean",
        "phys_ok": "mean",
        "diff_norm": ["mean", "std"],
        "t_forward_s": ["mean", "std"],
        "t_forceopt_s": ["mean", "std"],
        "vol_final": ["mean", "std"],
        "vol_final_ref": "first",
        "vol_rel_err": ["mean", "std"],
        "vol_abs_err": ["mean", "std"],
        "vol_traj_rmse": ["mean", "std"],
        "q_final_rms_err": ["mean", "std"],
        "forces_mean": ["mean", "std"],
        "forces_max": ["mean", "std"],
        "forceopt_lossT": ["mean", "std"],
    }

    group = df.groupby(keys, as_index=False).agg(agg)

    # flatten columns
    group.columns = [
        "_".join([c for c in col if c]) if isinstance(col, tuple) else col
        for col in group.columns
    ]
    # rename a few common ones
    rename = {
        "dt_us_first": "dt_us",
        "dt_first": "dt",
        "dt_ns_first": "dt_ns",
        "num_frames_first": "num_frames",
        "total_steps_first": "total_steps",
        "vol_final_ref_first": "vol_final_ref",
        "num_ok_mean": "num_ok_rate",
        "phys_ok_mean": "phys_ok_rate",
    }
    group = group.rename(columns=rename)

    # fill std NaN (happens if only one rep)
    for c in group.columns:
        if c.endswith("_std"):
            group[c] = group[c].fillna(0.0)

    # sort by dt
    group = group.sort_values("dt")

    # save summary table
    summary_path = os.path.join(OUT_DIR, "summary_grouped.csv")
    group.to_csv(summary_path, index=False)
    print(f"[OK] Wrote grouped summary to {summary_path}")

    # quick console summary
    print("\nGrouped summary (mean ± std):")
    for _, r in group.iterrows():
        fps = int(r["fps"])
        sub = int(r["sim_substeps"])
        print(
            f"  fps={fps} sub={sub} dt={r['dt']:.2e}  "
            f"vol_rel_err={r['vol_rel_err_mean']:.3e}±{r['vol_rel_err_std']:.1e}  "
            f"traj_rmse={r['vol_traj_rmse_mean']:.3e}±{r['vol_traj_rmse_std']:.1e}  "
            f"q_rms={r['q_final_rms_err_mean']:.3e}±{r['q_final_rms_err_std']:.1e}  "
            f"forces_max={r['forces_max_mean']:.2f}±{r['forces_max_std']:.2f}"
        )

    # Timing sanity warning: if per-step time decreases a lot when steps increase,
    # you probably measured async kernel launch time, not true GPU time.
    if "t_forward_s_mean" in group.columns:
        if len(group) >= 2:
            steps = group["total_steps"].to_numpy(dtype=float)
            tmean = group["t_forward_s_mean"].to_numpy(dtype=float)
            ms_per_step = 1e3 * (tmean / np.maximum(steps, 1.0))
            ratio = np.max(ms_per_step) / max(np.min(ms_per_step), 1e-12)
            if ratio > 3.0:
                print(
                    "\n[WARN] Forward timing looks unsynchronized (ms/step varies a lot across dt). "
                    "For real timing, add wp.synchronize() right after tendon.forward()."
                )

    # --- Plots you likely care about most ---
    plot_metric_vs_dt(df, group, "vol_rel_err", "Volume rel. error vs reference [-]",
                      "01_vol_rel_err_vs_dt.png", logy=True)

    plot_metric_vs_dt(df, group, "vol_traj_rmse", "Volume trajectory RMSE [vox units]",
                      "02_vol_traj_rmse_vs_dt.png", logy=True)

    plot_metric_vs_dt(df, group, "q_final_rms_err", "Final particle position RMS error [m]",
                      "03_q_final_rms_err_vs_dt.png", logy=True)

    # absolute volume final (with reference line)
    if "vol_final" in df.columns and "vol_final_ref" in df.columns:
        plt.figure(figsize=(7.5, 5))
        plt.scatter(df["dt_us"], df["vol_final"], alpha=0.7, label="rep")
        plt.errorbar(
            group["dt_us"],
            group["vol_final_mean"],
            yerr=group["vol_final_std"],
            marker="o",
            linestyle="-",
            capsize=3,
            label="mean±std",
        )
        # reference line (first non-nan ref)
        ref = group["vol_final_ref"].dropna()
        if len(ref) > 0:
            plt.axhline(float(ref.iloc[0]), linestyle="--", linewidth=1.5, label="reference vol_final")
        plt.xlabel("dt [µs]")
        plt.ylabel("vol_final")
        plt.title("Final enclosed volume vs dt")
        plt.grid(True, alpha=0.3)
        plt.xscale("log")
        plt.legend()
        savefig(os.path.join(OUT_DIR, "04_vol_final_vs_dt.png"))

    # force behaviour vs dt (this will highlight saturation)
    if "forces_max_mean" in group.columns and "forces_mean_mean" in group.columns:
        plt.figure(figsize=(7.5, 5))
        plt.plot(group["dt_us"], group["forces_mean_mean"], marker="o", linestyle="-", label="forces_mean")
        plt.plot(group["dt_us"], group["forces_max_mean"], marker="o", linestyle="-", label="forces_max")
        plt.xlabel("dt [µs]")
        plt.ylabel("Force [N]")
        plt.title("Force optimisation outcome vs dt")
        plt.grid(True, alpha=0.3)
        plt.xscale("log")
        plt.legend()
        savefig(os.path.join(OUT_DIR, "05_forces_vs_dt.png"))

    # time vs dt (measured)
    if "t_forward_s_mean" in group.columns and "t_forceopt_s_mean" in group.columns:
        plt.figure(figsize=(7.5, 5))
        plt.plot(group["dt_us"], group["t_forward_s_mean"], marker="o", linestyle="-", label="t_forward_s (measured)")
        plt.plot(group["dt_us"], group["t_forceopt_s_mean"], marker="o", linestyle="-", label="t_forceopt_s (measured)")
        plt.xlabel("dt [µs]")
        plt.ylabel("Time [s]")
        plt.title("Measured runtime vs dt (may be async if not synchronized)")
        plt.grid(True, alpha=0.3)
        plt.xscale("log")
        plt.legend()
        savefig(os.path.join(OUT_DIR, "06_time_vs_dt.png"))

    # Pareto-ish: error vs runtime
    if "t_forward_s_mean" in group.columns and "vol_traj_rmse_mean" in group.columns:
        plt.figure(figsize=(7.5, 5))
        plt.scatter(group["t_forward_s_mean"], group["vol_traj_rmse_mean"])
        for _, r in group.iterrows():
            plt.annotate(
                f"sub={int(r['sim_substeps'])}",
                (r["t_forward_s_mean"], r["vol_traj_rmse_mean"]),
                textcoords="offset points",
                xytext=(4, 4),
                fontsize=8,
            )
        plt.xlabel("t_forward_s mean [s] (measured)")
        plt.ylabel("vol_traj_rmse mean")
        plt.title("Pareto: volume traj error vs forward runtime")
        plt.grid(True, alpha=0.3)
        savefig(os.path.join(OUT_DIR, "07_pareto_rmse_vs_forward_time.png"))

    # Simple heatmap (fps x sim_substeps) for vol_rel_err_mean
    if "vol_rel_err_mean" in group.columns:
        piv = group.pivot(index="fps", columns="sim_substeps", values="vol_rel_err_mean")
        plt.figure(figsize=(7.5, 2.5 + 0.6 * len(piv.index)))
        plt.imshow(piv.values, aspect="auto", interpolation="nearest")
        plt.colorbar(label="vol_rel_err_mean")
        plt.xticks(range(len(piv.columns)), [str(c) for c in piv.columns])
        plt.yticks(range(len(piv.index)), [str(i) for i in piv.index])
        plt.xlabel("sim_substeps")
        plt.ylabel("fps")
        plt.title("Heatmap: mean volume relative error")
        savefig(os.path.join(OUT_DIR, "08_heatmap_vol_rel_err.png"))

    print(f"\n[OK] Plots saved to: {OUT_DIR}")


if __name__ == "__main__":
    main()
