import os
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

# Load data
CSV_PATH = "dt_sweep_light_cloth_v02_results_cleaned.csv"   # adapt if needed
df = pd.read_csv(CSV_PATH)

OUT_DIR = "dt_sweep_plots"
os.makedirs(OUT_DIR, exist_ok=True)

# Ensure dtypes are sane
df["dt"] = df["dt"].astype(float)
df["real_time_per_step_ms"] = df["real_time_per_step_ms"].astype(float)
df["diff_norm"] = df["diff_norm"].astype(float)

df["dt_us"] = df["dt"] * 1e6 # for nicer axis labels

# qualitative cloth flag (0/1)
if "cloth_ok" in df.columns:
    df["cloth_ok"] = df["cloth_ok"].astype(int)
else:
    df["cloth_ok"] = 1  # fallback: treat all as ok if column missing


# Aggregate over iterations: one row per (fps, sim_substeps)
group = (
    df.groupby(["fps", "sim_substeps"], as_index=False)
      .agg(
          dt=("dt", "first"),
          dt_us=("dt_us", "first"),
          real_time_per_step_ms=("real_time_per_step_ms", "mean"),
          diff_mean=("diff_norm", "mean"),
          diff_std=("diff_norm", "std"),
          phys_ok_rate=("phys_ok", "mean"),
          num_ok_rate=("num_ok", "mean"),
          cloth_ok_frac=("cloth_ok", "mean"),  # NEW
          total_steps=("total_steps", "first"),  # needed later
          runs=("diff_norm", "count"),
      )
)

# Fill NaN std (happens if only 1 run) with 0
group["diff_std"] = group["diff_std"].fillna(0.0)

GOOD_THRESH = 0.5  # majority of runs must be visually ok
group["cloth_ok_flag"] = group["cloth_ok_frac"] >= GOOD_THRESH

# total runtime per simulation
group["total_runtime_ms"] = group["real_time_per_step_ms"] * group["total_steps"]
group["total_runtime_s"] = group["total_runtime_ms"] / 1000.0


# Plot: diff_norm vs dt (one line per fps)
plt.figure(figsize=(8, 5))

for fps, g in group.groupby("fps"):
    g_sorted = g.sort_values("dt")

    # main line
    plt.errorbar(
        g_sorted["dt_us"],
        g_sorted["diff_mean"],
        yerr=g_sorted["diff_std"],
        marker="o",
        linestyle="-",
        label=f"{fps} fps",
        capsize=3,
    )

    # overlay crosses where cloth_ok_flag is False
    bad = g_sorted[~g_sorted["cloth_ok_flag"]]
    if not bad.empty:
        plt.scatter(
            bad["dt_us"],
            bad["diff_mean"],
            marker="x",
            s=80,
            color="red",
            linewidths=1.5,
        )

plt.xlabel("dt [µs]")
plt.ylabel("mean diff_norm [m]")
plt.title("Physical error vs dt\n(red × = cloth visually bad)")
plt.grid(True, alpha=0.3)
plt.legend(title="fps")
plt.tight_layout()
plt.savefig(os.path.join(OUT_DIR, "light_cloth_v02_cleaned_diff_vs_dt.png"), dpi=200)



# Plot: real_time_per_step_ms vs dt (one line per fps)
plt.figure(figsize=(8, 5))

for fps, g in group.groupby("fps"):
    g_sorted = g.sort_values("dt")

    plt.plot(
        g_sorted["dt_us"],
        g_sorted["real_time_per_step_ms"],
        marker="o",
        linestyle="-",
        label=f"{fps} fps",
    )

    bad = g_sorted[~g_sorted["cloth_ok_flag"]]
    if not bad.empty:
        plt.scatter(
            bad["dt_us"],
            bad["real_time_per_step_ms"],
            marker="x",
            s=80,
            color="red",
            linewidths=1.5,
        )

plt.xlabel("dt [µs]")
plt.ylabel("real_time_per_step [ms / integration step]")
plt.title("Runtime per integration step vs dt\n(red × = cloth visually bad)")
plt.grid(True, alpha=0.3)
plt.legend(title="fps")
plt.tight_layout()
plt.savefig(os.path.join(OUT_DIR, "light_cloth_v02_cleaned_runtime_vs_dt.png"), dpi=200)




# total runtime vs diff_norm

plt.figure(figsize=(8, 5))

good = group[group["cloth_ok_flag"]]
bad  = group[~group["cloth_ok_flag"]]

sc_good = plt.scatter(
    good["total_runtime_s"],
    good["diff_mean"],
    c=good["fps"],
    s=70,
    cmap="viridis",
    marker="o",
    label="cloth ok",
)

plt.scatter(
    bad["total_runtime_s"],
    bad["diff_mean"],
    c=bad["fps"],
    s=70,
    cmap="viridis",
    marker="x",
    label="cloth bad",
)

for _, row in good.iterrows():
    label = f"{int(row['fps'])}fps/{int(row['sim_substeps'])}sub"
    plt.annotate(
        label,
        (row["total_runtime_s"], row["diff_mean"]),
        textcoords="offset points",
        xytext=(3, 3),
        fontsize=7,
    )

plt.xlabel("total runtime per simulation [s]")
plt.ylabel("mean diff_norm [m]")
plt.title("diff_norm vs total runtime\n(circles = cloth ok, crosses = cloth bad)")
cbar = plt.colorbar(sc_good)
cbar.set_label("fps")
plt.grid(True, alpha=0.3)
plt.legend()
plt.tight_layout()
plt.savefig(os.path.join(OUT_DIR, "light_cloth_v02_cleaned_diff_vs_total_runtime.png"), dpi=200)







# Plot: diff_norm vs real_time_per_step_ms (Pareto-style)
plt.figure(figsize=(8, 5))

good = group[group["cloth_ok_flag"]]
bad  = group[~group["cloth_ok_flag"]]

sc_good = plt.scatter(
    good["real_time_per_step_ms"],
    good["diff_mean"],
    c=good["fps"],
    s=70,
    cmap="viridis",
    marker="o",
    label="cloth ok",
)

plt.scatter(
    bad["real_time_per_step_ms"],
    bad["diff_mean"],
    c=bad["fps"],
    s=70,
    cmap="viridis",
    marker="x",
    label="cloth bad",
)

for _, row in group.iterrows():
    label = f"{int(row['fps'])}fps/{int(row['sim_substeps'])}sub"
    plt.annotate(
        label,
        (row["real_time_per_step_ms"], row["diff_mean"]),
        textcoords="offset points",
        xytext=(3, 3),
        fontsize=7,
    )

plt.xlabel("real_time_per_step [ms / step]")
plt.ylabel("mean diff_norm [m]")
plt.title("diff_norm vs runtime per step\n(circles = cloth ok, crosses = cloth bad)")
cbar = plt.colorbar(sc_good)
cbar.set_label("fps")
plt.grid(True, alpha=0.3)
plt.tight_layout()
plt.savefig(os.path.join(OUT_DIR, "light_cloth_v02_cleaned_diff_vs_runtime.png"), dpi=200)



# heatmap (fps, sim_substeps)
try:
    import seaborn as sns

    heat_df = group.pivot(index="fps", columns="sim_substeps", values="diff_mean")
    cloth_df = group.pivot(index="fps", columns="sim_substeps", values="cloth_ok_frac")

    plt.figure(figsize=(7, 5))
    ax = sns.heatmap(
        heat_df,
        annot=True,
        fmt=".3f",
        cmap="mako",
        cbar_kws={"label": "mean diff_norm [m]"},
    )
    plt.xlabel("sim_substeps")
    plt.ylabel("fps")
    plt.title("Heatmap: mean diff_norm\nred × = cloth visually bad")

    for i, fps in enumerate(heat_df.index):
        for j, sub in enumerate(heat_df.columns):
            frac = cloth_df.loc[fps, sub]
            if frac < GOOD_THRESH:
                ax.text(
                    j + 0.5,
                    i + 0.5,
                    "×",
                    ha="center",
                    va="center",
                    color="red",
                    fontsize=14,
                    fontweight="bold",
                )

    plt.tight_layout()
    plt.savefig(os.path.join(OUT_DIR, "light_cloth_v02_cleaned_heatmap_diff_norm.png"), dpi=200)





except ImportError:
    print("seaborn not installed, skipping heatmap")


print(f"\nLogged results to: {OUT_DIR}")
