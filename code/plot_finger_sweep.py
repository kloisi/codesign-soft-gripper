# plot_finger_sweep.py
#
# Reads the finger count sweep CSV from run_sweep_voxvol.py and generates many plots:
#   Heatmaps:
#     - Final_VoxVol, Init_VoxVol, Delta_VoxVol
#     - Final/Init ratio, Delta/Init ratio
#     - Final_Loss, Avg_Force, Force imbalance (CV/range)
#     - Radius_Mean and radius spread (std/range)
#
#   Per object curves:
#     - Final_VoxVol vs Num_Fingers (spaghetti + best marker)
#     - Normalised performance vs fingers (Final / object max across fingers)
#
#   Aggregate across objects:
#     - mean ± std vs finger count for key metrics
#     - boxplots per finger count (Final_VoxVol, Final/Init, Loss, Avg_Force, Force_CV)
#
#   "Which finger count wins" type plots:
#     - best finger count per object (hist)
#     - win rate per finger count
#     - mean rank per finger count (lower is better)
#     - robustness curves (10th percentile normalised performance)
#
#   Tradeoffs + correlation:
#     - scatter: Final_VoxVol vs Avg_Force (coloured by Num_Fingers)
#     - scatter: Final_VoxVol vs Final_Loss
#     - scatter: Normalised performance vs force imbalance
#     - correlation heatmap between derived metrics
#
# Output:
#   Saves each plot twice:
#     - normal PNG
#     - *_presentation.png (black background, thicker lines, larger fonts)
#
# Run:
#   python plot_finger_sweep.py sweep_results
#   python plot_finger_sweep.py sweep_results/experiment_data_voxvol.csv
#
# Notes:
#   - This sweep logs only init and final voxel volumes (no time series), so "normalised over time"
#     is not possible here. Instead we plot "normalised across finger count per object":
#       Final_VoxVol / max_final_for_that_object

import os
import ast
import argparse
import numpy as np
import pandas as pd

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
from matplotlib.collections import PathCollection, PolyCollection


# -----------------------------
# Presentation plot saving
# -----------------------------
def _is_blackish(color) -> bool:
    try:
        r, g, b, a = mcolors.to_rgba(color)
        return (r + g + b) < 0.20 and a > 0.0
    except Exception:
        return False


def _apply_presentation_style(fig, font_scale=1.35, line_scale=1.8, marker_scale=1.35) -> None:
    try:
        fig.patch.set_facecolor("black")
    except Exception:
        pass

    try:
        w, h = fig.get_size_inches()
        fig.set_size_inches(w * 1.08, h * 1.08, forward=True)
    except Exception:
        pass

    for ax in fig.get_axes():
        try:
            ax.set_facecolor("black")
        except Exception:
            pass

        try:
            for sp in ax.spines.values():
                sp.set_color("white")
                sp.set_linewidth(max(1.0, float(sp.get_linewidth()) * 1.1))
        except Exception:
            pass

        try:
            ax.title.set_color("white")
            ax.title.set_fontsize(float(ax.title.get_fontsize()) * font_scale)
        except Exception:
            pass

        try:
            ax.xaxis.label.set_color("white")
            ax.yaxis.label.set_color("white")
            ax.xaxis.label.set_fontsize(float(ax.xaxis.label.get_fontsize()) * font_scale)
            ax.yaxis.label.set_fontsize(float(ax.yaxis.label.get_fontsize()) * font_scale)
        except Exception:
            pass

        try:
            ax.tick_params(axis="both", which="both", colors="white")
            for lab in ax.get_xticklabels() + ax.get_yticklabels():
                lab.set_color("white")
                lab.set_fontsize(float(lab.get_fontsize()) * font_scale)
        except Exception:
            pass

        try:
            for gl in ax.get_xgridlines() + ax.get_ygridlines():
                gl.set_color("white")
                gl.set_alpha(0.20)
                gl.set_linewidth(max(0.8, float(gl.get_linewidth()) * 1.1))
        except Exception:
            pass

        # Lines
        try:
            lines = ax.get_lines()
            n = len(lines)
            turbo_colors = plt.cm.turbo(np.linspace(0.05, 0.95, n)) if n >= 8 else None

            for i, ln in enumerate(lines):
                try:
                    lw = float(ln.get_linewidth())
                    ln.set_linewidth(max(1.6, lw * line_scale))
                except Exception:
                    pass

                try:
                    if turbo_colors is not None:
                        ln.set_color(turbo_colors[i])
                    else:
                        if _is_blackish(ln.get_color()):
                            ln.set_color("white")
                except Exception:
                    pass

                try:
                    a = ln.get_alpha()
                    if a is None:
                        a = 1.0
                    if a < 0.65:
                        ln.set_alpha(0.75 if turbo_colors is not None else 0.85)
                    else:
                        ln.set_alpha(min(1.0, float(a)))
                except Exception:
                    pass

                try:
                    ms = ln.get_markersize()
                    if ms is not None and float(ms) > 0:
                        ln.set_markersize(float(ms) * marker_scale)
                except Exception:
                    pass
        except Exception:
            pass

        # Collections
        try:
            for coll in ax.collections:
                # Special handling: dark tint overlay for "unreliable" single points
                try:
                    if getattr(coll, "get_gid", lambda: None)() == "unreliable_point_mask":
                        coll.set_facecolor("black")
                        coll.set_alpha(0.45)          # same vibe as the region overlay
                        coll.set_edgecolor("none")    # prevents the white halo
                        coll.set_linewidths([0.0])
                        coll.set_zorder(11)           # above the marker so it darkens it
                        continue
                except Exception:
                    pass
                if isinstance(coll, PolyCollection) and not isinstance(coll, PathCollection):
                    try:
                        coll.set_alpha(0.40)
                    except Exception:
                        pass
                    continue

                try:
                    a = coll.get_alpha()
                    if a is None or float(a) < 0.70:
                        coll.set_alpha(0.90)
                except Exception:
                    pass

                try:
                    if isinstance(coll, PathCollection):
                        coll.set_edgecolor("white")
                        lw = coll.get_linewidths()
                        if lw is None or len(lw) == 0:
                            coll.set_linewidths([0.6])
                        else:
                            coll.set_linewidths(np.maximum(0.6, np.asarray(lw, dtype=float) * 1.2))
                except Exception:
                    pass

                try:
                    sz = coll.get_sizes()
                    if sz is not None and len(sz):
                        coll.set_sizes(np.asarray(sz, dtype=float) * (marker_scale ** 2))
                except Exception:
                    pass
        except Exception:
            pass

        # Patches
        try:
            for p in ax.patches:
                # Special handling for our "unreliable region" band in presentation mode:
                try:
                    if getattr(p, "get_gid", lambda: None)() == "unreliable_band":
                        # Make it a black tint overlay that darkens underlying data
                        p.set_facecolor("black")
                        p.set_alpha(0.45)      # tune: 0.25 (subtle) .. 0.60 (strong)
                        p.set_zorder(10)       # put it above lines/fills so it actually darkens them
                        p.set_edgecolor("none")
                        p.set_linewidth(0.0)
                        continue
                except Exception:
                    pass

                # default patch styling (everything else)
                try:
                    lw = float(p.get_linewidth()) if p.get_linewidth() is not None else 0.0
                    p.set_linewidth(max(1.0, lw * line_scale))
                    p.set_edgecolor("white")
                    a = p.get_alpha()
                    if a is None or float(a) < 0.85:
                        p.set_alpha(0.95)
                except Exception:
                    pass
        except Exception:
            pass



        # Text
        try:
            for t in ax.texts:
                try:
                    t.set_color("white")
                    t.set_fontsize(float(t.get_fontsize()) * font_scale)
                    if t.get_bbox_patch() is None:
                        t.set_bbox(dict(facecolor="black", alpha=0.35, edgecolor="none", pad=0.4))
                except Exception:
                    pass
        except Exception:
            pass

        # Legend
        try:
            leg = ax.get_legend()
            if leg is not None:
                try:
                    frame = leg.get_frame()
                    frame.set_facecolor("black")
                    frame.set_edgecolor("white")
                    frame.set_alpha(0.65)
                except Exception:
                    pass
                try:
                    for txt in leg.get_texts():
                        txt.set_color("white")
                        txt.set_fontsize(float(txt.get_fontsize()) * font_scale)
                except Exception:
                    pass
        except Exception:
            pass


def savefig(path: str):
    plt.tight_layout()
    plt.savefig(path, dpi=200)

    fig = plt.gcf()
    _apply_presentation_style(fig)
    base, ext = os.path.splitext(path)
    pres_path = f"{base}_presentation{ext}"
    plt.tight_layout()
    plt.savefig(pres_path, dpi=220, facecolor=fig.get_facecolor(), edgecolor="none")
    plt.close()


def ensure_dir(d: str) -> None:
    os.makedirs(d, exist_ok=True)


# -----------------------------
# IO + parsing helpers
# -----------------------------
def _find_csv_from_log_path(log_path: str) -> str:
    """
    Accepts either:
      - a direct CSV path
      - a directory containing the CSV
    """
    log_path = os.path.expanduser(log_path)
    if os.path.isfile(log_path) and log_path.lower().endswith(".csv"):
        return log_path

    if os.path.isdir(log_path):
        # prefer the default name
        cand = os.path.join(log_path, "experiment_data_voxvol.csv")
        if os.path.exists(cand):
            return cand

        # else: pick the first csv
        cands = [f for f in os.listdir(log_path) if f.lower().endswith(".csv")]
        cands = sorted(cands)
        if len(cands) == 0:
            raise FileNotFoundError(f"No CSV files found in directory: {log_path}")
        return os.path.join(log_path, cands[0])

    raise FileNotFoundError(f"Could not find CSV. Given: {log_path}")


def _parse_list_cell(x):
    """
    Parses a CSV cell that looks like "[1.0, 2.0, 3.0]".
    Returns list[float]. Returns [] on failure.
    """
    if x is None or (isinstance(x, float) and np.isnan(x)):
        return []
    s = str(x).strip()
    if s == "" or s.lower() in ("nan", "none"):
        return []
    try:
        v = ast.literal_eval(s)
    except Exception:
        return []
    if isinstance(v, (list, tuple, np.ndarray)):
        out = []
        for a in v:
            try:
                out.append(float(a))
            except Exception:
                pass
        return out
    return []


def read_finger_sweep_csv(csv_path: str) -> pd.DataFrame:
    df = pd.read_csv(csv_path, quotechar='"')

    # keep only ok rows
    if "Status" in df.columns:
        df = df[df["Status"].astype(str).str.strip() == "ok"].copy()

    # numeric coercion
    num_cols = [
        "Num_Fingers",
        "PoseOptFrames", "PoseOptIters", "SimNumFrames",
        "ForceOptIters", "ForceOptFrames", "ForceOptLR",
        "Radius_Mean",
        "Final_Loss",
        "Avg_Force",
        "Init_VoxVol", "Final_VoxVol", "Delta_VoxVol",
    ]
    for c in num_cols:
        if c in df.columns:
            df[c] = pd.to_numeric(df[c], errors="coerce")

    # parse list columns
    df["Radius_List_parsed"] = df.get("Radius_List", "").apply(_parse_list_cell)
    df["All_Forces_parsed"] = df.get("All_Forces", "").apply(_parse_list_cell)

    # derived stats from radii and forces
    def _list_stats(v):
        if not isinstance(v, list) or len(v) == 0:
            return dict(mean=np.nan, std=np.nan, mn=np.nan, mx=np.nan, rng=np.nan, cv=np.nan)
        a = np.asarray(v, dtype=np.float64)
        a = a[np.isfinite(a)]
        if a.size == 0:
            return dict(mean=np.nan, std=np.nan, mn=np.nan, mx=np.nan, rng=np.nan, cv=np.nan)
        m = float(np.mean(a))
        s = float(np.std(a))
        mn = float(np.min(a))
        mx = float(np.max(a))
        rng = float(mx - mn)
        cv = float(s / m) if abs(m) > 1e-12 else np.nan
        return dict(mean=m, std=s, mn=mn, mx=mx, rng=rng, cv=cv)

    rad_stats = df["Radius_List_parsed"].apply(_list_stats)
    frc_stats = df["All_Forces_parsed"].apply(_list_stats)

    df["Radius_List_mean"] = rad_stats.apply(lambda d: d["mean"])
    df["Radius_List_std"] = rad_stats.apply(lambda d: d["std"])
    df["Radius_List_min"] = rad_stats.apply(lambda d: d["mn"])
    df["Radius_List_max"] = rad_stats.apply(lambda d: d["mx"])
    df["Radius_List_range"] = rad_stats.apply(lambda d: d["rng"])
    df["Radius_List_cv"] = rad_stats.apply(lambda d: d["cv"])

    df["Force_min"] = frc_stats.apply(lambda d: d["mn"])
    df["Force_max"] = frc_stats.apply(lambda d: d["mx"])
    df["Force_std"] = frc_stats.apply(lambda d: d["std"])
    df["Force_range"] = frc_stats.apply(lambda d: d["rng"])
    df["Force_cv"] = frc_stats.apply(lambda d: d["cv"])

    # more derived metrics
    df["Final_over_Init"] = df["Final_VoxVol"] / df["Init_VoxVol"]
    df["Delta_over_Init"] = df["Delta_VoxVol"] / df["Init_VoxVol"]
    df["Final_per_finger"] = df["Final_VoxVol"] / df["Num_Fingers"]
    df["Delta_per_finger"] = df["Delta_VoxVol"] / df["Num_Fingers"]

    # a simple "efficiency": final volume per unit average force
    df["Final_per_force"] = df["Final_VoxVol"] / df["Avg_Force"]

    # normalise within each object: Final / max Final across fingers for that object
    df["Final_norm_objectmax"] = np.nan
    for obj, g in df.groupby("Object"):
        mx = float(np.nanmax(g["Final_VoxVol"].to_numpy()))
        if np.isfinite(mx) and mx > 1e-12:
            df.loc[g.index, "Final_norm_objectmax"] = g["Final_VoxVol"] / mx

    return df


# -----------------------------
# Plot helpers
# -----------------------------
def _sorted_finger_counts(df: pd.DataFrame):
    return sorted([int(x) for x in df["Num_Fingers"].dropna().unique()])


def plot_heatmap(df: pd.DataFrame, out_dir: str, value_col: str, title: str, fname: str, annotate: bool = False):
    """
    Heatmap: rows = objects, cols = finger counts.
    """
    if value_col not in df.columns:
        return

    piv = df.pivot_table(index="Object", columns="Num_Fingers", values=value_col, aggfunc="mean")
    if piv.empty:
        return

    # sort objects by their mean value (helps readability)
    piv["__mean__"] = np.nanmean(piv.to_numpy(), axis=1)
    piv = piv.sort_values("__mean__", ascending=False).drop(columns="__mean__")

    mat = piv.to_numpy()
    objs = list(piv.index)
    fcs = [int(c) for c in piv.columns]

    plt.figure(figsize=(1.0 + 0.9 * len(fcs), max(4.5, 0.35 * len(objs))))
    im = plt.imshow(mat, aspect="auto")
    plt.colorbar(im, fraction=0.03, pad=0.03)
    plt.title(title)
    plt.xlabel("Number of fingers")
    plt.ylabel("Object")

    plt.xticks(ticks=np.arange(len(fcs)), labels=fcs)
    plt.yticks(ticks=np.arange(len(objs)), labels=objs)

    if annotate and mat.size <= 900:
        for i in range(mat.shape[0]):
            for j in range(mat.shape[1]):
                v = mat[i, j]
                if np.isfinite(v):
                    plt.text(j, i, f"{v:.3g}", ha="center", va="center", fontsize=8)

    savefig(os.path.join(out_dir, fname))


def plot_spaghetti(df: pd.DataFrame, out_dir: str, ycol: str, title: str, ylabel: str, fname: str):
    if ycol not in df.columns:
        return

    plt.figure(figsize=(11, 6))
    for obj, g in df.groupby("Object"):
        g = g.sort_values("Num_Fingers")
        plt.plot(g["Num_Fingers"], g[ycol], marker="o", alpha=0.55)

        # mark best finger count for that object (max of ycol)
        try:
            idx = int(np.nanargmax(g[ycol].to_numpy()))
            xf = float(g["Num_Fingers"].iloc[idx])
            yf = float(g[ycol].iloc[idx])
            if np.isfinite(xf) and np.isfinite(yf):
                plt.scatter([xf], [yf], s=55, zorder=3)
        except Exception:
            pass

    plt.title(title)
    plt.xlabel("Number of fingers")
    plt.ylabel(ylabel)
    plt.grid(True, alpha=0.25)
    savefig(os.path.join(out_dir, fname))


def plot_mean_std(
    df: pd.DataFrame,
    out_dir: str,
    ycol: str,
    title: str,
    ylabel: str,
    fname: str,
    unreliable_spans=None,      # e.g. [(3, 4)]
    unreliable_points=None,     # e.g. [3]
):
    if ycol not in df.columns:
        return

    agg = df.groupby("Num_Fingers")[ycol].agg(["mean", "std", "count"]).reset_index()
    agg = agg.sort_values("Num_Fingers")
    if agg.empty:
        return

    x = agg["Num_Fingers"].to_numpy(dtype=float)
    m = agg["mean"].to_numpy(dtype=float)
    s = agg["std"].to_numpy(dtype=float)

    neon_cyan = "#00E5FF"

    plt.figure(figsize=(6.0, 4.4))
    ax = plt.gca()

    # --- Unreliable region shading (behind everything) ---
    if unreliable_spans:
        for (x0, x1) in unreliable_spans:
            band = ax.axvspan(
                x0, x1,
                facecolor="0.75",   # light grey
                alpha=0.18,
                zorder=0,
            )
            band.set_gid("unreliable_band")
            try:
                band.set_edgecolor("none")
                band.set_linewidth(0.0)
            except Exception:
                pass

    # --- Mean ± std ---
    plt.fill_between(
        x, m - s, m + s,
        color=neon_cyan,
        alpha=0.18,
        linewidth=0.0,
        zorder=1,
    )

    plt.plot(
        x, m,
        color=neon_cyan,
        marker="o",
        linewidth=2.2,
        markersize=5.5,
        markerfacecolor=neon_cyan,
        markeredgecolor=neon_cyan,
        zorder=2,
    )

    # --- Optional: de-emphasise specific finger counts (e.g. 3 fingers) ---
    if unreliable_points:
        bad = np.isin(x, np.asarray(unreliable_points, dtype=float))
        if np.any(bad):
            ptmask = ax.scatter(
                x[bad], m[bad],
                s=130,                 # a bit larger than the marker so it "covers" it
                color="0.75",          # looks ok on white background
                alpha=0.18,
                edgecolors="none",
                linewidths=0.0,
                zorder=9,
            )
            ptmask.set_gid("unreliable_point_mask")


    plt.title(title)
    plt.xlabel("Number of fingers")
    plt.ylabel(ylabel)
    plt.grid(True, alpha=0.25)
    savefig(os.path.join(out_dir, fname))



def plot_box_by_fingers(df: pd.DataFrame, out_dir: str, ycol: str, title: str, ylabel: str, fname: str):
    if ycol not in df.columns:
        return
    fcs = _sorted_finger_counts(df)
    data = []
    labels = []
    for nf in fcs:
        v = df.loc[df["Num_Fingers"] == nf, ycol].to_numpy(dtype=float)
        v = v[np.isfinite(v)]
        if v.size == 0:
            continue
        data.append(v)
        labels.append(str(nf))

    if len(data) < 2:
        return

    plt.figure(figsize=(10.5, 5.5))
    plt.boxplot(data, labels=labels, showfliers=True)
    plt.title(title)
    plt.xlabel("Number of fingers")
    plt.ylabel(ylabel)
    plt.grid(True, axis="y", alpha=0.25)
    savefig(os.path.join(out_dir, fname))


def plot_scatter(df: pd.DataFrame, out_dir: str, xcol: str, ycol: str, title: str, xlabel: str, ylabel: str, fname: str):
    if xcol not in df.columns or ycol not in df.columns:
        return

    d = df[[xcol, ycol, "Num_Fingers"]].copy()
    d = d.replace([np.inf, -np.inf], np.nan).dropna(subset=[xcol, ycol, "Num_Fingers"])
    if d.empty:
        return

    plt.figure(figsize=(8.5, 6))
    # colour by finger count (simple categorical scatter)
    for nf in sorted(d["Num_Fingers"].unique()):
        sub = d[d["Num_Fingers"] == nf]
        plt.scatter(sub[xcol], sub[ycol], s=35, alpha=0.8, label=f"{int(nf)}")

    plt.title(title)
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    plt.grid(True, alpha=0.25)
    plt.legend(title="Fingers", fontsize="small", ncol=2)
    savefig(os.path.join(out_dir, fname))


def plot_best_finger_hist(df: pd.DataFrame, out_dir: str, metric: str, fname: str, title: str):
    """
    For each object, pick the finger count with max(metric), then histogram.
    """
    if metric not in df.columns:
        return

    best = []
    for obj, g in df.groupby("Object"):
        g = g.dropna(subset=[metric, "Num_Fingers"])
        if g.empty:
            continue
        try:
            idx = int(np.nanargmax(g[metric].to_numpy(dtype=float)))
            best_nf = int(g["Num_Fingers"].iloc[idx])
            best.append(best_nf)
        except Exception:
            continue

    if len(best) == 0:
        return

    vals, counts = np.unique(np.asarray(best, dtype=int), return_counts=True)
    order = np.argsort(vals)
    vals = vals[order]
    counts = counts[order]

    plt.figure(figsize=(9, 5))
    plt.bar(vals.astype(str), counts)
    plt.title(title)
    plt.xlabel("Best number of fingers")
    plt.ylabel("Number of objects")
    savefig(os.path.join(out_dir, fname))


def plot_win_rate(df: pd.DataFrame, out_dir: str, metric: str, fname: str, title: str):
    """
    Win rate = how often a finger count achieves the best metric for that object.
    """
    if metric not in df.columns:
        return

    fcs = _sorted_finger_counts(df)
    wins = {nf: 0 for nf in fcs}
    total = 0

    for obj, g in df.groupby("Object"):
        g = g.dropna(subset=[metric, "Num_Fingers"])
        if g.empty:
            continue
        total += 1
        try:
            idx = int(np.nanargmax(g[metric].to_numpy(dtype=float)))
            nf = int(g["Num_Fingers"].iloc[idx])
            if nf in wins:
                wins[nf] += 1
        except Exception:
            pass

    if total == 0:
        return

    xs = [str(nf) for nf in fcs]
    ys = [wins[nf] / total for nf in fcs]

    plt.figure(figsize=(9, 5))
    plt.bar(xs, ys)
    plt.ylim(0.0, 1.0)
    plt.title(title)
    plt.xlabel("Number of fingers")
    plt.ylabel("Win rate")
    plt.grid(True, axis="y", alpha=0.25)
    savefig(os.path.join(out_dir, fname))


def plot_mean_rank(df: pd.DataFrame, out_dir: str, metric: str, fname: str, title: str):
    """
    Per object: rank finger counts by metric (1 = best). Then average rank across objects.
    """
    if metric not in df.columns:
        return

    fcs = _sorted_finger_counts(df)
    ranks = {nf: [] for nf in fcs}

    for obj, g in df.groupby("Object"):
        g = g.dropna(subset=[metric, "Num_Fingers"])
        if g.empty:
            continue
        g = g.sort_values(metric, ascending=False)  # max is best
        # assign ranks among available finger counts for that object
        for r, (_, row) in enumerate(g.iterrows(), start=1):
            nf = int(row["Num_Fingers"])
            if nf in ranks:
                ranks[nf].append(r)

    xs = []
    ys = []
    ystd = []
    for nf in fcs:
        a = np.asarray(ranks[nf], dtype=float)
        a = a[np.isfinite(a)]
        if a.size == 0:
            continue
        xs.append(str(nf))
        ys.append(float(np.mean(a)))
        ystd.append(float(np.std(a)))

    if len(xs) < 2:
        return

    plt.figure(figsize=(9, 5))
    plt.bar(xs, ys)
    plt.title(title)
    plt.xlabel("Number of fingers")
    plt.ylabel("Mean rank (lower is better)")
    plt.grid(True, axis="y", alpha=0.25)
    savefig(os.path.join(out_dir, fname))


def plot_robustness_curve(df: pd.DataFrame, out_dir: str, fname: str):
    """
    Robustness curve based on normalised performance (Final_norm_objectmax):
      - mean
      - 10th percentile (rough worst case)
      - 25th percentile
    """
    col = "Final_norm_objectmax"
    if col not in df.columns:
        return

    rows = []
    for nf, g in df.groupby("Num_Fingers"):
        v = g[col].to_numpy(dtype=float)
        v = v[np.isfinite(v)]
        if v.size == 0:
            continue
        rows.append({
            "Num_Fingers": int(nf),
            "mean": float(np.mean(v)),
            "p10": float(np.percentile(v, 10)),
            "p25": float(np.percentile(v, 25)),
            "p50": float(np.percentile(v, 50)),
        })

    if len(rows) < 2:
        return

    a = pd.DataFrame(rows).sort_values("Num_Fingers")
    x = a["Num_Fingers"].to_numpy()
    plt.figure(figsize=(9.5, 5.5))
    plt.plot(x, a["mean"].to_numpy(), marker="o", label="mean")
    plt.plot(x, a["p50"].to_numpy(), marker="o", label="median")
    plt.plot(x, a["p25"].to_numpy(), marker="o", label="p25")
    plt.plot(x, a["p10"].to_numpy(), marker="o", label="p10 (robustness)")

    plt.ylim(0.0, 1.05)
    plt.title("Robustness across objects (normalised final volume)")
    plt.xlabel("Number of fingers")
    plt.ylabel("Final_VoxVol / object max across fingers")
    plt.grid(True, alpha=0.25)
    plt.legend()
    savefig(os.path.join(out_dir, fname))


def plot_corr_heatmap(df: pd.DataFrame, out_dir: str, fname: str):
    cols = [
        "Num_Fingers",
        "Radius_Mean",
        "Radius_List_std",
        "Radius_List_range",
        "Final_Loss",
        "Avg_Force",
        "Force_cv",
        "Force_range",
        "Init_VoxVol",
        "Final_VoxVol",
        "Final_over_Init",
        "Final_norm_objectmax",
    ]
    cols = [c for c in cols if c in df.columns]
    if len(cols) < 4:
        return

    d = df[cols].replace([np.inf, -np.inf], np.nan).dropna(axis=0, how="any")
    if d.shape[0] < 5:
        return

    C = d.corr(numeric_only=True).to_numpy()
    labels = cols

    plt.figure(figsize=(0.7 * len(labels) + 3, 0.7 * len(labels) + 3))
    im = plt.imshow(C, vmin=-1.0, vmax=1.0, aspect="auto")
    plt.colorbar(im, fraction=0.03, pad=0.03)
    plt.xticks(np.arange(len(labels)), labels, rotation=45, ha="right")
    plt.yticks(np.arange(len(labels)), labels)
    plt.title("Correlation heatmap (successful runs)")

    # annotate
    for i in range(C.shape[0]):
        for j in range(C.shape[1]):
            plt.text(j, i, f"{C[i,j]:.2f}", ha="center", va="center", fontsize=8)

    savefig(os.path.join(out_dir, fname))


# -----------------------------
# Main
# -----------------------------
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("log", nargs="?", default="sweep_results/experiment_data_voxvol.csv",
                    help="CSV path or directory containing it (e.g. sweep_results)")
    ap.add_argument("--out", default="", help="Output directory. Default: <csv_dir>/plots_finger_sweep")
    args = ap.parse_args()

    csv_path = _find_csv_from_log_path(args.log)
    csv_dir = os.path.dirname(csv_path)

    out_dir = args.out.strip() if args.out.strip() else os.path.join(csv_dir, "plots_finger_sweep")
    ensure_dir(out_dir)

    df = read_finger_sweep_csv(csv_path)
    if df.empty:
        print("[plot] No successful rows found (Status==ok). Nothing to plot.")
        return

    # Save a cleaned table with derived metrics
    df_out = df.copy()
    df_out.to_csv(os.path.join(out_dir, "finger_sweep_cleaned_with_derived_metrics.csv"), index=False)

    # Also save per finger count aggregates
    agg_cols = [
        "Final_VoxVol", "Init_VoxVol", "Delta_VoxVol",
        "Final_over_Init", "Delta_over_Init",
        "Final_Loss", "Avg_Force",
        "Force_cv", "Force_range",
        "Radius_Mean", "Radius_List_std", "Radius_List_range",
        "Final_norm_objectmax",
    ]
    agg_cols = [c for c in agg_cols if c in df.columns]
    agg = df.groupby("Num_Fingers")[agg_cols].agg(["mean", "std", "median", "count"])
    agg.to_csv(os.path.join(out_dir, "finger_sweep_aggregates_by_finger_count.csv"))

    # -----------------------------
    # Heatmaps (object x finger count)
    # -----------------------------
    plot_heatmap(df, out_dir, "Final_VoxVol", "Final enclosed volume (Final_VoxVol)", "heatmap_final_voxvol.png", annotate=False)
    plot_heatmap(df, out_dir, "Init_VoxVol", "Initial enclosed volume (Init_VoxVol)", "heatmap_init_voxvol.png", annotate=False)
    plot_heatmap(df, out_dir, "Delta_VoxVol", "Delta enclosed volume (Final - Init)", "heatmap_delta_voxvol.png", annotate=False)

    plot_heatmap(df, out_dir, "Final_over_Init", "Final / Init enclosed volume", "heatmap_final_over_init.png", annotate=False)
    plot_heatmap(df, out_dir, "Delta_over_Init", "Delta / Init enclosed volume", "heatmap_delta_over_init.png", annotate=False)

    plot_heatmap(df, out_dir, "Final_Loss", "Final loss after force optimisation", "heatmap_final_loss.png", annotate=False)
    plot_heatmap(df, out_dir, "Avg_Force", "Average optimised force (Avg_Force)", "heatmap_avg_force.png", annotate=False)
    plot_heatmap(df, out_dir, "Force_cv", "Force imbalance (CV of per finger forces)", "heatmap_force_cv.png", annotate=False)

    plot_heatmap(df, out_dir, "Radius_Mean", "Mean initial radius (Radius_Mean)", "heatmap_radius_mean.png", annotate=False)
    plot_heatmap(df, out_dir, "Radius_List_std", "Radius spread (std across fingers)", "heatmap_radius_std.png", annotate=False)

    # -----------------------------
    # Per object curves (spaghetti)
    # -----------------------------
    plot_spaghetti(df, out_dir, "Final_VoxVol", "Final_VoxVol vs number of fingers (per object)", "Final_VoxVol", "spaghetti_final_voxvol_vs_fingers.png")
    plot_spaghetti(df, out_dir, "Init_VoxVol", "Init_VoxVol vs number of fingers (per object)", "Init_VoxVol", "spaghetti_init_voxvol_vs_fingers.png")
    plot_spaghetti(df, out_dir, "Delta_VoxVol", "Delta_VoxVol vs number of fingers (per object)", "Delta_VoxVol", "spaghetti_delta_voxvol_vs_fingers.png")

    plot_spaghetti(df, out_dir, "Final_Loss", "Final loss vs number of fingers (per object)", "Final_Loss", "spaghetti_final_loss_vs_fingers.png")
    plot_spaghetti(df, out_dir, "Avg_Force", "Avg_Force vs number of fingers (per object)", "Avg_Force", "spaghetti_avg_force_vs_fingers.png")
    plot_spaghetti(df, out_dir, "Radius_Mean", "Radius_Mean vs number of fingers (per object)", "Radius_Mean", "spaghetti_radius_mean_vs_fingers.png")

    # Normalised performance across finger counts (per object)
    plot_spaghetti(
        df, out_dir, "Final_norm_objectmax",
        "Normalised final volume vs fingers (Final / object max)",
        "Final_VoxVol / object max",
        "spaghetti_final_norm_objectmax_vs_fingers.png"
    )

    # -----------------------------
    # Aggregate curves (mean ± std)
    # -----------------------------
    plot_mean_std(
        df, out_dir,
        "Final_VoxVol",
        "Final Time",
        "Final enclosed volume",
        "meanstd_final_voxvol_vs_fingers.png",
        unreliable_spans=[(3, 4)],     # grey band between 3 and 4
        unreliable_points=[3],         # grey marker on the 3-finger mean
    )
    plot_mean_std(df, out_dir, "Final_over_Init", "Mean Final/Init vs number of fingers", "Final/Init", "meanstd_final_over_init_vs_fingers.png")
    plot_mean_std(df, out_dir, "Final_Loss", "Mean final loss vs number of fingers", "Final_Loss", "meanstd_final_loss_vs_fingers.png")
    plot_mean_std(df, out_dir, "Avg_Force", "Mean Avg_Force vs number of fingers", "Avg_Force", "meanstd_avg_force_vs_fingers.png")
    plot_mean_std(df, out_dir, "Force_cv", "Mean force imbalance (CV) vs number of fingers", "Force CV", "meanstd_force_cv_vs_fingers.png")
    plot_mean_std(df, out_dir, "Radius_Mean", "Mean Radius_Mean vs number of fingers", "Radius_Mean", "meanstd_radius_mean_vs_fingers.png")
    plot_mean_std(df, out_dir, "Radius_List_std", "Mean radius spread (std) vs number of fingers", "Radius std", "meanstd_radius_std_vs_fingers.png")
    plot_mean_std(df, out_dir, "Final_norm_objectmax", "Mean normalised performance vs number of fingers", "Final / object max", "meanstd_final_norm_objectmax_vs_fingers.png")
    plot_mean_std(df, out_dir, "Init_VoxVol", "Initial Time", "Initial enclosed volume","meanstd_init_enclosed_volume_vs_fingers.png")


    plot_robustness_curve(df, out_dir, "robustness_curve_final_norm_objectmax.png")

    # -----------------------------
    # Boxplots by finger count
    # -----------------------------
    plot_box_by_fingers(df, out_dir, "Final_VoxVol", "Final_VoxVol distribution by finger count", "Final_VoxVol", "box_final_voxvol_by_fingers.png")
    plot_box_by_fingers(df, out_dir, "Final_over_Init", "Final/Init distribution by finger count", "Final/Init", "box_final_over_init_by_fingers.png")
    plot_box_by_fingers(df, out_dir, "Final_Loss", "Final_Loss distribution by finger count", "Final_Loss", "box_final_loss_by_fingers.png")
    plot_box_by_fingers(df, out_dir, "Avg_Force", "Avg_Force distribution by finger count", "Avg_Force", "box_avg_force_by_fingers.png")
    plot_box_by_fingers(df, out_dir, "Force_cv", "Force imbalance (CV) distribution by finger count", "Force CV", "box_force_cv_by_fingers.png")
    plot_box_by_fingers(df, out_dir, "Radius_List_std", "Radius spread (std) distribution by finger count", "Radius std", "box_radius_std_by_fingers.png")

    # -----------------------------
    # "Which finger count is best"
    # -----------------------------
    plot_best_finger_hist(
        df, out_dir,
        metric="Final_VoxVol",
        fname="best_finger_count_by_object_final_voxvol_hist.png",
        title="Best finger count per object (max Final_VoxVol)"
    )
    plot_win_rate(
        df, out_dir,
        metric="Final_VoxVol",
        fname="win_rate_by_fingers_final_voxvol.png",
        title="Win rate by finger count (max Final_VoxVol per object)"
    )
    plot_mean_rank(
        df, out_dir,
        metric="Final_VoxVol",
        fname="mean_rank_by_fingers_final_voxvol.png",
        title="Mean rank by finger count (Final_VoxVol, lower is better)"
    )

    plot_best_finger_hist(
        df, out_dir,
        metric="Final_norm_objectmax",
        fname="best_finger_count_by_object_normalised_hist.png",
        title="Best finger count per object (normalised final volume)"
    )
    plot_win_rate(
        df, out_dir,
        metric="Final_norm_objectmax",
        fname="win_rate_by_fingers_normalised.png",
        title="Win rate by finger count (normalised final volume)"
    )

    # -----------------------------
    # Tradeoff scatters (coloured by finger count)
    # -----------------------------
    plot_scatter(
        df, out_dir,
        xcol="Avg_Force", ycol="Final_VoxVol",
        title="Tradeoff: Final_VoxVol vs Avg_Force",
        xlabel="Avg_Force",
        ylabel="Final_VoxVol",
        fname="scatter_final_voxvol_vs_avg_force.png",
    )

    plot_scatter(
        df, out_dir,
        xcol="Final_Loss", ycol="Final_VoxVol",
        title="Final_VoxVol vs Final_Loss",
        xlabel="Final_Loss",
        ylabel="Final_VoxVol",
        fname="scatter_final_voxvol_vs_final_loss.png",
    )

    plot_scatter(
        df, out_dir,
        xcol="Force_cv", ycol="Final_norm_objectmax",
        title="Normalised performance vs force imbalance",
        xlabel="Force CV",
        ylabel="Final_VoxVol / object max",
        fname="scatter_norm_perf_vs_force_cv.png",
    )

    plot_scatter(
        df, out_dir,
        xcol="Radius_Mean", ycol="Final_VoxVol",
        title="Final_VoxVol vs Radius_Mean (initial pose)",
        xlabel="Radius_Mean",
        ylabel="Final_VoxVol",
        fname="scatter_final_voxvol_vs_radius_mean.png",
    )

    # Correlation heatmap
    plot_corr_heatmap(df, out_dir, "corr_heatmap_metrics.png")

    print(f"[plot] Read: {csv_path}")
    print(f"[plot] Saved plots to: {out_dir}")
    print("[plot] Presentation copies saved alongside originals with *_presentation.png")


if __name__ == "__main__":
    main()
