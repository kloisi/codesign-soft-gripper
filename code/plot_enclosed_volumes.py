# plot_enclosed_volumes.py
#
# Changes vs your original:
#  1) For EVERY plot we now ALSO save a 2nd "presentation" PNG:
#        same name + "_presentation.png"
#     with black background, thicker lines, larger fonts.
#     The original PNGs are saved exactly as before.
#
#  2) We filter out the YCB objects by default:
#        013_apple, 006_mustard_bottle, 019_pitcher_base
#     (i.e. only coral objects remain, assuming corals are “everything else” in your logs).
#     You can override with --include_ycb if you want them back.
#
# Usage:
#   python plot_enclosed_volumes.py
#   python plot_enclosed_volumes.py --out plots_enclosed_volume
#   python plot_enclosed_volumes.py --include_ycb
#   python plot_enclosed_volumes.py --objects coralA coralB ...

import os
import argparse
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

import matplotlib.colors as mcolors
from matplotlib.collections import PathCollection, PolyCollection


YCB_SKIP_DEFAULT = {"013_apple", "006_mustard_bottle", "019_pitcher_base", "acropora_cytherea"}


# -----------------------------
# IO
# -----------------------------
def read_summary(path: str) -> pd.DataFrame:
    df = pd.read_csv(path)

    if "error" not in df.columns:
        df["error"] = ""
    df["error"] = df["error"].fillna("").astype(str)

    # numeric coercion
    for c in ["vol_vox_m3", "voxel_size", "enclosed_voxels", "blocked_voxels", "y_bottom", "y_top"]:
        if c in df.columns:
            df[c] = pd.to_numeric(df[c], errors="coerce")

    if "object" in df.columns:
        df["object"] = df["object"].astype(str)

    return df


def read_timeseries(path: str) -> pd.DataFrame:
    ts = pd.read_csv(path)

    for c in ["step", "t", "frame", "substep", "vol_vox", "enclosed_voxels", "blocked_voxels", "y_bottom", "y_top"]:
        if c in ts.columns:
            ts[c] = pd.to_numeric(ts[c], errors="coerce")

    ts["object"] = ts["object"].astype(str)
    ts = ts.dropna(subset=["object", "frame", "vol_vox"])

    # ensure int frames
    ts["frame"] = ts["frame"].astype(int)
    if "step" in ts.columns and ts["step"].notna().any():
        ts["step"] = ts["step"].astype(int)

    return ts


def ensure_dir(d: str) -> None:
    os.makedirs(d, exist_ok=True)


def ok_rows_summary(df: pd.DataFrame) -> pd.DataFrame:
    ok = df.copy()
    ok["error"] = ok["error"].fillna("").astype(str)

    # treat "" and "nan" (string) as ok
    ok = ok[(ok["error"] == "") | (ok["error"].str.lower() == "nan")].copy()
    ok = ok.dropna(subset=["object", "vol_vox_m3", "voxel_size", "enclosed_voxels"])

    # sanity check columns
    ok["vol_check"] = ok["enclosed_voxels"] * (ok["voxel_size"] ** 3)
    ok["vol_abs_err"] = (ok["vol_vox_m3"] - ok["vol_check"]).abs()
    return ok


# -----------------------------
# Presentation plot saving
# -----------------------------
def _is_blackish(color) -> bool:
    """True if the given color is basically black (helps boxplots etc on dark bg)."""
    try:
        r, g, b, a = mcolors.to_rgba(color)
        return (r + g + b) < 0.20 and a > 0.0
    except Exception:
        return False


def _apply_presentation_style(fig, font_scale=1.35, line_scale=1.8, marker_scale=1.35) -> None:
    """
    Mutates the current figure to be more PPT friendly:
      - black background
      - larger fonts
      - thicker lines / larger markers
      - white text/ticks/spines
      - MORE POP: increase alpha + recolor many-line plots with a vivid cmap
    """
    try:
        fig.patch.set_facecolor("black")
    except Exception:
        pass

    # Slightly larger canvas (only affects presentation save because we call after original save)
    try:
        w, h = fig.get_size_inches()
        fig.set_size_inches(w * 1.08, h * 1.08, forward=True)
    except Exception:
        pass

    for ax in fig.get_axes():
        # Axes background
        try:
            ax.set_facecolor("black")
        except Exception:
            pass

        # Spines
        try:
            for sp in ax.spines.values():
                sp.set_color("white")
                sp.set_linewidth(max(1.0, float(sp.get_linewidth()) * 1.1))
        except Exception:
            pass

        # Titles/labels
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

        # Ticks
        try:
            ax.tick_params(axis="both", which="both", colors="white")
            for lab in ax.get_xticklabels() + ax.get_yticklabels():
                try:
                    lab.set_color("white")
                    lab.set_fontsize(float(lab.get_fontsize()) * font_scale)
                except Exception:
                    pass
        except Exception:
            pass

        # Grid
        try:
            for gl in ax.get_xgridlines() + ax.get_ygridlines():
                gl.set_color("white")
                gl.set_alpha(0.20)
                gl.set_linewidth(max(0.8, float(gl.get_linewidth()) * 1.1))
        except Exception:
            pass

        # --------
        # Lines (make them pop)
        # --------
        try:
            lines = ax.get_lines()
            n = len(lines)

            # If many lines (like timeseries overlays), recolor with vivid cmap
            turbo_colors = None
            if n >= 8:
                turbo_colors = plt.cm.turbo(np.linspace(0.05, 0.95, n))

            for i, ln in enumerate(lines):
                # thicker lines
                try:
                    lw = float(ln.get_linewidth())
                    ln.set_linewidth(max(1.6, lw * line_scale))
                except Exception:
                    pass

                # recolor many-line plots
                try:
                    if turbo_colors is not None:
                        ln.set_color(turbo_colors[i])
                    else:
                        # if it was black (boxplot etc), make it white on black bg
                        if _is_blackish(ln.get_color()):
                            ln.set_color("white")
                except Exception:
                    pass

                # increase alpha for presentation (especially important for overlays)
                try:
                    a = ln.get_alpha()
                    if a is None:
                        a = 1.0
                    # If originally drawn faintly (e.g. alpha=0.35), boost it
                    if a < 0.65:
                        ln.set_alpha(0.75 if turbo_colors is not None else 0.85)
                    else:
                        ln.set_alpha(min(1.0, float(a)))
                except Exception:
                    pass

                # markers bigger
                try:
                    ms = ln.get_markersize()
                    if ms is not None and float(ms) > 0:
                        ln.set_markersize(float(ms) * marker_scale)
                except Exception:
                    pass
        except Exception:
            pass

        # --------
        # Collections:
        #   PolyCollection = fill_between bands -> keep subtle
        #   PathCollection = scatter points -> make pop
        # --------
        try:
            for coll in ax.collections:
                # fill_between produces PolyCollection -> keep it LIGHT on slides
                if isinstance(coll, PolyCollection) and not isinstance(coll, PathCollection):
                    try:
                        coll.set_alpha(0.40)  # <- tweak: 0.06 to 0.15 depending on taste
                    except Exception:
                        pass
                    continue

                # For scatter points etc: bump alpha (helps on dark bg)
                try:
                    a = coll.get_alpha()
                    if a is None or float(a) < 0.70:
                        coll.set_alpha(0.90)
                except Exception:
                    pass

                # For scatter points: white edge improves contrast
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

                # scale marker sizes
                try:
                    sz = coll.get_sizes()
                    if sz is not None and len(sz):
                        coll.set_sizes(np.asarray(sz, dtype=float) * (marker_scale ** 2))
                except Exception:
                    pass
        except Exception:
            pass


        # Bars/patches: give a white edge so they pop on black
        try:
            for p in ax.patches:
                try:
                    lw = float(p.get_linewidth()) if p.get_linewidth() is not None else 0.0
                    p.set_linewidth(max(1.0, lw * line_scale))
                    p.set_edgecolor("white")
                    # also ensure patch is not too transparent
                    a = p.get_alpha()
                    if a is None or float(a) < 0.85:
                        p.set_alpha(0.95)
                except Exception:
                    pass
        except Exception:
            pass

        # Annotation/text
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
    """
    Saves:
      1) original plot exactly as before: path
      2) presentation version: same name + "_presentation" before extension
    """
    plt.tight_layout()

    # 1) Original
    plt.savefig(path, dpi=200)

    # 2) Presentation
    fig = plt.gcf()
    _apply_presentation_style(fig)
    base, ext = os.path.splitext(path)
    pres_path = f"{base}_presentation{ext}"
    plt.tight_layout()
    plt.savefig(pres_path, dpi=220, facecolor=fig.get_facecolor(), edgecolor="none")

    plt.close()


# -----------------------------
# Plot helpers
# -----------------------------
def set_unit_ylabel(unit_label: str) -> str:
    return f"Enclosed volume ({unit_label})"


def plot_bar_sorted(ok: pd.DataFrame, out_dir: str, unit_label: str, topk: int | None):
    d = ok.sort_values("vol_vox_m3", ascending=False).copy()
    if topk is not None and topk > 0:
        d = d.head(topk)

    fig_h = max(4.0, 0.35 * len(d))
    plt.figure(figsize=(12, fig_h))
    plt.barh(d["object"], d["vol_vox_m3"])
    plt.gca().invert_yaxis()
    plt.xlabel(set_unit_ylabel(unit_label))
    plt.title("Final enclosed volume per object")
    savefig(os.path.join(out_dir, "summary_enclosed_volume_bar.png"))

    # also write sorted csv
    cols = ["object", "vol_vox_m3", "voxel_size", "enclosed_voxels", "blocked_voxels", "y_bottom", "y_top", "shape", "vol_abs_err"]
    cols = [c for c in cols if c in d.columns]
    d[cols].to_csv(os.path.join(out_dir, "summary_sorted.csv"), index=False)


def plot_sanity_scatter(ok: pd.DataFrame, out_dir: str, unit_label: str):
    plt.figure(figsize=(6, 5))
    plt.scatter(ok["vol_check"], ok["vol_vox_m3"], s=18)
    plt.xlabel(f"enclosed_voxels * voxel_size^3 ({unit_label})")
    plt.ylabel(f"vol_vox_m3 ({unit_label})")
    plt.title("Sanity check (should lie on y = x)")

    mn = float(np.nanmin([ok["vol_check"].min(), ok["vol_vox_m3"].min()]))
    mx = float(np.nanmax([ok["vol_check"].max(), ok["vol_vox_m3"].max()]))
    plt.plot([mn, mx], [mn, mx])
    plt.xlim(mn, mx)
    plt.ylim(mn, mx)
    savefig(os.path.join(out_dir, "summary_sanity_scatter.png"))


def plot_hist_and_box(ok: pd.DataFrame, out_dir: str, unit_label: str, bins: int):
    v = ok["vol_vox_m3"].to_numpy()

    plt.figure(figsize=(8, 5))
    plt.hist(v, bins=bins)
    plt.xlabel(set_unit_ylabel(unit_label))
    plt.ylabel("Count")
    plt.title("Distribution of final enclosed volumes")
    savefig(os.path.join(out_dir, "summary_volume_hist.png"))

    plt.figure(figsize=(8, 2.8))
    plt.boxplot(v, vert=False, showfliers=True)
    plt.xlabel(set_unit_ylabel(unit_label))
    plt.title("Final enclosed volumes (boxplot)")
    savefig(os.path.join(out_dir, "summary_volume_boxplot.png"))


def plot_scatter_cols(ok: pd.DataFrame, out_dir: str, xcol: str, ycol: str, title: str, xlabel: str, ylabel: str):
    if xcol not in ok.columns or ycol not in ok.columns:
        return
    plt.figure(figsize=(6.5, 5))
    plt.scatter(ok[xcol], ok[ycol], s=18)
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    plt.title(title)
    savefig(os.path.join(out_dir, f"summary_scatter_{xcol}_vs_{ycol}.png"))


def downsample_ts(data: pd.DataFrame, stride: int) -> pd.DataFrame:
    if stride is None or stride <= 1:
        return data
    return data[data["frame"] % stride == 0].copy()


def plot_timeseries_overlay(ts: pd.DataFrame, out_dir: str, unit_label: str, xaxis: str):
    plt.figure(figsize=(12, 5))
    for obj, g in ts.groupby("object", sort=True):
        x = g["t"].to_numpy() if xaxis == "t" else g["frame"].to_numpy()
        plt.plot(x, g["vol_vox"].to_numpy(), alpha=0.35)
    plt.xlabel("time (s)" if xaxis == "t" else "frame")
    plt.ylabel(set_unit_ylabel(unit_label))
    plt.title("Enclosed volume over time (each line is one object)")
    savefig(os.path.join(out_dir, f"timeseries_overlay_{xaxis}.png"))


def plot_timeseries_mean_std(ts: pd.DataFrame, out_dir: str, unit_label: str, xaxis: str):
    key = "t" if xaxis == "t" else "frame"
    if ts["object"].nunique() < 2:
        return

    agg = ts.groupby(key, as_index=False)["vol_vox"].agg(["mean", "std"]).reset_index()
    x = agg[key].to_numpy()
    m = agg["mean"].to_numpy()
    s = agg["std"].to_numpy()

    plt.figure(figsize=(12, 5))

    # draw band FIRST (behind), then mean line on top
    plt.fill_between(x, m - s, m + s, alpha=0.2, zorder=1)
    plt.plot(x, m, zorder=2)

    plt.xlabel("time (s)" if xaxis == "t" else "frame")
    plt.ylabel(set_unit_ylabel(unit_label))
    plt.title("Enclosed volume over time (mean ± std across objects)")
    savefig(os.path.join(out_dir, f"timeseries_mean_std_{xaxis}.png"))



def plot_timeseries_normalised(ts: pd.DataFrame, out_dir: str, xaxis: str):
    plt.figure(figsize=(12, 5))
    for obj, g in ts.groupby("object", sort=True):
        g = g.sort_values("frame")
        v_final = float(g["vol_vox"].iloc[-1])
        if not np.isfinite(v_final) or abs(v_final) < 1e-12:
            continue
        x = g["t"].to_numpy() if xaxis == "t" else g["frame"].to_numpy()
        plt.plot(x, (g["vol_vox"].to_numpy() / v_final), alpha=0.35)
    plt.xlabel("time (s)" if xaxis == "t" else "frame")
    plt.ylabel("vol / vol_final")
    plt.title("Normalised enclosed volume over time (each line is one object)")
    # small on-plot note instead of a huge legend
    plt.text(
        0.99, 0.98,
        f"{ts['object'].nunique()} objects (one line each)\nlegend omitted",
        transform=plt.gca().transAxes,
        ha="right", va="top",
        fontsize=10,
        bbox=dict(facecolor="white", alpha=0.3, edgecolor="none", pad=3),
    )
    savefig(os.path.join(out_dir, f"timeseries_normalised_{xaxis}.png"))


def plot_per_object(ts: pd.DataFrame, out_dir: str, unit_label: str, xaxis: str, objects: list[str]):
    per_dir = os.path.join(out_dir, "per_object")
    ensure_dir(per_dir)

    for obj in objects:
        g = ts[ts["object"] == obj].sort_values("frame")
        if g.empty:
            continue
        x = g["t"].to_numpy() if xaxis == "t" else g["frame"].to_numpy()

        plt.figure(figsize=(12, 4))
        plt.plot(x, g["vol_vox"].to_numpy())
        plt.xlabel("time (s)" if xaxis == "t" else "frame")
        plt.ylabel(set_unit_ylabel(unit_label))
        plt.title(f"Enclosed volume over time: {obj}")
        savefig(os.path.join(per_dir, f"{obj}_vol_{xaxis}.png"))

        if "y_bottom" in g.columns and "y_top" in g.columns:
            plt.figure(figsize=(12, 4))
            plt.plot(x, g["y_bottom"].to_numpy(), label="y_bottom")
            plt.plot(x, g["y_top"].to_numpy(), label="y_top")
            plt.xlabel("time (s)" if xaxis == "t" else "frame")
            plt.ylabel("y")
            plt.title(f"Rim heights over time: {obj}")
            plt.legend()
            savefig(os.path.join(per_dir, f"{obj}_rim_y_{xaxis}.png"))


# -----------------------------
# Convergence metrics
# -----------------------------
def compute_convergence_metrics(ts: pd.DataFrame, frac: float = 0.95, tail_window: int = 50) -> pd.DataFrame:
    rows = []
    for obj, g in ts.groupby("object", sort=True):
        g = g.sort_values("frame")
        v = g["vol_vox"].to_numpy()
        if v.size < 2:
            continue

        v_final = float(v[-1])
        v_max = float(np.max(v))
        v_min = float(np.min(v))
        v0 = float(v[0])

        target = frac * v_final
        idx = np.argmax(v >= target) if np.any(v >= target) else -1

        if idx >= 0:
            frame_frac = int(g["frame"].iloc[idx])
            t_frac = float(g["t"].iloc[idx]) if "t" in g.columns else np.nan
        else:
            frame_frac = -1
            t_frac = np.nan

        tw = min(tail_window, v.size)
        tail = v[-tw:]
        tail_mean = float(np.mean(tail))
        tail_std = float(np.std(tail))
        tail_cv = float(tail_std / tail_mean) if abs(tail_mean) > 1e-12 else np.nan

        undershoot_rel = float((v_final - v_min) / v_min) if abs(v_min) > 1e-12 else np.nan
        final_to_min = float(v_final / v_min) if abs(v_min) > 1e-12 else np.nan

        rows.append({
            "object": obj,
            "v0": v0,
            "v_final": v_final,
            "v_min": v_min,
            "v_max": v_max,
            "t95": t_frac,
            "frame95": frame_frac,
            "tail_mean": tail_mean,
            "tail_std": tail_std,
            "tail_cv": tail_cv,
            "final_to_min": final_to_min,
        })

    return pd.DataFrame(rows).sort_values("v_final", ascending=False)


def plot_convergence_metrics(m: pd.DataFrame, out_dir: str, unit_label: str):
    if m.empty:
        return

    mpath = os.path.join(out_dir, "convergence_metrics.csv")
    m.to_csv(mpath, index=False)

    if m["t95"].notna().any():
        plt.figure(figsize=(8, 5))
        plt.hist(m["t95"].dropna().to_numpy(), bins=15)
        plt.xlabel("t95 (s)")
        plt.ylabel("count")
        plt.title("Time to reach 95% of final volume (t95)")
        savefig(os.path.join(out_dir, "convergence_t95_hist.png"))

        plt.figure(figsize=(6.5, 5))
        plt.scatter(m["v_final"], m["t95"], s=20)
        plt.xlabel(f"final volume ({unit_label})")
        plt.ylabel("t95 (s)")
        plt.title("Final volume vs t95")
        savefig(os.path.join(out_dir, "convergence_final_vs_t95.png"))

    plt.figure(figsize=(6.5, 5))
    plt.scatter(m["v_final"], m["tail_std"], s=20)
    plt.xlabel(f"final volume ({unit_label})")
    plt.ylabel(f"tail std ({unit_label})")
    plt.title("Tail stability: std of last window vs final volume")
    savefig(os.path.join(out_dir, "convergence_tailstd_vs_final.png"))


# -----------------------------
# Main
# -----------------------------
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--summary", default="logs/ycb_voxel_volume_sweep_f6.csv")
    ap.add_argument("--ts", default="logs/ycb_voxel_volume_timeseries_all_f6.csv")
    ap.add_argument("--out", default="plots_enclosed_volume")
    ap.add_argument("--stride", type=int, default=10, help="Downsample timeseries: keep frames where frame % stride == 0")
    ap.add_argument("--topk", type=int, default=0, help="If >0, only plot top K objects by final volume (summary bar + summary_sorted.csv)")
    ap.add_argument("--objects", nargs="*", default=None, help="If set, filter to these objects and also create per-object plots")
    ap.add_argument("--bins", type=int, default=20, help="Histogram bins")
    ap.add_argument("--xaxis", choices=["frame", "t"], default="frame", help="Use frame index or time in seconds on x-axis")
    ap.add_argument("--unit_label", default="m^3", help='Axis label for volume, e.g. "m^3" or "sim units^3"')
    ap.add_argument("--tail_window", type=int, default=50, help="Window size for tail stability metrics")
    ap.add_argument("--tfrac", type=float, default=0.95, help="Fraction of final volume for convergence metric (e.g. 0.95)")
    ap.add_argument("--include_ycb", action="store_true", help="If set, DO NOT filter out YCB objects (013_apple, 006_mustard_bottle, 019_pitcher_base)")
    args = ap.parse_args()

    ensure_dir(args.out)

    df = read_summary(args.summary)
    ts = read_timeseries(args.ts)

    # Default: filter out YCB objects (keep corals)
    if not args.include_ycb:
        if "object" in df.columns:
            df = df[~df["object"].isin(YCB_SKIP_DEFAULT)].copy()
        if "object" in ts.columns:
            ts = ts[~ts["object"].isin(YCB_SKIP_DEFAULT)].copy()

    # filter objects explicitly if requested (applies after YCB filtering unless --include_ycb)
    if args.objects and len(args.objects) > 0:
        df = df[df["object"].isin(args.objects)].copy()
        ts = ts[ts["object"].isin(args.objects)].copy()

    ok = ok_rows_summary(df)
    topk = args.topk if args.topk and args.topk > 0 else None

    # summary plots
    plot_bar_sorted(ok, args.out, args.unit_label, topk=topk)
    plot_sanity_scatter(ok, args.out, args.unit_label)
    plot_hist_and_box(ok, args.out, args.unit_label, bins=args.bins)

    # extra summary scatters
    plot_scatter_cols(
        ok, args.out,
        xcol="enclosed_voxels", ycol="vol_vox_m3",
        title="Final volume vs enclosed voxels",
        xlabel="enclosed_voxels",
        ylabel=f"final volume ({args.unit_label})",
    )
    if "blocked_voxels" in ok.columns:
        plot_scatter_cols(
            ok, args.out,
            xcol="blocked_voxels", ycol="vol_vox_m3",
            title="Final volume vs blocked voxels",
            xlabel="blocked_voxels",
            ylabel=f"final volume ({args.unit_label})",
        )

    # timeseries plots
    ts = ts.sort_values(["object", "frame"])
    ts_ds = downsample_ts(ts, args.stride)

    plot_timeseries_overlay(ts_ds, args.out, args.unit_label, xaxis=args.xaxis)
    plot_timeseries_mean_std(ts_ds, args.out, args.unit_label, xaxis=args.xaxis)
    plot_timeseries_normalised(ts_ds, args.out, xaxis=args.xaxis)

    # per-object
    if args.objects and len(args.objects) > 0:
        plot_per_object(ts_ds, args.out, args.unit_label, xaxis=args.xaxis, objects=args.objects)

    # convergence metrics + plots (use full-res ts for metrics)
    metrics = compute_convergence_metrics(ts, frac=args.tfrac, tail_window=args.tail_window)
    plot_convergence_metrics(metrics, args.out, args.unit_label)

    print(f"Saved plots to: {args.out}")
    print("Presentation copies saved alongside originals with *_presentation.png")


if __name__ == "__main__":
    main()
