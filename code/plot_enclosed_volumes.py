# plot_sweep_enclosed_volume_all.py
#
# Superset plotter for the new sweep logger output:
#   <run_dir>/summary.csv
#   <run_dir>/timeseries.csv
#   <run_dir>/npz/*.npz
#
# Goals:
#   - Keep ALL old plots (overlay, mean±std, normalised V/V_final, hist+box, sanity, scatters)
#   - Add NPZ-based plots (forces, torques, tendon forces, ratios, motion, init logs)
#   - Always produce presentation copies *_presentation.png
#   - Always produce voxel point projections by default (for a manageable subset)
#
# Usage:
#   python plot_sweep_enclosed_volume_all.py logs/sweep_20250101_123456_f6
#   python plot_sweep_enclosed_volume_all.py   # if you omit arg, it picks latest logs/sweep_* folder
#
import os
import glob
import json
import argparse
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

import matplotlib.colors as mcolors
from matplotlib.collections import PathCollection, PolyCollection


YCB_SKIP_DEFAULT = {"013_apple", "006_mustard_bottle", "019_pitcher_base", "acropora_cytherea", "acropora_palmata"}


# -----------------------------
# IO helpers
# -----------------------------
def ensure_dir(d: str) -> None:
    os.makedirs(d, exist_ok=True)


def _find_latest_run_dir(root="logs"):
    cands = sorted(glob.glob(os.path.join(root, "sweep_*_f*")))
    cands = [c for c in cands if os.path.isdir(c)]
    if not cands:
        return None
    cands.sort(key=lambda p: os.path.getmtime(p))
    return cands[-1]


def read_summary(path: str) -> pd.DataFrame:
    df = pd.read_csv(path)

    if "error" not in df.columns:
        df["error"] = ""
    df["error"] = df["error"].fillna("").astype(str)

    for c in [
        "runtime_s",
        "finger_num",
        "seed",
        "init_pose_ok",
        "force_opt_ok",
        "vol_vox_m3",
        "enclosed_over_object_vol",
        "mesh_vol_m3_scaled",
        "mesh_area_m2_scaled",
        "bbox_vol_m3_scaled",
        "voxel_size",
        "enclosed_voxels",
        "blocked_voxels",
        "y_bottom",
        "y_top",
        "obj_force_fx", "obj_force_fy", "obj_force_fz",
        "obj_torque_tx", "obj_torque_ty", "obj_torque_tz",
    ]:
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

    ts["frame"] = ts["frame"].astype(int)
    if "step" in ts.columns and ts["step"].notna().any():
        ts["step"] = ts["step"].astype(int)

    return ts


def downsample_ts(data: pd.DataFrame, stride: int) -> pd.DataFrame:
    if stride is None or stride <= 1:
        return data
    return data[data["frame"] % stride == 0].copy()


def ok_rows_summary(df: pd.DataFrame) -> pd.DataFrame:
    ok = df.copy()
    ok["error"] = ok["error"].fillna("").astype(str)
    ok = ok[(ok["error"] == "") | (ok["error"].str.lower() == "nan")].copy()

    # Old plotting expected these columns
    need = ["object", "vol_vox_m3", "voxel_size", "enclosed_voxels"]
    for c in need:
        if c not in ok.columns:
            ok[c] = np.nan

    ok = ok.dropna(subset=["object", "vol_vox_m3", "voxel_size", "enclosed_voxels"])

    ok["vol_check"] = ok["enclosed_voxels"] * (ok["voxel_size"] ** 3)
    ok["vol_abs_err"] = (ok["vol_vox_m3"] - ok["vol_check"]).abs()
    return ok


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

        # Texts
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
                frame = leg.get_frame()
                frame.set_facecolor("black")
                frame.set_edgecolor("white")
                frame.set_alpha(0.65)
                for txt in leg.get_texts():
                    txt.set_color("white")
                    txt.set_fontsize(float(txt.get_fontsize()) * font_scale)
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


# -----------------------------
# Old plot helpers (kept)
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

    neon_cyan = "#00E5FF"

    plt.figure(figsize=(6, 4.4))
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
        linewidth=2.2,
        zorder=2,
    )

    plt.xlabel("time (s)" if xaxis == "t" else "frame")
    plt.ylabel(set_unit_ylabel(unit_label))
    plt.title("Enclosed volume over time (mean ± std across corals)")
    savefig(os.path.join(out_dir, f"timeseries_mean_std_{xaxis}.png"))



def plot_timeseries_normalised_both(ts: pd.DataFrame, out_dir: str, xaxis: str):
    """
    Creates TWO plots:
      A) V(t) / V_final   (your old one)
      B) V_final / V(t)   (can be useful too, but can blow up near 0 -> clipped)
    """
    # A) V(t)/V_final
    plt.figure(figsize=(12, 5))
    n_ok = 0
    for obj, g in ts.groupby("object", sort=True):
        g = g.sort_values("frame")
        v = g["vol_vox"].to_numpy(dtype=float)
        if v.size < 2:
            continue
        v_final = float(v[-1])
        if not np.isfinite(v_final) or abs(v_final) < 1e-12:
            continue
        x = g["t"].to_numpy() if xaxis == "t" else g["frame"].to_numpy()
        plt.plot(x, (v / v_final), alpha=0.35)
        n_ok += 1
    plt.xlabel("time (s)" if xaxis == "t" else "frame")
    plt.ylabel("V(t) / V_final")
    plt.title("Normalised enclosed volume over time (each line is one coral)")
    plt.text(
        0.99, 0.98,
        f"{14} corals (one line each)\nlegend omitted",
        transform=plt.gca().transAxes,
        ha="right", va="top",
        fontsize=10,
        bbox=dict(facecolor="white", alpha=0.3, edgecolor="none", pad=3),
    )
    plt.grid(True, alpha=0.3)
    savefig(os.path.join(out_dir, f"timeseries_normalised_V_over_Vfinal_{xaxis}.png"))

    # B) V_final/V(t) (clipped)
    eps = 1e-12
    ratios = []
    per_obj = []
    for obj, g in ts.groupby("object", sort=True):
        g = g.sort_values("frame")
        v = g["vol_vox"].to_numpy(dtype=float)
        if v.size < 2:
            continue
        v_final = float(v[-1])
        if not np.isfinite(v_final) or abs(v_final) < 1e-12:
            continue
        r = v_final / np.maximum(v, eps)
        r = r[np.isfinite(r)]
        if r.size:
            ratios.append(np.quantile(r, 0.99))
            per_obj.append(obj)
    clip_val = float(np.nanmedian(ratios)) if ratios else 10.0
    clip_val = max(2.0, min(50.0, clip_val))

    plt.figure(figsize=(12, 5))
    n_ok = 0
    for obj, g in ts.groupby("object", sort=True):
        g = g.sort_values("frame")
        v = g["vol_vox"].to_numpy(dtype=float)
        if v.size < 2:
            continue
        v_final = float(v[-1])
        if not np.isfinite(v_final) or abs(v_final) < 1e-12:
            continue
        x = g["t"].to_numpy() if xaxis == "t" else g["frame"].to_numpy()
        r = v_final / np.maximum(v, eps)
        r = np.clip(r, 0.0, clip_val)
        plt.plot(x, r, alpha=0.35)
        n_ok += 1
    plt.xlabel("time (s)" if xaxis == "t" else "frame")
    plt.ylabel("V_final / V(t)  (clipped)")
    plt.title("Inverse normalised enclosed volume over time (each line is one object)")
    plt.text(
        0.99, 0.98,
        f"{n_ok} objects\nclip max ≈ {clip_val:.1f}\nlegend omitted",
        transform=plt.gca().transAxes,
        ha="right", va="top",
        fontsize=10,
        bbox=dict(facecolor="white", alpha=0.3, edgecolor="none", pad=3),
    )
    plt.grid(True, alpha=0.3)
    savefig(os.path.join(out_dir, f"timeseries_normalised_Vfinal_over_V_{xaxis}.png"))


def plot_timeseries_normalised_V_over_Vinit(ts: pd.DataFrame, out_dir: str, xaxis: str):
    """
    V(t) / V_initial so all lines START at 1.
    """
    plt.figure(figsize=(6, 4.4))
    n_ok = 0

    for obj, g in ts.groupby("object", sort=True):
        g = g.sort_values("frame")
        v = g["vol_vox"].to_numpy(dtype=float)
        if v.size < 2:
            continue

        v0 = float(v[0])
        if not np.isfinite(v0) or abs(v0) < 1e-12:
            continue

        x = g["t"].to_numpy() if xaxis == "t" else g["frame"].to_numpy()
        plt.plot(x, (v / v0), alpha=0.35)
        n_ok += 1

    plt.xlabel("time (s)" if xaxis == "t" else "frame")
    plt.ylabel("V(t) / V_initial")
    plt.title("Normalised enclosed volume over time (start at 1)")

    plt.text(
        0.99, 0.98,
        f"{n_ok+1} objects (one line each)\nlegend omitted",
        transform=plt.gca().transAxes,
        ha="right", va="top",
        fontsize=10,
        bbox=dict(facecolor="white", alpha=0.3, edgecolor="none", pad=3),
    )

    plt.grid(True, alpha=0.3)
    savefig(os.path.join(out_dir, f"timeseries_normalised_V_over_Vinit_{xaxis}.png"))




# -----------------------------
# NPZ reading + derived metrics
# -----------------------------
def _npz_json_scalar(z, key, default=None):
    if key not in z:
        return default
    v = z[key]
    try:
        if isinstance(v, np.ndarray) and v.size >= 1:
            v = v.reshape(-1)[0]
        if isinstance(v, bytes):
            v = v.decode("utf-8", errors="replace")
        if isinstance(v, str):
            return json.loads(v)
    except Exception:
        return default
    return default


def _npz_array(z, key):
    if key not in z:
        return None
    try:
        return np.asarray(z[key])
    except Exception:
        return None


def _safe_norm(v, axis=-1):
    try:
        v = np.asarray(v, dtype=np.float64)
        return np.linalg.norm(v, axis=axis)
    except Exception:
        return None


def _volume_metrics_from_curve(frame, t, vol, frac=0.95, tail_window=50):
    out = dict(
        v0=np.nan, v_final=np.nan, v_min=np.nan, v_max=np.nan,
        t95=np.nan, frame95=np.nan,
        tail_mean=np.nan, tail_std=np.nan, tail_cv=np.nan,
        overshoot_rel=np.nan, final_to_min=np.nan,
    )
    if vol is None:
        return out
    vol = np.asarray(vol, dtype=np.float64).ravel()
    good = np.isfinite(vol)
    vol = vol[good]
    if vol.size < 2:
        return out

    out["v0"] = float(vol[0])
    out["v_final"] = float(vol[-1])
    out["v_min"] = float(np.min(vol))
    out["v_max"] = float(np.max(vol))

    vf = out["v_final"]
    if np.isfinite(vf) and abs(vf) > 1e-12:
        out["overshoot_rel"] = float((out["v_max"] - vf) / vf)

    vmin = out["v_min"]
    if np.isfinite(vf) and np.isfinite(vmin) and abs(vmin) > 1e-12:
        out["final_to_min"] = float(vf / vmin)

    target = frac * vf if np.isfinite(vf) else np.nan
    idx = np.argmax(vol >= target) if np.isfinite(target) and np.any(vol >= target) else -1

    if idx >= 0:
        if frame is not None:
            fr = np.asarray(frame, dtype=np.float64).ravel()
            fr = fr[np.isfinite(fr)]
            if fr.size > idx:
                out["frame95"] = int(fr[idx])
        if t is not None:
            tt = np.asarray(t, dtype=np.float64).ravel()
            tt = tt[np.isfinite(tt)]
            if tt.size > idx:
                out["t95"] = float(tt[idx])

    tw = int(min(max(5, tail_window), vol.size))
    tail = vol[-tw:]
    tm = float(np.mean(tail))
    ts = float(np.std(tail))
    out["tail_mean"] = tm
    out["tail_std"] = ts
    out["tail_cv"] = float(ts / tm) if abs(tm) > 1e-12 else np.nan
    return out


def load_npz_metrics(npz_path, frac=0.95, tail_window=50):
    with np.load(npz_path, allow_pickle=True) as z:
        meta = _npz_json_scalar(z, "meta_json", default={}) or {}
        summ = _npz_json_scalar(z, "summary_json", default={}) or {}
        vox_meta = _npz_json_scalar(z, "voxel_debug_meta_json", default={}) or {}

        obj = str(summ.get("object", meta.get("object", "")))

        # volume timeseries stored as ts_* arrays
        ts_keys = _npz_json_scalar(z, "ts_keys_json", default=[]) or []
        ts = {}
        for k in ts_keys:
            a = _npz_array(z, f"ts_{k}")
            if a is not None:
                ts[k] = np.asarray(a, dtype=np.float64).ravel()

        frame = ts.get("frame", None)
        t = ts.get("t", None)
        vol = ts.get("vol_vox", None)

        vm = _volume_metrics_from_curve(frame, t, vol, frac=frac, tail_window=tail_window)

        # summary volumes
        vol_final = float(summ.get("vol_vox_m3", np.nan))
        if not np.isfinite(vol_final) and np.isfinite(vm["v_final"]):
            vol_final = float(vm["v_final"])

        mesh_vol_scaled = float(summ.get("mesh_vol_m3_scaled", np.nan))
        bbox_vol_scaled = float(summ.get("bbox_vol_m3_scaled", np.nan))
        ratio = float(summ.get("enclosed_over_object_vol", np.nan))

        # voxel sanity
        voxel_size = float(summ.get("voxel_size", vox_meta.get("voxel_size", np.nan)))
        enclosed_voxels = float(summ.get("enclosed_voxels", vox_meta.get("enclosed_voxels", np.nan)))
        vol_check = enclosed_voxels * (voxel_size ** 3) if np.isfinite(enclosed_voxels) and np.isfinite(voxel_size) else np.nan

        # tendon forces
        tendon_forces = _npz_array(z, "tendon_forces_final")
        tf_l2 = np.nan
        tf_max = np.nan
        tf_sumabs = np.nan
        tf_n = 0
        tf_full = None
        if tendon_forces is not None:
            tf_full = np.asarray(tendon_forces, dtype=np.float64).ravel()
            good = np.isfinite(tf_full)
            if np.any(good):
                tf = tf_full[good]
                tf_n = int(tf.size)
                tf_l2 = float(np.linalg.norm(tf))
                tf_max = float(np.max(np.abs(tf)))
                tf_sumabs = float(np.sum(np.abs(tf)))

        # object force/torque per frame
        obj_force_frames = _npz_array(z, "obj_force_frames")
        obj_torque_frames = _npz_array(z, "obj_torque_frames")

        force_final = np.nan
        force_max = np.nan
        force_rms = np.nan
        torque_final = np.nan
        torque_max = np.nan
        torque_rms = np.nan

        if obj_force_frames is not None and np.asarray(obj_force_frames).ndim == 2 and np.asarray(obj_force_frames).shape[1] == 3:
            fm = _safe_norm(obj_force_frames, axis=1)
            if fm is not None and fm.size:
                force_final = float(fm[-1])
                force_max = float(np.max(fm))
                force_rms = float(np.sqrt(np.mean(fm ** 2)))

        if obj_torque_frames is not None and np.asarray(obj_torque_frames).ndim == 2 and np.asarray(obj_torque_frames).shape[1] == 3:
            tm = _safe_norm(obj_torque_frames, axis=1)
            if tm is not None and tm.size:
                torque_final = float(tm[-1])
                torque_max = float(np.max(tm))
                torque_rms = float(np.sqrt(np.mean(tm ** 2)))

        # object pose motion
        obj_body_q = _npz_array(z, "obj_body_q_frames")
        disp_final = np.nan
        path_len = np.nan
        if obj_body_q is not None and np.asarray(obj_body_q).ndim == 2 and np.asarray(obj_body_q).shape[1] >= 3:
            pos = np.asarray(obj_body_q, dtype=np.float64)[:, :3]
            good = np.isfinite(pos).all(axis=1)
            pos = pos[good]
            if pos.shape[0] >= 2:
                disp = np.linalg.norm(pos - pos[0:1, :], axis=1)
                disp_final = float(disp[-1])
                path_len = float(np.sum(np.linalg.norm(pos[1:] - pos[:-1], axis=1)))

        # init pose loss
        init_loss_min = np.nan
        init_loss_last = np.nan
        init_loss_len = 0
        if "init_loss_history" in z:
            lh = _npz_array(z, "init_loss_history")
            if lh is not None:
                lh = np.asarray(lh, dtype=object).ravel()
                vals = []
                for v in lh:
                    try:
                        vals.append(float(v))
                    except Exception:
                        pass
                vals = np.asarray(vals, dtype=np.float64)
                vals = vals[np.isfinite(vals)]
                if vals.size:
                    init_loss_len = int(vals.size)
                    init_loss_min = float(np.min(vals))
                    init_loss_last = float(vals[-1])

        # presence flags
        has_voxel_enclosed = ("voxel_enclosed_pts" in z)
        has_voxel_blocked = ("voxel_blocked_pts" in z)

        row = dict(
            object=obj,
            npz_path=npz_path,
            optimizer=str(meta.get("optimizer", "")),
            scale=float(meta.get("scale", np.nan)) if "scale" in meta else np.nan,

            vol_final=vol_final,
            mesh_vol_scaled=mesh_vol_scaled,
            bbox_vol_scaled=bbox_vol_scaled,
            enclosed_over_object_vol=ratio,

            vol_check=vol_check,
            vol_abs_err=(abs(vol_final - vol_check) if np.isfinite(vol_final) and np.isfinite(vol_check) else np.nan),

            v0=vm["v0"],
            v_min=vm["v_min"],
            v_max=vm["v_max"],
            t95=vm["t95"],
            frame95=vm["frame95"],
            tail_std=vm["tail_std"],
            tail_cv=vm["tail_cv"],
            overshoot_rel=vm["overshoot_rel"],

            tendon_force_n=tf_n,
            tendon_force_l2=tf_l2,
            tendon_force_max=tf_max,
            tendon_force_sumabs=tf_sumabs,

            obj_force_final=force_final,
            obj_force_max=force_max,
            obj_force_rms=force_rms,

            obj_torque_final=torque_final,
            obj_torque_max=torque_max,
            obj_torque_rms=torque_rms,

            disp_final=disp_final,
            path_len=path_len,

            init_loss_min=init_loss_min,
            init_loss_last=init_loss_last,
            init_loss_len=init_loss_len,

            has_voxel_enclosed=int(bool(has_voxel_enclosed)),
            has_voxel_blocked=int(bool(has_voxel_blocked)),
        )

        # expand per finger tendon forces if present
        if tf_full is not None:
            for i in range(tf_full.size):
                row[f"tendon_f{i}"] = float(tf_full[i]) if np.isfinite(tf_full[i]) else np.nan

        return row


def load_npz_points(npz_path):
    with np.load(npz_path, allow_pickle=True) as z:
        enc = _npz_array(z, "voxel_enclosed_pts")
        blk = _npz_array(z, "voxel_blocked_pts")
        return enc, blk


# -----------------------------
# Generic plot helpers
# -----------------------------
def plot_hist(df, out_dir, col, title, xlabel, fname, bins=24, logx=False):
    if col not in df.columns:
        return
    v = pd.to_numeric(df[col], errors="coerce").to_numpy(dtype=float)
    v = v[np.isfinite(v)]
    if v.size < 2:
        return
    plt.figure(figsize=(8, 5))
    plt.hist(v, bins=bins)
    if logx:
        plt.xscale("log")
    plt.xlabel(xlabel)
    plt.ylabel("count")
    plt.title(title)
    savefig(os.path.join(out_dir, fname))


def plot_scatter(df, out_dir, xcol, ycol, title, xlabel, ylabel, fname, annotate_topk=0, size=22):
    if xcol not in df.columns or ycol not in df.columns:
        return
    d = df.copy()
    d[xcol] = pd.to_numeric(d[xcol], errors="coerce")
    d[ycol] = pd.to_numeric(d[ycol], errors="coerce")
    d = d[np.isfinite(d[xcol]) & np.isfinite(d[ycol])].copy()
    if d.empty:
        return

    plt.figure(figsize=(6.8, 5.6))
    plt.scatter(d[xcol], d[ycol], s=size, alpha=0.85)
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    plt.title(title)
    plt.grid(True, alpha=0.3)

    if annotate_topk and annotate_topk > 0:
        d2 = d.sort_values(ycol, ascending=False).head(int(annotate_topk))
        for _, r in d2.iterrows():
            plt.annotate(str(r["object"]), (r[xcol], r[ycol]), textcoords="offset points", xytext=(4, 3), fontsize=8, alpha=0.85)

    savefig(os.path.join(out_dir, fname))


def plot_tendon_force_heatmap(df, out_dir, fname="tendon_force_heatmap_objects_by_finger.png", topk=60):
    fcols = [c for c in df.columns if c.startswith("tendon_f") and c[8:].isdigit()]
    if not fcols:
        return
    d = df.copy()
    d["vol_final"] = pd.to_numeric(d["vol_final"], errors="coerce")
    d = d[np.isfinite(d["vol_final"])].sort_values("vol_final", ascending=False)
    if topk and topk > 0:
        d = d.head(int(topk))

    M = []
    labels = []
    for _, r in d.iterrows():
        row = []
        ok = False
        for c in sorted(fcols, key=lambda s: int(s[8:])):
            v = float(r.get(c, np.nan))
            row.append(v)
            ok = ok or np.isfinite(v)
        if ok:
            M.append(row)
            labels.append(str(r["object"]))

    if not M:
        return

    M = np.asarray(M, dtype=float)
    plt.figure(figsize=(9.6, max(4.8, 0.22 * len(labels))))
    im = plt.imshow(M, aspect="auto", interpolation="nearest")
    plt.colorbar(im, label="tendon force (final)")
    plt.yticks(np.arange(len(labels)), labels)
    plt.xticks(np.arange(M.shape[1]), [f"f{i}" for i in range(M.shape[1])])
    plt.xlabel("finger")
    plt.ylabel("object")
    plt.title("Final tendon forces per finger (objects sorted by vol_final)")
    savefig(os.path.join(out_dir, fname))


def plot_voxel_point_projections(npz_map, objects, out_dir, max_points=8000):
    vdir = os.path.join(out_dir, "voxel_points")
    ensure_dir(vdir)

    rng = np.random.default_rng(0)

    for obj in objects:
        p = npz_map.get(obj, None)
        if not p or not os.path.exists(p):
            continue

        enc, blk = load_npz_points(p)

        for kind, P in [("enclosed", enc), ("blocked", blk)]:
            if P is None:
                continue
            P = np.asarray(P, dtype=np.float64)
            if P.ndim != 2 or P.shape[1] != 3 or P.shape[0] < 50:
                continue

            if P.shape[0] > max_points:
                idx = rng.choice(P.shape[0], size=int(max_points), replace=False)
                P = P[idx]

            x, y, z = P[:, 0], P[:, 1], P[:, 2]

            # XY
            plt.figure(figsize=(6.2, 6.0))
            plt.scatter(x, y, s=2, alpha=0.7)
            plt.gca().set_aspect("equal", adjustable="box")
            plt.xlabel("x")
            plt.ylabel("y")
            plt.title(f"voxel_{kind}_pts projection XY: {obj}")
            plt.grid(True, alpha=0.2)
            savefig(os.path.join(vdir, f"{obj}_voxel_{kind}_xy.png"))

            # XZ
            plt.figure(figsize=(6.2, 6.0))
            plt.scatter(x, z, s=2, alpha=0.7)
            plt.gca().set_aspect("equal", adjustable="box")
            plt.xlabel("x")
            plt.ylabel("z")
            plt.title(f"voxel_{kind}_pts projection XZ: {obj}")
            plt.grid(True, alpha=0.2)
            savefig(os.path.join(vdir, f"{obj}_voxel_{kind}_xz.png"))

            # YZ
            plt.figure(figsize=(6.2, 6.0))
            plt.scatter(y, z, s=2, alpha=0.7)
            plt.gca().set_aspect("equal", adjustable="box")
            plt.xlabel("y")
            plt.ylabel("z")
            plt.title(f"voxel_{kind}_pts projection YZ: {obj}")
            plt.grid(True, alpha=0.2)
            savefig(os.path.join(vdir, f"{obj}_voxel_{kind}_yz.png"))


# -----------------------------
# Main
# -----------------------------
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("run_dir", nargs="?", default=None, help="logs/sweep_<runid>_f<finger> (if omitted: latest)")
    args = ap.parse_args()

    run_dir = args.run_dir
    if run_dir is None:
        run_dir = _find_latest_run_dir("logs")
        if run_dir is None:
            raise FileNotFoundError("No run_dir provided and no logs/sweep_* directories found.")
    if not os.path.isdir(run_dir):
        raise FileNotFoundError(f"run_dir not found: {run_dir}")

    summary_csv = os.path.join(run_dir, "summary.csv")
    ts_csv = os.path.join(run_dir, "timeseries.csv")
    npz_dir = os.path.join(run_dir, "npz")

    out_dir = os.path.join(run_dir, "plots_enclosed_volume_all")
    ensure_dir(out_dir)

    # --- defaults (no args needed) ---
    include_ycb = False
    stride = 10
    bins = 20
    xaxis = "frame"
    unit_label = "m^3"
    tail_window = 50
    tfrac = 0.95

    voxel_points_topk = 8
    voxel_points_max_points = 8000

    # Load CSVs
    df_sum = read_summary(summary_csv) if os.path.exists(summary_csv) else pd.DataFrame()
    ts = read_timeseries(ts_csv) if os.path.exists(ts_csv) else pd.DataFrame()

    # Default: filter out YCB objects (keep corals)
    if not include_ycb:
        if "object" in df_sum.columns:
            df_sum = df_sum[~df_sum["object"].isin(YCB_SKIP_DEFAULT)].copy()
        if "object" in ts.columns:
            ts = ts[~ts["object"].isin(YCB_SKIP_DEFAULT)].copy()

    # Old style "ok" rows and old plots
    if not df_sum.empty:
        ok = ok_rows_summary(df_sum)
        plot_bar_sorted(ok, out_dir, unit_label, topk=None)
        plot_sanity_scatter(ok, out_dir, unit_label)
        plot_hist_and_box(ok, out_dir, unit_label, bins=bins)

        plot_scatter_cols(
            ok, out_dir,
            xcol="enclosed_voxels", ycol="vol_vox_m3",
            title="Final volume vs enclosed voxels",
            xlabel="enclosed_voxels",
            ylabel=f"final volume ({unit_label})",
        )
        if "blocked_voxels" in ok.columns:
            plot_scatter_cols(
                ok, out_dir,
                xcol="blocked_voxels", ycol="vol_vox_m3",
                title="Final volume vs blocked voxels",
                xlabel="blocked_voxels",
                ylabel=f"final volume ({unit_label})",
            )

    # Old timeseries plots (CSV)
    if not ts.empty:
        ts = ts.sort_values(["object", "frame"])
        ts_ds = downsample_ts(ts, stride)

        plot_timeseries_overlay(ts_ds, out_dir, unit_label, xaxis=xaxis)
        plot_timeseries_mean_std(ts_ds, out_dir, unit_label, xaxis=xaxis)

        # Normalised plots (both directions) — NEW: restores your old one + adds inverse
        plot_timeseries_normalised_both(ts_ds, out_dir, xaxis=xaxis)
        plot_timeseries_normalised_V_over_Vinit(ts_ds, out_dir, xaxis=xaxis)


    # -----------------------------
    # NPZ based metrics + plots
    # -----------------------------
    if not os.path.isdir(npz_dir):
        print("[WARN] NPZ dir missing:", npz_dir)
        npz_files = []
    else:
        npz_files = sorted(glob.glob(os.path.join(npz_dir, "*.npz")))

    rows = []
    npz_map = {}
    for p in npz_files:
        try:
            r = load_npz_metrics(p, frac=tfrac, tail_window=tail_window)
            rows.append(r)
            npz_map[str(r["object"])] = p
        except Exception as e:
            print("[WARN] failed npz:", os.path.basename(p), type(e).__name__, str(e)[:160])

    if rows:
        df_npz = pd.DataFrame(rows)

        # merge error/runtime from summary.csv if present
        if not df_sum.empty and "object" in df_sum.columns:
            small = df_sum[["object", "error", "runtime_s", "init_pose_ok", "force_opt_ok"]].copy() if "runtime_s" in df_sum.columns else df_sum[["object", "error"]].copy()
            df_npz = df_npz.merge(small, on="object", how="left", suffixes=("", "_csv"))
        else:
            df_npz["error"] = ""

        # filter ok objects for NPZ plots
        e = df_npz["error"].fillna("").astype(str).str.lower()
        df_ok_npz = df_npz[(e == "") | (e == "nan")].copy()
        df_ok_npz["vol_final"] = pd.to_numeric(df_ok_npz["vol_final"], errors="coerce")
        df_ok_npz = df_ok_npz[np.isfinite(df_ok_npz["vol_final"]) & (df_ok_npz["vol_final"] > 0)].copy()

        # write derived metrics
        df_ok_npz.to_csv(os.path.join(out_dir, "derived_metrics_from_npz.csv"), index=False)

        # ratios as "percent"
        df_ok_npz["enclosed_over_object_pct"] = 100.0 * pd.to_numeric(df_ok_npz["enclosed_over_object_vol"], errors="coerce")

        # NOTE:
        # This is NOT true "overlap percentage inside enclosed volume".
        # It's a volume ratio enclosed/object.
        # True geometric overlap would require mesh geometry + voxel occupancy queries.
        plot_hist(df_ok_npz, out_dir, "enclosed_over_object_vol", "enclosed_over_object_vol distribution", "enclosed_over_object_vol", "npz_hist_enclosed_over_object_vol.png", bins=24)
        plot_hist(df_ok_npz, out_dir, "enclosed_over_object_pct", "enclosed volume as % of object volume (ratio * 100)", "%", "npz_hist_enclosed_over_object_pct.png", bins=24)

        plot_scatter(
            df_ok_npz, out_dir,
            "mesh_vol_scaled", "vol_final",
            "Enclosed volume vs mesh volume (scaled)",
            "mesh_vol_scaled (m^3)",
            f"vol_final ({unit_label})",
            "npz_scatter_volfinal_vs_meshvol.png",
            annotate_topk=10,
        )

        plot_scatter(
            df_ok_npz, out_dir,
            "bbox_vol_scaled", "vol_final",
            "Enclosed volume vs bbox volume (scaled)",
            "bbox_vol_scaled (m^3)",
            f"vol_final ({unit_label})",
            "npz_scatter_volfinal_vs_bboxvol.png",
            annotate_topk=10,
        )

        # convergence + stability
        plot_hist(df_ok_npz, out_dir, "t95", "Time to reach 95% of final volume (t95)", "t95 (s)", "npz_hist_t95.png", bins=20)
        plot_hist(df_ok_npz, out_dir, "tail_cv", "Tail stability (tail_cv)", "tail_cv", "npz_hist_tail_cv.png", bins=24, logx=True)
        plot_hist(df_ok_npz, out_dir, "overshoot_rel", "Overshoot relative to final volume", "overshoot_rel", "npz_hist_overshoot_rel.png", bins=24)

        # tendon force summary
        plot_hist(df_ok_npz, out_dir, "tendon_force_max", "Max tendon force across fingers (final)", "max |tendon_force|", "npz_hist_tendon_force_max.png", bins=24)
        plot_hist(df_ok_npz, out_dir, "tendon_force_l2", "L2 norm tendon forces (final)", "||tendon_forces||_2", "npz_hist_tendon_force_l2.png", bins=24)
        plot_tendon_force_heatmap(df_ok_npz, out_dir, topk=60)

        # object forces/torques (best effort: this is net rigid body force/torque, not cloth-only)
        plot_hist(df_ok_npz, out_dir, "obj_force_max", "Max object force magnitude over frames", "max ||force||", "npz_hist_obj_force_max.png", bins=24)
        plot_hist(df_ok_npz, out_dir, "obj_torque_max", "Max object torque magnitude over frames", "max ||torque||", "npz_hist_obj_torque_max.png", bins=24)

        plot_scatter(
            df_ok_npz, out_dir,
            "obj_force_max", "vol_final",
            "Tradeoff: vol_final vs max object force",
            "max ||force||",
            f"vol_final ({unit_label})",
            "npz_scatter_volfinal_vs_obj_force_max.png",
            annotate_topk=10,
        )
        plot_scatter(
            df_ok_npz, out_dir,
            "obj_force_max", "enclosed_over_object_vol",
            "Tradeoff: enclosed_over_object_vol vs max object force",
            "max ||force||",
            "enclosed_over_object_vol",
            "npz_scatter_ratio_vs_obj_force_max.png",
            annotate_topk=10,
        )

        # motion metrics
        plot_hist(df_ok_npz, out_dir, "disp_final", "Final displacement magnitude (object)", "disp_final", "npz_hist_disp_final.png", bins=24)
        plot_hist(df_ok_npz, out_dir, "path_len", "Path length (object)", "path_len", "npz_hist_path_len.png", bins=24)
        plot_scatter(
            df_ok_npz, out_dir,
            "disp_final", "obj_force_max",
            "Object motion vs max force",
            "disp_final",
            "max ||force||",
            "npz_scatter_dispfinal_vs_obj_force_max.png",
            annotate_topk=10,
        )

        # init pose loss if present
        if df_ok_npz["init_loss_len"].fillna(0).astype(int).max() > 0:
            plot_scatter(
                df_ok_npz, out_dir,
                "init_loss_min", "vol_final",
                "Init pose loss (min) vs vol_final",
                "init_loss_min",
                f"vol_final ({unit_label})",
                "npz_scatter_initlossmin_vs_volfinal.png",
                annotate_topk=10,
            )
            plot_scatter(
                df_ok_npz, out_dir,
                "init_loss_min", "obj_force_max",
                "Init pose loss (min) vs max object force",
                "init_loss_min",
                "max ||force||",
                "npz_scatter_initlossmin_vs_objforcemax.png",
                annotate_topk=10,
            )

        # -----------------------------
        # Voxel point projections ON BY DEFAULT
        # We'll plot for a subset (top volume + high force + worst tail_cv), otherwise you get hundreds of images.
        # -----------------------------
        sel = []

        d = df_ok_npz.sort_values("vol_final", ascending=False)
        sel += d.head(voxel_points_topk)["object"].astype(str).tolist()

        d2 = df_ok_npz.copy()
        d2["obj_force_max"] = pd.to_numeric(d2["obj_force_max"], errors="coerce")
        d2 = d2[np.isfinite(d2["obj_force_max"])].sort_values("obj_force_max", ascending=False)
        sel += d2.head(3)["object"].astype(str).tolist()

        d3 = df_ok_npz.copy()
        d3["tail_cv"] = pd.to_numeric(d3["tail_cv"], errors="coerce")
        d3 = d3[np.isfinite(d3["tail_cv"])].sort_values("tail_cv", ascending=False)
        sel += d3.head(3)["object"].astype(str).tolist()

        # unique preserve order
        seen = set()
        sel_u = []
        for x in sel:
            if x not in seen:
                sel_u.append(x)
                seen.add(x)

        plot_voxel_point_projections(npz_map, sel_u, out_dir, max_points=voxel_points_max_points)

    print(f"Saved plots to: {out_dir}")
    print("Presentation copies saved alongside originals with *_presentation.png")


if __name__ == "__main__":
    main()
