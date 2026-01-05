# plot_stiffness_sweep.py
#
# Drop in replacement that uses the per run .npz files produced by sweep_stiffness.py
#
# Key change vs previous:
#   - Seeds are treated as redundancy: for ALL plots we only use "successful" runs.
#   - Successful is defined primarily via volume validity (vol_final > vol_eps).
#   - If scalars like vol_final/t95/overshoot/tail_cv are broken but the saved curve exists,
#     we recompute them from the curve and use those instead.
#
# Usage examples:
#   python plot_stiffness_sweep.py --npz_dir stiffness_sweep_npz_acropora_cervicornis_forwardlike
#   python plot_stiffness_sweep.py --out plots_stiffness_npz
#
import os
import re
import glob
import argparse
import numpy as np
import pandas as pd

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt


# -------------------------
# Helpers
# -------------------------
def _is_num(x):
    try:
        return np.isfinite(float(x))
    except Exception:
        return False


def _as_float(x, default=np.nan):
    try:
        return float(x)
    except Exception:
        return default


def _as_int(x, default=-1):
    try:
        return int(x)
    except Exception:
        return default


def _get_scalar(z, key, default=np.nan):
    if key not in z:
        return default
    v = z[key]
    if isinstance(v, np.ndarray):
        if v.shape == ():
            v = v.item()
        elif v.size == 1:
            v = v.reshape(()).item()
        else:
            return default
    if isinstance(v, (str, bytes)):
        return v.decode("utf-8") if isinstance(v, bytes) else v
    if isinstance(v, (int, np.integer)):
        return int(v)
    if isinstance(v, (float, np.floating)):
        return float(v)
    try:
        return float(v)
    except Exception:
        return default


def _get_str(z, key, default=""):
    if key not in z:
        return default
    v = z[key]
    if isinstance(v, np.ndarray):
        if v.shape == ():
            v = v.item()
        elif v.size == 1:
            v = v.reshape(()).item()
        else:
            try:
                return str(v)
            except Exception:
                return default
    if isinstance(v, bytes):
        return v.decode("utf-8", errors="replace")
    return str(v)


def _sanitize_token(x):
    try:
        s = f"{float(x)}"
    except Exception:
        s = str(x)
    return s.replace(".", "p").replace("-", "m")


def _presentation_filename(name, suffix="_presentation"):
    """
    Insert suffix before extension.
    Example: foo.png -> foo_presentation.png
    """
    base, ext = os.path.splitext(name)
    if ext == "":
        return f"{name}{suffix}"
    return f"{base}{suffix}{ext}"


def _apply_presentation_style(fig, font_scale=1.35, line_scale=1.8, marker_scale=1.35):
    """
    Mutates the current figure to be more PPT friendly:
      - black background
      - larger fonts
      - thicker lines
      - white text/ticks/spines
    """
    # Figure background
    try:
        fig.patch.set_facecolor("black")
    except Exception:
        pass

    # Optional: slightly larger overall canvas (keeps aspect)
    try:
        w, h = fig.get_size_inches()
        fig.set_size_inches(w * 1.05, h * 1.05, forward=True)
    except Exception:
        pass

    axes = fig.get_axes()
    for ax in axes:
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

        # Titles and labels
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

        # Tick params and tick label sizes
        try:
            ax.tick_params(axis="both", which="both", colors="white")
            for lab in (ax.get_xticklabels() + ax.get_yticklabels()):
                try:
                    lab.set_color("white")
                    lab.set_fontsize(float(lab.get_fontsize()) * font_scale)
                except Exception:
                    pass
        except Exception:
            pass

        # Gridlines
        try:
            for gl in ax.get_xgridlines() + ax.get_ygridlines():
                gl.set_color("white")
                gl.set_alpha(0.20)
                gl.set_linewidth(max(0.8, float(gl.get_linewidth()) * 1.1))
        except Exception:
            pass

        # Lines
        try:
            for ln in ax.get_lines():
                try:
                    lw = float(ln.get_linewidth())
                    ln.set_linewidth(max(1.4, lw * line_scale))
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

        # Scatter/collections
        try:
            for coll in ax.collections:
                try:
                    lw = coll.get_linewidths()
                    if lw is not None and len(lw):
                        coll.set_linewidths(np.maximum(1.0, np.asarray(lw, dtype=float) * line_scale))
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

        # Hist bars and other patches
        try:
            for p in ax.patches:
                try:
                    lw = float(p.get_linewidth()) if p.get_linewidth() is not None else 0.0
                    p.set_linewidth(max(1.0, lw * line_scale))
                    p.set_edgecolor("white")
                except Exception:
                    pass
        except Exception:
            pass

        # Images (heatmaps): make "bad"/masked values darker for black background
        try:
            for im in ax.images:
                try:
                    cmap = im.get_cmap().copy()
                    cmap.set_bad(color=(0.25, 0.25, 0.25, 1.0))
                    im.set_cmap(cmap)
                except Exception:
                    pass
        except Exception:
            pass

        # Text annotations: make white + bump font
        try:
            for t in ax.texts:
                try:
                    t.set_color("white")
                    t.set_fontsize(float(t.get_fontsize()) * font_scale)
                    # add subtle bbox to keep readable on bright heatmap cells
                    if t.get_bbox_patch() is None:
                        t.set_bbox(dict(facecolor="black", alpha=0.50, edgecolor="none", pad=0.4))
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


def savefig(out_dir, name):
    """
    Saves:
      1) the original plot exactly as before
      2) a presentation version (black background, bigger fonts, thicker lines)
         with suffix "_presentation" inserted before the extension
    """
    os.makedirs(out_dir, exist_ok=True)

    # 1) Original (unchanged)
    path = os.path.join(out_dir, name)
    plt.tight_layout()
    plt.savefig(path, dpi=220)
    print("[OK] wrote", path)

    # 2) Presentation version
    pres_name = _presentation_filename(name, suffix="_presentation")
    pres_path = os.path.join(out_dir, pres_name)

    fig = plt.gcf()
    _apply_presentation_style(fig)

    plt.tight_layout()
    plt.savefig(pres_path, dpi=220, facecolor=fig.get_facecolor(), edgecolor="none")
    print("[OK] wrote", pres_path)

    plt.close()


def _marker_for_mass(m):
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
    annot_fmt="{:.3g}",
):
    x = list(piv.columns)
    y = list(piv.index)
    Z = piv.values.astype(float)

    # Mask invalid entries so they render as "missing" instead of default white blocks.
    Zm = np.ma.masked_invalid(Z)

    plt.figure(figsize=(8.4, 6.0))
    im = plt.imshow(Zm, aspect="auto", interpolation="nearest")

    # Make masked (missing) values show as light gray instead of white.
    try:
        cmap = im.get_cmap().copy()
        cmap.set_bad(color=(0.88, 0.88, 0.88, 1.0))
        im.set_cmap(cmap)
    except Exception:
        pass

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
                    plt.text(j, i, annot_fmt.format(val), ha="center", va="center", fontsize=8)

    savefig(out_dir, filename)


def _find_default_npz_dir(curr_dir):
    cands = [d for d in glob.glob(os.path.join(curr_dir, "stiffness_sweep_npz_*")) if os.path.isdir(d)]
    if not cands:
        return None
    cands.sort(key=lambda p: os.path.getmtime(p))
    return cands[-1]


def _force_stats(forces):
    if forces is None or len(forces) == 0:
        return dict(
            forces_mean=np.nan,
            forces_std=np.nan,
            forces_min=np.nan,
            forces_max=np.nan,
            forces_cv=np.nan,
            forces_l2=np.nan,
        )
    forces = np.asarray(forces, dtype=np.float64).ravel()
    mu = float(np.mean(forces))
    sd = float(np.std(forces))
    mn = float(np.min(forces))
    mx = float(np.max(forces))
    cv = float(sd / mu) if abs(mu) > 1e-12 else np.nan
    l2 = float(np.linalg.norm(forces))
    return dict(forces_mean=mu, forces_std=sd, forces_min=mn, forces_max=mx, forces_cv=cv, forces_l2=l2)


def _volume_curve_metrics(vol_t, vol_v):
    """
    Metrics purely from the saved volume curve (if present).
    Returns dict including robust fallbacks for v_final, t95, overshoot, tail_cv.
    """
    out = dict(
        v_final_curve=np.nan,
        t95_curve=np.nan,
        tail_cv_curve=np.nan,
        overshoot_rel_curve=np.nan,
        vol_t10=np.nan,
        vol_t90=np.nan,
        vol_risetime_10_90=np.nan,
        vol_tail_std=np.nan,
        vol_tail_mean=np.nan,
        vol_max_curve=np.nan,
        vol_min_curve=np.nan,
        vol_mean_curve=np.nan,
    )
    if vol_t is None or vol_v is None:
        return out

    vol_t = np.asarray(vol_t, dtype=np.float64).ravel()
    vol_v = np.asarray(vol_v, dtype=np.float64).ravel()
    good = np.isfinite(vol_t) & np.isfinite(vol_v)
    vol_t = vol_t[good]
    vol_v = vol_v[good]
    if vol_v.size < 5:
        return out

    order = np.argsort(vol_t)
    vol_t = vol_t[order]
    vol_v = vol_v[order]

    v_final = float(vol_v[-1])
    out["v_final_curve"] = v_final

    out["vol_max_curve"] = float(np.max(vol_v))
    out["vol_min_curve"] = float(np.min(vol_v))
    out["vol_mean_curve"] = float(np.mean(vol_v))

    if not np.isfinite(v_final) or abs(v_final) < 1e-12:
        return out

    # t95
    v95 = 0.95 * v_final
    idx95 = np.argmax(vol_v >= v95) if np.any(vol_v >= v95) else -1
    out["t95_curve"] = float(vol_t[idx95]) if idx95 >= 0 else np.nan

    # overshoot
    out["overshoot_rel_curve"] = float((np.max(vol_v) - v_final) / v_final)

    # tail stats -> tail_cv
    tw = min(50, vol_v.size)
    tail = vol_v[-tw:]
    tail_mean = float(np.mean(tail))
    tail_std = float(np.std(tail))
    out["vol_tail_mean"] = tail_mean
    out["vol_tail_std"] = tail_std
    out["tail_cv_curve"] = float(tail_std / tail_mean) if abs(tail_mean) > 1e-12 else np.nan

    # risetime 10-90
    v10 = 0.10 * v_final
    v90 = 0.90 * v_final
    idx10 = np.argmax(vol_v >= v10) if np.any(vol_v >= v10) else -1
    idx90 = np.argmax(vol_v >= v90) if np.any(vol_v >= v90) else -1
    t10 = float(vol_t[idx10]) if idx10 >= 0 else np.nan
    t90 = float(vol_t[idx90]) if idx90 >= 0 else np.nan
    out["vol_t10"] = t10
    out["vol_t90"] = t90
    out["vol_risetime_10_90"] = float(t90 - t10) if _is_num(t10) and _is_num(t90) else np.nan

    return out


def _end_translation_mismatch(obj_xyz, p0_xyz):
    if obj_xyz is None or p0_xyz is None:
        return np.nan
    obj_xyz = np.asarray(obj_xyz, dtype=np.float64)
    p0_xyz = np.asarray(p0_xyz, dtype=np.float64)
    if obj_xyz.ndim != 2 or obj_xyz.shape[1] != 3:
        return np.nan
    if p0_xyz.ndim != 2 or p0_xyz.shape[1] != 3:
        return np.nan
    if obj_xyz.shape[0] < 2 or p0_xyz.shape[0] < 2:
        return np.nan
    if not (np.isfinite(obj_xyz[[0, -1]]).all() and np.isfinite(p0_xyz[[0, -1]]).all()):
        return np.nan
    obj_trans = obj_xyz[-1] - obj_xyz[0]
    p_trans = p0_xyz[-1] - p0_xyz[0]
    return float(np.linalg.norm(obj_trans - p_trans))


def _disp_mag(xyz):
    if xyz is None:
        return np.nan
    xyz = np.asarray(xyz, dtype=np.float64)
    if xyz.ndim != 2 or xyz.shape[1] != 3 or xyz.shape[0] < 2:
        return np.nan
    if not np.isfinite(xyz[[0, -1]]).all():
        return np.nan
    d = xyz[-1] - xyz[0]
    return float(np.linalg.norm(d))


def load_npz_run_summary(npz_path, expect_fingers=None):
    """
    Loads one NPZ and returns a per-run summary dict.
    Important: tries to FIX broken scalar volume metrics using the saved volume curve if available.
    """
    with np.load(npz_path, allow_pickle=True) as z:
        obj = _get_str(z, "object", "")
        finger_num = _as_int(_get_scalar(z, "finger_num", np.nan), default=-1)

        stiff = _as_float(_get_scalar(z, "cloth_stiff_scale", np.nan))
        mass = _as_float(_get_scalar(z, "cloth_mass_scale", np.nan))
        damp = _as_float(_get_scalar(z, "cloth_damp_scale", np.nan))
        seed = _as_int(_get_scalar(z, "kernel_seed", np.nan), default=-1)

        fps = _as_int(_get_scalar(z, "fps", np.nan), default=-1)
        sim_substeps = _as_int(_get_scalar(z, "sim_substeps", np.nan), default=-1)
        dt = _as_float(_get_scalar(z, "dt", np.nan))
        num_frames = _as_int(_get_scalar(z, "num_frames", np.nan), default=-1)
        total_steps = _as_int(_get_scalar(z, "total_steps", np.nan), default=-1)

        timestamp = _get_str(z, "timestamp", "")
        optimizer = _get_str(z, "optimizer", "")
        opt_frames = _as_int(_get_scalar(z, "opt_frames", np.nan), default=-1)

        # saved scalars (may be broken in some runs)
        vol_final = _as_float(_get_scalar(z, "vol_final", np.nan))
        t95 = _as_float(_get_scalar(z, "t95", np.nan))
        tail_cv = _as_float(_get_scalar(z, "tail_cv", np.nan))
        overshoot_rel = _as_float(_get_scalar(z, "overshoot_rel", np.nan))
        vol_max = _as_float(_get_scalar(z, "vol_max", np.nan))
        vol_min = _as_float(_get_scalar(z, "vol_min", np.nan))
        vol_mean = _as_float(_get_scalar(z, "vol_mean", np.nan))

        # volume curve fallbacks
        vol_t = z["vol_t"] if "vol_t" in z else None
        vol_v = z["vol_vol_vox"] if "vol_vol_vox" in z else None
        vextra = _volume_curve_metrics(vol_t, vol_v)

        # Fix scalar volume metrics if curve provides better values
        # (common case: scalar ended up 0/NaN but curve is fine)
        if (not np.isfinite(vol_final)) or (vol_final <= 0.0):
            if np.isfinite(vextra.get("v_final_curve", np.nan)) and vextra["v_final_curve"] > 0.0:
                vol_final = float(vextra["v_final_curve"])
        if (not np.isfinite(vol_max)) and np.isfinite(vextra.get("vol_max_curve", np.nan)):
            vol_max = float(vextra["vol_max_curve"])
        if (not np.isfinite(vol_min)) and np.isfinite(vextra.get("vol_min_curve", np.nan)):
            vol_min = float(vextra["vol_min_curve"])
        if (not np.isfinite(vol_mean)) and np.isfinite(vextra.get("vol_mean_curve", np.nan)):
            vol_mean = float(vextra["vol_mean_curve"])

        if not np.isfinite(t95) and np.isfinite(vextra.get("t95_curve", np.nan)):
            t95 = float(vextra["t95_curve"])
        if not np.isfinite(overshoot_rel) and np.isfinite(vextra.get("overshoot_rel_curve", np.nan)):
            overshoot_rel = float(vextra["overshoot_rel_curve"])
        if not np.isfinite(tail_cv) and np.isfinite(vextra.get("tail_cv_curve", np.nan)):
            tail_cv = float(vextra["tail_cv_curve"])


        # --- rebound metric: how far final is above the minimum (>=1, lower is better)
        denom = float(vol_min) if np.isfinite(vol_min) else np.nan
        if np.isfinite(vol_final) and np.isfinite(denom) and denom > 1e-12:
            undershoot_rel = float((vol_final - denom) / denom)   # == final_to_min - 1
            final_to_min = float(vol_final / denom)              # your preferred one
        else:
            undershoot_rel = np.nan
            final_to_min = np.nan


        # forces
        forces_final = z["forces_final"] if "forces_final" in z else None
        if forces_final is not None:
            forces_final = np.asarray(forces_final, dtype=np.float64).ravel()
        fstats = _force_stats(forces_final)

        # per finger forces columns
        force_cols = {}
        if forces_final is not None and forces_final.size > 0:
            nF = int(finger_num) if finger_num > 0 else int(forces_final.size)
            if expect_fingers is not None:
                nF = int(expect_fingers)
            for i in range(min(nF, forces_final.size)):
                force_cols[f"force_f{i}"] = float(forces_final[i])
            for i in range(min(nF, max(nF, 0))):
                force_cols.setdefault(f"force_f{i}", np.nan)
        elif expect_fingers is not None and expect_fingers > 0:
            for i in range(int(expect_fingers)):
                force_cols[f"force_f{i}"] = np.nan

        # trajectories
        p0_xyz = z["p0_xyz"] if "p0_xyz" in z else None
        obj_xyz = z["obj_xyz"] if "obj_xyz" in z else None
        diff_norm_end = _end_translation_mismatch(obj_xyz, p0_xyz)
        obj_disp = _disp_mag(obj_xyz)
        p0_disp = _disp_mag(p0_xyz)

        # optimiser loss
        loss_curve = z["loss_curve"] if "loss_curve" in z else None
        loss_len = int(np.asarray(loss_curve).size) if loss_curve is not None else 0
        loss0 = np.nan
        lossT = np.nan
        if loss_curve is not None and loss_len > 0:
            lc = np.asarray(loss_curve, dtype=np.float64).ravel()
            if lc.size:
                loss0 = float(lc[0])
                lossT = float(lc[-1])

        out = dict(
            npz_path=npz_path,
            timestamp=timestamp,
            object=obj,
            finger_num=finger_num,
            cloth_stiff_scale=stiff,
            cloth_mass_scale=mass,
            cloth_damp_scale=damp,
            kernel_seed=seed,
            fps=fps,
            sim_substeps=sim_substeps,
            dt=dt,
            num_frames=num_frames,
            total_steps=total_steps,
            optimizer=optimizer,
            opt_frames=opt_frames,
            vol_final=vol_final,
            t95=t95,
            tail_cv=tail_cv,
            overshoot_rel=overshoot_rel,
            undershoot_rel=undershoot_rel,
            final_to_min=final_to_min,
            vol_max=vol_max,
            vol_min=vol_min,
            vol_mean=vol_mean,
            diff_norm_end=diff_norm_end,
            obj_disp=obj_disp,
            p0_disp=p0_disp,
            loss0=loss0,
            lossT=lossT,
            loss_len=loss_len,
            **fstats,
            **force_cols,
            # keep curve-derived extras too (useful for plotting)
            **{k: vextra[k] for k in [
                "vol_t10", "vol_t90", "vol_risetime_10_90", "vol_tail_std", "vol_tail_mean"
            ] if k in vextra},
        )
        return out


def load_npz_timeseries(npz_path):
    with np.load(npz_path, allow_pickle=True) as z:
        out = {}
        out["vol_t"] = np.asarray(z["vol_t"], dtype=np.float64) if "vol_t" in z else None
        out["vol_v"] = np.asarray(z["vol_vol_vox"], dtype=np.float64) if "vol_vol_vox" in z else None

        out["obj_t"] = np.asarray(z["obj_t"], dtype=np.float64) if "obj_t" in z else None
        out["obj_xyz"] = np.asarray(z["obj_xyz"], dtype=np.float64) if "obj_xyz" in z else None
        out["obj_quat"] = np.asarray(z["obj_quat"], dtype=np.float64) if "obj_quat" in z else None

        out["p0_t"] = np.asarray(z["p0_t"], dtype=np.float64) if "p0_t" in z else None
        out["p0_xyz"] = np.asarray(z["p0_xyz"], dtype=np.float64) if "p0_xyz" in z else None

        out["loss_curve"] = np.asarray(z["loss_curve"], dtype=np.float64) if "loss_curve" in z else None

        out["object"] = _get_str(z, "object", "")
        out["finger_num"] = _as_int(_get_scalar(z, "finger_num", np.nan), default=-1)
        out["stiff"] = _as_float(_get_scalar(z, "cloth_stiff_scale", np.nan))
        out["mass"] = _as_float(_get_scalar(z, "cloth_mass_scale", np.nan))
        out["damp"] = _as_float(_get_scalar(z, "cloth_damp_scale", np.nan))
        out["seed"] = _as_int(_get_scalar(z, "kernel_seed", np.nan), default=-1)
        return out


def pick_representative_paths(runs_df, by="vol_final"):
    """
    For each (stiff,mass,damp), pick the run closest to the median of `by`.
    IMPORTANT: you should pass in runs_df already filtered to successful runs.
    """
    reps = {}
    keys = ["cloth_stiff_scale", "cloth_mass_scale", "cloth_damp_scale"]
    for key, grp in runs_df.groupby(keys):
        g = grp.copy()
        g[by] = pd.to_numeric(g[by], errors="coerce")
        g = g[np.isfinite(g[by].astype(float))]
        if g.empty:
            reps[key] = grp.iloc[0]["npz_path"]
            continue
        med = float(np.median(g[by].astype(float)))
        g["abs_dev"] = (g[by].astype(float) - med).abs()
        reps[key] = g.sort_values("abs_dev").iloc[0]["npz_path"]
    return reps


# -------------------------
# Plot suite
# -------------------------
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--npz_dir", type=str, default=None, help="directory containing stiffness_sweep_npz_... files")
    parser.add_argument("--out", type=str, default="plots_stiffness_sweep_npz")
    parser.add_argument("--no_annot", action="store_true", help="disable heatmap annotations")
    parser.add_argument("--no_timeseries", action="store_true", help="skip time series plots (volume/object/loss)")
    parser.add_argument("--no_per_finger", action="store_true", help="skip per finger force plots/heatmaps")
    parser.add_argument("--filter_object", type=str, default="", help="only keep this object name (exact match)")
    parser.add_argument("--filter_finger_num", type=int, default=-1, help="only keep this finger_num")
    parser.add_argument("--vol_eps", type=float, default=1e-9, help="volume validity threshold: vol_final must be > vol_eps to count as successful")
    args = parser.parse_args()

    curr_dir = os.path.dirname(os.path.abspath(__file__))
    npz_dir = args.npz_dir
    if npz_dir is None:
        npz_dir = _find_default_npz_dir(curr_dir)
        if npz_dir is None:
            raise FileNotFoundError("No --npz_dir provided and no stiffness_sweep_npz_* directory found next to this script.")
    if not os.path.isabs(npz_dir):
        npz_dir = os.path.join(curr_dir, npz_dir)

    out_dir = args.out
    if not os.path.isabs(out_dir):
        out_dir = os.path.join(curr_dir, out_dir)
    os.makedirs(out_dir, exist_ok=True)

    npz_files = sorted(glob.glob(os.path.join(npz_dir, "*.npz")))
    if not npz_files:
        raise FileNotFoundError(f"No .npz files found in: {npz_dir}")

    print("[INFO] reading npz from:", npz_dir)
    print("[INFO] found files:", len(npz_files))

    # determine finger hint
    finger_hint = None
    for p in npz_files[:10]:
        try:
            with np.load(p, allow_pickle=True) as z:
                fn = _as_int(_get_scalar(z, "finger_num", np.nan), default=-1)
                if fn > 0:
                    finger_hint = fn
                    break
        except Exception:
            pass

    # load per run summaries
    rows = []
    for p in npz_files:
        try:
            r = load_npz_run_summary(p, expect_fingers=finger_hint)
            rows.append(r)
        except Exception as e:
            print("[WARN] failed loading:", p, "err:", type(e).__name__, str(e)[:200])

    if not rows:
        raise RuntimeError("Could not load any npz runs successfully.")

    runs = pd.DataFrame(rows)

    # filters
    if args.filter_object:
        runs = runs[runs["object"] == args.filter_object].copy()
    if args.filter_finger_num > 0:
        runs = runs[runs["finger_num"] == int(args.filter_finger_num)].copy()

    # numeric coercion
    for c in ["cloth_stiff_scale", "cloth_mass_scale", "cloth_damp_scale", "kernel_seed",
              "vol_final", "t95", "tail_cv", "overshoot_rel", "vol_min", "vol_max", "final_to_min"]:
        if c in runs.columns:
            runs[c] = pd.to_numeric(runs[c], errors="coerce")

    runs = runs[
        np.isfinite(runs["cloth_stiff_scale"])
        & np.isfinite(runs["cloth_mass_scale"])
        & np.isfinite(runs["cloth_damp_scale"])
    ].copy()

    # derived columns
    runs["tail_cv_safe"] = np.maximum(pd.to_numeric(runs["tail_cv"], errors="coerce").astype(float), 1e-12)
    runs["log10_tail_cv"] = np.log10(runs["tail_cv_safe"])

    # SUCCESS FLAG:
    # Treat seeds as redundancy: only "successful" runs contribute to ALL plots.
    vol_eps = float(args.vol_eps)
    runs["vol_valid"] = (
        np.isfinite(pd.to_numeric(runs["vol_final"], errors="coerce").astype(float))
        & (pd.to_numeric(runs["vol_final"], errors="coerce").astype(float) > vol_eps)
    )

    runs_all = runs.copy()
    runs_used = runs_all[runs_all["vol_valid"]].copy()

    runs_csv = os.path.join(out_dir, "runs_summary_npz_all.csv")
    runs_all.to_csv(runs_csv, index=False)
    print("[OK] wrote", runs_csv)

    runs_used_csv = os.path.join(out_dir, "runs_summary_npz_used_successful.csv")
    runs_used.to_csv(runs_used_csv, index=False)
    print("[OK] wrote", runs_used_csv)

    dropped = len(runs_all) - len(runs_used)
    print(f"[INFO] successful runs: {len(runs_used)} / {len(runs_all)}  (dropped {dropped} due to vol_final <= {vol_eps} or NaN)")

    # -------------------------
    # Group across seeds (present vs used)
    # -------------------------
    keys = ["cloth_stiff_scale", "cloth_mass_scale", "cloth_damp_scale"]

    present = (
        runs_all.groupby(keys, as_index=False)
        .agg(
            seed_count_present=("kernel_seed", "nunique"),
            run_count_present=("npz_path", "count"),
        )
    )

    used_counts = (
        runs_used.groupby(keys, as_index=False)
        .agg(
            seed_count_used=("kernel_seed", "nunique"),
            run_count_used=("npz_path", "count"),
        )
    )

    # metrics aggregated ONLY on successful runs
    agg_metrics = {
        "vol_final": ["mean", "std"],
        "vol_max": ["mean", "std"],
        "vol_min": ["mean", "std"],
        "vol_mean": ["mean", "std"],
        "t95": ["mean", "std"],
        "tail_cv": ["mean", "std"],
        "log10_tail_cv": ["mean", "std"],
        "overshoot_rel": ["mean", "std"],
        "final_to_min": ["mean", "std"], 
        "forces_mean": ["mean", "std"],
        "forces_max": ["mean", "std"],
        "forces_std": ["mean", "std"],
        "forces_l2": ["mean", "std"],
        "diff_norm_end": ["mean", "std"],
        "obj_disp": ["mean", "std"],
        "p0_disp": ["mean", "std"],
        "vol_risetime_10_90": ["mean", "std"],
        "vol_tail_std": ["mean", "std"],
        "loss0": ["mean", "std"],
        "lossT": ["mean", "std"],
        "loss_len": ["mean", "std"],
    }

    # include per finger columns in aggregation if available
    force_cols = [c for c in runs_all.columns if re.match(r"^force_f\d+$", c)]
    for c in force_cols:
        agg_metrics[c] = ["mean", "std"]

    if len(runs_used) > 0:
        g_metrics = runs_used.groupby(keys, as_index=False).agg(agg_metrics)

        # flatten multiindex columns
        flat_cols = []
        for col in g_metrics.columns:
            if isinstance(col, tuple):
                a, b = col
                if b == "":
                    flat_cols.append(a)
                else:
                    flat_cols.append(f"{a}_{b}")
            else:
                flat_cols.append(col)
        g_metrics.columns = flat_cols
    else:
        # no successful runs at all
        g_metrics = present.copy()

    # merge present + used counts + metrics
    g = present.merge(used_counts, on=keys, how="left").merge(g_metrics, on=keys, how="left")
    g["seed_count_used"] = g["seed_count_used"].fillna(0).astype(int)
    g["run_count_used"] = g["run_count_used"].fillna(0).astype(int)

    # For combos with exactly 1 used run: std columns become NaN -> set to 0
    for c in g.columns:
        if c.endswith("_std"):
            mask_one = g["run_count_used"] == 1
            g.loc[mask_one, c] = g.loc[mask_one, c].fillna(0.0)

    grouped_csv = os.path.join(out_dir, "grouped_summary_npz.csv")
    g.to_csv(grouped_csv, index=False)
    print("[OK] wrote", grouped_csv)

    annotate = (not args.no_annot)

    # axes values (from PRESENT combos so layout stays consistent)
    mass_vals = sorted(g["cloth_mass_scale"].unique().tolist())
    damp_vals = sorted(g["cloth_damp_scale"].unique().tolist())
    stiff_vals = sorted(g["cloth_stiff_scale"].unique().tolist())

    # -------------------------
    # Heatmaps per mass scale (SUCCESSFUL ONLY)
    # -------------------------
    def heatmap_metric(metric_mean_col, title_prefix, cbar, fname_prefix):
        if metric_mean_col not in g.columns:
            print("[WARN] missing column for heatmap:", metric_mean_col)
            return
        for m in mass_vals:
            gm = g[np.isclose(g["cloth_mass_scale"], m)].copy()
            piv = gm.pivot(
                index="cloth_stiff_scale",
                columns="cloth_damp_scale",
                values=metric_mean_col,
            ).reindex(index=stiff_vals, columns=damp_vals)

            fmt = "{:.4f}" if metric_mean_col.startswith("final_to_min") else "{:.3g}" # shows 1.0000, 1.0030, etc

            make_heatmap(
                piv=piv,
                title=f"{title_prefix}  mass_scale={m}",
                xlabel="cloth_damp_scale (kd)",
                ylabel="cloth_stiff_scale (ke/ka)",
                cbar_label=cbar,
                out_dir=out_dir,
                filename=f"{fname_prefix}_mass{_sanitize_token(m)}.png",
                annotate=annotate,
                annot_fmt=fmt,
            )

    heatmap_metric("vol_final_mean", "vol_final", "vol_final", "heat_vol_final")
    #heatmap_metric("overshoot_rel_mean", "overshoot_rel", "overshoot_rel", "heat_overshoot")
    heatmap_metric("final_to_min_mean", "Final/Min (rebound ratio)", "Final/Min", "heat_final_to_min")
    heatmap_metric("log10_tail_cv_mean", "log10(tail_cv)", "log10(tail_cv)", "heat_logtailcv")
    heatmap_metric("t95_mean", "t95", "t95 [s]", "heat_t95")
    heatmap_metric("vol_risetime_10_90_mean", "volume risetime 10 to 90", "t90 - t10 [s]", "heat_risetime_10_90")
    heatmap_metric("vol_tail_std_mean", "volume tail std (last 50)", "tail std", "heat_vol_tail_std")

    heatmap_metric("forces_mean_mean", "forces mean (final)", "force", "heat_forces_mean")
    heatmap_metric("forces_max_mean", "forces max (final)", "force", "heat_forces_max")
    heatmap_metric("forces_std_mean", "forces std (final)", "force", "heat_forces_std")
    heatmap_metric("diff_norm_end_mean", "end translation mismatch", "||d_obj - d_p0||", "heat_diff_norm_end")
    heatmap_metric("obj_disp_mean", "object displacement magnitude", "||obj_end - obj0||", "heat_obj_disp")
    heatmap_metric("p0_disp_mean", "first particle displacement magnitude", "||p0_end - p00||", "heat_p0_disp")

    # coverage heatmaps
    for m in mass_vals:
        gm = g[np.isclose(g["cloth_mass_scale"], m)].copy()

        piv = gm.pivot(index="cloth_stiff_scale", columns="cloth_damp_scale", values="seed_count_present").reindex(index=stiff_vals, columns=damp_vals)
        make_heatmap(
            piv,
            title=f"seed_count_present  mass_scale={m}",
            xlabel="cloth_damp_scale",
            ylabel="cloth_stiff_scale",
            cbar_label="seed_count_present",
            out_dir=out_dir,
            filename=f"heat_seed_count_present_mass{_sanitize_token(m)}.png",
            annotate=annotate,
        )

        piv = gm.pivot(index="cloth_stiff_scale", columns="cloth_damp_scale", values="seed_count_used").reindex(index=stiff_vals, columns=damp_vals)
        make_heatmap(
            piv,
            title=f"seed_count_used (successful)  mass_scale={m}",
            xlabel="cloth_damp_scale",
            ylabel="cloth_stiff_scale",
            cbar_label="seed_count_used",
            out_dir=out_dir,
            filename=f"heat_seed_count_used_mass{_sanitize_token(m)}.png",
            annotate=annotate,
        )

    # per finger heatmaps (optional)
    if (not args.no_per_finger) and force_cols:
        for m in mass_vals:
            gm = g[np.isclose(g["cloth_mass_scale"], m)].copy()
            for fc in sorted(force_cols, key=lambda s: int(s.split("force_f")[-1])):
                mean_col = f"{fc}_mean"
                if mean_col not in gm.columns:
                    continue
                piv = gm.pivot(index="cloth_stiff_scale", columns="cloth_damp_scale", values=mean_col).reindex(index=stiff_vals, columns=damp_vals)
                make_heatmap(
                    piv,
                    title=f"{fc} (mean final force, successful only)  mass_scale={m}",
                    xlabel="cloth_damp_scale",
                    ylabel="cloth_stiff_scale",
                    cbar_label=f"{fc} mean",
                    out_dir=out_dir,
                    filename=f"heat_{fc}_mass{_sanitize_token(m)}.png",
                    annotate=annotate,
                )

    # -------------------------
    # Trend plots vs stiffness (SUCCESSFUL ONLY)
    # -------------------------
    def plot_trends(metric_mean_col, ylabel, fname, logy=False):
        if metric_mean_col not in g.columns:
            print("[WARN] missing column for trend:", metric_mean_col)
            return
        plt.figure(figsize=(8.6, 5.6))
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
        plt.title(f"{ylabel} vs stiffness")
        plt.grid(True, alpha=0.3)
        plt.xscale("log")
        if logy:
            plt.yscale("log")
        plt.legend(fontsize=8, ncol=2)
        savefig(out_dir, fname)

    plot_trends("vol_final_mean", "vol_final mean", "trend_vol_final_vs_stiff.png", logy=False)
    #plot_trends("overshoot_rel_mean", "overshoot_rel mean", "trend_overshoot_vs_stiff.png", logy=False)
    plot_trends("final_to_min_mean", "Final/Min mean (lower is better)", "trend_final_to_min_vs_stiff.png", logy=False)
    plot_trends("tail_cv_mean", "tail_cv mean", "trend_tailcv_vs_stiff.png", logy=True)
    plot_trends("t95_mean", "t95 mean [s]", "trend_t95_vs_stiff.png", logy=False)
    plot_trends("forces_mean_mean", "forces mean (final)", "trend_forces_mean_vs_stiff.png", logy=False)
    plot_trends("forces_max_mean", "forces max (final)", "trend_forces_max_vs_stiff.png", logy=False)
    plot_trends("diff_norm_end_mean", "end translation mismatch", "trend_diff_norm_end_vs_stiff.png", logy=False)

    # -------------------------
    # Pareto scatter plots (SUCCESSFUL ONLY)
    # -------------------------

    # only consider mass 1.0
    g_p = g.copy()
    g_p = g_p[g_p["run_count_used"] > 0].copy()
    g_p = g_p[np.isclose(g_p["cloth_mass_scale"], 1.0)].copy()

    plt.figure(figsize=(8.4, 5.8))
    for _, r in g_p.iterrows():
        x = _as_float(r.get("final_to_min_mean", np.nan))
        y = _as_float(r.get("vol_final_mean", np.nan))
        if not (np.isfinite(x) and np.isfinite(y)):
            continue
        m = _as_float(r.get("cloth_mass_scale", np.nan))
        s = r.get("cloth_stiff_scale", np.nan)
        d = r.get("cloth_damp_scale", np.nan)
        plt.scatter(x, y, marker=_marker_for_mass(m), alpha=0.9)
        plt.annotate(f"s={s}, d={d}, m={m}", (x, y), textcoords="offset points", xytext=(4, 3), fontsize=7, alpha=0.8)
    plt.xlabel("V_final/V_min (lower is better)")
    plt.ylabel("V_final (lower is better)")
    plt.title("Pareto view: Final volume (V_final) vs V_final/V_min")
    plt.grid(True, alpha=0.3)
    savefig(out_dir, "pareto_vol_vs_final_to_min.png")

    plt.figure(figsize=(8.4, 5.8))
    for _, r in g_p.iterrows():
        x = _as_float(r.get("tail_cv_mean", np.nan))
        y = _as_float(r.get("vol_final_mean", np.nan))
        if not (np.isfinite(x) and np.isfinite(y)):
            continue
        x = max(float(x), 1e-12)
        m = _as_float(r.get("cloth_mass_scale", np.nan))
        s = r.get("cloth_stiff_scale", np.nan)
        d = r.get("cloth_damp_scale", np.nan)
        plt.scatter(x, y, marker=_marker_for_mass(m), alpha=0.9)
        plt.annotate(f"s={s}, d={d}, m={m}", (x, y), textcoords="offset points", xytext=(4, 3), fontsize=7, alpha=0.8)
    plt.xscale("log")
    plt.xlabel("tail_cv (log scale, lower is better)")
    plt.ylabel("vol_final (lower is better)")
    plt.title("Pareto view: vol_final vs tail_cv")
    plt.grid(True, alpha=0.3)
    savefig(out_dir, "pareto_vol_vs_tailcv.png")

    plt.figure(figsize=(8.4, 5.8))
    for _, r in g_p.iterrows():
        x = _as_float(r.get("forces_max_mean", np.nan))
        y = _as_float(r.get("vol_final_mean", np.nan))
        if not (np.isfinite(x) and np.isfinite(y)):
            continue
        m = _as_float(r.get("cloth_mass_scale", np.nan))
        plt.scatter(x, y, marker=_marker_for_mass(m), alpha=0.9)
    plt.xlabel("forces max (final)")
    plt.ylabel("vol_final")
    plt.title("Tradeoff: final force peak vs volume stability")
    plt.grid(True, alpha=0.3)
    savefig(out_dir, "pareto_vol_vs_forces_max.png")

    # -------------------------
    # Force distribution plots (successful only)
    # -------------------------
    if force_cols and len(runs_used) > 0:
        all_forces = []
        for fc in force_cols:
            v = pd.to_numeric(runs_used[fc], errors="coerce").to_numpy(dtype=float)
            v = v[np.isfinite(v)]
            if v.size:
                all_forces.append(v)
        if all_forces:
            all_forces = np.concatenate(all_forces, axis=0)
            plt.figure(figsize=(8.2, 5.2))
            plt.hist(all_forces, bins=40)
            plt.xlabel("final per finger force")
            plt.ylabel("count")
            plt.title("Distribution of final per finger forces (successful runs only)")
            plt.grid(True, alpha=0.3)
            savefig(out_dir, "hist_all_final_forces_successful.png")

    # per finger vs stiffness (baseline mass=1, damp=1 if exists)
    if (not args.no_per_finger) and force_cols:
        baseline_mass = 1.0 if 1.0 in mass_vals else mass_vals[len(mass_vals)//2]
        baseline_damp = 1.0 if 1.0 in damp_vals else damp_vals[len(damp_vals)//2]
        gb = g[np.isclose(g["cloth_mass_scale"], baseline_mass) & np.isclose(g["cloth_damp_scale"], baseline_damp)].copy()
        if not gb.empty:
            plt.figure(figsize=(9.0, 5.8))
            gb = gb.sort_values("cloth_stiff_scale")
            for fc in sorted(force_cols, key=lambda s: int(s.split("force_f")[-1])):
                mean_col = f"{fc}_mean"
                if mean_col not in gb.columns:
                    continue
                plt.plot(
                    gb["cloth_stiff_scale"],
                    gb[mean_col],
                    marker="o",
                    linestyle="-",
                    alpha=0.9,
                    label=fc,
                )
            plt.xscale("log")
            plt.xlabel("cloth_stiff_scale")
            plt.ylabel("mean final force")
            plt.title(f"Per finger mean final force vs stiffness (successful only, mass={baseline_mass}, damp={baseline_damp})")
            plt.grid(True, alpha=0.3)
            plt.legend(fontsize=8, ncol=3)
            savefig(out_dir, f"perfinger_force_vs_stiff_mass{_sanitize_token(baseline_mass)}_damp{_sanitize_token(baseline_damp)}.png")

    # -------------------------
    # Time series plots (successful only)
    # -------------------------
    if not args.no_timeseries:
        if len(runs_used) == 0:
            print("[WARN] no successful runs -> skipping time series plots.")
        else:
            reps = pick_representative_paths(runs_used, by="vol_final")

            # Volume curves per mass
            for m in mass_vals:
                fig = plt.figure(figsize=(11.5, 8.2))
                any_plotted = False
                for j, d in enumerate(damp_vals, start=1):
                    ax = fig.add_subplot(len(damp_vals), 1, j)
                    for s in stiff_vals:
                        key = (float(s), float(m), float(d))
                        if key not in reps:
                            continue
                        ts = load_npz_timeseries(reps[key])
                        vt = ts.get("vol_t", None)
                        vv = ts.get("vol_v", None)
                        if vt is None or vv is None:
                            continue
                        vt = np.asarray(vt, dtype=np.float64).ravel()
                        vv = np.asarray(vv, dtype=np.float64).ravel()
                        good = np.isfinite(vt) & np.isfinite(vv)
                        vt = vt[good]
                        vv = vv[good]
                        if vt.size < 2:
                            continue
                        ax.plot(vt, vv, linestyle="-", marker=None, alpha=0.9, label=f"stiff={s}")
                        any_plotted = True
                    ax.set_title(f"mass={m} damp={d}")
                    ax.set_xlabel("t [s]")
                    ax.set_ylabel("vol_vox")
                    ax.grid(True, alpha=0.3)
                    ax.legend(fontsize=8, ncol=2)
                if any_plotted:
                    savefig(out_dir, f"timeseries_volume_mass{_sanitize_token(m)}.png")
                else:
                    plt.close(fig)

            # Object displacement magnitude over time
            for m in mass_vals:
                fig = plt.figure(figsize=(11.5, 8.2))
                any_plotted = False
                for j, d in enumerate(damp_vals, start=1):
                    ax = fig.add_subplot(len(damp_vals), 1, j)
                    for s in stiff_vals:
                        key = (float(s), float(m), float(d))
                        if key not in reps:
                            continue
                        ts = load_npz_timeseries(reps[key])
                        ot = ts.get("obj_t", None)
                        ox = ts.get("obj_xyz", None)
                        if ot is None or ox is None:
                            continue
                        ot = np.asarray(ot, dtype=np.float64).ravel()
                        ox = np.asarray(ox, dtype=np.float64)
                        if ox.ndim != 2 or ox.shape[1] != 3 or ot.size != ox.shape[0]:
                            continue
                        if ox.shape[0] < 2:
                            continue
                        dxyz = ox - ox[0:1, :]
                        mag = np.linalg.norm(dxyz, axis=1)
                        ax.plot(ot, mag, linestyle="-", marker=None, alpha=0.9, label=f"stiff={s}")
                        any_plotted = True
                    ax.set_title(f"mass={m} damp={d}  (|obj(t)-obj(0)|)")
                    ax.set_xlabel("t [s]")
                    ax.set_ylabel("displacement magnitude")
                    ax.grid(True, alpha=0.3)
                    ax.legend(fontsize=8, ncol=2)
                if any_plotted:
                    savefig(out_dir, f"timeseries_object_disp_mass{_sanitize_token(m)}.png")
                else:
                    plt.close(fig)

            # Loss curves: pick a few representative combos based on g (successful-only metrics)
            g_rank = g.copy()
            g_rank["vol_final_mean"] = pd.to_numeric(g_rank.get("vol_final_mean", np.nan), errors="coerce")
            #g_rank["overshoot_rel_mean"] = pd.to_numeric(g_rank.get("overshoot_rel_mean", np.nan), errors="coerce")
            g_rank["final_to_min_mean"] = pd.to_numeric(g_rank.get("final_to_min_mean", np.nan), errors="coerce")

            # only combos that actually have successful runs
            g_rank = g_rank[g_rank["run_count_used"] > 0].copy()
            g_rank = g_rank[np.isfinite(g_rank["vol_final_mean"])].copy()

            picks = []
            if not g_rank.empty:
                picks += g_rank.sort_values("vol_final_mean").head(2).to_dict("records")
                picks += g_rank.sort_values("vol_final_mean", ascending=False).head(2).to_dict("records")
                #picks += g_rank.sort_values("overshoot_rel_mean", ascending=False).head(2).to_dict("records")
                picks += g_rank.sort_values("final_to_min_mean", ascending=False).head(2).to_dict("records")

            seen = set()
            uniq = []
            for r in picks:
                k = (float(r["cloth_stiff_scale"]), float(r["cloth_mass_scale"]), float(r["cloth_damp_scale"]))
                if k in seen:
                    continue
                seen.add(k)
                uniq.append(r)

            if uniq:
                plt.figure(figsize=(9.2, 5.8))
                for r in uniq:
                    k = (float(r["cloth_stiff_scale"]), float(r["cloth_mass_scale"]), float(r["cloth_damp_scale"]))
                    if k not in reps:
                        continue
                    ts = load_npz_timeseries(reps[k])
                    lc = ts.get("loss_curve", None)
                    if lc is None:
                        continue
                    lc = np.asarray(lc, dtype=np.float64).ravel()
                    if lc.size < 2:
                        continue
                    plt.plot(lc, linestyle="-", marker=None, alpha=0.9, label=f"s={k[0]}, m={k[1]}, d={k[2]}")
                plt.xlabel("optimiser step")
                plt.ylabel("loss")
                plt.title("Representative optimiser loss curves (successful only)")
                plt.grid(True, alpha=0.3)
                plt.legend(fontsize=8, ncol=2)
                savefig(out_dir, "timeseries_loss_curves_representative.png")

    # -------------------------
    # Console summary
    # -------------------------
    print("\n[INFO] combos present:", len(g))
    print("[INFO] runs present:", len(runs_all))
    print("[INFO] successful runs used:", len(runs_used))

    if len(g):
        cols = [
            "cloth_stiff_scale", "cloth_mass_scale", "cloth_damp_scale",
            "run_count_used", "seed_count_used",
            "vol_final_mean", "final_to_min_mean", "tail_cv_mean", "t95_mean", "forces_max_mean"
        ]
        keep = [c for c in cols if c in g.columns]

        g_ok = g[g["run_count_used"] > 0].copy()
        if not g_ok.empty and "vol_final_mean" in g_ok.columns:
            print("\n[INFO] best (lowest vol_final_mean, successful only):")
            print(g_ok.sort_values("vol_final_mean").head(8)[keep].to_string(index=False))

        if not g_ok.empty and "final_to_min_mean" in g_ok.columns:
            print("\n[INFO] worst final_to_min_mean (successful only):")
            print(g_ok.sort_values("final_to_min_mean", ascending=False).head(8)[keep].to_string(index=False))

    print(f"\n[OK] All plots saved to: {out_dir}")
    print(f"[OK] NPZ dir was: {npz_dir}")


if __name__ == "__main__":
    main()
