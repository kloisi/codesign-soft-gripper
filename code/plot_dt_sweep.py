# plot_dt_sweep.py
#
# Load dt sweep NPZ logs produced by sweep_dt.py and generate plots.
#
# Expected inputs (default paths relative to this script):
#   - dt_sweep_data/run_rep??_fps*_sub*_seed*.npz
#   - dt_sweep_data/sweep_meta.json  (optional, used for context)
#
# Outputs (default):
#   - dt_sweep_plots/*.png
#   - dt_sweep_plots/summary_table.csv
#
# NOTE (added):
#   - For each plot, we now ALSO save a 2nd "presentation" PNG:
#       same name + "_presentation.png"
#     with black background, thicker lines, and larger text.
#   - The original plots are saved exactly as before.
#
# Usage examples:
#   python plot_dt_sweep.py
#   python plot_dt_sweep.py --data_dir dt_sweep_data --out_dir dt_sweep_plots --show
#   python plot_dt_sweep.py --ref sub=20
#   python plot_dt_sweep.py --ref file=dt_sweep_data/run_rep01_fps4000_sub50_seed12346.npz

from __future__ import annotations

import os
import json
import glob
import argparse
from dataclasses import dataclass
from typing import Any, Dict, List, Optional

import numpy as np
import matplotlib.pyplot as plt


# -------------------------
# Utilities
# -------------------------
def _safe_float(x, default=np.nan) -> float:
    if x is None:
        return float(default)
    try:
        return float(x)
    except Exception:
        return float(default)


def _safe_int(x, default=-1) -> int:
    if x is None:
        return int(default)
    try:
        return int(x)
    except Exception:
        return int(default)


def _safe_bool(x, default=False) -> bool:
    if x is None:
        return bool(default)
    if isinstance(x, bool):
        return x
    if isinstance(x, (int, np.integer)):
        return bool(int(x))
    if isinstance(x, str):
        return x.strip().lower() in ["1", "true", "yes", "y", "ok"]
    return bool(x)


def _load_json_scalar(arr) -> Dict[str, Any]:
    """
    sweep_dt.py stored dicts as json strings inside npz:
      meta_json=np.array(json_string)
    so this comes out as a 0d numpy array or string.
    """
    if arr is None:
        return {}
    try:
        if isinstance(arr, np.ndarray):
            s = arr.item()
        else:
            s = arr
        if isinstance(s, bytes):
            s = s.decode("utf-8", errors="replace")
        return json.loads(str(s))
    except Exception:
        return {}


def _interp_rmse(
    t_ref: np.ndarray,
    y_ref: np.ndarray,
    t_run: np.ndarray,
    y_run: np.ndarray,
    t0: float,
    t1: float,
    n: int = 400,
) -> float:
    if t_ref.size < 2 or t_run.size < 2:
        return float("nan")
    t0 = float(t0)
    t1 = float(t1)
    if not np.isfinite(t0) or not np.isfinite(t1) or t1 <= t0:
        return float("nan")
    grid = np.linspace(t0, t1, int(n))
    y_ref_i = np.interp(grid, t_ref, y_ref)
    y_run_i = np.interp(grid, t_run, y_run)
    return float(np.sqrt(np.mean((y_run_i - y_ref_i) ** 2)))


def _ensure_dir(path: str) -> None:
    os.makedirs(path, exist_ok=True)


def _apply_presentation_style(fig, font_scale=1.35, line_scale=1.8, marker_scale=1.35) -> None:
    """
    Mutates the current figure to be more PPT friendly:
      - black background
      - larger fonts
      - thicker lines / larger markers
      - white text/ticks/spines
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

        # Scatter/collections (markers)
        try:
            for coll in ax.collections:
                try:
                    sz = coll.get_sizes()
                    if sz is not None and len(sz):
                        coll.set_sizes(np.asarray(sz, dtype=float) * (marker_scale ** 2))
                except Exception:
                    pass
                try:
                    lw = coll.get_linewidths()
                    if lw is not None and len(lw):
                        coll.set_linewidths(np.maximum(1.0, np.asarray(lw, dtype=float) * line_scale))
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


def _save_fig(out_dir: str, name: str) -> None:
    """
    Saves:
      1) original PNG exactly as before: {name}.png
      2) presentation PNG: {name}_presentation.png (black bg, thicker lines, larger text)
    """
    _ensure_dir(out_dir)

    # 1) Original (unchanged)
    png = os.path.join(out_dir, f"{name}.png")
    plt.savefig(png, dpi=200, bbox_inches="tight")

    # 2) Presentation version
    fig = plt.gcf()
    _apply_presentation_style(fig)
    png_pres = os.path.join(out_dir, f"{name}_presentation.png")
    plt.savefig(
        png_pres,
        dpi=220,
        bbox_inches="tight",
        facecolor=fig.get_facecolor(),
        edgecolor="none",
    )


# -------------------------
# Data structures
# -------------------------
@dataclass
class Run:
    path: str
    meta: Dict[str, Any]
    stats: Dict[str, Any]

    # arrays (may be empty)
    vol_t: np.ndarray
    vol_v: np.ndarray
    q_end: np.ndarray
    qd_end: np.ndarray
    body_q0: np.ndarray
    body_q_end: np.ndarray

    # convenience
    fps: int
    sub: int
    rep: int
    seed: int
    sim_dt: float
    frame_dt: float
    T_target: float
    total_steps: int

    # key stats
    t_forward_s: float
    real_time_per_step_ms: float
    num_ok: bool
    phys_ok: bool
    max_speed: float
    max_abs_pos: float
    vol_final: float


# -------------------------
# Loading NPZ runs
# -------------------------
def load_runs(data_dir: str) -> List[Run]:
    pattern = os.path.join(data_dir, "run_rep*_fps*_sub*_seed*.npz")
    files = sorted(glob.glob(pattern))
    if not files:
        raise FileNotFoundError(f"No NPZ files found at: {pattern}")

    runs: List[Run] = []
    for p in files:
        try:
            with np.load(p, allow_pickle=True) as z:
                meta = _load_json_scalar(z.get("meta_json", None))
                stats = _load_json_scalar(z.get("stats_json", None))

                vol_t = np.asarray(z.get("vol_t", np.array([], dtype=np.float64)), dtype=np.float64)
                vol_v = np.asarray(z.get("vol_v", np.array([], dtype=np.float64)), dtype=np.float64)

                q_end = np.asarray(z.get("q_end", np.array([], dtype=np.float64)), dtype=np.float64)
                qd_end = np.asarray(z.get("qd_end", np.array([], dtype=np.float64)), dtype=np.float64)
                body_q0 = np.asarray(z.get("body_q0", np.array([], dtype=np.float64)), dtype=np.float64)
                body_q_end = np.asarray(z.get("body_q_end", np.array([], dtype=np.float64)), dtype=np.float64)

            fps = _safe_int(meta.get("fps", None), -1)
            sub = _safe_int(meta.get("sim_substeps", None), -1)
            rep = _safe_int(meta.get("rep", None), -1)
            seed = _safe_int(meta.get("kernel_seed", None), -1)
            sim_dt = _safe_float(meta.get("sim_dt", None), np.nan)
            frame_dt = _safe_float(meta.get("frame_dt", None), np.nan)
            T_target = _safe_float(meta.get("T_target", None), np.nan)
            total_steps = _safe_int(meta.get("total_steps", None), -1)

            t_forward_s = _safe_float(stats.get("t_forward_s", None), np.nan)
            real_time_per_step_ms = _safe_float(stats.get("real_time_per_step_ms", None), np.nan)
            num_ok = _safe_bool(stats.get("num_ok", None), False)
            phys_ok = _safe_bool(stats.get("phys_ok", None), False)
            max_speed = _safe_float(stats.get("max_speed", None), np.nan)
            max_abs_pos = _safe_float(stats.get("max_abs_pos", None), np.nan)
            vol_final = _safe_float(stats.get("vol_final", None), np.nan)

            runs.append(
                Run(
                    path=p,
                    meta=meta,
                    stats=stats,
                    vol_t=vol_t,
                    vol_v=vol_v,
                    q_end=q_end,
                    qd_end=qd_end,
                    body_q0=body_q0,
                    body_q_end=body_q_end,
                    fps=fps,
                    sub=sub,
                    rep=rep,
                    seed=seed,
                    sim_dt=sim_dt,
                    frame_dt=frame_dt,
                    T_target=T_target,
                    total_steps=total_steps,
                    t_forward_s=t_forward_s,
                    real_time_per_step_ms=real_time_per_step_ms,
                    num_ok=num_ok,
                    phys_ok=phys_ok,
                    max_speed=max_speed,
                    max_abs_pos=max_abs_pos,
                    vol_final=vol_final,
                )
            )
        except Exception as e:
            print(f"[WARN] Failed to load {p}: {type(e).__name__}: {e}")

    if not runs:
        raise RuntimeError("All NPZ loads failed.")
    return runs


# -------------------------
# Reference selection
# -------------------------
def pick_reference(runs: List[Run], ref_spec: str) -> Run:
    ok_runs = [r for r in runs if r.num_ok and r.phys_ok and np.isfinite(r.sim_dt)]
    if not ok_runs:
        ok_runs = [r for r in runs if np.isfinite(r.sim_dt)]
    if not ok_runs:
        raise RuntimeError("No runs with finite sim_dt found.")

    ref_spec = (ref_spec or "").strip().lower()
    if ref_spec in ["", "min_dt", "mindt", "auto"]:
        # smallest dt among ok runs
        ok_runs = sorted(ok_runs, key=lambda r: (r.sim_dt, r.rep, r.sub))
        return ok_runs[0]

    if ref_spec.startswith("sub="):
        target_sub = int(ref_spec.split("=", 1)[1])
        cand = [r for r in ok_runs if r.sub == target_sub]
        if not cand:
            raise ValueError(f"No runs found with sub={target_sub}")
        cand = sorted(cand, key=lambda r: (r.rep, r.sim_dt))
        return cand[0]

    if ref_spec.startswith("rep="):
        target_rep = int(ref_spec.split("=", 1)[1])
        cand = [r for r in ok_runs if r.rep == target_rep]
        if not cand:
            raise ValueError(f"No runs found with rep={target_rep}")
        cand = sorted(cand, key=lambda r: (r.sim_dt, r.sub))
        return cand[0]

    if ref_spec.startswith("file="):
        target = ref_spec.split("=", 1)[1]
        target = os.path.normpath(target)
        for r in runs:
            if os.path.normpath(r.path) == target:
                return r
        raise ValueError(f"Reference file not found in loaded runs: {target}")

    raise ValueError(f"Unknown --ref spec: {ref_spec}")


# -------------------------
# Metrics vs reference
# -------------------------
def compute_comparisons(runs: List[Run], ref: Run) -> List[Dict[str, Any]]:
    """
    Returns a list of dict rows for summary table and plotting.
    """
    rows: List[Dict[str, Any]] = []

    # Reference signals
    t_ref = ref.vol_t
    v_ref = ref.vol_v

    # Determine time interval for trajectory RMSE
    t0 = 0.0
    t1 = float(ref.T_target) if np.isfinite(ref.T_target) else (float(t_ref[-1]) if t_ref.size else np.nan)

    # Reference volume normaliser
    v_ref_final = ref.vol_final
    if not np.isfinite(v_ref_final) and v_ref.size:
        v_ref_final = float(v_ref[-1])
    v_ref_final = float(v_ref_final) if np.isfinite(v_ref_final) else float("nan")

    for r in runs:
        # Final volume error
        v_run_final = r.vol_final
        if not np.isfinite(v_run_final) and r.vol_v.size:
            v_run_final = float(r.vol_v[-1])

        vol_abs_err = float("nan")
        vol_rel_err = float("nan")
        if np.isfinite(v_run_final) and np.isfinite(v_ref_final):
            vol_abs_err = float(v_run_final - v_ref_final)
            if abs(v_ref_final) > 1e-12:
                vol_rel_err = float(vol_abs_err / v_ref_final)

        # Trajectory RMSE
        vol_traj_rmse = float("nan")
        if r.vol_t.size >= 2 and r.vol_v.size >= 2 and t_ref.size >= 2 and v_ref.size >= 2:
            t1_use = t1
            if np.isfinite(t1_use):
                t1_use = min(t1_use, float(r.T_target) if np.isfinite(r.T_target) else t1_use)
                t1_use = min(t1_use, float(r.vol_t[-1]))
                t1_use = min(t1_use, float(t_ref[-1]))
            vol_traj_rmse = _interp_rmse(t_ref, v_ref, r.vol_t, r.vol_v, t0, t1_use, n=500)

        # q_end RMS error
        q_rms_err = float("nan")
        if ref.q_end.size and r.q_end.size and ref.q_end.shape == r.q_end.shape:
            diff = (r.q_end - ref.q_end).astype(np.float64, copy=False)
            q_rms_err = float(np.sqrt(np.mean(diff * diff)))

        # Cost metrics
        sim_rate = float("nan")  # simulated seconds per real second
        time_per_sim_s = float("nan")
        if np.isfinite(r.t_forward_s) and np.isfinite(r.T_target) and r.t_forward_s > 0 and r.T_target > 0:
            sim_rate = float(r.T_target / r.t_forward_s)
            time_per_sim_s = float(r.t_forward_s / r.T_target)

        rows.append(
            dict(
                path=r.path,
                fps=r.fps,
                sub=r.sub,
                rep=r.rep,
                seed=r.seed,
                sim_dt=r.sim_dt,
                frame_dt=r.frame_dt,
                total_steps=r.total_steps,
                T_target=r.T_target,
                ok=int(bool(r.num_ok and r.phys_ok)),
                num_ok=int(bool(r.num_ok)),
                phys_ok=int(bool(r.phys_ok)),
                t_forward_s=r.t_forward_s,
                real_time_per_step_ms=r.real_time_per_step_ms,
                time_per_sim_s=time_per_sim_s,
                sim_rate=sim_rate,
                max_speed=r.max_speed,
                max_abs_pos=r.max_abs_pos,
                vol_final=v_run_final,
                vol_abs_err=vol_abs_err,
                vol_rel_err=vol_rel_err,
                vol_traj_rmse=vol_traj_rmse,
                q_end_rms_err=q_rms_err,
            )
        )

    rows.sort(key=lambda d: (d["sim_dt"], d["rep"], d["sub"]))
    return rows


# -------------------------
# Plot helpers
# -------------------------
def plot_volume_trajectories(
    runs: List[Run],
    ref: Run,
    out_dir: str,
    subs_to_show: Optional[List[int]] = None,
) -> None:
    """
    Overlay voxel volume trajectories for a few substeps (dt values) to show drift and shape changes.
    """
    all_subs = sorted({r.sub for r in runs if r.vol_t.size >= 2 and r.vol_v.size >= 2})
    if not all_subs:
        print("[WARN] No volume trajectories found to plot.")
        return

    if subs_to_show:
        pick = [s for s in subs_to_show if s in all_subs]
        if not pick:
            pick = all_subs
    else:
        pick = all_subs

    # For each sub, use the first rep found
    chosen: List[Run] = []
    for s in pick:
        cand = [r for r in runs if r.sub == s and r.vol_t.size >= 2 and r.vol_v.size >= 2]
        cand = sorted(cand, key=lambda r: (r.rep, r.sim_dt))
        if cand:
            chosen.append(cand[0])

    # Ensure reference is included
    if ref.sub not in [r.sub for r in chosen] and ref.vol_t.size >= 2:
        chosen.append(ref)
    chosen = sorted(chosen, key=lambda r: r.sim_dt)

    plt.figure()
    for r in chosen:
        lab = f"sub={r.sub} (dt={r.sim_dt:.2e})"
        if os.path.normpath(r.path) == os.path.normpath(ref.path):
            lab += "  [ref]"
        plt.plot(r.vol_t, r.vol_v, label=lab)

    plt.xlabel("time [s]")
    plt.ylabel("voxel volume")
    plt.title("Voxel volume trajectory for different dt")
    plt.legend(fontsize=9)
    plt.grid(True, alpha=0.3)
    _save_fig(out_dir, "volume_trajectory_overlay")
    plt.close()


def plot_errors_vs_dt(rows: List[Dict[str, Any]], ref: Run, out_dir: str) -> None:
    dt = np.array([r["sim_dt"] for r in rows], dtype=float)
    sub = np.array([r["sub"] for r in rows], dtype=int)
    ok = np.array([r["ok"] for r in rows], dtype=int)

    vol_rel = np.array([r["vol_rel_err"] for r in rows], dtype=float)
    vol_rmse = np.array([r["vol_traj_rmse"] for r in rows], dtype=float)
    q_rms = np.array([r["q_end_rms_err"] for r in rows], dtype=float)

    # final volume rel error vs dt
    plt.figure()
    plt.scatter(dt, np.abs(vol_rel), c=ok)
    for i in range(len(rows)):
        y = abs(vol_rel[i]) if np.isfinite(vol_rel[i]) else 1e-16
        plt.annotate(f"{sub[i]}", (dt[i], max(1e-16, y)), fontsize=8)
    plt.xscale("log")
    plt.yscale("log")
    plt.xlabel("dt [s] (log)")
    plt.ylabel("|final volume relative error| vs reference (log)")
    plt.title(f"Final volume error vs dt (reference: sub={ref.sub}, dt={ref.sim_dt:.2e})")
    plt.grid(True, which="both", alpha=0.3)
    _save_fig(out_dir, "error_final_volume_rel_vs_dt")
    plt.close()

    # trajectory RMSE vs dt
    plt.figure()
    plt.scatter(dt, vol_rmse, c=ok)
    for i in range(len(rows)):
        if np.isfinite(vol_rmse[i]):
            plt.annotate(f"{sub[i]}", (dt[i], vol_rmse[i]), fontsize=8)
    plt.xscale("log")
    plt.yscale("log")
    plt.xlabel("dt [s] (log)")
    plt.ylabel("volume trajectory RMSE vs reference (log)")
    plt.title("Volume trajectory difference vs dt")
    plt.grid(True, which="both", alpha=0.3)
    _save_fig(out_dir, "error_volume_traj_rmse_vs_dt")
    plt.close()

    # q_end rms error vs dt
    if np.isfinite(q_rms).any():
        plt.figure()
        plt.scatter(dt, q_rms, c=ok)
        for i in range(len(rows)):
            if np.isfinite(q_rms[i]):
                plt.annotate(f"{sub[i]}", (dt[i], q_rms[i]), fontsize=8)
        plt.xscale("log")
        plt.yscale("log")
        plt.xlabel("dt [s] (log)")
        plt.ylabel("q_end RMS error vs reference (log)")
        plt.title("Final cloth state difference vs dt")
        plt.grid(True, which="both", alpha=0.3)
        _save_fig(out_dir, "error_qend_rms_vs_dt")
        plt.close()


def plot_cost_vs_dt(rows: List[Dict[str, Any]], out_dir: str) -> None:
    dt = np.array([r["sim_dt"] for r in rows], dtype=float)
    sub = np.array([r["sub"] for r in rows], dtype=int)

    t_forward = np.array([r["t_forward_s"] for r in rows], dtype=float)
    time_per_sim_s = np.array([r["time_per_sim_s"] for r in rows], dtype=float)

    plt.figure()
    plt.scatter(dt, t_forward)
    for i in range(len(rows)):
        if np.isfinite(t_forward[i]):
            plt.annotate(f"{sub[i]}", (dt[i], t_forward[i]), fontsize=8)
    plt.xscale("log")
    plt.xlabel("dt [s] (log)")
    plt.ylabel("forward runtime [s]")
    plt.title("Forward runtime vs dt")
    plt.grid(True, which="both", alpha=0.3)
    _save_fig(out_dir, "cost_forward_runtime_vs_dt")
    plt.close()

    if np.isfinite(time_per_sim_s).any():
        plt.figure()
        plt.scatter(dt, time_per_sim_s)
        for i in range(len(rows)):
            if np.isfinite(time_per_sim_s[i]):
                plt.annotate(f"{sub[i]}", (dt[i], time_per_sim_s[i]), fontsize=8)
        plt.xscale("log")
        plt.yscale("log")
        plt.xlabel("dt [s] (log)")
        plt.ylabel("real seconds per simulated second (log)")
        plt.title("How expensive is one simulated second")
        plt.grid(True, which="both", alpha=0.3)
        _save_fig(out_dir, "cost_time_per_sim_second_vs_dt")
        plt.close()


def plot_tradeoff(rows: List[Dict[str, Any]], ref: Run, out_dir: str) -> None:
    """
    Accuracy vs compute, with each point labelled by substeps.
    """
    sub = np.array([r["sub"] for r in rows], dtype=int)
    ok = np.array([r["ok"] for r in rows], dtype=int)

    cost = np.array([r["t_forward_s"] for r in rows], dtype=float)

    vol_rmse = np.array([r["vol_traj_rmse"] for r in rows], dtype=float)
    q_rms = np.array([r["q_end_rms_err"] for r in rows], dtype=float)

    use_q = np.isfinite(q_rms).any()
    if use_q:
        vr = vol_rmse.copy()
        qr = q_rms.copy()
        vr = vr / (np.nanmedian(vr[np.isfinite(vr)]) + 1e-12)
        qr = qr / (np.nanmedian(qr[np.isfinite(qr)]) + 1e-12)
        err = np.sqrt(vr * vr + qr * qr)
    else:
        err = np.abs(vol_rmse)

    plt.figure()
    plt.scatter(cost, err, c=ok)
    for i in range(len(rows)):
        if np.isfinite(cost[i]) and np.isfinite(err[i]):
            plt.annotate(f"{sub[i]}", (cost[i], err[i]), fontsize=8)
    plt.yscale("log")
    plt.xlabel("forward runtime [s]")
    plt.ylabel("combined error vs reference (log)")
    plt.title(f"Speed accuracy tradeoff (reference sub={ref.sub})")
    plt.grid(True, which="both", alpha=0.3)
    _save_fig(out_dir, "tradeoff_speed_accuracy")
    plt.close()

    finite = np.isfinite(cost) & np.isfinite(err)
    if np.any(finite):
        c = cost[finite]
        e = err[finite]
        s = sub[finite]

        c_n = (c - np.min(c)) / (np.max(c) - np.min(c) + 1e-12)
        e_n = (e - np.min(e)) / (np.max(e) - np.min(e) + 1e-12)
        score = np.sqrt(c_n * c_n + e_n * e_n)
        j = int(np.argmin(score))
        print(f"[suggestion] knee candidate: sub={int(s[j])}  runtime={c[j]:.3g}s  err={e[j]:.3g}")


def plot_stability(rows: List[Dict[str, Any]], out_dir: str) -> None:
    dt = np.array([r["sim_dt"] for r in rows], dtype=float)
    sub = np.array([r["sub"] for r in rows], dtype=int)
    ok = np.array([r["ok"] for r in rows], dtype=int)
    num_ok = np.array([r["num_ok"] for r in rows], dtype=int)
    phys_ok = np.array([r["phys_ok"] for r in rows], dtype=int)

    plt.figure()
    plt.scatter(dt, num_ok, label="num_ok")
    plt.scatter(dt, phys_ok, label="phys_ok")
    plt.scatter(dt, ok, label="both ok")
    for i in range(len(rows)):
        plt.annotate(f"{sub[i]}", (dt[i], ok[i]), fontsize=8)
    plt.xscale("log")
    plt.yticks([0, 1], ["fail", "ok"])
    plt.xlabel("dt [s] (log)")
    plt.ylabel("sanity checks")
    plt.title("Stability and sanity checks vs dt")
    plt.legend()
    plt.grid(True, which="both", alpha=0.3)
    _save_fig(out_dir, "stability_checks_vs_dt")
    plt.close()


def plot_max_speed_vs_dt(rows: List[Dict[str, Any]], out_dir: str) -> None:
    dt = np.array([r["sim_dt"] for r in rows], dtype=float)
    sub = np.array([r["sub"] for r in rows], dtype=int)
    max_speed = np.array([r["max_speed"] for r in rows], dtype=float)

    if not np.isfinite(max_speed).any():
        return

    plt.figure()
    plt.scatter(dt, max_speed)
    for i in range(len(rows)):
        if np.isfinite(max_speed[i]):
            plt.annotate(f"{sub[i]}", (dt[i], max_speed[i]), fontsize=8)
    plt.xscale("log")
    plt.yscale("log")
    plt.xlabel("dt [s] (log)")
    plt.ylabel("max particle speed (log)")
    plt.title("Max speed vs dt (often reveals instability onset)")
    plt.grid(True, which="both", alpha=0.3)
    _save_fig(out_dir, "max_speed_vs_dt")
    plt.close()


def write_summary_csv(rows: List[Dict[str, Any]], out_dir: str) -> None:
    import csv

    path = os.path.join(out_dir, "summary_table.csv")
    cols = [
        "rep",
        "sub",
        "sim_dt",
        "t_forward_s",
        "time_per_sim_s",
        "total_steps",
        "ok",
        "num_ok",
        "phys_ok",
        "vol_final",
        "vol_abs_err",
        "vol_rel_err",
        "vol_traj_rmse",
        "q_end_rms_err",
        "max_speed",
        "max_abs_pos",
        "path",
    ]
    with open(path, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(cols)
        for r in rows:
            w.writerow([r.get(c, "") for c in cols])
    print(f"[saved] {path}")


# -------------------------
# Main
# -------------------------
def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--data_dir", default="dt_sweep_data", help="directory containing run_*.npz")
    ap.add_argument("--out_dir", default="dt_sweep_plots", help="directory to save plots")
    ap.add_argument("--ref", default="min_dt", help="reference selection: min_dt | sub=50 | rep=1 | file=PATH")
    ap.add_argument("--show", action="store_true", help="also show figures interactively")
    ap.add_argument(
        "--traj_subs",
        default="",
        help="comma separated substeps to overlay in trajectory plot, eg: 5,20,100 (empty = auto)",
    )
    args = ap.parse_args()

    data_dir = os.path.abspath(args.data_dir)
    out_dir = os.path.abspath(args.out_dir)
    _ensure_dir(out_dir)

    runs = load_runs(data_dir)
    ref = pick_reference(runs, args.ref)
    rows = compute_comparisons(runs, ref)

    print(f"[info] loaded runs: {len(runs)}")
    print(f"[info] reference: sub={ref.sub} dt={ref.sim_dt:.2e} rep={ref.rep} file={os.path.basename(ref.path)}")

    write_summary_csv(rows, out_dir)

    subs_to_show = None
    if args.traj_subs.strip():
        subs_to_show = [int(s.strip()) for s in args.traj_subs.split(",") if s.strip()]

    plot_volume_trajectories(runs, ref, out_dir, subs_to_show=subs_to_show)
    plot_errors_vs_dt(rows, ref, out_dir)
    plot_cost_vs_dt(rows, out_dir)
    plot_tradeoff(rows, ref, out_dir)
    plot_stability(rows, out_dir)
    plot_max_speed_vs_dt(rows, out_dir)

    print(f"[done] plots saved to: {out_dir}")
    print(f"[done] presentation plots saved alongside originals with *_presentation.png")

    if args.show:
        print("[note] This script saves plots and closes figures. For interactive viewing, open the PNG files.")


if __name__ == "__main__":
    main()
