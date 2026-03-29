# sweep_cloth_params.py
#
# sweep for cloth-parameter evaluation.

# run like: python sweep_cloth_params.py

# Saves to logs/sweep_cloth_params_YYYYMMDD_HHMMSS/ :
# - summary.csv : one row per parameter setting and repeat
# - timeseries.csv : enclosed volume over time for all runs
# - aggregate_stats.csv : mean/std/median/IQR for each cloth-parameter combination
# - optional mp4 visualizations if --make_viz is used

import os
import gc
import math
import argparse
from datetime import datetime
from itertools import product

import numpy as np
import pandas as pd
import warp as wp

from forward import FEMTendon
from init_pose import InitializeFingers
from quick_viz import quick_visualize


STIFF_SCALES = [0.1, 0.5, 1.0, 5.0]
MASS_SCALES = [1.0]
DAMP_SCALES = [0.5, 1.0, 2.0]


def cleanup():
    gc.collect()
    try:
        wp.synchronize()
    except Exception:
        pass
    try:
        import torch
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
    except Exception:
        pass


def safe_np(x):
    if hasattr(x, "numpy"):
        return x.numpy()
    return np.asarray(x)


def compute_stats(values):
    values = pd.to_numeric(pd.Series(values), errors="coerce").dropna().to_numpy()
    if len(values) == 0:
        return {
            "mean": np.nan,
            "std": np.nan,
            "median": np.nan,
            "iqr": np.nan,
        }

    return {
        "mean": float(np.mean(values)),
        "std": float(np.std(values, ddof=1)) if len(values) > 1 else np.nan,
        "median": float(np.median(values)),
        "iqr": float(np.percentile(values, 75) - np.percentile(values, 25)),
    }


def get_cloth_ids(tendon):
    if getattr(tendon, "cloth_ids", None) is not None:
        ids = np.asarray(tendon.cloth_ids, dtype=np.int64).ravel()
        if len(ids) > 0:
            return ids

    if getattr(tendon.model, "cloth_particle_ids", None) is not None:
        ids = safe_np(tendon.model.cloth_particle_ids).astype(np.int64).ravel()
        if len(ids) > 0:
            return ids

    if getattr(tendon.builder, "cloth_particle_ids", None) is not None:
        ids = np.asarray(tendon.builder.cloth_particle_ids, dtype=np.int64).ravel()
        if len(ids) > 0:
            return ids

    return None


def scale_cloth_mass(tendon, mass_scale):
    cloth_ids = get_cloth_ids(tendon)
    if cloth_ids is None or len(cloth_ids) == 0:
        return

    inv_mass = safe_np(tendon.model.particle_inv_mass).astype(np.float32)

    for idx in cloth_ids:
        if inv_mass[idx] > 0.0:
            mass = 1.0 / inv_mass[idx]
            mass *= float(mass_scale)
            inv_mass[idx] = 1.0 / max(mass, 1e-12)

    tendon.model.particle_inv_mass = wp.array(
        inv_mass,
        dtype=wp.float32,
        device=wp.get_device(),
    )


def scale_cloth_stiffness(tendon, stiff_scale, damp_scale):
    cloth_ids = get_cloth_ids(tendon)
    if cloth_ids is None or len(cloth_ids) == 0:
        return

    tri_indices = safe_np(tendon.model.tri_indices).astype(np.int64)
    if tri_indices.ndim == 1:
        tri_indices = tri_indices.reshape(-1, 3)

    cloth_mask = np.isin(tri_indices, cloth_ids).any(axis=1)
    if not np.any(cloth_mask):
        return

    tri_materials = safe_np(tendon.model.tri_materials).astype(np.float32)
    tri_materials[cloth_mask, 0] *= float(stiff_scale)
    tri_materials[cloth_mask, 1] *= float(stiff_scale)
    tri_materials[cloth_mask, 2] *= float(damp_scale)

    tendon.model.tri_materials = wp.array(
        tri_materials,
        dtype=wp.float32,
        device=wp.get_device(),
    )


def run_init_pose(object_name, finger_num, pose_iters, scale, finger_len, finger_rot, finger_width, object_rot, device):
    with wp.ScopedDevice(device):
        init_finger = InitializeFingers(
            stage_path="init_pose_sweep_tmp.usd",
            finger_len=finger_len,
            finger_rot=finger_rot,
            finger_width=finger_width,
            stop_margin=0.0005,
            num_frames=30,
            iterations=pose_iters,
            scale=scale,
            num_envs=1,
            ycb_object_name=object_name,
            object_rot=object_rot,
            is_render=False,
            verbose=False,
            is_triangle=False,
            finger_num=finger_num,
            add_random=False,
            consider_cloth=True,
        )

        finger_transform, _ = init_finger.get_initial_position()

    del init_finger
    cleanup()

    if finger_transform is None:
        raise RuntimeError("Pose initialization failed.")

    return finger_transform


def build_tendon(object_name, object_density, finger_num, finger_transform, seed, args, finger_len, finger_rot, finger_width, scale, object_rot):
    tendon = FEMTendon(
        stage_path=None,
        num_frames=args.num_frames,
        verbose=False,
        save_log=False,
        is_render=False,
        use_graph=False,
        kernel_seed=seed,
        train_iters=args.pose_iters,
        object_rot=object_rot,
        object_density=object_density,
        ycb_object_name=object_name,
        finger_len=finger_len,
        finger_rot=finger_rot,
        finger_width=finger_width,
        scale=scale,
        finger_transform=finger_transform,
        finger_num=finger_num,
        requires_grad=True,
        init_finger=None,
        no_cloth=False,
        no_voxvol=False,
    )

    tendon.vol_logger.to_csv = lambda path: None
    return tendon


def extract_force_values(tendon):
    if getattr(tendon, "tendon_forces", None) is None:
        return []
    return safe_np(tendon.tendon_forces).astype(float).ravel().tolist()


def extract_volume_timeseries(tendon, alpha_k, alpha_m, alpha_d, rep):
    rows = []

    v0 = tendon.init_voxel_volume
    if v0 is not None and np.isfinite(v0):
        rows.append({
            "alpha_k": alpha_k,
            "alpha_m": alpha_m,
            "alpha_d": alpha_d,
            "rep": rep,
            "frame": -1,
            "t": 0.0,
            "vol_vox": float(v0),
            "vol_norm_init": 1.0,
        })

    for row in tendon.vol_logger.rows:
        vol = float(row["vol_vox"])
        rows.append({
            "alpha_k": alpha_k,
            "alpha_m": alpha_m,
            "alpha_d": alpha_d,
            "rep": rep,
            "frame": int(row["frame"]),
            "t": float(row["t"]),
            "vol_vox": vol,
            "vol_norm_init": vol / v0 if v0 is not None and abs(v0) > 1e-12 else np.nan,
        })

    return rows


def run_one_case(alpha_k, alpha_m, alpha_d, rep, finger_transform, args, finger_len, finger_rot, finger_width, scale, object_rot):
    seed = int(args.seed_base + 1000 * rep + 100 * alpha_k + 10 * alpha_m + alpha_d)

    result = {
        "object": args.object_name,
        "finger_num": args.finger_num,
        "alpha_k": alpha_k,
        "alpha_m": alpha_m,
        "alpha_d": alpha_d,
        "rep": rep,
        "seed": seed,
        "init_pose_ok": 1,
        "force_opt_ok": 0,
        "init_vol_vox_m3": np.nan,
        "final_vol_vox_m3": np.nan,
        "min_vol_vox_m3": np.nan,
        "delta_vol_vox_m3": np.nan,
        "pct_change": np.nan,
        "pct_reduction": np.nan,
        "rebound_ratio": np.nan,
        "force_mean": np.nan,
        "force_max": np.nan,
        "force_values": "",
        "force_opt_runtime_s": np.nan,
        "rollout_runtime_s": np.nan,
        "error": "",
    }

    timeseries = []

    with wp.ScopedDevice(args.device):
        tendon = None
        try:
            tendon = build_tendon(
                object_name=args.object_name,
                object_density=args.object_density,
                finger_num=args.finger_num,
                finger_transform=finger_transform,
                seed=seed,
                args=args,
                finger_len=finger_len,
                finger_rot=finger_rot,
                finger_width=finger_width,
                scale=scale,
                object_rot=object_rot,
            )

            scale_cloth_mass(tendon, alpha_m)
            scale_cloth_stiffness(tendon, alpha_k, alpha_d)

            try:
                if not args.no_force_opt:
                    t0 = datetime.now()
                    tendon.optimize_forces_lbfgs(
                        iterations=1,
                        learning_rate=1.0,
                        opt_frames=args.opt_frames,
                    )
                    result["force_opt_runtime_s"] = (datetime.now() - t0).total_seconds()
                result["force_opt_ok"] = 1
            except Exception as e:
                result["error"] = f"force optimization failed: {type(e).__name__}: {e}"

            if result["error"] == "":
                t0 = datetime.now()
                tendon.forward()
                result["rollout_runtime_s"] = (datetime.now() - t0).total_seconds()

            if result["error"] == "":
                v0 = tendon.init_voxel_volume
                vend = tendon.last_voxel_volume
                series = extract_volume_timeseries(tendon, alpha_k, alpha_m, alpha_d, rep)

                volumes = pd.to_numeric(pd.DataFrame(series)["vol_vox"], errors="coerce").dropna().to_numpy()
                vmin = float(np.min(volumes)) if len(volumes) > 0 else np.nan

                result["init_vol_vox_m3"] = float(v0)
                result["final_vol_vox_m3"] = float(vend)
                result["min_vol_vox_m3"] = vmin
                result["delta_vol_vox_m3"] = float(vend - v0)

                if abs(v0) > 1e-12:
                    pct_change = 100.0 * (vend - v0) / v0
                    result["pct_change"] = pct_change
                    result["pct_reduction"] = -pct_change

                if np.isfinite(vmin) and abs(vmin) > 1e-12:
                    result["rebound_ratio"] = float(vend / vmin)

                forces = extract_force_values(tendon)
                if len(forces) > 0:
                    result["force_mean"] = float(np.mean(forces))
                    result["force_max"] = float(np.max(forces))
                    result["force_values"] = str([float(f) for f in forces])

                timeseries = series

                if args.make_viz and rep == 1:
                    viz_name = f"viz_k{str(alpha_k).replace('.', 'p')}_m{str(alpha_m).replace('.', 'p')}_d{str(alpha_d).replace('.', 'p')}.mp4"
                    quick_visualize(
                        tendon,
                        stride=50,
                        interval=30,
                        save_path=os.path.join(args.out_dir, viz_name),
                        elev=30,
                        azim=45,
                    )

        except Exception as e:
            result["error"] = f"{type(e).__name__}: {e}"

        if tendon is not None:
            del tendon
        cleanup()

    return result, timeseries


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--device", type=str, default="cuda:0")
    parser.add_argument("--object_name", type=str, default="acropora_cervicornis")
    parser.add_argument("--object_density", type=float, default=2.0)
    parser.add_argument("--finger_num", type=int, default=6)
    parser.add_argument("--pose_iters", type=int, default=1000)
    parser.add_argument("--num_frames", type=int, default=1000)
    parser.add_argument("--opt_frames", type=int, default=100)
    parser.add_argument("--repeats", type=int, default=1)
    parser.add_argument("--seed_base", type=int, default=12345)
    parser.add_argument("--no_force_opt", action="store_true")
    parser.add_argument("--make_viz", action="store_true")
    args = parser.parse_args()

    finger_len = 11
    finger_rot = np.pi / 30
    finger_width = 0.08
    scale = 5.0
    object_rot = wp.quat_rpy(-math.pi / 2, 0.0, 0.0)

    run_name = datetime.now().strftime("sweep_cloth_params_%Y%m%d_%H%M%S")
    out_dir = os.path.join("logs", f"{run_name}_f{args.finger_num}")
    os.makedirs(out_dir, exist_ok=True)
    args.out_dir = out_dir

    print("Running pose initialization once")
    finger_transform = run_init_pose(
        object_name=args.object_name,
        finger_num=args.finger_num,
        pose_iters=args.pose_iters,
        scale=scale,
        finger_len=finger_len,
        finger_rot=finger_rot,
        finger_width=finger_width,
        object_rot=object_rot,
        device=args.device,
    )

    summary_rows = []
    timeseries_rows = []

    cases = list(product(STIFF_SCALES, MASS_SCALES, DAMP_SCALES))
    total = len(cases) * args.repeats

    print(f"Running {total} cloth-parameter cases")
    print(f"Saving results to {out_dir}")

    case_idx = 0
    for rep in range(1, args.repeats + 1):
        for alpha_k, alpha_m, alpha_d in cases:
            case_idx += 1
            print(
                f"[{case_idx}/{total}] "
                f"k={alpha_k}, m={alpha_m}, d={alpha_d}, rep={rep}"
            )

            summary, timeseries = run_one_case(
                alpha_k=alpha_k,
                alpha_m=alpha_m,
                alpha_d=alpha_d,
                rep=rep,
                finger_transform=finger_transform,
                args=args,
                finger_len=finger_len,
                finger_rot=finger_rot,
                finger_width=finger_width,
                scale=scale,
                object_rot=object_rot,
            )

            summary_rows.append(summary)
            timeseries_rows.extend(timeseries)

            pd.DataFrame(summary_rows).to_csv(os.path.join(out_dir, "summary.csv"), index=False)
            pd.DataFrame(timeseries_rows).to_csv(os.path.join(out_dir, "timeseries.csv"), index=False)

    summary_df = pd.DataFrame(summary_rows)
    ok = summary_df[summary_df["error"].fillna("") == ""].copy()

    aggregate_rows = []
    for (alpha_k, alpha_m, alpha_d), group in ok.groupby(["alpha_k", "alpha_m", "alpha_d"]):
        final_stats = compute_stats(group["final_vol_vox_m3"])
        reduction_stats = compute_stats(group["pct_reduction"])
        rebound_stats = compute_stats(group["rebound_ratio"])
        runtime_stats = compute_stats(group["rollout_runtime_s"])

        aggregate_rows.append({
            "alpha_k": alpha_k,
            "alpha_m": alpha_m,
            "alpha_d": alpha_d,
            "n_runs": len(group),

            "final_mean": final_stats["mean"],
            "final_std": final_stats["std"],
            "final_median": final_stats["median"],
            "final_iqr": final_stats["iqr"],

            "reduction_mean": reduction_stats["mean"],
            "reduction_std": reduction_stats["std"],
            "reduction_median": reduction_stats["median"],
            "reduction_iqr": reduction_stats["iqr"],

            "rebound_mean": rebound_stats["mean"],
            "rebound_std": rebound_stats["std"],
            "rebound_median": rebound_stats["median"],
            "rebound_iqr": rebound_stats["iqr"],

            "runtime_mean": runtime_stats["mean"],
            "runtime_std": runtime_stats["std"],
            "runtime_median": runtime_stats["median"],
            "runtime_iqr": runtime_stats["iqr"],
        })

    aggregate_df = pd.DataFrame(aggregate_rows).sort_values(["alpha_m", "alpha_k", "alpha_d"])

    summary_df.to_csv(os.path.join(out_dir, "summary.csv"), index=False)
    pd.DataFrame(timeseries_rows).to_csv(os.path.join(out_dir, "timeseries.csv"), index=False)
    aggregate_df.to_csv(os.path.join(out_dir, "aggregate_stats.csv"), index=False)

    print("Done.")
    print("Saved:")
    print(" ", os.path.join(out_dir, "summary.csv"))
    print(" ", os.path.join(out_dir, "timeseries.csv"))
    print(" ", os.path.join(out_dir, "aggregate_stats.csv"))


if __name__ == "__main__":
    try:
        wp.init()
    except Exception:
        pass
    main()