# sweep_volume_across_corals.py
#
# sweep for enclosure-volume evaluation across coral objects.

# run like: python sweep_volume_across_corals.py

# Saves:
# - summary.csv : one row per coral
# - timeseries.csv : enclosed volume over time for all corals
# - aggregate_stats.csv : mean/std/median/IQR for report table

import os
import gc
import math
import argparse
from datetime import datetime

import numpy as np
import pandas as pd
import warp as wp

from forward import FEMTendon
from init_pose import InitializeFingers
from object_loader import ObjectLoader


def list_coral_objects():
    """
    Return all non-YCB object folders.
    the YCB folders start with 000_, 001_, etc.
    """
    loader = ObjectLoader()
    names = []

    for name in sorted(os.listdir(loader.data_dir)):
        full = os.path.join(loader.data_dir, name)
        if not os.path.isdir(full):
            continue
        if len(name) >= 4 and name[:3].isdigit() and name[3] == "_":
            continue
        names.append(name)

    return names


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


def run_init_pose(object_name, finger_num, pose_iters, scale, finger_len, finger_rot, finger_width, object_rot):
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
    ok = finger_transform is not None

    del init_finger
    cleanup()

    return finger_transform, ok


def run_one_object(name, args):
    finger_len = 11
    finger_rot = np.pi / 30
    finger_width = 0.08
    scale = 5.0
    object_rot = wp.quat_rpy(-math.pi / 2, 0.0, 0.0)

    seed = int(np.random.randint(0, 1_000_000_000))

    result = {
        "object": name,
        "finger_num": args.finger_num,
        "seed": seed,
        "init_pose_ok": 0,
        "force_opt_ok": 0,
        "init_vol_vox_m3": np.nan,
        "final_vol_vox_m3": np.nan,
        "delta_vol_vox_m3": np.nan,
        "pct_change": np.nan,
        "pct_reduction": np.nan,
        "error": "",
    }

    timeseries = []

    with wp.ScopedDevice(args.device):
        np.random.seed(seed)

        # 1) initialize fingers
        finger_transform, init_ok = run_init_pose(
            object_name=name,
            finger_num=args.finger_num,
            pose_iters=args.pose_iters,
            scale=scale,
            finger_len=finger_len,
            finger_rot=finger_rot,
            finger_width=finger_width,
            object_rot=object_rot,
        )
        result["init_pose_ok"] = int(init_ok)

        # 2) build simulation
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
            object_density=2.0,
            ycb_object_name=name,
            finger_len=finger_len,
            finger_rot=finger_rot,
            finger_width=finger_width,
            scale=scale,
            finger_transform=finger_transform,
            finger_num=args.finger_num,
            requires_grad=True,
            init_finger=None,
            no_cloth=False,
            no_voxvol=False,
        )

        # prevent per-object CSV spam from forward.py
        tendon.vol_logger.to_csv = lambda path: None

        # 3) optimize tendon forces
        try:
            if not args.no_force_opt:
                tendon.optimize_forces_lbfgs(
                    iterations=1,
                    learning_rate=1.0,
                    opt_frames=args.opt_frames,
                )
            result["force_opt_ok"] = 1
        except Exception as e:
            result["error"] = f"force optimization failed: {type(e).__name__}: {e}"

        # 4) rollout
        if result["error"] == "":
            try:
                tendon.forward(
                    save_stride=1,
                    save_for_viz=False,
                    vox_every=args.vox_every,
                )
            except Exception as e:
                result["error"] = f"forward failed: {type(e).__name__}: {e}"

        # 5) collect results
        if result["error"] == "":
            v0 = tendon.init_voxel_volume
            vend = tendon.last_voxel_volume

            result["init_vol_vox_m3"] = float(v0)
            result["final_vol_vox_m3"] = float(vend)
            result["delta_vol_vox_m3"] = float(vend - v0)

            if abs(v0) > 1e-12:
                pct_change = 100.0 * (vend - v0) / v0
                result["pct_change"] = pct_change
                result["pct_reduction"] = -pct_change

            # explicit t=0 row
            timeseries.append({
                "object": name,
                "finger_num": args.finger_num,
                "frame": -1,
                "t": 0.0,
                "vol_vox": float(v0),
                "vol_norm_init": 1.0,
            })

            for row in tendon.vol_logger.rows:
                vol = float(row["vol_vox"])
                timeseries.append({
                    "object": name,
                    "finger_num": args.finger_num,
                    "frame": int(row["frame"]),
                    "t": float(row["t"]),
                    "vol_vox": vol,
                    "vol_norm_init": vol / v0 if abs(v0) > 1e-12 else np.nan,
                })

        del tendon
        cleanup()

    return result, timeseries


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--device", type=str, default="cuda:0")
    parser.add_argument("--finger_num", type=int, default=6)
    parser.add_argument("--pose_iters", type=int, default=1000)
    parser.add_argument("--num_frames", type=int, default=1000)
    parser.add_argument("--opt_frames", type=int, default=100)
    parser.add_argument("--vox_every", type=int, default=10)
    parser.add_argument("--no_force_opt", action="store_true")
    parser.add_argument(
        "--objects",
        type=str,
        default="",
        help="Comma-separated list of coral object names. If empty, all non-YCB objects are used.",
    )
    args = parser.parse_args()

    if args.objects.strip():
        object_names = [x.strip() for x in args.objects.split(",") if x.strip()]
    else:
        object_names = list_coral_objects()

    run_name = datetime.now().strftime("sweep_vol_corals_%Y%m%d_%H%M%S")
    out_dir = os.path.join("logs", f"{run_name}_f{args.finger_num}")
    os.makedirs(out_dir, exist_ok=True)

    summary_rows = []
    timeseries_rows = []

    print(f"Running sweep for {len(object_names)} coral objects")
    print(f"Saving results to {out_dir}")

    for i, name in enumerate(object_names, start=1):
        print(f"[{i}/{len(object_names)}] {name}")

        try:
            summary, timeseries = run_one_object(name, args)
        except Exception as e:
            summary = {
                "object": name,
                "finger_num": args.finger_num,
                "seed": np.nan,
                "init_pose_ok": 0,
                "force_opt_ok": 0,
                "init_vol_vox_m3": np.nan,
                "final_vol_vox_m3": np.nan,
                "delta_vol_vox_m3": np.nan,
                "pct_change": np.nan,
                "pct_reduction": np.nan,
                "error": f"{type(e).__name__}: {e}",
            }
            timeseries = []

        summary_rows.append(summary)
        timeseries_rows.extend(timeseries)

        pd.DataFrame(summary_rows).to_csv(os.path.join(out_dir, "summary.csv"), index=False)
        pd.DataFrame(timeseries_rows).to_csv(os.path.join(out_dir, "timeseries.csv"), index=False)

    summary_df = pd.DataFrame(summary_rows)
    ok = summary_df[summary_df["error"].fillna("") == ""].copy()

    initial_stats = compute_stats(ok["init_vol_vox_m3"])
    final_stats = compute_stats(ok["final_vol_vox_m3"])
    change_stats = compute_stats(ok["pct_change"])

    aggregate_df = pd.DataFrame([
        {
            "Quantity": "Mean",
            "Initial": initial_stats["mean"],
            "Final": final_stats["mean"],
            "Relative change (%)": change_stats["mean"],
        },
        {
            "Quantity": "Standard deviation",
            "Initial": initial_stats["std"],
            "Final": final_stats["std"],
            "Relative change (%)": change_stats["std"],
        },
        {
            "Quantity": "Median",
            "Initial": initial_stats["median"],
            "Final": final_stats["median"],
            "Relative change (%)": change_stats["median"],
        },
        {
            "Quantity": "Interquartile range",
            "Initial": initial_stats["iqr"],
            "Final": final_stats["iqr"],
            "Relative change (%)": change_stats["iqr"],
        },
    ])

    summary_df.to_csv(os.path.join(out_dir, "summary.csv"), index=False)
    pd.DataFrame(timeseries_rows).to_csv(os.path.join(out_dir, "timeseries.csv"), index=False)
    aggregate_df.to_csv(os.path.join(out_dir, "aggregate_stats.csv"), index=False)

    # useful clean per-object table
    per_object_df = ok[[
        "object",
        "init_vol_vox_m3",
        "final_vol_vox_m3",
        "delta_vol_vox_m3",
        "pct_change",
        "pct_reduction",
    ]].sort_values("object")
    per_object_df.to_csv(os.path.join(out_dir, "per_object_stats.csv"), index=False)

    print("Done.")
    print("Saved:")
    print(" ", os.path.join(out_dir, "summary.csv"))
    print(" ", os.path.join(out_dir, "timeseries.csv"))
    print(" ", os.path.join(out_dir, "aggregate_stats.csv"))
    print(" ", os.path.join(out_dir, "per_object_stats.csv"))


if __name__ == "__main__":
    try:
        wp.init()
    except Exception:
        pass
    main()