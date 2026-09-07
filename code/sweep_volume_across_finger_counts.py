# sweep_volume_across_finger_counts.py
#
# sweep for enclosure-volume evaluation across coral objects and finger counts.

# run like: python sweep_volume_across_finger_counts.py

# Saves to logs/sweep_vol_fingers_YYYYMMDD_HHMMSS/ :
# - summary.csv : one row per (coral, finger count)
#
# saved quantities include:
# - initialized and final enclosed volume
# - volume reduction during actuation
# - final-to-initial volume ratio
# - mean initialized radius
# - optimized tendon forces
#
# this is enough to later quantify diminishing returns, for example from:
# - marginal gain between consecutive finger counts
# - saturation relative to the best finger count

import os
import gc
import math
import time
import argparse
from datetime import datetime

import numpy as np
import pandas as pd
import warp as wp

from forward import FEMTendon
from init_pose import InitializeFingers
from object_loader import ObjectLoader


OBJECT_LIST_FULL = [
    "acropora_cervicornis",
    "acropora_florida",
    "acropora_loripes",
    "acropora_millepora",
    "acropora_nobilis",
    "acropora_palmata",
    "acropora_sarmentosa",
    "acropora_tenuis",
    "fungia_scutaria",
    "goniastrea_aspera",
    "montipora_capitata",
    "platygyra_daedalea",
    "platygyra_lamellina",
    "pocillopora_meandrina",
]

FINGER_COUNTS_FULL = [3, 4, 5, 6, 7, 8, 9]


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


def list_available_corals():
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


def extract_radii(finger_transform):
    radii = []

    for tf in finger_transform:
        if hasattr(tf, "p"):
            p = tf.p
            pos = np.array([p[0], p[1], p[2]], dtype=float)
        else:
            pos = np.array(tf[0], dtype=float)

        radii.append(float(np.linalg.norm(pos)))

    return radii, float(np.mean(radii))


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
    if finger_transform is None:
        raise RuntimeError("Pose initialization failed.")

    radii, radius_mean = extract_radii(finger_transform)

    del init_finger
    cleanup()

    return finger_transform, radii, radius_mean


def run_one_case(object_name, finger_num, args):
    finger_len = 11
    finger_rot = np.pi / 30
    finger_width = 0.08
    scale = 5.0
    object_rot = wp.quat_rpy(-math.pi / 2, 0.0, 0.0)

    result = {
        "object": object_name,
        "finger_num": finger_num,
        "seed": int(np.random.randint(0, 1_000_000_000)),
        "radius_mean": np.nan,
        "radius_list": "",
        "final_loss": np.nan,
        "avg_force": np.nan,
        "all_forces": "",
        "init_vol_vox": np.nan,
        "final_vol_vox": np.nan,
        "delta_vol_vox": np.nan,
        "reduction_pct": np.nan,
        "final_over_init": np.nan,
        "runtime_s": np.nan,
        "status": "failed",
        "error": "",
    }

    tendon = None

    with wp.ScopedDevice(args.device):
        np.random.seed(result["seed"])

        try:
            finger_transform, radii, radius_mean = run_init_pose(
                object_name=object_name,
                finger_num=finger_num,
                pose_iters=args.pose_iters,
                scale=scale,
                finger_len=finger_len,
                finger_rot=finger_rot,
                finger_width=finger_width,
                object_rot=object_rot,
            )

            result["radius_mean"] = radius_mean
            result["radius_list"] = str(radii)

            tendon = FEMTendon(
                stage_path=None,
                num_frames=args.num_frames,
                verbose=False,
                save_log=False,
                is_render=False,
                use_graph=False,
                kernel_seed=result["seed"],
                train_iters=args.pose_iters,
                object_rot=object_rot,
                object_density=2.0,
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

            history = tendon.optimize_forces_lbfgs(
                iterations=args.force_opt_iters,
                learning_rate=args.force_opt_lr,
                opt_frames=args.force_opt_frames,
            )

            if history is not None and "loss" in history and len(history["loss"]) > 0:
                result["final_loss"] = float(history["loss"][-1])

            if history is not None and "forces" in history and len(history["forces"]) > 0:
                final_forces = history["forces"][-1]
                result["avg_force"] = float(np.mean(final_forces))
                result["all_forces"] = str(final_forces)

            t0 = time.perf_counter()
            tendon.forward()
            result["runtime_s"] = time.perf_counter() - t0

            v0 = tendon.init_voxel_volume
            vend = tendon.last_voxel_volume

            result["init_vol_vox"] = float(v0)
            result["final_vol_vox"] = float(vend)
            result["delta_vol_vox"] = float(vend - v0)

            if abs(v0) > 1e-12:
                result["reduction_pct"] = 100.0 * (v0 - vend) / v0
                result["final_over_init"] = vend / v0

            result["status"] = "ok"

        except Exception as e:
            result["error"] = f"{type(e).__name__}: {e}"

        if tendon is not None:
            del tendon
        cleanup()

    return result


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--device", type=str, default="cuda:0")
    parser.add_argument("--pose_iters", type=int, default=2000)
    parser.add_argument("--num_frames", type=int, default=1000)
    parser.add_argument("--force_opt_iters", type=int, default=1)
    parser.add_argument("--force_opt_frames", type=int, default=100)
    parser.add_argument("--force_opt_lr", type=float, default=1.0)
    parser.add_argument(
        "--objects",
        type=str,
        default="",
        help="Comma-separated coral names. If empty, use default full list.",
    )
    parser.add_argument(
        "--finger_counts",
        type=str,
        default="",
        help="Comma-separated finger counts. If empty, use default full list.",
    )
    args = parser.parse_args()

    if args.objects.strip():
        object_names = [x.strip() for x in args.objects.split(",") if x.strip()]
    else:
        object_names = OBJECT_LIST_FULL

    if args.finger_counts.strip():
        finger_counts = [int(x.strip()) for x in args.finger_counts.split(",") if x.strip()]
    else:
        finger_counts = FINGER_COUNTS_FULL

    run_name = datetime.now().strftime("sweep_vol_fingers_%Y%m%d_%H%M%S")
    out_dir = os.path.join("logs", run_name)
    os.makedirs(out_dir, exist_ok=True)

    summary_path = os.path.join(out_dir, "summary.csv")
    summary_rows = []

    print(f"Running finger-count sweep for {len(object_names)} coral objects")
    print(f"Finger counts: {finger_counts}")
    print(f"Saving results to {out_dir}")

    total = len(object_names) * len(finger_counts)
    run_id = 0

    for object_name in object_names:
        for finger_num in finger_counts:
            run_id += 1
            print(f"[{run_id}/{total}] {object_name}, fingers={finger_num}")

            try:
                row = run_one_case(object_name, finger_num, args)
            except Exception as e:
                row = {
                    "object": object_name,
                    "finger_num": finger_num,
                    "seed": np.nan,
                    "radius_mean": np.nan,
                    "radius_list": "",
                    "final_loss": np.nan,
                    "avg_force": np.nan,
                    "all_forces": "",
                    "init_vol_vox": np.nan,
                    "final_vol_vox": np.nan,
                    "delta_vol_vox": np.nan,
                    "reduction_pct": np.nan,
                    "final_over_init": np.nan,
                    "runtime_s": np.nan,
                    "status": "failed",
                    "error": f"{type(e).__name__}: {e}",
                }

            summary_rows.append(row)
            pd.DataFrame(summary_rows).to_csv(summary_path, index=False)

    print("Done.")
    print("Saved:")
    print(" ", summary_path)


if __name__ == "__main__":
    try:
        wp.init()
    except Exception:
        pass
    main()