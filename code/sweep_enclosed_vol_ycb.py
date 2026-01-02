# sweep_enclosed_vol_ycb.py

import os
import csv
import glob
import traceback
import math
import numpy as np
import warp as wp

from forward import FEMTendon
from object_loader import ObjectLoader

def list_ycb_objects(require_mesh=True):
    loader = ObjectLoader()
    root = loader.data_dir  # ../models/ycb/

    names = []
    for name in sorted(os.listdir(root)):
        p = os.path.join(root, name)
        if not os.path.isdir(p):
            continue

        if require_mesh:
            mesh_path = os.path.join(p, "google_16k", "nontextured.ply")
            mesh_path_simple = os.path.join(p, "google_16k", "simple_nontextured.ply")
            if not (os.path.exists(mesh_path) or os.path.exists(mesh_path_simple)):
                continue

        names.append(name)

    return names


def run_one_object(name, device=None):
    # use the same settings you use in forward.py
    finger_len = 11
    finger_rot = np.pi / 30
    finger_width = 0.08
    scale = 5.0
    object_rot = wp.quat_rpy(-math.pi/2, 0.0, 0.0)

    with wp.ScopedDevice(device):
        tendon = FEMTendon(
            stage_path=None,           # no USD output for sweep
            num_frames=1000,           # or reduce if you only need the final state
            verbose=False,
            save_log=False,            # set false to avoid writing per frame CSVs
            is_render=False,
            use_graph=False,
            kernel_seed=123,
            train_iters=1,
            object_rot=object_rot,
            object_density=2.0,
            ycb_object_name=name,
            finger_len=finger_len,
            finger_rot=finger_rot,
            finger_width=finger_width,
            scale=scale,
            finger_transform=None,     # optional: if you want init_pose per object, see note below
            finger_num=2,
            requires_grad=False,       # sweep doesn’t need gradients
            init_finger=None,
            no_cloth=False,
        )

        tendon.forward()

        vol = tendon.last_voxel_volume
        dbg = tendon.last_voxel_debug or {}

        timeseries = tendon.vol_logger.rows  # list of dicts

        summary = {
            "object": name,
            "vol_vox_m3": vol,
            "voxel_size": dbg.get("voxel_size", np.nan),
            "enclosed_voxels": dbg.get("enclosed_voxels", np.nan),
            "blocked_voxels": dbg.get("blocked_voxels", np.nan),
            "y_bottom": dbg.get("y_bottom", np.nan),
            "y_top": dbg.get("y_top", np.nan),
            "shape": str(dbg.get("shape", "")),
        }
        return summary, timeseries


def main():
    out_summary = "logs/ycb_voxel_volume_sweep.csv"
    out_ts = "logs/ycb_voxel_volume_timeseries_all.csv"
    os.makedirs("logs", exist_ok=True)

    names = list_ycb_objects()
    print(f"Found {len(names)} YCB objects")

    summary_fieldnames = [
        "object",
        "vol_vox_m3",
        "voxel_size",
        "enclosed_voxels",
        "blocked_voxels",
        "y_bottom",
        "y_top",
        "shape",
        "error",
    ]

    ts_fieldnames = [
        "object",
        "step",
        "t",
        "frame",
        "substep",
        "vol_vox",
        "enclosed_voxels",
        "blocked_voxels",
        "y_bottom",
        "y_top",
    ]

    device = None  # or "cuda:0"

    with open(out_summary, "w", newline="") as fs, open(out_ts, "w", newline="") as ft:
        ws = csv.DictWriter(fs, fieldnames=summary_fieldnames)
        wt = csv.DictWriter(ft, fieldnames=ts_fieldnames)
        ws.writeheader()
        wt.writeheader()

        for k, name in enumerate(names):
            print(f"[{k+1}/{len(names)}] {name}")

            # summary row default
            srow = {fn: "" for fn in summary_fieldnames}
            srow["object"] = name

            try:
                summary, timeseries = run_one_object(name, device=device)
                srow.update(summary)
                ws.writerow(srow)
                fs.flush()

                # append timeseries rows
                for r in timeseries:
                    trow = {fn: "" for fn in ts_fieldnames}
                    trow["object"] = name
                    # copy matching keys from logger rows
                    for key in ("step","t","frame","substep","vol_vox","enclosed_voxels","blocked_voxels","y_bottom","y_top"):
                        if key in r:
                            trow[key] = r[key]
                    wt.writerow(trow)
                ft.flush()

            except Exception as e:
                srow["error"] = f"{type(e).__name__}: {e}"
                ws.writerow(srow)
                fs.flush()
                # optional but safest:
                # os.fsync(fs.fileno())
                traceback.print_exc()


    print(f"Saved summary to: {out_summary}")
    print(f"Saved timeseries to: {out_ts}")


if __name__ == "__main__":
    main()
