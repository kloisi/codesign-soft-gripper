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
from init_pose import InitializeFingers

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


def run_one_object(
    name,
    finger_num,
    device=None,
    pose_iters=1000,
    optimizer="lbfgs",
    do_force_opt=True,
    opt_frames=100,
    disable_per_object_forward_csv=True,
):
    # match forward.py settings
    finger_len = 11
    finger_rot = np.pi / 30
    finger_width = 0.08
    scale = 5.0
    object_rot = wp.quat_rpy(-math.pi / 2, 0.0, 0.0)

    no_cloth = False
    consider_cloth = not no_cloth

    # init pose optimisation settings (match forward.py)
    init_pose_num_frames = 30
    init_pose_stop_margin = 0.0005

    with wp.ScopedDevice(device):
        # --------------------------------------------------
        # 1) Initialise fingers like forward.py
        # --------------------------------------------------
        finger_transform = None
        init_finger = None

        try:
            init_finger = InitializeFingers(
                stage_path="init_pose_sweep_tmp.usd",
                finger_len=finger_len,
                finger_rot=finger_rot,
                finger_width=finger_width,
                stop_margin=init_pose_stop_margin,
                num_frames=init_pose_num_frames,
                iterations=pose_iters,          # <- match forward.py default
                scale=scale,
                num_envs=1,
                ycb_object_name=name,
                object_rot=object_rot,
                is_render=False,
                verbose=False,
                is_triangle=False,
                finger_num=finger_num,
                add_random=False,
                consider_cloth=consider_cloth,
            )

            finger_transform, _ = init_finger.get_initial_position()

            # same extra step as forward.py (doesn't affect physics, but keep parity)
            init_finger.capture_proxy_points_frozen()

            if finger_transform is None:
                print(f"[WARN] init_pose failed for {name}, using default transforms (finger_transform=None).")
                finger_transform = None
                init_finger = None

        except Exception as e:
            print(f"[WARN] init_pose crashed for {name}: {type(e).__name__}: {e}")
            finger_transform = None
            init_finger = None

        # --------------------------------------------------
        # 2) Build FEMTendon like forward.py
        # --------------------------------------------------
        tendon = FEMTendon(
            stage_path=None,               # no USD output in sweep
            num_frames=1000,
            verbose=False,
            save_log=False,
            is_render=False,
            use_graph=False,
            kernel_seed=np.random.randint(0, 10000),  # closer to forward.py than fixed 123
            train_iters=pose_iters,        # match forward.py
            object_rot=object_rot,
            object_density=2.0,            # match forward.py default CLI
            ycb_object_name=name,
            finger_len=finger_len,
            finger_rot=finger_rot,
            finger_width=finger_width,
            scale=scale,
            finger_transform=finger_transform,
            finger_num=finger_num,
            requires_grad=True,            # match forward.py
            init_finger=init_finger,       # match forward.py
            no_cloth=no_cloth,
        )

        # same proxy pts attachment as forward.py (no physics effect)
        if init_finger is not None and getattr(init_finger, "proxy_pts_frozen", None) is not None:
            tendon.proxy_pts_frozen = init_finger.proxy_pts_frozen

        # IMPORTANT: forward.py writes one CSV; sweep would write thousands.
        # This disables that internal per-object CSV write without changing simulation.
        if disable_per_object_forward_csv:
            tendon.vol_logger.to_csv = lambda _path: None

        # --------------------------------------------------
        # 3) Force optimisation like forward.py (default LBFGS)
        # --------------------------------------------------
        if do_force_opt:
            if optimizer == "sgd":
                tendon.optimize_forces(iterations=10, learning_rate=0.1, opt_frames=opt_frames)
            elif optimizer == "adam":
                tendon.optimize_forces_adam(iterations=20, learning_rate=10.0, opt_frames=opt_frames)
            elif optimizer == "autodiff":
                tendon.optimize_forces_autodiff(iterations=10, learning_rate=0.01, opt_frames=opt_frames)
            elif optimizer == "lbfgs":
                tendon.optimize_forces_lbfgs(iterations=1, learning_rate=1.0, opt_frames=opt_frames)
            else:
                raise ValueError(f"Unknown optimizer: {optimizer}")

        # --------------------------------------------------
        # 4) Run the actual sim + volume logging
        # --------------------------------------------------
        tendon.forward()

        vol = tendon.last_voxel_volume
        dbg = tendon.last_voxel_debug or {}
        timeseries = tendon.vol_logger.rows

        summary = {
            "object": name,
            "finger_num": int(finger_num),
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
    finger_num = 6

    # match usual forward.py defaults
    pose_iters = 1000
    optimizer = "lbfgs"
    do_force_opt = True
    opt_frames = 100

    # prevent writing logs/volume_timeseries_{obj}_f{finger_num}.csv for every object
    disable_per_object_forward_csv = True


    out_summary = f"logs/ycb_voxel_volume_sweep_f{finger_num}.csv"
    out_ts = f"logs/ycb_voxel_volume_timeseries_all_f{finger_num}.csv"
    os.makedirs("logs", exist_ok=True)

    names = list_ycb_objects()
    print(f"Found {len(names)} YCB objects")

    summary_fieldnames = [
        "object",
        "finger_num",
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
        "finger_num",
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

            srow = {fn: "" for fn in summary_fieldnames}
            srow["object"] = name
            srow["finger_num"] = int(finger_num)

            try:
                summary, timeseries = run_one_object(
                    name,
                    finger_num=finger_num,
                    device=device,
                    pose_iters=pose_iters,
                    optimizer=optimizer,
                    do_force_opt=do_force_opt,
                    opt_frames=opt_frames,
                    disable_per_object_forward_csv=disable_per_object_forward_csv,
                )

                srow.update(summary)
                ws.writerow(srow)
                fs.flush()

                for r in timeseries:
                    trow = {fn: "" for fn in ts_fieldnames}
                    trow["object"] = name
                    trow["finger_num"] = int(finger_num)
                    for key in ("step","t","frame","substep","vol_vox","enclosed_voxels","blocked_voxels","y_bottom","y_top"):
                        if key in r:
                            trow[key] = r[key]
                    wt.writerow(trow)
                ft.flush()

            except Exception as e:
                srow["error"] = f"{type(e).__name__}: {e}"
                ws.writerow(srow)
                fs.flush()
                traceback.print_exc()

    print(f"Saved summary to: {out_summary}")
    print(f"Saved timeseries to: {out_ts}")


if __name__ == "__main__":
    main()
