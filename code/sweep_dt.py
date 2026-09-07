# sweep_dt.py
#
# sweep for time-step evaluation by varying the number of physics substeps.

# run like: python sweep_dt.py

# Saves to logs/sweep_dt_YYYYMMDD_HHMMSS/ :
# - summary.csv : one row per tested substep setting
# - volume_repXX_subYYY.csv : enclosed volume over time for each run
# - optional mp4 visualizations if MAKE_VIZ = True

import os
import gc
import math
import time
import csv
from datetime import datetime

import numpy as np
import warp as wp

from forward import FEMTendon
from init_pose import InitializeFingers
from quick_viz import quick_visualize


# ------------------------------------------------------------
# Configuration
# ------------------------------------------------------------
OBJECT_NAME = "acropora_cervicornis"
OBJECT_DENSITY = 2.0

FINGER_NUM = 6
NO_CLOTH = False

POSE_ITERS = 1000

FPS = 4000
NUM_FRAMES = 1000
SUBSTEPS_LIST = [5, 10, 20, 50, 100]

TENDON_FORCE = 100.0
REPEATS = 1

MAKE_VIZ = False
DEVICE = "cuda:0"

SUMMARY_CSV = "summary.csv"


# ------------------------------------------------------------
# Helpers
# ------------------------------------------------------------
def extract_volume_series(tendon):
    rows = getattr(tendon.vol_logger, "rows", [])
    if not rows:
        return np.array([]), np.array([])

    t = np.array([row["t"] for row in rows], dtype=float)
    v = np.array([row["vol_vox"] for row in rows], dtype=float)

    valid = np.isfinite(t) & np.isfinite(v)
    t = t[valid]
    v = v[valid]

    if len(t) == 0:
        return np.array([]), np.array([])

    order = np.argsort(t)
    return t[order], v[order]


def save_volume_series_csv(path, t, v):
    with open(path, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["time_s", "volume"])
        for ti, vi in zip(t, v):
            writer.writerow([f"{ti:.8f}", f"{vi:.8f}"])


def run_pose_initialization(object_name, finger_num, pose_iters, device, no_cloth):
    finger_len = 11
    finger_rot = np.pi / 30
    finger_width = 0.08
    scale = 5.0
    object_rot = wp.quat_rpy(-math.pi / 2, 0.0, 0.0)

    with wp.ScopedDevice(device):
        init_finger = InitializeFingers(
            stage_path="dt_sweep_init.usd",
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
            consider_cloth=not no_cloth,
        )

        finger_transform, _ = init_finger.get_initial_position()
        init_finger.capture_proxy_points_frozen()

    if finger_transform is None:
        raise RuntimeError("Pose initialization failed.")

    return {
        "finger_len": finger_len,
        "finger_rot": finger_rot,
        "finger_width": finger_width,
        "scale": scale,
        "object_rot": object_rot,
        "finger_transform": finger_transform,
        "init_finger": init_finger,
    }


def build_tendon(scene, object_name, object_density, finger_num, kernel_seed, device, no_cloth):
    with wp.ScopedDevice(device):
        tendon = FEMTendon(
            stage_path=None,
            num_frames=1,
            verbose=False,
            save_log=False,
            is_render=False,
            use_graph=False,
            kernel_seed=kernel_seed,
            train_iters=POSE_ITERS,
            object_rot=scene["object_rot"],
            object_density=object_density,
            ycb_object_name=object_name,
            finger_len=scene["finger_len"],
            finger_rot=scene["finger_rot"],
            finger_width=scene["finger_width"],
            scale=scene["scale"],
            finger_transform=scene["finger_transform"],
            finger_num=finger_num,
            requires_grad=False,
            init_finger=scene["init_finger"],
            no_cloth=no_cloth,
        )

        if getattr(scene["init_finger"], "proxy_pts_frozen", None) is not None:
            tendon.proxy_pts_frozen = scene["init_finger"].proxy_pts_frozen

        # Disable the automatic csv write inside forward.py
        tendon.vol_logger.to_csv = lambda path: None

        tendon.tendon_forces = wp.array(
            [TENDON_FORCE] * finger_num,
            dtype=wp.float32,
            requires_grad=False,
        )

    return tendon


def reset_time_stepping(tendon, fps, substeps, num_frames):
    tendon.frame_dt = 1.0 / float(fps)
    tendon.num_frames = int(num_frames)

    tendon.sim_substeps = int(substeps)
    tendon.sim_dt = tendon.frame_dt / float(tendon.sim_substeps)

    tendon.sim_time = 0.0
    tendon.render_time = 0.0

    tendon.states = []
    n_states = tendon.num_frames * tendon.sim_substeps + 1
    for _ in range(n_states):
        tendon.states.append(tendon.model.state(requires_grad=False))

    tendon.last_voxel_volume = None
    tendon.last_voxel_debug = None
    tendon._last_voxel_q = None
    tendon._vox_calibrated = False
    tendon.vol_logger.rows = []


# ------------------------------------------------------------
# Main
# ------------------------------------------------------------
def main():
    run_name = datetime.now().strftime("sweep_dt_%Y%m%d_%H%M%S")
    out_dir = os.path.join("logs", run_name)
    os.makedirs(out_dir, exist_ok=True)

    summary_path = os.path.join(out_dir, SUMMARY_CSV)

    scene = run_pose_initialization(
        object_name=OBJECT_NAME,
        finger_num=FINGER_NUM,
        pose_iters=POSE_ITERS,
        device=DEVICE,
        no_cloth=NO_CLOTH,
    )

    with open(summary_path, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow([
            "object",
            "finger_num",
            "rep",
            "fps",
            "substeps",
            "dt",
            "rollout_time_s",
            "runtime_s",
            "runtime_per_step_ms",
            "initial_volume",
            "final_volume",
            "min_volume",
            "reduction_percent",
            "timeseries_csv",
            "status",
            "error",
        ])

        with wp.ScopedDevice(DEVICE):
            for rep in range(1, REPEATS + 1):
                kernel_seed = 12345 + rep

                for substeps in SUBSTEPS_LIST:
                    tendon = None
                    error_msg = ""

                    try:
                        print(f"Running rep={rep}, substeps={substeps}")

                        tendon = build_tendon(
                            scene=scene,
                            object_name=OBJECT_NAME,
                            object_density=OBJECT_DENSITY,
                            finger_num=FINGER_NUM,
                            kernel_seed=kernel_seed,
                            device=DEVICE,
                            no_cloth=NO_CLOTH,
                        )

                        reset_time_stepping(
                            tendon=tendon,
                            fps=FPS,
                            substeps=substeps,
                            num_frames=NUM_FRAMES,
                        )

                        t0 = time.perf_counter()
                        tendon.forward()
                        runtime_s = time.perf_counter() - t0

                        wp.synchronize()

                        t, v = extract_volume_series(tendon)

                        initial_volume = tendon.init_voxel_volume
                        final_volume = tendon.last_voxel_volume
                        min_volume = float(np.min(v)) if len(v) > 0 else np.nan

                        if np.isfinite(initial_volume) and initial_volume > 0 and np.isfinite(final_volume):
                            reduction_percent = 100.0 * (initial_volume - final_volume) / initial_volume
                        else:
                            reduction_percent = np.nan

                        total_steps = tendon.num_frames * tendon.sim_substeps
                        runtime_per_step_ms = 1000.0 * runtime_s / total_steps
                        rollout_time_s = tendon.num_frames / FPS

                        ts_name = f"volume_rep{rep:02d}_sub{substeps:03d}.csv"
                        ts_path = os.path.join(out_dir, ts_name)
                        save_volume_series_csv(ts_path, t, v)

                        if MAKE_VIZ and rep == 1:
                            viz_path = os.path.join(out_dir, f"viz_sub{substeps:03d}.mp4")
                            quick_visualize(
                                tendon,
                                stride=50,
                                interval=30,
                                save_path=viz_path,
                                elev=30,
                                azim=45,
                            )

                        writer.writerow([
                            OBJECT_NAME,
                            FINGER_NUM,
                            rep,
                            FPS,
                            substeps,
                            f"{tendon.sim_dt:.8e}",
                            f"{rollout_time_s:.6f}",
                            f"{runtime_s:.6f}",
                            f"{runtime_per_step_ms:.6f}",
                            f"{initial_volume:.8f}" if np.isfinite(initial_volume) else "",
                            f"{final_volume:.8f}" if np.isfinite(final_volume) else "",
                            f"{min_volume:.8f}" if np.isfinite(min_volume) else "",
                            f"{reduction_percent:.4f}" if np.isfinite(reduction_percent) else "",
                            ts_name,
                            "ok",
                            "",
                        ])

                        print(
                            f"  dt={tendon.sim_dt:.2e}, "
                            f"V0={initial_volume:.6f}, "
                            f"Vend={final_volume:.6f}, "
                            f"runtime={runtime_s:.2f}s"
                        )

                    except Exception as e:
                        error_msg = f"{type(e).__name__}: {e}"
                        print("  failed:", error_msg)

                        writer.writerow([
                            OBJECT_NAME,
                            FINGER_NUM,
                            rep,
                            FPS,
                            substeps,
                            "",
                            f"{NUM_FRAMES / FPS:.6f}",
                            "",
                            "",
                            "",
                            "",
                            "",
                            "",
                            "",
                            "fail",
                            error_msg,
                        ])

                    finally:
                        if tendon is not None:
                            del tendon
                        wp.synchronize()
                        gc.collect()

    print(f"Saved results to {out_dir}")
    print(f"Saved summary to {summary_path}")


if __name__ == "__main__":
    main()