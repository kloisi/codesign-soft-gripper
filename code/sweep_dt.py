# sweep_dt.py
#
# DT sweep that matches your "usual forward.py pipeline":
#   1) InitializeFingers pose optimisation (once, like forward.py)
#   2) Build FEMTendon from forward.py (connecting cloth etc)
#   3) (Optional) Force optimisation (LBFGS/Adam/SGD/Autodiff like forward.py)
#   4) Run tendon.forward() and log metrics
#
# Key detail:
# Your forward.FEMTendon hardcodes fps=4000 and sim_substeps=100 in __init__.
# For a dt sweep we therefore "reconfigure" the instance after construction:
#   - set frame_dt, sim_substeps, sim_dt, num_frames
#   - reallocate states to size (num_frames*sim_substeps + 1)
#   - reset voxel logging flags
#
# Recommended default for final-system dt sweep:
#   - keep fps fixed at 4000 (matches forward.py and avoids frame-scheduling artefacts)
#   - sweep sim_substeps only (dt changes cleanly)
#
# If you also sweep fps, be aware that control.update_target_vel(frame) is frame-indexed,
# so changing fps changes the physical time when your scripted control happens,
# unless update_target_vel internally compensates.

import os
import csv
import time
import math
import numpy as np
import warp as wp

from forward import FEMTendon
from init_pose import InitializeFingers
from quick_viz import quick_visualize


# -------------------------
# Small helpers
# -------------------------
def _safe_np(x):
    if hasattr(x, "numpy"):
        return x.numpy()
    return np.asarray(x)


def _as_N3(arr):
    """Accept (N,3) or (1,N,3) or (N*3,) and return (N,3)."""
    a = _safe_np(arr)
    if a.ndim == 3:
        a = a[0]
    if a.ndim == 1:
        a = a.reshape(-1, 3)
    return a.astype(np.float64, copy=False)


def _first_particle_xyz(state_particle_q):
    q = _as_N3(state_particle_q)
    if q.shape[0] == 0:
        return np.array([np.nan, np.nan, np.nan], dtype=np.float64)
    return q[0].copy()


def _extract_q_end(tendon):
    return _as_N3(tendon.states[-1].particle_q)


def _extract_max_speed(tendon):
    st = tendon.states[-1]
    if not hasattr(st, "particle_qd"):
        return np.nan
    qd = _as_N3(st.particle_qd)
    if qd.size == 0:
        return np.nan
    return float(np.nanmax(np.linalg.norm(qd, axis=1)))


def _extract_vol_ts(tendon):
    rows = getattr(getattr(tendon, "vol_logger", None), "rows", None)
    if rows is None or len(rows) == 0:
        return None, None

    t = np.array([r.get("t", np.nan) for r in rows], dtype=float)
    v = np.array([r.get("vol_vox", np.nan) for r in rows], dtype=float)

    good = np.isfinite(t) & np.isfinite(v)
    t = t[good]
    v = v[good]
    if t.size < 2:
        return None, None

    order = np.argsort(t)
    t = t[order]
    v = v[order]

    # remove duplicate t for interp
    t_unique, idx = np.unique(t, return_index=True)
    v_unique = v[idx]
    return t_unique, v_unique


def _traj_rmse(t_ref, v_ref, t_run, v_run, T_target, n=200):
    if t_ref is None or t_run is None:
        return np.nan
    grid = np.linspace(0.0, float(T_target), int(n))
    v_ref_i = np.interp(grid, t_ref, v_ref)
    v_run_i = np.interp(grid, t_run, v_run)
    return float(np.sqrt(np.mean((v_run_i - v_ref_i) ** 2)))


# -------------------------
# Forward-like setup
# -------------------------
def run_init_pose_forward_like(
    object_name: str,
    finger_num: int,
    pose_iters: int,
    device: str | None,
    no_cloth: bool,
):
    # forward.py constants
    finger_len = 11
    finger_rot = np.pi / 30
    finger_width = 0.08
    scale = 5.0
    object_rot = wp.quat_rpy(-math.pi / 2, 0.0, 0.0)

    consider_cloth = not no_cloth

    init_pose_num_frames = 30
    init_pose_stop_margin = 0.0005

    with wp.ScopedDevice(device):
        init_finger = InitializeFingers(
            stage_path="init_pose_sweep_tmp.usd",
            finger_len=finger_len,
            finger_rot=finger_rot,
            finger_width=finger_width,
            stop_margin=init_pose_stop_margin,
            num_frames=init_pose_num_frames,
            iterations=pose_iters,
            scale=scale,
            num_envs=1,
            ycb_object_name=object_name,
            object_rot=object_rot,
            is_render=False,
            verbose=False,
            is_triangle=False,          # match forward.py
            finger_num=finger_num,
            add_random=False,
            consider_cloth=consider_cloth,
        )

        finger_transform, _ = init_finger.get_initial_position()
        if init_finger is not None:
            init_finger.capture_proxy_points_frozen()

        if finger_transform is None:
            print("[WARN] InitializeFingers returned None. Falling back to default transforms.")
            init_finger = None

    scene = dict(
        finger_len=finger_len,
        finger_rot=finger_rot,
        finger_width=finger_width,
        scale=scale,
        object_rot=object_rot,
        finger_transform=finger_transform,
        init_finger=init_finger,
    )
    return scene


def reconfigure_time_stepping(tendon: FEMTendon, fps: int, sim_substeps: int, num_frames: int):
    """
    Patch a forward.FEMTendon instance to use custom fps/substeps/frames.

    IMPORTANT: We rebuild tendon.states because forward.forward() indexing depends on
    sim_substeps*num_frames + 1.
    """
    tendon.frame_dt = 1.0 / float(fps)
    tendon.num_frames = int(num_frames)

    tendon.sim_substeps = int(sim_substeps)
    tendon.sim_dt = tendon.frame_dt / float(tendon.sim_substeps)

    tendon.sim_time = 0.0
    tendon.render_time = 0.0

    # rebuild states (this is the critical bit)
    tendon.states = []
    n_states = tendon.sim_substeps * tendon.num_frames + 1
    for _ in range(n_states):
        tendon.states.append(tendon.model.state(requires_grad=tendon.requires_grad))

    # refresh initial references used by your sanity checks
    tendon.init_particle_q = _first_particle_xyz(tendon.states[0].particle_q)
    if getattr(tendon, "has_object", False):
        tendon.init_body_q = _safe_np(tendon.states[0].body_q)[0, :].copy()
        tendon.object_body_f = tendon.states[0].body_f
        tendon.object_q = tendon.states[0].body_q
    else:
        tendon.init_body_q = None
        tendon.object_body_f = None
        tendon.object_q = None

    # reset voxel bookkeeping for this run
    tendon.last_voxel_volume = None
    tendon.last_voxel_debug = None
    tendon._last_voxel_q = None
    tendon._vox_calibrated = False

    # reset volume logger rows
    if hasattr(tendon, "vol_logger") and tendon.vol_logger is not None:
        tendon.vol_logger.rows = []


def build_tendon_forward_like(
    object_name: str,
    object_density: float,
    finger_num: int,
    scene: dict,
    device: str | None,
    kernel_seed: int,
    no_cloth: bool,
):
    # Keep construction cheap: we will reconfigure time stepping afterwards.
    # num_frames here only affects some paths and the initial allocation we overwrite.
    cheap_num_frames = 1

    with wp.ScopedDevice(device):
        tendon = FEMTendon(
            stage_path=None,
            num_frames=cheap_num_frames,
            verbose=False,
            save_log=False,
            is_render=False,
            use_graph=False,
            kernel_seed=kernel_seed,
            train_iters=scene.get("pose_iters", 1000),
            object_rot=scene["object_rot"],
            object_density=object_density,
            ycb_object_name=object_name,
            finger_len=scene["finger_len"],
            finger_rot=scene["finger_rot"],
            finger_width=scene["finger_width"],
            scale=scene["scale"],
            finger_transform=scene["finger_transform"],
            finger_num=finger_num,
            requires_grad=True,
            init_finger=scene["init_finger"],
            no_cloth=no_cloth,
        )

        # attach proxy pts for viz like forward.py
        init_finger = scene.get("init_finger", None)
        if init_finger is not None and getattr(init_finger, "proxy_pts_frozen", None) is not None:
            tendon.proxy_pts_frozen = init_finger.proxy_pts_frozen

        # disable per-run CSV writes inside forward.forward()
        # (keeps your sweep clean and avoids overwriting the same file many times)
        if hasattr(tendon, "vol_logger") and tendon.vol_logger is not None:
            tendon.vol_logger.to_csv = lambda _path: None

    return tendon


def run_force_optimisation_forward_like(
    tendon: FEMTendon,
    optimizer: str,
    opt_frames: int,
    do_force_opt: bool,
):
    if not do_force_opt:
        return dict(loss0=np.nan, lossT=np.nan, forces_mean=np.nan, forces_max=np.nan)

    opt_frames = int(max(1, min(opt_frames, tendon.num_frames)))

    history = None
    if optimizer == "sgd":
        history = tendon.optimize_forces(iterations=10, learning_rate=0.1, opt_frames=opt_frames)
    elif optimizer == "adam":
        history = tendon.optimize_forces_adam(iterations=20, learning_rate=10.0, opt_frames=opt_frames)
    elif optimizer == "autodiff":
        history = tendon.optimize_forces_autodiff(iterations=10, learning_rate=0.01, opt_frames=opt_frames)
    elif optimizer == "lbfgs":
        history = tendon.optimize_forces_lbfgs(iterations=1, learning_rate=1.0, opt_frames=opt_frames)
    else:
        raise ValueError(f"Unknown optimizer: {optimizer}")

    # pull simple stats
    forces = _safe_np(tendon.tendon_forces).astype(float).ravel()
    forces_mean = float(np.mean(forces)) if forces.size else np.nan
    forces_max = float(np.max(forces)) if forces.size else np.nan

    # pull losses if available
    loss0 = np.nan
    lossT = np.nan
    if isinstance(history, dict) and "loss" in history and len(history["loss"]) > 0:
        loss0 = float(history["loss"][0])
        lossT = float(history["loss"][-1])

    return dict(loss0=loss0, lossT=lossT, forces_mean=forces_mean, forces_max=forces_max)


# -------------------------
# Main sweep
# -------------------------
def run_dt_sweep(
    device="cuda:0",
    log_filename="dt_sweep_results_forwardlike.csv",
    make_viz=True,
):
    # -------------------------
    # Sweep configuration
    # -------------------------
    object_name = "006_mustard_bottle"
    object_density = 2.0
    finger_num = 6
    no_cloth = False

    # Pose optimisation settings (matches your forward.py defaults)
    pose_iters = 1000

    # Force optimisation settings (matches forward.py behaviour)
    do_force_opt = True
    optimizer = "lbfgs"
    opt_frames = 100

    # Match forward.py default horizon:
    BASE_FPS = 4000
    NUM_FRAMES = 1000                 # forward.py default --num_frames 1000
    T_target = NUM_FRAMES / BASE_FPS  # 0.25s, only used for RMSE grid + logging

    fps_list = [BASE_FPS]             # keep fixed like forward.py
    sim_substeps_list = [10, 50, 100] # sweep dt by substeps

    # Reference run: finest dt among substeps
    fps_ref = BASE_FPS
    sub_ref = max(sim_substeps_list)

    # repeats per setting (use >1 if you want to detect stochasticity / rare blowups)
    repeats = 3

    # Deterministic-ish seeds (GPU may still be nondeterministic)
    np.random.seed(0)
    kernel_seed_base = 12345

    # -------------------------
    # Logging
    # -------------------------
    curr_dir = os.path.dirname(os.path.abspath(__file__))
    log_path = os.path.join(curr_dir, log_filename)

    file_exists = os.path.exists(log_path)
    os.makedirs(os.path.dirname(log_path) or ".", exist_ok=True)

    log_file = open(log_path, "a", newline="")
    writer = csv.writer(log_file)

    if not file_exists:
        writer.writerow([
            "timestamp",
            "object",
            "finger_num",
            "fps",
            "sim_substeps",
            "rep",
            "dt",
            "num_frames",
            "total_steps",

            "t_pose_s",
            "t_forceopt_s",
            "t_forward_s",
            "real_time_per_step_ms",

            "num_ok",
            "phys_ok",
            "diff_norm",
            "max_abs_pos",
            "max_speed",

            "forceopt_loss0",
            "forceopt_lossT",
            "forces_mean",
            "forces_max",

            "vol_final",
            "vol_final_ref",
            "vol_abs_err",
            "vol_rel_err",
            "vol_traj_rmse",
            "q_final_rms_err",

            "kernel_seed",
        ])

    viz_dir = os.path.join(curr_dir, "dt_sweep_viz_forwardlike")
    if make_viz:
        os.makedirs(viz_dir, exist_ok=True)

    # -------------------------
    # 1) Pose optimisation once (like you usually run forward.py)
    # -------------------------
    t_pose0 = time.perf_counter()
    scene = run_init_pose_forward_like(
        object_name=object_name,
        finger_num=finger_num,
        pose_iters=pose_iters,
        device=device,
        no_cloth=no_cloth,
    )
    t_pose = time.perf_counter() - t_pose0
    scene["pose_iters"] = pose_iters

    # -------------------------
    # 2) Reference run
    # -------------------------
    num_frames_ref = NUM_FRAMES


    print(f"\n[dt_sweep] Reference run fps={fps_ref} sub={sub_ref} frames={num_frames_ref}")

    kernel_seed_ref = kernel_seed_base + 999

    tendon_ref = build_tendon_forward_like(
        object_name=object_name,
        object_density=object_density,
        finger_num=finger_num,
        scene=scene,
        device=device,
        kernel_seed=kernel_seed_ref,
        no_cloth=no_cloth,
    )
    reconfigure_time_stepping(tendon_ref, fps=fps_ref, sim_substeps=sub_ref, num_frames=num_frames_ref)

    t_force0 = time.perf_counter()
    opt_stats_ref = run_force_optimisation_forward_like(
        tendon_ref,
        optimizer=optimizer,
        opt_frames=opt_frames,
        do_force_opt=do_force_opt,
    )
    t_force_ref = time.perf_counter() - t_force0

    t_fwd0 = time.perf_counter()
    tendon_ref.forward()
    t_fwd_ref = time.perf_counter() - t_fwd0

    q_ref = _extract_q_end(tendon_ref)
    t_ref, v_ref = _extract_vol_ts(tendon_ref)

    vol_final_ref = getattr(tendon_ref, "last_voxel_volume", np.nan)
    if not np.isfinite(vol_final_ref) and v_ref is not None and v_ref.size:
        vol_final_ref = float(v_ref[-1])

    print(f"[dt_sweep] Reference done. V_ref={vol_final_ref:.6g}  dt_ref={tendon_ref.sim_dt:.2e}")

    # optional: video for reference
    if make_viz:
        save_path = os.path.join(viz_dir, f"viz_REF_fps{fps_ref}_sub{sub_ref}.mp4")
        print("  making viz:", save_path)
        quick_visualize(
            tendon_ref,
            stride=50,
            interval=30,
            save_path=save_path,
            elev=30,
            azim=45,
        )

    # -------------------------
    # 3) Sweep
    # -------------------------
    with wp.ScopedDevice(device):
        for fps in fps_list:
            num_frames = NUM_FRAMES

            for sub in sim_substeps_list:
                for rep in range(1, repeats + 1):
                    print(f"\n[dt_sweep] fps={fps} sub={sub} rep={rep}")

                    # make seed deterministic per setting
                    kernel_seed = kernel_seed_base + (fps * 100000) + (sub * 1000) + rep

                    tendon = build_tendon_forward_like(
                        object_name=object_name,
                        object_density=object_density,
                        finger_num=finger_num,
                        scene=scene,
                        device=device,
                        kernel_seed=kernel_seed,
                        no_cloth=no_cloth,
                    )
                    reconfigure_time_stepping(tendon, fps=fps, sim_substeps=sub, num_frames=num_frames)

                    # force optimisation
                    t_force0 = time.perf_counter()
                    opt_stats = run_force_optimisation_forward_like(
                        tendon,
                        optimizer=optimizer,
                        opt_frames=opt_frames,
                        do_force_opt=do_force_opt,
                    )
                    t_force = time.perf_counter() - t_force0

                    # forward sim
                    t_fwd0 = time.perf_counter()
                    tendon.forward()
                    t_fwd = time.perf_counter() - t_fwd0

                    total_steps = tendon.num_frames * tendon.sim_substeps
                    real_time_per_step = (t_fwd / max(1, total_steps))

                    # numeric checks
                    q_end = _extract_q_end(tendon)
                    max_abs_pos = float(np.nanmax(np.abs(q_end))) if q_end.size else np.nan
                    max_speed = _extract_max_speed(tendon)
                    num_ok = bool(np.isfinite(q_end).all() and (max_abs_pos < 2.0))

                    # physical sanity check (your diff_q style)
                    p_end = _first_particle_xyz(tendon.states[-1].particle_q)
                    p0 = np.asarray(tendon.init_particle_q, dtype=np.float64).reshape(3,)
                    particle_trans = p_end - p0

                    if getattr(tendon, "has_object", False) and tendon.object_q is not None and tendon.init_body_q is not None:
                        obj_xyz = _safe_np(tendon.object_q)[0, 0:3].astype(np.float64)
                        obj0_xyz = _safe_np(tendon.init_body_q)[0:3].astype(np.float64)
                        object_trans = obj_xyz - obj0_xyz
                        diff_q = object_trans - particle_trans
                        diff_norm = float(np.linalg.norm(diff_q))
                        phys_ok = bool(diff_norm < 0.5)
                    else:
                        diff_norm = np.nan
                        phys_ok = True

                    # volume errors
                    t_run, v_run = _extract_vol_ts(tendon)
                    vol_final = getattr(tendon, "last_voxel_volume", np.nan)
                    if not np.isfinite(vol_final) and v_run is not None and v_run.size:
                        vol_final = float(v_run[-1])

                    vol_abs_err = float(vol_final - vol_final_ref) if (np.isfinite(vol_final) and np.isfinite(vol_final_ref)) else np.nan
                    vol_rel_err = float(vol_abs_err / vol_final_ref) if (np.isfinite(vol_abs_err) and abs(vol_final_ref) > 1e-12) else np.nan
                    vol_traj_rmse = _traj_rmse(t_ref, v_ref, t_run, v_run, T_target)

                    # final state error vs reference (requires same particle count)
                    if q_end.shape == q_ref.shape:
                        q_final_rms_err = float(np.sqrt(np.mean((q_end - q_ref) ** 2)))
                    else:
                        q_final_rms_err = np.nan

                    print(
                        f"[dt_sweep] dt={tendon.sim_dt:.2e} steps={total_steps} "
                        f"t/step={real_time_per_step*1e3:.3f}ms "
                        f"num_ok={num_ok} phys_ok={phys_ok} "
                        f"V={vol_final:.6g} Verr_rel={vol_rel_err:.3e} traj_rmse={vol_traj_rmse:.3e} q_rms={q_final_rms_err:.3e}"
                    )

                    writer.writerow([
                        time.strftime("%Y-%m-%d %H:%M:%S"),
                        object_name,
                        finger_num,
                        fps,
                        sub,
                        rep,
                        f"{tendon.sim_dt:.6e}",
                        tendon.num_frames,
                        total_steps,

                        f"{t_pose:.6f}",         # pose optimisation time (same for all runs)
                        f"{t_force:.6f}",        # force optimisation time for this run
                        f"{t_fwd:.6f}",          # forward sim time
                        f"{real_time_per_step*1e3:.6f}",

                        int(num_ok),
                        int(phys_ok),
                        f"{diff_norm:.6g}" if np.isfinite(diff_norm) else "",
                        f"{max_abs_pos:.6g}" if np.isfinite(max_abs_pos) else "",
                        f"{max_speed:.6g}" if np.isfinite(max_speed) else "",

                        f"{opt_stats.get('loss0', np.nan):.6g}" if np.isfinite(opt_stats.get("loss0", np.nan)) else "",
                        f"{opt_stats.get('lossT', np.nan):.6g}" if np.isfinite(opt_stats.get("lossT", np.nan)) else "",
                        f"{opt_stats.get('forces_mean', np.nan):.6g}" if np.isfinite(opt_stats.get("forces_mean", np.nan)) else "",
                        f"{opt_stats.get('forces_max', np.nan):.6g}" if np.isfinite(opt_stats.get("forces_max", np.nan)) else "",

                        f"{vol_final:.6g}" if np.isfinite(vol_final) else "",
                        f"{vol_final_ref:.6g}" if np.isfinite(vol_final_ref) else "",
                        f"{vol_abs_err:.6g}" if np.isfinite(vol_abs_err) else "",
                        f"{vol_rel_err:.6g}" if np.isfinite(vol_rel_err) else "",
                        f"{vol_traj_rmse:.6g}" if np.isfinite(vol_traj_rmse) else "",
                        f"{q_final_rms_err:.6g}" if np.isfinite(q_final_rms_err) else "",

                        kernel_seed,
                    ])
                    log_file.flush()

                    # Videos: only rep==1, or failures
                    if make_viz and (rep == 1 or (not num_ok) or (not phys_ok)):
                        vid_name = f"viz_fps{fps}_sub{sub}_rep{rep}.mp4"
                        save_path = os.path.join(viz_dir, vid_name)
                        print("  making viz:", save_path)
                        quick_visualize(
                            tendon,
                            stride=50,
                            interval=30,
                            save_path=save_path,
                            elev=30,
                            azim=45,
                        )

    log_file.close()
    print(f"\nLogged results to: {log_path}")


if __name__ == "__main__":
    run_dt_sweep(device="cuda:0", make_viz=True)
