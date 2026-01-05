# sweep_dt.py
#
# DT sweep that matches your "usual forward.py pipeline":
#   1) InitializeFingers pose optimisation (once)
#   2) Build FEMTendon from forward.py (cloth, object, fingers)
#   3) Optional force optimisation (usually OFF for dt sweep)
#   4) Run tendon.forward() and log a lot of data to disk
#
# Design goals for this version:
#   - No hardcoded "reference" run during the sweep
#   - Save enough per run to choose any reference later in plotting
#   - Save trajectory data you can use later (volume time series always)
#   - Save final full particle state so you can do q_end RMS comparisons later
#   - Save optional downsampled snapshots (subset of particles) for extra analysis
#   - Avoid OOM by running without gradients
#   - Avoid accumulating GPU memory by reusing one tendon per rep and rebuilding states each run
#
# Outputs:
#   - CSV summary: dt_sweep_results_forwardlike.csv
#   - NPZ per run: dt_sweep_data/run_rep{rep}_fps{fps}_sub{sub}_seed{seed}.npz
#   - Optional videos: dt_sweep_viz_forwardlike/*.mp4

import os
import csv
import time
import math
import json
import gc
import traceback
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


def _extract_qd_end(tendon):
    st = tendon.states[-1]
    if not hasattr(st, "particle_qd"):
        return None
    return _as_N3(st.particle_qd)


def _extract_max_speed(tendon):
    st = tendon.states[-1]
    if not hasattr(st, "particle_qd"):
        return np.nan
    qd = _as_N3(st.particle_qd)
    if qd.size == 0:
        return np.nan
    return float(np.nanmax(np.linalg.norm(qd, axis=1)))


def _extract_body_q_end(tendon):
    st = tendon.states[-1]
    if not hasattr(st, "body_q") or st.body_q is None:
        return None
    bq = _safe_np(st.body_q)
    return bq.copy()


def _extract_body_q0(tendon):
    st = tendon.states[0]
    if not hasattr(st, "body_q") or st.body_q is None:
        return None
    bq = _safe_np(st.body_q)
    return bq.copy()


def _extract_vol_ts(tendon):
    rows = getattr(getattr(tendon, "vol_logger", None), "rows", None)
    if rows is None or len(rows) == 0:
        return None, None

    t = np.array([r.get("t", np.nan) for r in rows], dtype=float)
    v = np.array([r.get("vol_vox", np.nan) for r in rows], dtype=float)

    good = np.isfinite(t) & np.isfinite(v)
    t = t[good]
    v = v[good]
    if t.size < 1:
        return None, None

    order = np.argsort(t)
    t = t[order]
    v = v[order]

    # remove duplicate t for interp
    t_unique, idx = np.unique(t, return_index=True)
    v_unique = v[idx]
    return t_unique, v_unique


def _evenly_spaced_indices(n: int, k: int):
    if n <= 0:
        return np.array([], dtype=np.int64)
    if k <= 0:
        return np.array([], dtype=np.int64)
    if k >= n:
        return np.arange(n, dtype=np.int64)
    idx = np.linspace(0, n - 1, k).astype(np.int64)
    idx = np.unique(idx)
    return idx


def _npz_name(rep, fps, sub, seed):
    return f"run_rep{rep:02d}_fps{fps}_sub{sub}_seed{seed}.npz"


def _now_timestamp():
    return time.strftime("%Y-%m-%d %H:%M:%S")


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
            is_triangle=False,
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


def build_tendon_forward_like(
    object_name: str,
    object_density: float,
    finger_num: int,
    scene: dict,
    device: str | None,
    kernel_seed: int,
    no_cloth: bool,
    init_force: float = 100.0,
    requires_grad: bool = False,
):
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
            requires_grad=requires_grad,
            init_finger=scene["init_finger"],
            no_cloth=no_cloth,
        )

        init_finger = scene.get("init_finger", None)
        if init_finger is not None and getattr(init_finger, "proxy_pts_frozen", None) is not None:
            tendon.proxy_pts_frozen = init_finger.proxy_pts_frozen

        if hasattr(tendon, "vol_logger") and tendon.vol_logger is not None:
            tendon.vol_logger.to_csv = lambda _path: None

        tendon.tendon_forces = wp.array(
            [float(init_force)] * int(tendon.finger_num),
            dtype=wp.float32,
            requires_grad=requires_grad,
        )

    return tendon


def reconfigure_time_stepping(tendon: FEMTendon, fps: int, sim_substeps: int, num_frames: int):
    tendon.frame_dt = 1.0 / float(fps)
    tendon.num_frames = int(num_frames)

    tendon.sim_substeps = int(sim_substeps)
    tendon.sim_dt = tendon.frame_dt / float(tendon.sim_substeps)

    tendon.sim_time = 0.0
    tendon.render_time = 0.0

    # rebuild states (critical)
    tendon.states = []
    n_states = tendon.sim_substeps * tendon.num_frames + 1
    for _ in range(n_states):
        tendon.states.append(tendon.model.state(requires_grad=tendon.requires_grad))

    tendon.init_particle_q = _first_particle_xyz(tendon.states[0].particle_q)

    if getattr(tendon, "has_object", False):
        tendon.init_body_q = _safe_np(tendon.states[0].body_q)[0, :].copy()
        tendon.object_body_f = tendon.states[0].body_f
        tendon.object_q = tendon.states[0].body_q
    else:
        tendon.init_body_q = None
        tendon.object_body_f = None
        tendon.object_q = None

    tendon.last_voxel_volume = None
    tendon.last_voxel_debug = None
    tendon._last_voxel_q = None
    tendon._vox_calibrated = False

    if hasattr(tendon, "vol_logger") and tendon.vol_logger is not None:
        tendon.vol_logger.rows = []


def run_force_optimisation_forward_like(
    tendon: FEMTendon,
    optimizer: str,
    opt_frames: int,
    do_force_opt: bool,
):
    forces = _safe_np(tendon.tendon_forces).astype(float).ravel()
    forces_mean = float(np.mean(forces)) if forces.size else np.nan
    forces_max = float(np.max(forces)) if forces.size else np.nan

    if not do_force_opt:
        return dict(loss0=np.nan, lossT=np.nan, forces_mean=forces_mean, forces_max=forces_max)

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

    loss0 = np.nan
    lossT = np.nan
    if isinstance(history, dict) and "loss" in history and len(history["loss"]) > 0:
        loss0 = float(history["loss"][0])
        lossT = float(history["loss"][-1])

    forces = _safe_np(tendon.tendon_forces).astype(float).ravel()
    forces_mean = float(np.mean(forces)) if forces.size else np.nan
    forces_max = float(np.max(forces)) if forces.size else np.nan

    return dict(loss0=loss0, lossT=lossT, forces_mean=forces_mean, forces_max=forces_max)


def save_run_npz(
    npz_path: str,
    meta: dict,
    stats: dict,
    vol_t: np.ndarray | None,
    vol_v: np.ndarray | None,
    q_end: np.ndarray | None,
    qd_end: np.ndarray | None,
    body_q0: np.ndarray | None,
    body_q_end: np.ndarray | None,
    snap_state_idx: np.ndarray | None,
    snap_time: np.ndarray | None,
    snap_particle_idx: np.ndarray | None,
    snap_q: np.ndarray | None,
    snap_qd: np.ndarray | None,
):
    # ensure numpy arrays (empty is fine)
    def _arr(x, dtype=None):
        if x is None:
            return np.array([], dtype=np.float64 if dtype is None else dtype)
        a = np.asarray(x)
        if dtype is not None:
            a = a.astype(dtype, copy=False)
        return a

    # store dicts as json strings inside npz
    meta_json = json.dumps(meta, indent=None, sort_keys=True)
    stats_json = json.dumps(stats, indent=None, sort_keys=True)

    np.savez_compressed(
        npz_path,
        meta_json=np.array(meta_json),
        stats_json=np.array(stats_json),
        vol_t=_arr(vol_t, np.float64),
        vol_v=_arr(vol_v, np.float64),
        q_end=_arr(q_end, np.float64),
        qd_end=_arr(qd_end, np.float64),
        body_q0=_arr(body_q0, np.float64),
        body_q_end=_arr(body_q_end, np.float64),
        snap_state_idx=_arr(snap_state_idx, np.int64),
        snap_time=_arr(snap_time, np.float64),
        snap_particle_idx=_arr(snap_particle_idx, np.int64),
        snap_q=_arr(snap_q, np.float64),
        snap_qd=_arr(snap_qd, np.float64),
    )


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
    object_name = "acropora_cervicornis"
    object_density = 2.0
    finger_num = 6
    no_cloth = False

    pose_iters = 1000

    do_force_opt = False
    optimizer = "lbfgs"
    opt_frames = 100

    INIT_FORCE = 100.0

    # IMPORTANT: keep gradients off to avoid OOM
    SWEEP_REQUIRES_GRAD = False

    BASE_FPS = 4000
    NUM_FRAMES = 1000

    fps_list = [BASE_FPS]
    sim_substeps_list = [5, 10, 20, 50, 100]   # you can add 200 if you want, but it increases memory

    repeats = 3
    np.random.seed(0)
    kernel_seed_base = 12345

    # How much extra trajectory state to save (optional but recommended)
    SAVE_SNAPSHOTS = True
    MAX_SNAPSHOTS = 120          # number of time snapshots taken from tendon.states
    MAX_PARTICLES_SAVED = 512    # subset of particles per snapshot, plus full q_end saved always
    SAVE_SNAPSHOT_QD = True

    # Video policy
    # Keeping videos for all runs will be huge and slow.
    # This saves for rep 1 only, and also if the run fails sanity checks.
    VIZ_REP1_ONLY = True

    # -------------------------
    # Output directories
    # -------------------------
    curr_dir = os.path.dirname(os.path.abspath(__file__))
    log_path = os.path.join(curr_dir, log_filename)

    data_dir = os.path.join(curr_dir, "dt_sweep_data")
    os.makedirs(data_dir, exist_ok=True)

    viz_dir = os.path.join(curr_dir, "dt_sweep_viz_forwardlike")
    if make_viz:
        os.makedirs(viz_dir, exist_ok=True)

    # write a sweep metadata file (useful later)
    sweep_meta_path = os.path.join(data_dir, "sweep_meta.json")
    with open(sweep_meta_path, "w") as f:
        json.dump(
            dict(
                timestamp=_now_timestamp(),
                object_name=object_name,
                object_density=object_density,
                finger_num=finger_num,
                no_cloth=no_cloth,
                pose_iters=pose_iters,
                do_force_opt=do_force_opt,
                optimizer=optimizer,
                opt_frames=opt_frames,
                init_force=INIT_FORCE,
                base_fps=BASE_FPS,
                num_frames=NUM_FRAMES,
                fps_list=fps_list,
                sim_substeps_list=sim_substeps_list,
                repeats=repeats,
                kernel_seed_base=kernel_seed_base,
                save_snapshots=SAVE_SNAPSHOTS,
                max_snapshots=MAX_SNAPSHOTS,
                max_particles_saved=MAX_PARTICLES_SAVED,
                save_snapshot_qd=SAVE_SNAPSHOT_QD,
            ),
            f,
            indent=2,
            sort_keys=True,
        )

    # -------------------------
    # CSV logging
    # -------------------------
    file_exists = os.path.exists(log_path)
    log_file = open(log_path, "a", newline="")
    writer = csv.writer(log_file)

    if not file_exists:
        writer.writerow([
            "timestamp",
            "status",
            "error",

            "object",
            "object_density",
            "finger_num",
            "no_cloth",

            "fps",
            "sim_substeps",
            "rep",
            "dt",
            "num_frames",
            "total_steps",
            "kernel_seed",

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
            "vol_n",

            "npz_path",
            "snap_n",
            "snap_particles_n",
        ])

    # -------------------------
    # 1) Pose optimisation once
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
    # 2) Sweep
    # -------------------------
    with wp.ScopedDevice(device):
        for rep in range(1, repeats + 1):
            print(f"\n[dt_sweep] ===== REP {rep}/{repeats} =====")

            # One seed per rep, shared across all substeps so only dt changes
            kernel_seed_common = kernel_seed_base + rep

            for fps in fps_list:
                num_frames = NUM_FRAMES
                T_target = num_frames / float(fps)

                for sub in sim_substeps_list:
                    print(f"\n[dt_sweep] fps={fps} sub={sub} rep={rep}")

                    status = "ok"
                    err_msg = ""
                    npz_path = ""
                    snap_n = 0
                    snap_particles_n = 0

                    tendon = None

                    try:
                        # Build a fresh tendon for THIS run (rep,fps,sub)
                        tendon = build_tendon_forward_like(
                            object_name=object_name,
                            object_density=object_density,
                            finger_num=finger_num,
                            scene=scene,
                            device=device,
                            kernel_seed=kernel_seed_common,
                            no_cloth=no_cloth,
                            init_force=INIT_FORCE,
                            requires_grad=SWEEP_REQUIRES_GRAD,
                        )

                        # reset and allocate states for this dt
                        reconfigure_time_stepping(tendon, fps=fps, sim_substeps=sub, num_frames=num_frames)

                        # force optimisation (usually off)
                        t_force0 = time.perf_counter()
                        opt_stats = run_force_optimisation_forward_like(
                            tendon,
                            optimizer=optimizer,
                            opt_frames=opt_frames,
                            do_force_opt=do_force_opt,
                        )
                        t_force = time.perf_counter() - t_force0

                        # forward simulation
                        t_fwd0 = time.perf_counter()
                        tendon.forward()
                        t_fwd = time.perf_counter() - t_fwd0

                        total_steps = tendon.num_frames * tendon.sim_substeps
                        real_time_per_step = (t_fwd / max(1, total_steps))

                        # numeric checks
                        q_end = _extract_q_end(tendon)
                        qd_end = _extract_qd_end(tendon)
                        max_abs_pos = float(np.nanmax(np.abs(q_end))) if q_end.size else np.nan
                        max_speed = _extract_max_speed(tendon)
                        num_ok = bool(np.isfinite(q_end).all() and (max_abs_pos < 2.0))

                        # physical sanity check (diff between object and particle motion)
                        p_end = _first_particle_xyz(tendon.states[-1].particle_q)
                        p0 = np.asarray(tendon.init_particle_q, dtype=np.float64).reshape(3,)
                        particle_trans = p_end - p0

                        body_q0 = _extract_body_q0(tendon)
                        body_q_end = _extract_body_q_end(tendon)

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

                        # volume time series and final volume
                        vol_t, vol_v = _extract_vol_ts(tendon)
                        vol_n = int(vol_t.size) if vol_t is not None else 0

                        vol_final = getattr(tendon, "last_voxel_volume", np.nan)
                        if (not np.isfinite(vol_final)) and (vol_v is not None) and vol_v.size:
                            vol_final = float(vol_v[-1])

                        # snapshots (optional)
                        snap_state_idx = None
                        snap_time = None
                        snap_particle_idx = None
                        snap_q = None
                        snap_qd = None

                        if SAVE_SNAPSHOTS:
                            n_states = len(tendon.states)
                            snap_state_idx = _evenly_spaced_indices(n_states, MAX_SNAPSHOTS)
                            snap_time = snap_state_idx.astype(np.float64) * float(tendon.sim_dt)

                            n_particles = int(q_end.shape[0]) if q_end is not None else 0
                            snap_particle_idx = _evenly_spaced_indices(n_particles, MAX_PARTICLES_SAVED)

                            snap_n = int(snap_state_idx.size)
                            snap_particles_n = int(snap_particle_idx.size)

                            if snap_n > 0 and snap_particles_n > 0:
                                snap_q = np.empty((snap_n, snap_particles_n, 3), dtype=np.float64)
                                if SAVE_SNAPSHOT_QD:
                                    snap_qd = np.empty((snap_n, snap_particles_n, 3), dtype=np.float64)

                                for i, si in enumerate(snap_state_idx):
                                    qi = _as_N3(tendon.states[int(si)].particle_q)
                                    snap_q[i, :, :] = qi[snap_particle_idx, :]

                                    if SAVE_SNAPSHOT_QD and hasattr(tendon.states[int(si)], "particle_qd"):
                                        qdi = _as_N3(tendon.states[int(si)].particle_qd)
                                        snap_qd[i, :, :] = qdi[snap_particle_idx, :]

                        # save npz
                        npz_name = _npz_name(rep=rep, fps=fps, sub=sub, seed=kernel_seed_common)
                        npz_path = os.path.join(data_dir, npz_name)

                        meta = dict(
                            timestamp=_now_timestamp(),
                            object_name=object_name,
                            object_density=float(object_density),
                            finger_num=int(finger_num),
                            no_cloth=bool(no_cloth),
                            pose_iters=int(pose_iters),

                            fps=int(fps),
                            sim_substeps=int(sub),
                            num_frames=int(num_frames),
                            frame_dt=float(tendon.frame_dt),
                            sim_dt=float(tendon.sim_dt),
                            T_target=float(T_target),
                            total_steps=int(total_steps),

                            rep=int(rep),
                            kernel_seed=int(kernel_seed_common),

                            init_force=float(INIT_FORCE),
                            do_force_opt=bool(do_force_opt),
                            optimizer=str(optimizer),
                            opt_frames=int(opt_frames),

                            save_snapshots=bool(SAVE_SNAPSHOTS),
                            max_snapshots=int(MAX_SNAPSHOTS),
                            max_particles_saved=int(MAX_PARTICLES_SAVED),
                            save_snapshot_qd=bool(SAVE_SNAPSHOT_QD),
                        )

                        stats = dict(
                            t_pose_s=float(t_pose),
                            t_forceopt_s=float(t_force),
                            t_forward_s=float(t_fwd),
                            real_time_per_step_ms=float(real_time_per_step * 1e3),

                            num_ok=bool(num_ok),
                            phys_ok=bool(phys_ok),
                            diff_norm=float(diff_norm) if np.isfinite(diff_norm) else None,
                            max_abs_pos=float(max_abs_pos) if np.isfinite(max_abs_pos) else None,
                            max_speed=float(max_speed) if np.isfinite(max_speed) else None,

                            forceopt_loss0=float(opt_stats.get("loss0", np.nan)) if np.isfinite(opt_stats.get("loss0", np.nan)) else None,
                            forceopt_lossT=float(opt_stats.get("lossT", np.nan)) if np.isfinite(opt_stats.get("lossT", np.nan)) else None,
                            forces_mean=float(opt_stats.get("forces_mean", np.nan)) if np.isfinite(opt_stats.get("forces_mean", np.nan)) else None,
                            forces_max=float(opt_stats.get("forces_max", np.nan)) if np.isfinite(opt_stats.get("forces_max", np.nan)) else None,

                            vol_final=float(vol_final) if np.isfinite(vol_final) else None,
                            vol_n=int(vol_n),
                        )

                        save_run_npz(
                            npz_path=npz_path,
                            meta=meta,
                            stats=stats,
                            vol_t=vol_t,
                            vol_v=vol_v,
                            q_end=q_end,
                            qd_end=qd_end,
                            body_q0=body_q0,
                            body_q_end=body_q_end,
                            snap_state_idx=snap_state_idx,
                            snap_time=snap_time,
                            snap_particle_idx=snap_particle_idx,
                            snap_q=snap_q,
                            snap_qd=snap_qd,
                        )

                        print(
                            f"[dt_sweep] dt={tendon.sim_dt:.2e} steps={total_steps} "
                            f"t/step={real_time_per_step*1e3:.3f}ms "
                            f"num_ok={num_ok} phys_ok={phys_ok} "
                            f"V={vol_final:.6g} vol_n={vol_n} "
                            f"npz={os.path.basename(npz_path)}"
                        )

                        # video policy
                        make_this_viz = False
                        if make_viz:
                            if (not num_ok) or (not phys_ok):
                                make_this_viz = True
                            elif (not VIZ_REP1_ONLY) or (rep == 1):
                                make_this_viz = True

                        if make_viz and make_this_viz:
                            vid_name = f"viz_rep{rep}_fps{fps}_sub{sub}.mp4"
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

                        # write CSV
                        writer.writerow([
                            _now_timestamp(),
                            "ok",
                            "",

                            object_name,
                            f"{object_density:.6g}",
                            finger_num,
                            int(no_cloth),

                            fps,
                            sub,
                            rep,
                            f"{tendon.sim_dt:.6e}",
                            tendon.num_frames,
                            total_steps,
                            kernel_seed_common,

                            f"{t_pose:.6f}",
                            f"{t_force:.6f}",
                            f"{t_fwd:.6f}",
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
                            vol_n,

                            npz_path,
                            snap_n,
                            snap_particles_n,
                        ])
                        log_file.flush()

                    except Exception as e:
                        status = "fail"
                        err_msg = f"{type(e).__name__}: {e}"
                        print("[dt_sweep][ERROR]", err_msg)
                        traceback.print_exc()

                        writer.writerow([
                            _now_timestamp(),
                            status,
                            err_msg.replace("\n", " ").strip(),

                            object_name,
                            f"{object_density:.6g}",
                            finger_num,
                            int(no_cloth),

                            fps,
                            sub,
                            rep,
                            "",
                            num_frames,
                            "",
                            kernel_seed_common,

                            f"{t_pose:.6f}",
                            "",
                            "",
                            "",

                            "",
                            "",
                            "",
                            "",
                            "",

                            "",
                            "",
                            "",
                            "",

                            "",
                            "",

                            "",
                            "",
                            "",
                        ])
                        log_file.flush()

                    finally:
                        wp.synchronize()
                        gc.collect()
                        if tendon is not None:
                            del tendon
                        wp.synchronize()
                        gc.collect()

    log_file.close()
    print(f"\nLogged CSV to: {log_path}")
    print(f"Saved per run NPZ to: {data_dir}")
    print(f"Sweep meta: {sweep_meta_path}")


if __name__ == "__main__":
    run_dt_sweep(device="cuda:0", make_viz=True)
