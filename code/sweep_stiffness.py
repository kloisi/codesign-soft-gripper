# sweep_stiffness.py (forward-like, memory-safe, with per-finger force logging)
import os
import csv
import time
import math
import gc
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


def _extract_q_qd_end(tendon):
    st = tendon.states[-1]
    q = _safe_np(st.particle_q)
    qd = _safe_np(st.particle_qd) if hasattr(st, "particle_qd") else None

    if q.ndim == 3:
        q = q[0]
    if q.ndim == 1:
        q = q.reshape(-1, 3)

    if qd is not None:
        if qd.ndim == 3:
            qd = qd[0]
        if qd.ndim == 1:
            qd = qd.reshape(-1, 3)

    return q, qd


def _get_cloth_ids(tendon):
    # Prefer tendon.cloth_ids (forward caches this), else model.cloth_particle_ids, else builder
    if getattr(tendon, "cloth_ids", None) is not None:
        ids = np.asarray(tendon.cloth_ids, dtype=np.int64).ravel()
        if ids.size:
            return ids

    model = getattr(tendon, "model", None)
    if model is not None and getattr(model, "cloth_particle_ids", None) is not None:
        try:
            ids = _safe_np(model.cloth_particle_ids).astype(np.int64).ravel()
            if ids.size:
                return ids
        except Exception:
            pass

    builder = getattr(tendon, "builder", None)
    if builder is not None and getattr(builder, "cloth_particle_ids", None) is not None:
        ids = np.asarray(builder.cloth_particle_ids, dtype=np.int64).ravel()
        if ids.size:
            return ids

    return None


def _extract_tendon_forces(tendon):
    # tendon.tendon_forces is a wp.array of length finger_num
    if not hasattr(tendon, "tendon_forces") or tendon.tendon_forces is None:
        return None
    f = _safe_np(tendon.tendon_forces).astype(np.float64).ravel()
    if f.size == 0:
        return None
    return f


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


def _fmt(x, fmt="{:.6g}"):
    if x is None:
        return ""
    try:
        if not np.isfinite(float(x)):
            return ""
    except Exception:
        return ""
    return fmt.format(float(x))


# -------------------------
# Forward-like init pose
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
            consider_cloth=consider_cloth,
        )

        finger_transform, _ = init_finger.get_initial_position()

        if init_finger is not None:
            init_finger.capture_proxy_points_frozen()

        if finger_transform is None:
            print("[WARN] InitializeFingers returned None. Falling back to default transforms.")
            init_finger = None

    return dict(
        object_rot=object_rot,
        ycb_object_name=object_name,
        object_density=2.0,
        finger_len=finger_len,
        finger_rot=finger_rot,
        finger_width=finger_width,
        scale=scale,
        finger_transform=finger_transform,
        init_finger=init_finger,
        finger_num=finger_num,
    )


# -------------------------
# Reconfigure time stepping like dt sweep
# -------------------------
def reconfigure_time_stepping(tendon: FEMTendon, fps: int, sim_substeps: int, num_frames: int):
    tendon.frame_dt = 1.0 / float(fps)
    tendon.num_frames = int(num_frames)

    tendon.sim_substeps = int(sim_substeps)
    tendon.sim_dt = tendon.frame_dt / float(tendon.sim_substeps)

    tendon.sim_time = 0.0
    tendon.render_time = 0.0

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


def build_tendon_forward_like(scene: dict, device: str | None, kernel_seed: int, no_cloth: bool):
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
            object_density=scene["object_density"],
            ycb_object_name=scene["ycb_object_name"],
            finger_len=scene["finger_len"],
            finger_rot=scene["finger_rot"],
            finger_width=scene["finger_width"],
            scale=scene["scale"],
            finger_transform=scene["finger_transform"],
            finger_num=scene["finger_num"],
            requires_grad=True,
            init_finger=scene["init_finger"],
            no_cloth=no_cloth,
        )

        init_finger = scene.get("init_finger", None)
        if init_finger is not None and getattr(init_finger, "proxy_pts_frozen", None) is not None:
            tendon.proxy_pts_frozen = init_finger.proxy_pts_frozen

        if hasattr(tendon, "vol_logger") and tendon.vol_logger is not None:
            tendon.vol_logger.to_csv = lambda _path: None

    return tendon


def run_force_optimisation_forward_like(
    tendon: FEMTendon,
    optimizer: str = "lbfgs",
    opt_frames: int = 100,
    do_force_opt: bool = True,
):
    if not do_force_opt:
        return dict(loss0=np.nan, lossT=np.nan)

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

    return dict(loss0=loss0, lossT=lossT)


# -------------------------
# Cloth scaling
# -------------------------
def scale_cloth_stiffness(tendon, factor_ke_ka=1.0, factor_kd=1.0):
    model = getattr(tendon, "model", None)
    if model is None or getattr(model, "tri_indices", None) is None:
        print("[scale_cloth_stiffness] model or tri_indices missing")
        return

    cloth_ids = _get_cloth_ids(tendon)
    if cloth_ids is None or cloth_ids.size == 0:
        print("[scale_cloth_stiffness] No cloth ids found; nothing to scale.")
        return

    tri_indices = _safe_np(model.tri_indices).astype(np.int64)
    if tri_indices.ndim == 1:
        tri_indices = tri_indices.reshape(-1, 3)

    mask = np.isin(tri_indices, cloth_ids).any(axis=1)
    if not np.any(mask):
        print("[scale_cloth_stiffness] No cloth tris found by ANY rule; nothing to scale.")
        return

    tri_mats = _safe_np(model.tri_materials).astype(np.float32)
    tri_mats[mask, 0] *= float(factor_ke_ka)
    tri_mats[mask, 1] *= float(factor_ke_ka)
    tri_mats[mask, 2] *= float(factor_kd)

    model.tri_materials = wp.array(tri_mats, dtype=wp.float32, device=wp.get_device())
    print(f"[scale_cloth_stiffness] scaled {int(mask.sum())} cloth tris: ke/ka x{factor_ke_ka} kd x{factor_kd}")


def scale_cloth_mass(tendon, mass_factor: float):
    model = getattr(tendon, "model", None)
    if model is None or getattr(model, "particle_inv_mass", None) is None:
        print("[scale_cloth_mass] model or particle_inv_mass missing")
        return

    cloth_ids = _get_cloth_ids(tendon)
    if cloth_ids is None or cloth_ids.size == 0:
        print("[scale_cloth_mass] No cloth ids found; nothing to scale.")
        return

    inv_m = _safe_np(model.particle_inv_mass).astype(np.float32)

    for gid in cloth_ids:
        if inv_m[gid] > 0.0:
            m = 1.0 / inv_m[gid]
            m *= float(mass_factor)
            inv_m[gid] = 1.0 / max(m, 1e-12)

    model.particle_inv_mass = wp.array(inv_m, dtype=wp.float32, device=wp.get_device())
    print(f"[scale_cloth_mass] scaled {len(cloth_ids)} cloth vertices by mass_factor={mass_factor}")


# -------------------------
# Volume metrics
# -------------------------
def volume_metrics_from_rows(rows, frac=0.95, tail_window=50):
    if rows is None or len(rows) == 0:
        return dict(vol_final=np.nan, t_frac=np.nan, tail_cv=np.nan, overshoot_rel=np.nan)

    t = np.array([r.get("t", np.nan) for r in rows], dtype=float)
    v = np.array([r.get("vol_vox", np.nan) for r in rows], dtype=float)

    good = np.isfinite(t) & np.isfinite(v)
    t = t[good]
    v = v[good]
    if v.size < 2:
        return dict(vol_final=np.nan, t_frac=np.nan, tail_cv=np.nan, overshoot_rel=np.nan)

    order = np.argsort(t)
    t = t[order]
    v = v[order]

    v_final = float(v[-1])
    v_max = float(np.max(v))

    target = frac * v_final
    idx = np.argmax(v >= target) if np.any(v >= target) else -1
    t_frac = float(t[idx]) if idx >= 0 else np.nan

    tw = min(int(tail_window), v.size)
    tail = v[-tw:]
    tail_mean = float(np.mean(tail))
    tail_std = float(np.std(tail))
    tail_cv = float(tail_std / tail_mean) if abs(tail_mean) > 1e-12 else np.nan

    overshoot_rel = float((v_max - v_final) / v_final) if abs(v_final) > 1e-12 else np.nan

    return dict(vol_final=v_final, t_frac=t_frac, tail_cv=tail_cv, overshoot_rel=overshoot_rel)


def _ensure_logfile_with_header(path, expected_header):
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)

    if not os.path.exists(path):
        f = open(path, "a", newline="")
        w = csv.writer(f)
        w.writerow(expected_header)
        f.flush()
        return f, w, path

    # If exists, check header compatibility.
    try:
        with open(path, "r", newline="") as r:
            first = r.readline().strip()
        existing = [c.strip() for c in first.split(",")] if first else []
        if existing == expected_header:
            f = open(path, "a", newline="")
            w = csv.writer(f)
            return f, w, path
    except Exception:
        pass

    # Header mismatch: write to new file
    base, ext = os.path.splitext(path)
    new_path = base + "_forces_v2" + ext
    print(f"[stiff_sweep] Header mismatch in {path}. Writing to {new_path} instead.")
    f = open(new_path, "a", newline="")
    w = csv.writer(f)
    w.writerow(expected_header)
    f.flush()
    return f, w, new_path


# -------------------------
# Main sweep
# -------------------------
def run_stiffness_sweep(
    device="cuda:0",
    log_filename="stiffness_sweep_forwardlike.csv",
    make_viz=True,
):
    BASE_FPS = 4000
    SIM_SUBSTEPS = 100
    NUM_FRAMES = 1000

    cloth_stiff_scales = [10.0, 5.0, 2.0, 1.0, 0.75, 0.5, 0.3, 0.1]
    cloth_mass_scales  = [0.5, 1.0, 2.0]
    cloth_damp_scales  = [0.5, 1.0, 2.0]

    object_name = "006_mustard_bottle"
    object_density = 2.0
    finger_num = 6
    no_cloth = False
    pose_iters = 1000

    do_force_opt = True
    optimizer = "lbfgs"
    opt_frames = 100

    kernel_seed_base = 12345

    curr_dir = os.path.dirname(os.path.abspath(__file__))
    log_path = os.path.join(curr_dir, log_filename)

    # Header includes per-finger forces (final after optimisation)
    force_cols = [f"force_f{i}" for i in range(int(finger_num))]
    header = [
        "timestamp",
        "object",
        "finger_num",
        "cloth_stiff_scale",
        "cloth_mass_scale",
        "cloth_damp_scale",
        "fps",
        "sim_substeps",
        "dt",
        "num_frames",
        "total_steps",
        "t_forceopt_s",
        "real_time_per_step_ms",
        "num_ok",
        "phys_ok",
        "diff_norm",
        "max_abs_pos",
        "max_speed",
        "vol_final",
        "t95",
        "tail_cv",
        "overshoot_rel",
        "forceopt_loss0",
        "forceopt_lossT",
        "forces_mean",
        "forces_std",
        "forces_min",
        "forces_max",
        "forces_cv",
        "forces_l2",
    ] + force_cols + [
        "kernel_seed",
    ]

    log_file, writer, actual_log_path = _ensure_logfile_with_header(log_path, header)

    viz_dir = os.path.join(curr_dir, "stiffness_sweep_viz_forwardlike")
    if make_viz:
        os.makedirs(viz_dir, exist_ok=True)

    with wp.ScopedDevice(device):
        scene = run_init_pose_forward_like(
            object_name=object_name,
            finger_num=finger_num,
            pose_iters=pose_iters,
            device=device,
            no_cloth=no_cloth,
        )
        scene["pose_iters"] = pose_iters
        scene["object_density"] = object_density

        for stiff_scale in cloth_stiff_scales:
            for mass_scale in cloth_mass_scales:
                for damp_scale in cloth_damp_scales:
                    print(f"\n=== stiff={stiff_scale} mass={mass_scale} damp={damp_scale} ===")

                    kernel_seed = kernel_seed_base
                    tendon = build_tendon_forward_like(scene, device=device, kernel_seed=kernel_seed, no_cloth=no_cloth)
                    reconfigure_time_stepping(tendon, fps=BASE_FPS, sim_substeps=SIM_SUBSTEPS, num_frames=NUM_FRAMES)

                    scale_cloth_mass(tendon, mass_scale)
                    scale_cloth_stiffness(tendon, factor_ke_ka=stiff_scale, factor_kd=damp_scale)

                    # Force optimisation (and capture final per-finger forces)
                    t_opt0 = time.perf_counter()
                    opt_stats = run_force_optimisation_forward_like(
                        tendon,
                        optimizer=optimizer,
                        opt_frames=opt_frames,
                        do_force_opt=do_force_opt,
                    )
                    t_forceopt = time.perf_counter() - t_opt0

                    forces_final = _extract_tendon_forces(tendon)
                    fstats = _force_stats(forces_final)

                    # Run forward sim
                    t0 = time.perf_counter()
                    tendon.forward()
                    elapsed = time.perf_counter() - t0

                    total_steps = tendon.num_frames * tendon.sim_substeps
                    real_time_per_step = elapsed / max(total_steps, 1)

                    # Numeric checks
                    q_end, qd_end = _extract_q_qd_end(tendon)
                    max_abs_pos = float(np.nanmax(np.abs(q_end))) if q_end.size else np.nan
                    max_speed = np.nan
                    if qd_end is not None and qd_end.size:
                        max_speed = float(np.nanmax(np.linalg.norm(qd_end, axis=1)))
                    num_ok = bool(np.isfinite(q_end).all() and (max_abs_pos < 2.0))

                    # Physical sanity check (diff_q style)
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

                    # Volume metrics
                    rows = getattr(getattr(tendon, "vol_logger", None), "rows", None)
                    vm = volume_metrics_from_rows(rows, frac=0.95, tail_window=50)

                    vol_final = getattr(tendon, "last_voxel_volume", np.nan)
                    if not np.isfinite(vol_final) and np.isfinite(vm["vol_final"]):
                        vol_final = vm["vol_final"]

                    print(
                        f"[stiff_sweep] dt={tendon.sim_dt:.2e} steps={total_steps} "
                        f"t/step={real_time_per_step*1e3:.3f}ms "
                        f"num_ok={num_ok} phys_ok={phys_ok} "
                        f"V={vol_final:.6g} t95={vm['t_frac']:.4f} tail_cv={vm['tail_cv']:.3e} "
                        f"forces_mean={fstats['forces_mean']:.3g} forces_max={fstats['forces_max']:.3g}"
                    )

                    # Per-finger forces, padded/truncated to finger_num
                    forces_row = [""] * int(finger_num)
                    if forces_final is not None:
                        ff = np.asarray(forces_final, dtype=np.float64).ravel()
                        for i in range(min(len(ff), int(finger_num))):
                            forces_row[i] = _fmt(ff[i])

                    writer.writerow([
                        time.strftime("%Y-%m-%d %H:%M:%S"),
                        scene["ycb_object_name"],
                        scene["finger_num"],
                        stiff_scale,
                        mass_scale,
                        damp_scale,
                        BASE_FPS,
                        SIM_SUBSTEPS,
                        _fmt(tendon.sim_dt, "{:.6e}"),
                        NUM_FRAMES,
                        total_steps,
                        _fmt(t_forceopt, "{:.6f}"),
                        _fmt(real_time_per_step * 1e3, "{:.6f}"),
                        int(num_ok),
                        int(phys_ok),
                        _fmt(diff_norm),
                        _fmt(max_abs_pos),
                        _fmt(max_speed),
                        _fmt(vol_final),
                        _fmt(vm["t_frac"]),
                        _fmt(vm["tail_cv"]),
                        _fmt(vm["overshoot_rel"]),
                        _fmt(opt_stats.get("loss0", np.nan)),
                        _fmt(opt_stats.get("lossT", np.nan)),
                        _fmt(fstats["forces_mean"]),
                        _fmt(fstats["forces_std"]),
                        _fmt(fstats["forces_min"]),
                        _fmt(fstats["forces_max"]),
                        _fmt(fstats["forces_cv"]),
                        _fmt(fstats["forces_l2"]),
                        *forces_row,
                        kernel_seed,
                    ])
                    log_file.flush()

                    # Visualisation
                    if make_viz:
                        def _fmtname(x):
                            return str(float(x)).replace(".", "p")

                        vid = f"viz_stiff{_fmtname(stiff_scale)}_mass{_fmtname(mass_scale)}_damp{_fmtname(damp_scale)}.mp4"
                        save_path = os.path.join(viz_dir, vid)
                        print("  making viz:", save_path)
                        quick_visualize(
                            tendon,
                            stride=50,
                            interval=30,
                            save_path=save_path,
                            elev=30,
                            azim=45,
                        )

                    # Free GPU memory per run
                    del tendon
                    gc.collect()
                    wp.synchronize()

    log_file.close()
    print(f"\nLogged results to: {actual_log_path}")


if __name__ == "__main__":
    run_stiffness_sweep(device="cuda:0", make_viz=True)
