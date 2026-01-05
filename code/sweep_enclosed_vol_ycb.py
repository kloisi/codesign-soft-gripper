# sweep_enclosed_vol_ycb.py
#
# Robust overnight sweep logger:
# - summary.csv (one row per object)
# - timeseries.csv (volume logger rows for all objects)
# - npz per object (heavy dump, safe + json-safe)
#
# Fixes:
# - Do NOT try to cast tendon.control (TendonControl object) to float32
# - Do NOT json-dump voxel debug dict containing numpy arrays (enclosed_pts, blocked_pts)
#
# Extra saved metrics:
# - per-frame object body force/torque (raw6 + split)
# - per-frame object body_q (pose)
# - final tendon forces
# - voxel debug meta + points arrays

import os
import csv
import json
import time
import math
import glob
import hashlib
import traceback
import platform
import subprocess
from datetime import datetime

import numpy as np
import warp as wp

from forward import FEMTendon
from object_loader import ObjectLoader
from init_pose import InitializeFingers


# -----------------------------
# Small utilities
# -----------------------------

def now_tag():
    return datetime.now().strftime("%Y%m%d_%H%M%S")

def safe_makedirs(p):
    os.makedirs(p, exist_ok=True)

def sha1_file(path, chunk=1 << 20):
    h = hashlib.sha1()
    with open(path, "rb") as f:
        while True:
            b = f.read(chunk)
            if not b:
                break
            h.update(b)
    return h.hexdigest()

def try_git_commit():
    try:
        out = subprocess.check_output(["git", "rev-parse", "HEAD"], stderr=subprocess.DEVNULL).decode().strip()
        return out
    except Exception:
        return ""

def _is_numeric_ndarray(a: np.ndarray) -> bool:
    return isinstance(a, np.ndarray) and (np.issubdtype(a.dtype, np.number))

def _structured_vec3_to_floatNx3(a: np.ndarray):
    # Warp sometimes returns structured arrays for vec3
    if not isinstance(a, np.ndarray) or a.dtype.fields is None:
        return None
    keys = set(a.dtype.fields.keys())
    if {"x", "y", "z"}.issubset(keys):
        return np.stack([a["x"], a["y"], a["z"]], axis=-1).astype(np.float32)
    return None

def wp_any_to_numpy(x):
    # Warp array -> numpy best effort
    if x is None:
        return None
    try:
        if hasattr(x, "numpy"):
            return x.numpy()
    except Exception:
        pass
    try:
        return np.asarray(x)
    except Exception:
        return None

def to_float_array(x):
    """
    Convert x into a numeric numpy array if possible.
    Returns None if x is not safely numeric.
    """
    a = wp_any_to_numpy(x)
    if a is None:
        return None

    if isinstance(a, np.ndarray):
        if _is_numeric_ndarray(a):
            return a.astype(np.float32, copy=False)

        # structured vec3 -> (N,3)
        v = _structured_vec3_to_floatNx3(a)
        if v is not None:
            return v

        # object arrays (e.g. list of vec3) -> try
        if a.dtype == object:
            try:
                b = np.asarray(a.tolist(), dtype=np.float32)
                if b.ndim >= 1:
                    return b
            except Exception:
                return None

        return None

    # scalar
    try:
        b = np.asarray(a, dtype=np.float32)
        return b
    except Exception:
        return None

def any_state_from_tendon(tendon):
    # Your FEMTendon uses tendon.states list, so we will use that.
    # Still keep generic fallback.
    for name in ("states", "state", "state1", "state0"):
        if hasattr(tendon, name):
            st = getattr(tendon, name)
            if st is not None:
                return st
    return None

def any_model_from_tendon(tendon):
    for name in ("model", "wp_model"):
        if hasattr(tendon, name):
            m = getattr(tendon, name)
            if m is not None:
                return m
    return None


# -----------------------------
# Mesh helpers (YCB .ply)
# -----------------------------

def find_ycb_mesh_path(loader: ObjectLoader, name: str):
    root = loader.data_dir
    p = os.path.join(root, name, "google_16k")
    cand = [
        os.path.join(p, "nontextured.ply"),
        os.path.join(p, "simple_nontextured.ply"),
    ]
    for c in cand:
        if os.path.exists(c):
            return c
    return None

def _parse_ply_header(f):
    first = f.readline().decode("utf-8", errors="ignore").strip()
    if first != "ply":
        raise ValueError("Not a PLY file")

    fmt = None
    vertex_count = 0
    face_count = 0

    raw = []
    while True:
        line = f.readline()
        if not line:
            raise ValueError("Unexpected EOF in PLY header")
        s = line.decode("utf-8", errors="ignore").strip()
        raw.append(s)
        if s.startswith("format"):
            fmt = s.split()[1]
        elif s.startswith("element vertex"):
            vertex_count = int(s.split()[-1])
        elif s.startswith("element face"):
            face_count = int(s.split()[-1])
        elif s == "end_header":
            header_bytes = f.tell()
            break

    if fmt not in ("ascii", "binary_little_endian"):
        raise ValueError(f"Unsupported PLY format: {fmt}")

    return {
        "fmt": fmt,
        "vertex_count": vertex_count,
        "face_count": face_count,
        "header_bytes": header_bytes,
        "raw": raw,
    }

def load_ply_vertices_faces(path):
    """
    Minimal PLY reader for typical YCB meshes.
    Returns:
      V: (N,3) float32
      F: (M,3) int32 (triangulated)
    """
    with open(path, "rb") as f:
        h = _parse_ply_header(f)
        fmt = h["fmt"]
        nv = h["vertex_count"]
        nf = h["face_count"]

        if fmt == "ascii":
            verts = []
            for _ in range(nv):
                parts = f.readline().decode("utf-8", errors="ignore").strip().split()
                verts.append([float(parts[0]), float(parts[1]), float(parts[2])])
            verts = np.asarray(verts, dtype=np.float32)

            faces = []
            for _ in range(nf):
                parts = f.readline().decode("utf-8", errors="ignore").strip().split()
                k = int(parts[0])
                idx = [int(x) for x in parts[1:1 + k]]
                for j in range(1, k - 1):
                    faces.append([idx[0], idx[j], idx[j + 1]])
            faces = np.asarray(faces, dtype=np.int32)
            return verts, faces

        # binary_little_endian: parse vertex props to get stride
        raw = h["raw"]
        vprops = []
        in_vertex = False
        in_face = False
        face_count_type = None
        face_index_type = None

        for s in raw:
            if s.startswith("element vertex"):
                in_vertex = True
                in_face = False
                vprops = []
            elif s.startswith("element face"):
                in_vertex = False
                in_face = True
            elif s.startswith("element "):
                in_vertex = False
                in_face = False
            elif s.startswith("property") and in_vertex:
                parts = s.split()
                if len(parts) >= 3:
                    vprops.append((parts[1], parts[2]))
            elif s.startswith("property") and in_face:
                parts = s.split()
                if len(parts) >= 5 and parts[1] == "list":
                    face_count_type = parts[2]
                    face_index_type = parts[3]

        type_map = {
            "char": np.int8,
            "uchar": np.uint8,
            "short": np.int16,
            "ushort": np.uint16,
            "int": np.int32,
            "uint": np.uint32,
            "float": np.float32,
            "double": np.float64,
        }

        v_dtype = []
        for t, name in vprops:
            if t not in type_map:
                raise ValueError(f"Unsupported vertex prop type: {t}")
            v_dtype.append((name, type_map[t]))
        v_dtype = np.dtype(v_dtype)

        if face_count_type is None or face_index_type is None:
            face_count_type = "uchar"
            face_index_type = "int"

        if face_count_type not in type_map or face_index_type not in type_map:
            raise ValueError(f"Unsupported face list types: {face_count_type}, {face_index_type}")

        f_count_dtype = type_map[face_count_type]
        f_index_dtype = type_map[face_index_type]

        payload = f.read()

        v_rec_bytes = v_dtype.itemsize
        need = nv * v_rec_bytes
        if len(payload) < need:
            raise ValueError("PLY payload too small for vertices")
        vbuf = payload[:need]
        Vfull = np.frombuffer(vbuf, dtype=v_dtype, count=nv)

        if not all(k in Vfull.dtype.names for k in ("x", "y", "z")):
            names = list(Vfull.dtype.names)
            V = np.stack([Vfull[names[0]], Vfull[names[1]], Vfull[names[2]]], axis=1).astype(np.float32)
        else:
            V = np.stack([Vfull["x"], Vfull["y"], Vfull["z"]], axis=1).astype(np.float32)

        faces = []
        off = need
        n_payload = len(payload)

        count_size = np.dtype(f_count_dtype).itemsize
        idx_size = np.dtype(f_index_dtype).itemsize

        for _ in range(nf):
            if off + count_size > n_payload:
                break
            k = np.frombuffer(payload, dtype=f_count_dtype, count=1, offset=off)[0]
            off += count_size
            if off + int(k) * idx_size > n_payload:
                break
            idx = np.frombuffer(payload, dtype=f_index_dtype, count=int(k), offset=off).astype(np.int64)
            off += int(k) * idx_size

            if int(k) < 3:
                continue
            i0 = int(idx[0])
            for j in range(1, int(k) - 1):
                faces.append([i0, int(idx[j]), int(idx[j + 1])])

        F = np.asarray(faces, dtype=np.int32)
        return V, F

def mesh_volume_area_bbox(V, F):
    if V is None or F is None or len(F) == 0:
        return dict(
            mesh_vol=np.nan,
            mesh_area=np.nan,
            bbox_min=np.full(3, np.nan),
            bbox_max=np.full(3, np.nan),
            bbox_vol=np.nan,
        )

    v0 = V[F[:, 0]]
    v1 = V[F[:, 1]]
    v2 = V[F[:, 2]]

    cross = np.cross(v1, v2)
    vol6 = np.einsum("ij,ij->i", v0, cross)
    vol = abs(vol6.sum()) / 6.0

    area = 0.5 * np.linalg.norm(np.cross(v1 - v0, v2 - v0), axis=1).sum()

    bmin = V.min(axis=0)
    bmax = V.max(axis=0)
    bbox_vol = float(np.prod(bmax - bmin))

    return dict(mesh_vol=float(vol), mesh_area=float(area), bbox_min=bmin, bbox_max=bmax, bbox_vol=bbox_vol)


# -----------------------------
# YCB list
# -----------------------------

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


# -----------------------------
# Force reading from Warp state
# -----------------------------

def body_f_to_raw6_and_split(bf_row):
    """
    bf_row could be:
      - shape (6,) numeric
      - structured / object
    We return:
      raw6 (6,) float32 (or NaNs)
      torque3, force3 (best guess: torque then force)
    """
    raw = np.full(6, np.nan, dtype=np.float32)

    try:
        a = np.asarray(bf_row)
        a = a.reshape(-1)
        if a.size == 6 and np.issubdtype(a.dtype, np.number):
            raw = a.astype(np.float32)
    except Exception:
        pass

    torque = raw[0:3].copy()
    force = raw[3:6].copy()
    return raw, torque, force


# -----------------------------
# Main per-object runner
# -----------------------------

def run_one_object(
    name,
    finger_num,
    device=None,
    pose_iters=1000,
    optimizer="lbfgs",
    do_force_opt=True,
    opt_frames=100,
    disable_per_object_forward_csv=True,
    scale=5.0,
    save_npz_path=None,
    save_npz_compress=True,
    particle_stride=10,  # save particle_q every N frames (None or 0 disables)
    verbose=False,
):
    finger_len = 11
    finger_rot = np.pi / 30
    finger_width = 0.08
    object_rot = wp.quat_rpy(-math.pi / 2, 0.0, 0.0)

    no_cloth = False
    consider_cloth = not no_cloth

    init_pose_num_frames = 30
    init_pose_stop_margin = 0.0005

    seed = int(np.random.randint(0, 1_000_000_000))

    # mesh stats
    loader = ObjectLoader()
    mesh_path = find_ycb_mesh_path(loader, name)
    mesh_sha1 = sha1_file(mesh_path) if mesh_path and os.path.exists(mesh_path) else ""

    mesh_vol = np.nan
    mesh_area = np.nan
    bbox_min = np.full(3, np.nan, dtype=np.float32)
    bbox_max = np.full(3, np.nan, dtype=np.float32)
    bbox_vol = np.nan

    if mesh_path and os.path.exists(mesh_path):
        try:
            V, F = load_ply_vertices_faces(mesh_path)
            m = mesh_volume_area_bbox(V, F)
            mesh_vol = float(m["mesh_vol"])
            mesh_area = float(m["mesh_area"])
            bbox_min = m["bbox_min"].astype(np.float32)
            bbox_max = m["bbox_max"].astype(np.float32)
            bbox_vol = float(m["bbox_vol"])
        except Exception as e:
            if verbose:
                print(f"[WARN] mesh parse failed for {name}: {type(e).__name__}: {e}")

    mesh_vol_scaled = mesh_vol * (scale ** 3) if np.isfinite(mesh_vol) else np.nan
    mesh_area_scaled = mesh_area * (scale ** 2) if np.isfinite(mesh_area) else np.nan
    bbox_min_scaled = bbox_min * float(scale)
    bbox_max_scaled = bbox_max * float(scale)
    bbox_vol_scaled = bbox_vol * (scale ** 3) if np.isfinite(bbox_vol) else np.nan

    t0 = time.time()

    with wp.ScopedDevice(device):
        np.random.seed(seed)

        # --------------------------------------------------
        # 1) init pose
        # --------------------------------------------------
        finger_transform = None
        init_finger = None
        init_pose_ok = False
        init_pose_logs = {}
        proxy_pts_frozen = None
        init_jq = None

        try:
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
                ycb_object_name=name,
                object_rot=object_rot,
                is_render=False,
                verbose=False,
                is_triangle=False,
                finger_num=finger_num,
                add_random=False,
                consider_cloth=consider_cloth,
            )

            finger_transform, init_jq = init_finger.get_initial_position()
            init_finger.capture_proxy_points_frozen()
            proxy_pts_frozen = getattr(init_finger, "proxy_pts_frozen", None)

            # capture whatever histories exist
            for k in (
                "loss_history",
                "radius_history",
                "iter_history",
                "finger_loss_history",
                "cloth_loss_raw_history",
                "cloth_loss_norm_history",
            ):
                if hasattr(init_finger, k):
                    init_pose_logs[k] = np.asarray(getattr(init_finger, k), dtype=object)

            init_pose_ok = finger_transform is not None
            if finger_transform is None and verbose:
                print(f"[WARN] init_pose failed for {name} (returned None).")

        except Exception as e:
            if verbose:
                print(f"[WARN] init_pose crashed for {name}: {type(e).__name__}: {e}")
            finger_transform = None
            init_finger = None

        # --------------------------------------------------
        # 2) build FEMTendon
        # --------------------------------------------------
        tendon = FEMTendon(
            stage_path=None,
            num_frames=1000,
            verbose=False,
            save_log=False,
            is_render=False,
            use_graph=False,
            kernel_seed=seed,
            train_iters=pose_iters,
            object_rot=object_rot,
            object_density=2.0,
            ycb_object_name=name,
            finger_len=finger_len,
            finger_rot=finger_rot,
            finger_width=finger_width,
            scale=scale,
            finger_transform=finger_transform,
            finger_num=finger_num,
            requires_grad=True,
            init_finger=init_finger,
            no_cloth=no_cloth,
        )

        if init_finger is not None and proxy_pts_frozen is not None:
            tendon.proxy_pts_frozen = proxy_pts_frozen

        # disable internal per-object voxel CSV writing
        if disable_per_object_forward_csv and hasattr(tendon, "vol_logger") and hasattr(tendon.vol_logger, "to_csv"):
            tendon.vol_logger.to_csv = lambda _path: None

        # --------------------------------------------------
        # 3) force optimisation
        # --------------------------------------------------
        force_opt_info = {"force_opt_ok": False, "force_opt_kind": optimizer, "opt_frames": int(opt_frames)}
        try:
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
            force_opt_info["force_opt_ok"] = True
        except Exception as e:
            force_opt_info["force_opt_ok"] = False
            force_opt_info["force_opt_error"] = f"{type(e).__name__}: {e}"
            if verbose:
                print(f"[WARN] force optimisation failed for {name}: {type(e).__name__}: {e}")

        # --------------------------------------------------
        # 4) forward
        # --------------------------------------------------
        tendon.forward()

        vol = getattr(tendon, "last_voxel_volume", np.nan)
        dbg = getattr(tendon, "last_voxel_debug", None) or {}
        rows = getattr(getattr(tendon, "vol_logger", None), "rows", []) or []

        model = any_model_from_tendon(tendon)

        # In your forward.py the object is body 0 if present.
        obj_body_id = 0 if getattr(tendon, "has_object", False) else -1

        # final tendon forces
        tendon_forces_final = to_float_array(getattr(tendon, "tendon_forces", None))

        # control arrays (avoid saving tendon.control object)
        ctrl_waypoint_forces = None
        ctrl_vel_values = None
        ctrl_waypoint_ids = None
        try:
            if hasattr(tendon, "control") and tendon.control is not None:
                ctrl_waypoint_forces = to_float_array(getattr(tendon.control, "waypoint_forces", None))
                ctrl_vel_values = to_float_array(getattr(tendon.control, "vel_values", None))
                ctrl_waypoint_ids = to_float_array(getattr(tendon.control, "waypoint_ids", None))
        except Exception:
            pass

        # per-frame object body forces + body pose
        obj_bf_raw6 = None
        obj_force3 = None
        obj_torque3 = None
        obj_body_q = None

        if hasattr(tendon, "states") and isinstance(tendon.states, list) and len(tendon.states) > 0 and obj_body_id >= 0:
            nf = int(getattr(tendon, "num_frames", 0))
            sub = int(getattr(tendon, "sim_substeps", 1))
            if nf > 0 and sub > 0:
                obj_bf_raw6 = np.full((nf, 6), np.nan, dtype=np.float32)
                obj_force3 = np.full((nf, 3), np.nan, dtype=np.float32)
                obj_torque3 = np.full((nf, 3), np.nan, dtype=np.float32)
                obj_body_q = np.full((nf, 7), np.nan, dtype=np.float32)

                for frame in range(nf):
                    idx = (frame + 1) * sub
                    if idx >= len(tendon.states):
                        break
                    st = tendon.states[idx]

                    # body_f
                    try:
                        bf = st.body_f.numpy()
                        raw6, tq, fc = body_f_to_raw6_and_split(bf[obj_body_id])
                        obj_bf_raw6[frame, :] = raw6
                        obj_torque3[frame, :] = tq
                        obj_force3[frame, :] = fc
                    except Exception:
                        pass

                    # body_q
                    try:
                        bq = st.body_q.numpy()
                        bq0 = np.asarray(bq[obj_body_id]).reshape(-1)
                        if bq0.size >= 7 and np.issubdtype(bq0.dtype, np.number):
                            obj_body_q[frame, :] = bq0[:7].astype(np.float32)
                    except Exception:
                        pass

        # particle snapshots (stride)
        particle_q_stride = None
        particle_q_stride_frames = None
        if particle_stride is not None and int(particle_stride) > 0:
            stride = int(particle_stride)
            if hasattr(tendon, "states") and isinstance(tendon.states, list) and len(tendon.states) > 0:
                nf = int(getattr(tendon, "num_frames", 0))
                sub = int(getattr(tendon, "sim_substeps", 1))
                frames = list(range(0, nf, stride))
                if (nf - 1) not in frames and nf > 0:
                    frames.append(nf - 1)
                frames = sorted(set(frames))

                snaps = []
                for frame in frames:
                    idx = (frame + 1) * sub
                    if idx >= len(tendon.states):
                        continue
                    st = tendon.states[idx]
                    try:
                        q = st.particle_q.numpy()
                        if q.ndim == 3:
                            q = q[0]
                        q = np.asarray(q, dtype=np.float32)
                        snaps.append(q)
                    except Exception:
                        continue

                if len(snaps) > 0:
                    particle_q_stride = np.stack(snaps, axis=0)  # (K, N, 3)
                    particle_q_stride_frames = np.asarray(frames[:particle_q_stride.shape[0]], dtype=np.int32)

        # voxel debug: split meta vs arrays so JSON never crashes
        dbg_meta = {}
        voxel_enclosed_pts = None
        voxel_blocked_pts = None
        if isinstance(dbg, dict):
            for k, v in dbg.items():
                if isinstance(v, np.ndarray):
                    if k == "enclosed_pts":
                        voxel_enclosed_pts = np.asarray(v, dtype=np.float32)
                    elif k == "blocked_pts":
                        voxel_blocked_pts = np.asarray(v, dtype=np.float32)
                    else:
                        # store only shape info in meta
                        dbg_meta[k] = {"ndarray_shape": list(v.shape), "dtype": str(v.dtype)}
                else:
                    # json-safe scalars / tuples / lists
                    try:
                        json.dumps(v)
                        dbg_meta[k] = v
                    except Exception:
                        dbg_meta[k] = str(v)

            # always include point shapes in meta
            if voxel_enclosed_pts is not None:
                dbg_meta["enclosed_pts_shape"] = list(voxel_enclosed_pts.shape)
            if voxel_blocked_pts is not None:
                dbg_meta["blocked_pts_shape"] = list(voxel_blocked_pts.shape)

        # ratios
        enclosed_over_object = float(vol) / float(mesh_vol_scaled) if np.isfinite(vol) and np.isfinite(mesh_vol_scaled) and mesh_vol_scaled > 0 else np.nan

        # object final force (from last frame arrays if available)
        obj_force_final = np.full(3, np.nan, dtype=np.float32)
        obj_torque_final = np.full(3, np.nan, dtype=np.float32)
        if obj_force3 is not None and obj_force3.shape[0] > 0:
            obj_force_final = obj_force3[-1].copy()
        if obj_torque3 is not None and obj_torque3.shape[0] > 0:
            obj_torque_final = obj_torque3[-1].copy()

        summary = {
            "object": name,
            "finger_num": int(finger_num),
            "seed": int(seed),

            "init_pose_ok": int(bool(init_pose_ok)),
            "force_opt_ok": int(bool(force_opt_info.get("force_opt_ok", False))),

            "vol_vox_m3": float(vol) if np.isfinite(vol) else np.nan,
            "enclosed_over_object_vol": float(enclosed_over_object),

            "mesh_path": mesh_path or "",
            "mesh_sha1": mesh_sha1,
            "mesh_vol_m3_unscaled": float(mesh_vol) if np.isfinite(mesh_vol) else np.nan,
            "mesh_vol_m3_scaled": float(mesh_vol_scaled) if np.isfinite(mesh_vol_scaled) else np.nan,
            "mesh_area_m2_scaled": float(mesh_area_scaled) if np.isfinite(mesh_area_scaled) else np.nan,
            "bbox_vol_m3_scaled": float(bbox_vol_scaled) if np.isfinite(bbox_vol_scaled) else np.nan,

            "obj_body_id": int(obj_body_id),

            "obj_force_fx": float(obj_force_final[0]) if np.isfinite(obj_force_final[0]) else np.nan,
            "obj_force_fy": float(obj_force_final[1]) if np.isfinite(obj_force_final[1]) else np.nan,
            "obj_force_fz": float(obj_force_final[2]) if np.isfinite(obj_force_final[2]) else np.nan,
            "obj_torque_tx": float(obj_torque_final[0]) if np.isfinite(obj_torque_final[0]) else np.nan,
            "obj_torque_ty": float(obj_torque_final[1]) if np.isfinite(obj_torque_final[1]) else np.nan,
            "obj_torque_tz": float(obj_torque_final[2]) if np.isfinite(obj_torque_final[2]) else np.nan,

            "voxel_size": float(dbg_meta.get("voxel_size", np.nan)) if isinstance(dbg_meta, dict) else np.nan,
            "enclosed_voxels": float(dbg_meta.get("enclosed_voxels", np.nan)) if isinstance(dbg_meta, dict) else np.nan,
            "blocked_voxels": float(dbg_meta.get("blocked_voxels", np.nan)) if isinstance(dbg_meta, dict) else np.nan,
            "y_bottom": float(dbg_meta.get("y_bottom", np.nan)) if isinstance(dbg_meta, dict) else np.nan,
            "y_top": float(dbg_meta.get("y_top", np.nan)) if isinstance(dbg_meta, dict) else np.nan,
        }

        # heavy NPZ dump
        if save_npz_path:
            meta = {
                "object": name,
                "finger_num": int(finger_num),
                "seed": int(seed),
                "device": str(wp.get_device()),
                "platform": platform.platform(),
                "python": platform.python_version(),
                "git_commit": try_git_commit(),
                "scale": float(scale),
                "finger_len": int(finger_len),
                "finger_rot": float(finger_rot),
                "finger_width": float(finger_width),
                "optimizer": optimizer,
                "do_force_opt": bool(do_force_opt),
                "opt_frames": int(opt_frames),
                "mesh_path": mesh_path or "",
                "mesh_sha1": mesh_sha1,
                "particle_stride": int(particle_stride) if particle_stride else 0,
            }

            npz_dict = {}
            npz_dict["meta_json"] = np.array([json.dumps(meta)], dtype=object)
            npz_dict["summary_json"] = np.array([json.dumps(summary)], dtype=object)
            npz_dict["force_opt_info_json"] = np.array([json.dumps(force_opt_info)], dtype=object)

            # mesh scalars
            npz_dict["mesh_bbox_min_scaled"] = bbox_min_scaled.astype(np.float32)
            npz_dict["mesh_bbox_max_scaled"] = bbox_max_scaled.astype(np.float32)
            npz_dict["mesh_vol_scaled"] = np.array([mesh_vol_scaled], dtype=np.float64)
            npz_dict["mesh_area_scaled"] = np.array([mesh_area_scaled], dtype=np.float64)

            # init pose outputs
            if finger_transform is not None:
                npz_dict["finger_transform"] = np.asarray(finger_transform, dtype=np.float32)
            if init_jq is not None:
                npz_dict["init_joint_q"] = np.asarray(init_jq, dtype=np.float32)
            if proxy_pts_frozen is not None:
                npz_dict["proxy_pts_frozen"] = np.asarray(proxy_pts_frozen, dtype=np.float32)

            for k, v in init_pose_logs.items():
                npz_dict[f"init_{k}"] = v

            # voxel outputs
            npz_dict["last_voxel_volume"] = np.array([vol], dtype=np.float64)
            npz_dict["voxel_debug_meta_json"] = np.array([json.dumps(dbg_meta)], dtype=object)
            if voxel_enclosed_pts is not None:
                npz_dict["voxel_enclosed_pts"] = voxel_enclosed_pts
            if voxel_blocked_pts is not None:
                npz_dict["voxel_blocked_pts"] = voxel_blocked_pts

            # volume logger time series (as column arrays)
            if isinstance(rows, list) and len(rows) > 0:
                keys = sorted({k for r in rows if isinstance(r, dict) for k in r.keys()})
                npz_dict["ts_keys_json"] = np.array([json.dumps(keys)], dtype=object)
                for k in keys:
                    col = []
                    for r in rows:
                        v = r.get(k, np.nan) if isinstance(r, dict) else np.nan
                        try:
                            col.append(float(v))
                        except Exception:
                            col.append(np.nan)
                    npz_dict[f"ts_{k}"] = np.asarray(col, dtype=np.float64)
            else:
                npz_dict["ts_keys_json"] = np.array([json.dumps([])], dtype=object)

            # forces + control arrays
            if tendon_forces_final is not None:
                npz_dict["tendon_forces_final"] = tendon_forces_final
            if ctrl_waypoint_forces is not None:
                npz_dict["control_waypoint_forces"] = ctrl_waypoint_forces
            if ctrl_vel_values is not None:
                npz_dict["control_vel_values"] = ctrl_vel_values
            if ctrl_waypoint_ids is not None:
                npz_dict["control_waypoint_ids"] = ctrl_waypoint_ids

            # per-frame object forces/poses
            if obj_bf_raw6 is not None:
                npz_dict["obj_body_f_raw6_frames"] = obj_bf_raw6
            if obj_force3 is not None:
                npz_dict["obj_force_frames"] = obj_force3
            if obj_torque3 is not None:
                npz_dict["obj_torque_frames"] = obj_torque3
            if obj_body_q is not None:
                npz_dict["obj_body_q_frames"] = obj_body_q

            # particle snapshots
            if particle_q_stride is not None:
                npz_dict["particle_q_stride_frames"] = particle_q_stride_frames
                npz_dict["particle_q_stride"] = particle_q_stride

            if save_npz_compress:
                np.savez_compressed(save_npz_path, **npz_dict)
            else:
                np.savez(save_npz_path, **npz_dict)

        dt = time.time() - t0
        summary["runtime_s"] = float(dt)

        return summary, rows


# -----------------------------
# Main sweep
# -----------------------------

def main():
    # --- knobs ---
    finger_num = 6
    pose_iters = 1000
    optimizer = "lbfgs"
    do_force_opt = True
    opt_frames = 100

    device = "cuda:0"   # set to None to use default
    scale = 5.0

    disable_per_object_forward_csv = True

    # save particle positions every N frames (set to 0/None to disable)
    particle_stride = 10

    run_id = now_tag()
    out_root = os.path.join("logs", f"sweep_{run_id}_f{finger_num}")
    out_npz_dir = os.path.join(out_root, "npz")
    safe_makedirs(out_root)
    safe_makedirs(out_npz_dir)

    out_summary = os.path.join(out_root, "summary.csv")
    out_ts = os.path.join(out_root, "timeseries.csv")
    out_errors = os.path.join(out_root, "errors.txt")

    names = list_ycb_objects()
    print(f"Found {len(names)} YCB objects")
    print(f"Writing to: {out_root}")

    summary_fieldnames = [
        "object",
        "finger_num",
        "seed",
        "runtime_s",
        "init_pose_ok",
        "force_opt_ok",
        "vol_vox_m3",
        "enclosed_over_object_vol",
        "mesh_path",
        "mesh_sha1",
        "mesh_vol_m3_unscaled",
        "mesh_vol_m3_scaled",
        "mesh_area_m2_scaled",
        "bbox_vol_m3_scaled",
        "obj_body_id",
        "obj_force_fx",
        "obj_force_fy",
        "obj_force_fz",
        "obj_torque_tx",
        "obj_torque_ty",
        "obj_torque_tz",
        "voxel_size",
        "enclosed_voxels",
        "blocked_voxels",
        "y_bottom",
        "y_top",
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

    with open(out_summary, "w", newline="") as fs, open(out_ts, "w", newline="") as ft, open(out_errors, "w") as fe:
        ws = csv.DictWriter(fs, fieldnames=summary_fieldnames)
        wt = csv.DictWriter(ft, fieldnames=ts_fieldnames)
        ws.writeheader()
        wt.writeheader()

        for k, name in enumerate(names):
            print(f"[{k+1}/{len(names)}] {name}")

            srow = {fn: "" for fn in summary_fieldnames}
            srow["object"] = name
            srow["finger_num"] = int(finger_num)

            npz_path = os.path.join(out_npz_dir, f"{name}_f{finger_num}.npz")

            try:
                summary, timeseries_rows = run_one_object(
                    name=name,
                    finger_num=finger_num,
                    device=device,
                    pose_iters=pose_iters,
                    optimizer=optimizer,
                    do_force_opt=do_force_opt,
                    opt_frames=opt_frames,
                    disable_per_object_forward_csv=disable_per_object_forward_csv,
                    scale=scale,
                    save_npz_path=npz_path,
                    save_npz_compress=True,
                    particle_stride=particle_stride,
                    verbose=False,
                )

                srow.update(summary)
                ws.writerow(srow)
                fs.flush()

                if isinstance(timeseries_rows, list):
                    for r in timeseries_rows:
                        trow = {fn: "" for fn in ts_fieldnames}
                        trow["object"] = name
                        trow["finger_num"] = int(finger_num)
                        if isinstance(r, dict):
                            for key in ("step", "t", "frame", "substep", "vol_vox", "enclosed_voxels", "blocked_voxels", "y_bottom", "y_top"):
                                if key in r:
                                    trow[key] = r[key]
                        wt.writerow(trow)
                ft.flush()

            except Exception as e:
                srow["error"] = f"{type(e).__name__}: {e}"
                ws.writerow(srow)
                fs.flush()

                fe.write(f"\n[{name}] {type(e).__name__}: {e}\n")
                fe.write(traceback.format_exc())
                fe.flush()

                traceback.print_exc()

    print(f"Saved summary to: {out_summary}")
    print(f"Saved timeseries to: {out_ts}")
    print(f"Saved errors to: {out_errors}")
    print(f"Saved per-object NPZ to: {out_npz_dir}")


if __name__ == "__main__":
    try:
        wp.init()
    except Exception:
        pass
    main()
