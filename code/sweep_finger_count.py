# sweep_finger_count.py
#
# Sweep over coral meshes and finger counts.
# For each (coral, n_fingers):
#   1) Run InitializeFingers to get an individually optimised initial pose (radius not fixed)
#   2) Build ONE FEMTendon (same idea as forward.py)
#   3) Run L-BFGS force optimisation on that tendon
#   4) Run tendon.forward() once and read Init_VoxVol and Final_VoxVol
#
# Safety and robustness:
#   - Writes a CSV row after EACH run
#   - Resume skips only Status == "ok"
#   - CSV is schema locked and fully quoted, so commas in lists do not break it
#   - Default mode runs each case in a fresh subprocess (best protection against CUDA fragmentation)
#   - Enclosed volume is computed only at t0 and final frame (intermediate calls are cached)
#
# NOTE:
#   This does NOT change fps, sim_substeps, sim_dt etc inside FEMTendon.
#   Those stay exactly as in forward.py.

import os
import sys
import math
import gc
import time
import csv
import argparse
import subprocess
from datetime import datetime

import numpy as np
import pandas as pd
import warp as wp
import torch

import matplotlib
matplotlib.use("Agg")  # safe for ssh/headless
import matplotlib.pyplot as plt

from forward import FEMTendon, InitializeFingers


# ----------------------------
# CONFIG (keep your commented full lists)
# ----------------------------

# --- TEST SET (keep small while debugging) ---
OBJECT_LIST_TEST = ["acropora_cervicornis", "acropora_florida"]

# --- FULL SWEEP (DO NOT DELETE; uncomment when ready) ---
OBJECT_LIST_FULL = [
     "acropora_cervicornis", "acropora_florida", "acropora_loripes", "acropora_millepora",
    "acropora_nobilis", "acropora_palmata", "acropora_sarmentosa", "acropora_tenuis",
    "fungia_scutaria", "goniastrea_aspera", "montipora_capitata", "platygyra_daedalea",
    "platygyra_lamellina", "pocillopora_meandrina"
]

OBJECT_LIST_REPAIR = [
    "montipora_capitata"
]

# --- TEST SET ---
FINGER_COUNTS_TEST = [4, 5, 6]

# --- FULL SWEEP ---
FINGER_COUNTS_FULL = [3, 4, 5, 6, 7, 8, 9]

FINGER_COUNTS_REPAIR = [3, 7]


# ----------------------------
# CSV schema (fixed columns)
# ----------------------------
CSV_COLUMNS = [
    "Object",
    "Num_Fingers",

    # settings (handy later)
    "Device",
    "PoseOptFrames",
    "PoseOptIters",
    "SimNumFrames",
    "ForceOptIters",
    "ForceOptFrames",
    "ForceOptLR",

    # pose outputs
    "Radius_Mean",
    "Radius_List",

    # optimisation outputs
    "Final_Loss",
    "Avg_Force",
    "All_Forces",

    # volume outputs
    "Init_VoxVol",
    "Final_VoxVol",
    "Delta_VoxVol",

    # bookkeeping
    "Status",
    "Error",
    "Timestamp",
]


def _extract_radii(finger_transform):
    """
    Returns:
      radii: list[float]
      radius_mean: float
    Works for:
      - list/tuple of wp.transform (per finger)
      - single wp.transform
      - tuple(pos, quat) variants
    """
    def _pos_from_tf(tf):
        if hasattr(tf, "p"):  # wp.transform
            p = tf.p
            return np.array([p[0], p[1], p[2]], dtype=np.float64)
        if isinstance(tf, (list, tuple)) and len(tf) >= 1:  # (pos, quat)
            pos = tf[0]
            return np.array([pos[0], pos[1], pos[2]], dtype=np.float64)
        raise TypeError(f"Unsupported finger_transform element type: {type(tf)}")

    radii = []
    if isinstance(finger_transform, (list, tuple)) and len(finger_transform) > 0:
        if hasattr(finger_transform[0], "p"):
            for tf in finger_transform:
                radii.append(float(np.linalg.norm(_pos_from_tf(tf))))
        else:
            radii.append(float(np.linalg.norm(_pos_from_tf(finger_transform))))
    else:
        radii.append(float(np.linalg.norm(_pos_from_tf(finger_transform))))

    radius_mean = float(np.mean(radii)) if radii else float("nan")
    return radii, radius_mean


def _ensure_csv_schema(csv_path: str):
    """
    Ensures CSV exists with the exact schema.
    If an existing CSV has a different header, it is renamed and a new file is created.
    """
    os.makedirs(os.path.dirname(csv_path), exist_ok=True)

    if not os.path.exists(csv_path):
        with open(csv_path, "w", newline="") as f:
            w = csv.DictWriter(f, fieldnames=CSV_COLUMNS, quoting=csv.QUOTE_ALL)
            w.writeheader()
        return

    # check header matches
    try:
        with open(csv_path, "r", newline="") as f:
            r = csv.reader(f)
            header = next(r, None)
        if header != CSV_COLUMNS:
            ts = datetime.now().strftime("%Y%m%d_%H%M%S")
            new_name = csv_path.replace(".csv", f"_old_{ts}.csv")
            os.rename(csv_path, new_name)
            print(f"[csv] Existing CSV header mismatch. Renamed to: {new_name}")
            with open(csv_path, "w", newline="") as f2:
                w = csv.DictWriter(f2, fieldnames=CSV_COLUMNS, quoting=csv.QUOTE_ALL)
                w.writeheader()
    except Exception as e:
        # if something is weird, rotate it anyway
        ts = datetime.now().strftime("%Y%m%d_%H%M%S")
        new_name = csv_path.replace(".csv", f"_broken_{ts}.csv")
        try:
            os.rename(csv_path, new_name)
            print(f"[csv] Could not validate CSV, renamed to: {new_name} ({e})")
        except Exception:
            print(f"[csv] Could not validate CSV and could not rename it ({e})")
        with open(csv_path, "w", newline="") as f2:
            w = csv.DictWriter(f2, fieldnames=CSV_COLUMNS, quoting=csv.QUOTE_ALL)
            w.writeheader()


def _append_row_csv(row: dict, csv_path: str):
    """
    Appends a single row using a fixed schema and QUOTE_ALL, so commas never break parsing.
    """
    _ensure_csv_schema(csv_path)

    safe_row = {k: row.get(k, "") for k in CSV_COLUMNS}
    with open(csv_path, "a", newline="") as f:
        w = csv.DictWriter(f, fieldnames=CSV_COLUMNS, quoting=csv.QUOTE_ALL)
        w.writerow(safe_row)


def _read_done_ok(csv_path: str):
    """
    Reads completed successful runs: Status == ok.
    Returns a set of (Object, Num_Fingers).
    """
    done = set()
    if not os.path.exists(csv_path):
        return done

    try:
        with open(csv_path, "r", newline="") as f:
            r = csv.DictReader(f)
            for row in r:
                try:
                    if str(row.get("Status", "")).strip() == "ok":
                        obj = str(row.get("Object", ""))
                        nf = int(float(row.get("Num_Fingers", "nan")))
                        done.add((obj, nf))
                except Exception:
                    continue
    except Exception as e:
        print(f"[resume] Could not read CSV for resume: {e}")
    return done


def _patch_voxvol_init_final_only(tendon: FEMTendon):
    """
    Makes enclosed volume computation cheap:
    - compute real volume at t0
    - compute real volume at final frame
    - intermediate volume calls return cached t0 volume fast
    Also disables volume logger and timeseries CSV saving.

    Does NOT change fps/substeps or simulation timing.
    """
    # disable logger writes and storage
    try:
        tendon.vol_logger.log = lambda *args, **kwargs: None
        tendon.vol_logger.to_csv = lambda *args, **kwargs: None
    except Exception:
        pass

    if getattr(tendon, "voxvol", None) is None:
        return

    real_compute = tendon.voxvol.compute
    cache = {"have_t0": False, "vol0": None, "dbg0": None}

    def compute_wrapped(q, solid_tri_sets=None, return_points=False, max_points=40000):
        # final frame call uses return_points=True in your forward.py
        if return_points:
            # compute final but do not generate point clouds (saves time and memory)
            vol, dbg = real_compute(q, solid_tri_sets=solid_tri_sets, return_points=False, max_points=max_points)
            return vol, dbg

        # first call is the t0 compute in forward()
        if not cache["have_t0"]:
            vol, dbg = real_compute(q, solid_tri_sets=solid_tri_sets, return_points=False, max_points=max_points)
            cache["have_t0"] = True
            cache["vol0"] = float(vol)
            cache["dbg0"] = dbg
            return vol, dbg

        # intermediate calls: return cached t0 quickly
        if cache["dbg0"] is None:
            return float(cache["vol0"] or 0.0), {}
        return float(cache["vol0"]), cache["dbg0"]

    tendon.voxvol.compute = compute_wrapped


def run_one_case(
    obj_name: str,
    n_fingers: int,
    device: str,
    csv_path: str,
    # fixed sim params
    finger_len: int,
    finger_rot: float,
    finger_width: float,
    scale: float,
    object_rot,
    object_density: float,
    # pose optimisation
    pose_opt_frames: int,
    pose_opt_iters: int,
    # simulation
    sim_num_frames: int,
    # force optimisation
    force_opt_iters: int,
    force_opt_lr: float,
    force_opt_frames: int,
):
    """
    Runs one (object, finger count) case and appends a row to CSV no matter what.
    Intended to be executed in a fresh process (worker).
    """

    row = {
        "Object": obj_name,
        "Num_Fingers": int(n_fingers),

        "Device": device,
        "PoseOptFrames": int(pose_opt_frames),
        "PoseOptIters": int(pose_opt_iters),
        "SimNumFrames": int(sim_num_frames),
        "ForceOptIters": int(force_opt_iters),
        "ForceOptFrames": int(force_opt_frames),
        "ForceOptLR": float(force_opt_lr),

        "Radius_Mean": np.nan,
        "Radius_List": "",
        "Final_Loss": np.nan,
        "Avg_Force": np.nan,
        "All_Forces": "",
        "Init_VoxVol": np.nan,
        "Final_VoxVol": np.nan,
        "Delta_VoxVol": np.nan,

        "Status": "started",
        "Error": "",
        "Timestamp": datetime.now().isoformat(timespec="seconds"),
    }

    init_finger = None
    tendon = None

    try:
        with wp.ScopedDevice(device):
            print(f"\n=== {obj_name} | fingers={n_fingers} ===")

            # --------------------------
            # A) Pose optimisation (NO fixed_radius)
            # --------------------------
            init_finger = InitializeFingers(
                stage_path="temp_init.usd",
                finger_len=finger_len,
                finger_rot=finger_rot,
                finger_width=finger_width,
                stop_margin=0.0005,
                num_frames=pose_opt_frames,
                iterations=pose_opt_iters,
                scale=scale,
                num_envs=1,
                ycb_object_name=obj_name,
                object_rot=object_rot,
                is_render=False,
                verbose=False,
                is_triangle=False,
                finger_num=n_fingers,
                add_random=False,
                consider_cloth=True,
                # IMPORTANT: no fixed_radius here
            )

            finger_transform, _ = init_finger.get_initial_position()
            if finger_transform is None:
                raise RuntimeError("InitializeFingers returned None finger_transform")

            radii, radius_mean = _extract_radii(finger_transform)
            row["Radius_Mean"] = float(radius_mean)
            row["Radius_List"] = str(radii)
            print(f"   -> Pose done. mean radius={radius_mean:.4f}")

            # Optional memory safety:
            # init_finger can hold GPU memory. We do not need it anymore after extracting finger_transform.
            # This does not change the sim, since finger_transform is the only thing FEMTendon needs.
            try:
                del init_finger
                init_finger = None
                gc.collect()
                torch.cuda.empty_cache()
                torch.cuda.ipc_collect()
            except Exception:
                pass

            # --------------------------
            # B) ONE FEMTendon, like forward.py
            # --------------------------
            tendon = FEMTendon(
                stage_path=None,
                num_frames=sim_num_frames,      # full sim frames, same as your forward run
                verbose=False,
                save_log=False,
                train_iters=1,
                is_render=False,
                object_rot=object_rot,
                ycb_object_name=obj_name,
                object_density=object_density,
                finger_len=finger_len,
                finger_rot=finger_rot,
                finger_width=finger_width,
                scale=scale,
                finger_transform=finger_transform,
                finger_num=n_fingers,
                requires_grad=True,             # like forward.py
                init_finger=None,               # keep None to avoid keeping init pose GPU buffers
                no_cloth=False,
                no_voxvol=False,
            )

            # patch voxvol so it computes only t0 and final
            _patch_voxvol_init_final_only(tendon)

            # --------------------------
            # C) Force optimisation on the SAME tendon
            # --------------------------
            history = tendon.optimize_forces_lbfgs(
                iterations=force_opt_iters,
                learning_rate=force_opt_lr,
                opt_frames=force_opt_frames,
            )

            final_loss = float(history["loss"][-1]) if history and history.get("loss") else np.nan
            final_forces = history["forces"][-1] if history and history.get("forces") else []
            avg_force = float(np.mean(final_forces)) if len(final_forces) else np.nan

            row["Final_Loss"] = float(final_loss)
            row["Avg_Force"] = float(avg_force)
            row["All_Forces"] = str(final_forces)

            print(f"   -> Force opt done. loss={final_loss:.4f} avg_force={avg_force:.2f}")

            # --------------------------
            # D) Run forward once on SAME tendon and read volumes
            # --------------------------
            tendon.forward()

            init_voxvol = float(tendon.init_voxel_volume) if tendon.init_voxel_volume is not None else np.nan
            final_voxvol = float(tendon.last_voxel_volume) if tendon.last_voxel_volume is not None else np.nan

            row["Init_VoxVol"] = init_voxvol
            row["Final_VoxVol"] = final_voxvol
            row["Delta_VoxVol"] = (final_voxvol - init_voxvol) if np.isfinite(init_voxvol) and np.isfinite(final_voxvol) else np.nan

            print(f"   -> VoxVol: init={init_voxvol:.6g} final={final_voxvol:.6g}")

            row["Status"] = "ok"

    except Exception as e:
        row["Status"] = "failed"
        row["Error"] = f"{type(e).__name__}: {e}"
        print(f"   [FAIL] {row['Error']}")

    finally:
        # always write row
        _append_row_csv(row, csv_path)

        # cleanup hard
        try:
            del tendon
        except Exception:
            pass
        try:
            del init_finger
        except Exception:
            pass

        gc.collect()
        try:
            wp.synchronize()
        except Exception:
            pass

        # clear torch caching allocator
        try:
            torch.cuda.empty_cache()
            torch.cuda.ipc_collect()
        except Exception:
            pass

    return row


def run_master_sweep(
    device: str,
    csv_path: str,
    use_subprocess: bool,
    # config
    object_list,
    finger_counts,
    finger_len,
    finger_rot,
    finger_width,
    scale,
    object_rot,
    object_density,
    pose_opt_frames,
    pose_opt_iters,
    sim_num_frames,
    force_opt_iters,
    force_opt_lr,
    force_opt_frames,
):
    _ensure_csv_schema(csv_path)

    done_ok = _read_done_ok(csv_path)
    total = len(object_list) * len(finger_counts)
    print(f"[resume] Found {len(done_ok)} successful runs in CSV")
    print(f"Starting VOX-VOL sweep: {total} runs")

    for obj_name in object_list:
        for n_fingers in finger_counts:
            key = (obj_name, int(n_fingers))
            if key in done_ok:
                print(f"[skip] {obj_name} | fingers={n_fingers} already ok in CSV")
                continue

            if use_subprocess:
                # Run one case in a fresh process to avoid CUDA fragmentation and OOM cascades.
                cmd = [
                    sys.executable,
                    os.path.abspath(__file__),
                    "--worker",
                    "--device", device,
                    "--csv", csv_path,
                    "--object", obj_name,
                    "--fingers", str(int(n_fingers)),
                ]
                print(f"\n[master] launching worker: {obj_name} fingers={n_fingers}")
                p = subprocess.run(cmd)

                # If worker crashed (should not, but can happen on hard CUDA faults),
                # ensure there is at least a failed row in the CSV.
                if p.returncode != 0:
                    print(f"[master] worker returned code {p.returncode}, writing fallback failed row")
                    fallback = {
                        "Object": obj_name,
                        "Num_Fingers": int(n_fingers),
                        "Device": device,
                        "PoseOptFrames": int(pose_opt_frames),
                        "PoseOptIters": int(pose_opt_iters),
                        "SimNumFrames": int(sim_num_frames),
                        "ForceOptIters": int(force_opt_iters),
                        "ForceOptFrames": int(force_opt_frames),
                        "ForceOptLR": float(force_opt_lr),
                        "Radius_Mean": np.nan,
                        "Radius_List": "",
                        "Final_Loss": np.nan,
                        "Avg_Force": np.nan,
                        "All_Forces": "",
                        "Init_VoxVol": np.nan,
                        "Final_VoxVol": np.nan,
                        "Delta_VoxVol": np.nan,
                        "Status": "failed",
                        "Error": f"WorkerCrashed: returncode={p.returncode}",
                        "Timestamp": datetime.now().isoformat(timespec="seconds"),
                    }
                    _append_row_csv(fallback, csv_path)
            else:
                # In-process mode (less safe for long overnight runs on CUDA)
                run_one_case(
                    obj_name=obj_name,
                    n_fingers=int(n_fingers),
                    device=device,
                    csv_path=csv_path,
                    finger_len=finger_len,
                    finger_rot=finger_rot,
                    finger_width=finger_width,
                    scale=scale,
                    object_rot=object_rot,
                    object_density=object_density,
                    pose_opt_frames=pose_opt_frames,
                    pose_opt_iters=pose_opt_iters,
                    sim_num_frames=sim_num_frames,
                    force_opt_iters=force_opt_iters,
                    force_opt_lr=force_opt_lr,
                    force_opt_frames=force_opt_frames,
                )

    print(f"\nSweep complete. Saved: {csv_path}")


def plot_results(csv_path: str, out_dir: str):
    os.makedirs(out_dir, exist_ok=True)

    try:
        df = pd.read_csv(csv_path, quotechar='"')
    except Exception as e:
        print(f"[plot] Could not read CSV for plotting: {e}")
        return

    if "Status" in df.columns:
        df = df[df["Status"].astype(str) == "ok"].copy()

    if df.empty:
        print("[plot] No successful rows to plot.")
        return

    df["Num_Fingers"] = df["Num_Fingers"].astype(int)

    # Final VoxVol vs Fingers per object
    plt.figure(figsize=(10, 6))
    for obj in sorted(df["Object"].unique()):
        sub = df[df["Object"] == obj].sort_values("Num_Fingers")
        plt.plot(sub["Num_Fingers"], sub["Final_VoxVol"], marker="o", label=obj)
    plt.title("Final enclosed volume (voxel)")
    plt.xlabel("Number of fingers")
    plt.ylabel("Final_VoxVol")
    plt.grid(True, alpha=0.3)
    plt.legend(fontsize="small", ncol=2)
    plt.tight_layout()
    out_path = os.path.join(out_dir, "final_voxvol_vs_fingers.png")
    plt.savefig(out_path, dpi=200)
    plt.close()
    print(f"[plot] Saved: {out_path}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--device", type=str, default="cuda")
    parser.add_argument("--csv", type=str, default=os.path.join("sweep_results", "experiment_data_voxvol.csv"))
    parser.add_argument("--out_dir", type=str, default="sweep_results")

    # worker mode
    parser.add_argument("--worker", action="store_true")
    parser.add_argument("--object", type=str, default="")
    parser.add_argument("--fingers", type=int, default=-1)

    # safety toggle
    parser.add_argument("--no_subprocess", action="store_true", help="Run in one process (less safe for long CUDA sweeps).")

    args = parser.parse_args()

    # Fixed sim params (match forward.py usage)
    finger_len = 11
    finger_rot = np.pi / 30
    finger_width = 0.08
    scale = 5.0
    object_rot = wp.quat_rpy(-math.pi / 2, 0.0, 0.0)
    object_density = 2e0

    # Pose optimisation
    pose_opt_frames = 30
    pose_opt_iters = 2000

    # Sim length (this is the same as your forward style run)
    sim_num_frames = 1000

    # Force optimisation
    force_opt_iters = 1
    force_opt_lr = 1.0
    force_opt_frames = 100

    # keep your test and full lists
    object_list = OBJECT_LIST_FULL
    finger_counts = FINGER_COUNTS_FULL

    # Worker: run exactly one case and exit
    if args.worker:
        if not args.object or args.fingers < 0:
            print("[worker] missing --object or --fingers")
            return

        run_one_case(
            obj_name=args.object,
            n_fingers=int(args.fingers),
            device=args.device,
            csv_path=args.csv,
            finger_len=finger_len,
            finger_rot=finger_rot,
            finger_width=finger_width,
            scale=scale,
            object_rot=object_rot,
            object_density=object_density,
            pose_opt_frames=pose_opt_frames,
            pose_opt_iters=pose_opt_iters,
            sim_num_frames=sim_num_frames,
            force_opt_iters=force_opt_iters,
            force_opt_lr=force_opt_lr,
            force_opt_frames=force_opt_frames,
        )
        return

    # Master: sweep
    run_master_sweep(
        device=args.device,
        csv_path=args.csv,
        use_subprocess=(not args.no_subprocess),
        object_list=object_list,
        finger_counts=finger_counts,
        finger_len=finger_len,
        finger_rot=finger_rot,
        finger_width=finger_width,
        scale=scale,
        object_rot=object_rot,
        object_density=object_density,
        pose_opt_frames=pose_opt_frames,
        pose_opt_iters=pose_opt_iters,
        sim_num_frames=sim_num_frames,
        force_opt_iters=force_opt_iters,
        force_opt_lr=force_opt_lr,
        force_opt_frames=force_opt_frames,
    )

    plot_results(args.csv, args.out_dir)


if __name__ == "__main__":
    main()
