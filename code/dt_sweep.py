import time
import os
import csv
import math
import numpy as np
import warp as wp
from sim_gen import FEMTendon
from quick_viz import quick_visualize
from init_pose import InitializeFingers


def build_default_scene(object_name="006_mustard_bottle",
                        object_density=2.0,
                        ood=False):
    """
    Reproduce the same setup as sim_gen.py __main__:
    - same object_rot (rotated mustard bottle),
    - same finger_len / finger_rot / finger_width / scale,
    - same finger_transform (from pose file if possible, otherwise InitializeFingers).
    """

    finger_len   = 11
    finger_rot   = np.pi / 30.0
    finger_width = 0.08
    scale        = 5.0
    object_rot   = wp.quat_rpy(-math.pi/2, 0.0, 0.0)   # same as sim_gen.py
    pose_id      = 5                                   # same default pose

    curr_dir = os.path.dirname(os.path.abspath(__file__))

    finger_transform = None
    init_finger = None

    # Try to load the precomputed pose, same path as sim_gen.py
    pose_file = os.path.join(curr_dir, "..", "pose_info", "init_opt", f"{object_name}.npz")
    try:
        data = np.load(pose_file, allow_pickle=True)
        finger_transform_list = data["finger_transform"]
        this_trans = finger_transform_list[pose_id]

        if np.linalg.norm(this_trans) > 0.0:
            finger_transform = [
                wp.transform(this_trans[0, :3], this_trans[0, 3:]),
                wp.transform(this_trans[1, :3], this_trans[1, 3:]),
            ]
        print(f"[dt_sweep] Loaded pose from {pose_file}, pose_id={pose_id}")
    except Exception as e:
        print(f"[dt_sweep] Could not load pose file ({e}), falling back to InitializeFingers")
        finger_transform = None

    # If no pose file / invalid, compute initial transform exactly like sim_gen.py
    if finger_transform is None:
        init_finger = InitializeFingers(
            finger_len=finger_len,
            finger_rot=finger_rot,
            finger_width=finger_width,
            stop_margin=0.0005,
            scale=scale,
            num_envs=1,
            ycb_object_name=object_name,
            object_rot=object_rot,
            num_frames=30,
            iterations=15000,
            is_render=False,
            is_triangle=True,
            pose_id=pose_id,
            is_ood=ood,
        )
        finger_transform, jq = init_finger.get_initial_position()
        print("[dt_sweep] Initialized fingers with InitializeFingers")

    return dict(
        object_rot=object_rot,
        ycb_object_name=object_name,
        object_density=object_density,
        finger_len=finger_len,
        finger_rot=finger_rot,
        finger_width=finger_width,
        scale=scale,
        finger_transform=finger_transform,
        init_finger=init_finger,
    )



def run_dt_sweep(device="cuda:0", log_filename="dt_sweep_results.csv", make_viz=True,):

    fps_list = [5000, 4000, 3000, 2000, 1000, 500, 100] # [5000, 2500, 1000, 500, 50]
    sim_substeps_list = [250, 100, 75, 50, 25, 10]  # [200, 100, 50, 20, 10]

    # choose how long we want to simulate in physical time
    # total_sim_time = num_frames / fps  (seconds)
    num_frames = 5000  # e.g. 1000 frames at 1000 fps -> 1 s

    # where to store the log (same folder as this file)
    curr_dir = os.path.dirname(os.path.abspath(__file__))
    log_path = os.path.join(curr_dir, log_filename)

    # open file once in append mode
    file_exists = os.path.exists(log_path)
    log_file = open(log_path, "a", newline="")
    writer = csv.writer(log_file)

    # write header only if file did not exist
    if not file_exists:
        writer.writerow([
            "timestamp",
            "fps",
            "sim_substeps",
            "iteration",
            "dt",
            "num_frames",
            "total_steps",
            "real_time_per_step_ms",
            "num_ok",
            "phys_ok",
            "diff_norm",
            "kernel_seed",
        ])

    # directory for videos
    viz_dir = os.path.join(curr_dir, "dt_sweep_viz")
    if make_viz:
        os.makedirs(viz_dir, exist_ok=True)

    with wp.ScopedDevice(device):

        # build scene once (object_rot, finger_transform, etc.)
        scene_kwargs = build_default_scene(
            object_name="006_mustard_bottle",
            object_density=2.0,
            ood=False,
        )

        T_target = 1.0  # 1 second simulated time
        for fps in fps_list:
            num_frames = int(T_target * fps)  # keep T_sim same

            for sub in sim_substeps_list:
                for it in range(1, 5+1):

                    print("\n === Testing fps =", fps, "=== Testing sim_substeps =", sub, "===", "iteration", it,"===")

                    # randomize seed so we can later see which run exploded
                    kernel_seed = int(np.random.randint(0, 1_000_000))

                    tendon = FEMTendon(
                        # stage_path=None, # no USD for this experiment
                        num_frames=num_frames,
                        fps=fps,
                        sim_substeps=sub,
                        verbose=False,
                        save_log=False,
                        is_render=False,
                        use_graph=False,
                        kernel_seed=kernel_seed,
                        **scene_kwargs, # same steup as sim_gen
                    )

                    t0 = time.perf_counter()
                    tendon.forward()
                    elapsed = time.perf_counter() - t0 # real time on machine

                    total_steps = tendon.num_frames * tendon.sim_substeps # total integration steps done
                    real_time_per_step = elapsed / total_steps # real time seconds per integration step

                    # ----- numeric stability check (no NaNs, positions not huge) -----
                    q_end = tendon.states[-1].particle_q.numpy()
                    num_ok = np.isfinite(q_end).all() and np.abs(q_end).max() < 1.5

                    # ----- physical sanity: object vs finger displacement difference -----
                    end_particle_q = tendon.states[-1].particle_q.numpy()[0, :]
                    particle_trans_q = end_particle_q[0:3] - tendon.init_particle_q[0:3]
                    object_trans_q = tendon.object_q.numpy()[0, 0:3] - tendon.init_body_q[0:3]
                    diff_q = object_trans_q - particle_trans_q
                    diff_norm = np.linalg.norm(diff_q)
                    phys_ok = diff_norm < 0.5  # choose threshold; <0.5m here

                    print(
                        f"[dt sweep] dt={tendon.sim_dt:.2e}, "
                        f"steps={total_steps}, "
                        f"real_time_per_step={real_time_per_step*1e3:.3f} ms, "
                        f"num_ok={num_ok}, phys_ok={phys_ok}, diff_norm={diff_norm:.3f}"
                    )

                    # write to CSV
                    writer.writerow([
                        time.strftime("%Y-%m-%d %H:%M:%S"),
                        fps,
                        sub,
                        it,
                        f"{tendon.sim_dt:.6e}",
                        num_frames,
                        total_steps,
                        real_time_per_step * 1e3, # ms
                        int(num_ok),
                        int(phys_ok),
                        diff_norm,
                        kernel_seed,
                    ])

                    log_file.flush() # make sure each run is saved to disk

                    # optional: make a video for this run
                    if make_viz:
                        vid_name = f"viz_heavy_cloth_fps{fps}_sub{sub}_it{it}.mp4"
                        save_path = os.path.join(viz_dir, vid_name)
                        print("  making viz:", save_path)

                        quick_visualize(
                            tendon,
                            stride=50, # use larger stride to keep files small
                            interval=30, # ms between video frames
                            save_path=save_path,
                            elev=30,
                            azim=45,
                        )


        log_file.close()
        print(f"\nLogged results to: {log_path}")


if __name__ == "__main__":
    # for CPU: run_dt_sweep(device="cpu")
    run_dt_sweep(device="cuda:0")