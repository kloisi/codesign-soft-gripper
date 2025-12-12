import os
import csv
import time
import math
import numpy as np
import warp as wp

from sim_gen import FEMTendon
from init_pose import InitializeFingers
from quick_viz import quick_visualize


def build_default_scene(object_name="006_mustard_bottle",
                        object_density=2.0,
                        pose_id=5,
                        ood=False):
    """
    Same logic as sim_gen.__main__ to get:
    - object_rot
    - finger_len/rot/width/scale
    - finger_transform from pose_info or InitializeFingers
    """
    finger_len   = 11
    finger_rot   = np.pi / 30.0
    finger_width = 0.08
    scale        = 5.0
    object_rot   = wp.quat_rpy(-math.pi/2, 0.0, 0.0) # same as sim_gen.py

    curr_dir = os.path.dirname(os.path.abspath(__file__))
    finger_transform = None
    init_finger = None

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
        print(f"[stiff_sweep] Loaded pose from {pose_file}, pose_id={pose_id}")
    except Exception as e:
        print(f"[stiff_sweep] Could not load pose file ({e}), falling back to InitializeFingers")
        finger_transform = None

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
        print("[stiff_sweep] Initialized fingers with InitializeFingers")

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




def scale_cloth_stiffness(tendon, factor_ke_ka=1.0, factor_kd=1.0):
    """
    Scale only the cloth triangle stiffness, leaving finger FEM untouched.

    - factor_ke_ka: multiplies tri_ke and tri_ka (stretch + area)
    - factor_kd:    multiplies tri_kd (internal damping)
    """
    builder = tendon.builder
    model   = tendon.model

    cloth_ids = getattr(builder, "cloth_particle_ids", None)
    if cloth_ids is None:
        print("[scale_cloth_stiffness] No cloth_particle_ids found on builder; nothing to scale.")
        return

    cloth_ids = np.asarray(cloth_ids, dtype=np.int32)

    # triangles (N, 3) with vertex indices
    tri_indices = model.tri_indices.numpy() # shape (n_tris, 3)
    # mask for triangles whose *all* vertices are cloth vertices
    mask = np.all(np.isin(tri_indices, cloth_ids), axis=1)

    if not np.any(mask):
        print("[scale_cloth_stiffness] No triangles fully on cloth; nothing to scale.")
        return

    tri_mats = model.tri_materials.numpy() # shape (n_tris, 5) typically: [ke, ka, kd, drag, lift]

    # scale only cloth rows:
    tri_mats[mask, 0] *= factor_ke_ka # tri_ke
    tri_mats[mask, 1] *= factor_ke_ka # tri_ka
    tri_mats[mask, 2] *= factor_kd # tri_kd (damping)

    model.tri_materials = wp.array(tri_mats, dtype=wp.float32)
    print(
        f"[scale_cloth_stiffness] scaled {mask.sum()} cloth tris: "
        f"ke,ka x {factor_ke_ka}, kd x {factor_kd}"
    )


def scale_cloth_mass(tendon, mass_factor: float):
    """
    Scale only the cloth vertex mass, leaving fingers + object unchanged.
    """
    builder = tendon.builder
    model   = tendon.model

    cloth_ids = getattr(builder, "cloth_particle_ids", None)
    if cloth_ids is None:
        print("[scale_cloth_mass] No cloth_particle_ids found; nothing to scale.")
        return

    inv_m = model.particle_inv_mass.numpy()  # 1/m

    for gid in cloth_ids:
        if inv_m[gid] > 0.0:
            m = 1.0 / inv_m[gid]
            m *= mass_factor
            inv_m[gid] = 1.0 / m

    model.particle_inv_mass = wp.array(inv_m, dtype=wp.float32)
    print(f"[scale_cloth_mass] scaled {len(cloth_ids)} cloth vertices by mass_factor={mass_factor}")



def run_stiffness_sweep(
    device="cuda:0",
    log_filename="stiffness_sweep_results.csv",
    make_viz=True,
):
    # fixed time stepping
    fps          = 5000
    sim_substeps = 50
    num_frames   = 5000

    cloth_scales  = [1.0, 0.5, 0.1, 0.05, 0.01] # keep internal stiffness baseline
    mass_scales   = [0.5, 1.0, 2.0] # lighter, baseline, heavier
    damp_scales   = [1.0, 2.0] # baseline and more damping

    curr_dir = os.path.dirname(os.path.abspath(__file__))
    log_path = os.path.join(curr_dir, log_filename)

    file_exists = os.path.exists(log_path)
    log_file = open(log_path, "a", newline="")
    writer = csv.writer(log_file)

    if not file_exists:
        writer.writerow([
            "timestamp",
            "cloth_stiffness_scale",
            "fps",
            "sim_substeps",
            "dt",
            "num_frames",
            "total_steps",
            "real_time_per_step_ms",
            "num_ok",
            "phys_ok",
            "diff_norm",
            "kernel_seed",
        ])

    viz_dir = os.path.join(curr_dir, "stiffness_sweep_viz")
    if make_viz:
        os.makedirs(viz_dir, exist_ok=True)

    with wp.ScopedDevice(device):
        scene_kwargs = build_default_scene(
            object_name="006_mustard_bottle",
            object_density=2.0,
            pose_id=5,
            ood=False,
        )

        for stiff_scale in cloth_scales:
            for mass_scale in mass_scales:
                for damp_scale in damp_scales:
                    
                    print(f"\n=== cloth stiffness scale = {stiff_scale} ===")

                    kernel_seed = int(np.random.randint(0, 1_000_000))

                    tendon = FEMTendon(
                        # stage_path=None,
                        num_frames=num_frames,
                        fps=fps,
                        sim_substeps=sim_substeps,
                        verbose=False,
                        save_log=False,
                        is_render=False,
                        use_graph=False,
                        kernel_seed=kernel_seed,
                        **scene_kwargs,
                    )

                    # only cloth stiffness changes here
                    scale_cloth_mass(tendon, mass_scale)
                    damp_boost = min(4.0, 1.0 / np.sqrt(stiff_scale))  # cap so it doesn’t explode
                    scale_cloth_stiffness(tendon, factor_ke_ka=stiff_scale, factor_kd=damp_boost)

                    t0 = time.perf_counter()
                    tendon.forward()
                    elapsed = time.perf_counter() - t0

                    # same checks as dt_sweep for now: num_ok, phys_ok, diff_norm
                    total_steps = tendon.num_frames * tendon.sim_substeps
                    real_time_per_step = elapsed / total_steps

                    # numeric sanity
                    q_end = tendon.states[-1].particle_q.numpy()
                    num_ok = np.isfinite(q_end).all() and np.abs(q_end).max() < 2.0 # slightly relaxed

                    # physical sanity: object vs finger translation
                    end_particle_q = tendon.states[-1].particle_q.numpy()[0, :]
                    particle_trans_q = end_particle_q[0:3] - tendon.init_particle_q[0:3]
                    object_trans_q = tendon.object_q.numpy()[0, 0:3] - tendon.init_body_q[0:3]
                    diff_q = object_trans_q - particle_trans_q
                    diff_norm = np.linalg.norm(diff_q)
                    phys_ok = diff_norm < 0.5

                    dt = tendon.sim_dt

                    print(
                        f"[stiff_sweep] cloth_stiffness_scale={stiff_scale:+.2f}, dt={dt:.2e}, "
                        f"steps={total_steps}, "
                        f"t/step={real_time_per_step*1e3:.3f} ms, "
                        f"num_ok={num_ok}, phys_ok={phys_ok}, diff_norm={diff_norm:.3f}"
                    )

                    writer.writerow([
                        time.strftime("%Y-%m-%d %H:%M:%S"),
                        stiff_scale,
                        fps,
                        sim_substeps,
                        f"{dt:.6e}",
                        num_frames,
                        total_steps,
                        real_time_per_step * 1e3,
                        int(num_ok),
                        int(phys_ok),
                        diff_norm,
                        kernel_seed,
                    ])
                    log_file.flush()

                    if make_viz:
                        vid_name = f"viz_cloth_stiffness_scale_{stiff_scale:+.2f}.mp4".replace("+", "p").replace("-", "m")
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
    run_stiffness_sweep(device="cuda:0", make_viz=True)
