# forward.py

import math
import os
import time
import datetime
import torch
import numpy as np
import warp as wp
import warp.sim
import warp.sim.render
import warp.optim
from enum import Enum
import pandas as pd
import matplotlib.pyplot as plt

import utils
from object_loader import ObjectLoader
from integrator_euler_fem import FEMIntegrator
from tendon_model import TendonModelBuilder, TendonRenderer, TendonHolder
from init_pose import InitializeFingers
from force_optimizer import FEMForceOptimization

from matplotlib.animation import FuncAnimation, FFMpegWriter
from scipy.spatial.transform import Rotation as R

from quick_viz import quick_visualize, plot_last_frame_with_voxels

from enclosed_volume_voxel import (
    VoxelFillConfig,
    VoxelVolumeEstimator,
    prepare_vox_topology_from_model,
)


@wp.kernel
def update_materials(
    # frame_id: wp.int32,
    log_K: wp.array(dtype=wp.float32),
    opt_v: wp.array(dtype=wp.float32),
    block_ids: wp.array(dtype=wp.int32),
    tet_materials: wp.array2d(dtype=wp.float32)):
    tid = wp.tid()
    this_log_k = log_K[block_ids[tid]]
    K = wp.exp(this_log_k)
    v = opt_v[0]
    k_mu = K / (2.0 * (1.0 + v))
    k_lambda = K * v / ((1.0 + v) * (1.0 - 2.0 * v))

    tet_materials[tid, 0] = k_mu
    tet_materials[tid, 1] = k_lambda


class FEMTendon:
    def __init__(self, 
                 stage_path="sim", num_frames=30, 
                 verbose=True, save_log=False,
                 train_iters=100,
                 log_prefix="", 
                 is_render=True,
                 use_graph=False,
                 kernel_seed=42,
                 object_rot=wp.quat_identity(),
                 ycb_object_name='',
                 object_density=1e1,
                 finger_len=9, finger_rot=np.pi/9, finger_width=0.08, scale=4.0, finger_transform=None,
                 finger_num=6,
                 requires_grad=True,
                 init_finger=None,
                 no_cloth=False,
                 no_voxvol=False):
        self.verbose = verbose
        self.save_log = save_log
        fps = 4000
        self.frame_dt = 1.0 / fps
        self.num_frames = num_frames

        self.sim_substeps = 100
        self.sim_dt = self.frame_dt / self.sim_substeps
        self.sim_time = 0.0
        self.render_time = 0.0
        self.iter = 0
        self.train_iters = train_iters
        self.requires_grad = requires_grad

        self.obj_loader = ObjectLoader()
        self.finger_len = finger_len # need to be odd number
        self.finger_num = finger_num
        self.finger_rot = finger_rot
        self.finger_width = finger_width
        self.finger_transform = finger_transform
        self.init_finger = init_finger
        self.scale = scale
        self.obj_name = 'ycb'
        self.ycb_object_name = ycb_object_name
        self.object_density = object_density
        self.has_object = bool(self.ycb_object_name)

        self.no_cloth = bool(no_cloth)
        self.no_voxvol = bool(no_voxvol)

        # mw_added, todo: check again if these are good
        self.init_voxel_volume = None
        self.last_voxel_volume = None
        self.last_voxel_debug = None
        self._last_voxel_q = None

        self.curr_dir = os.path.dirname(os.path.realpath(__file__))
        if stage_path is None: # so we can run sweeps without rendering
            self.stage_path = None
        else:
            self.stage_path = self.curr_dir + "/../output/" + stage_path + "_" + f"{ycb_object_name}_{log_prefix}_frame{num_frames}" + ".usd"

        save_dir = self.curr_dir + "/../data_gen/" + f"{ycb_object_name}_frame{num_frames}/rand/"
        if save_log and (not os.path.exists(save_dir)):
            os.makedirs(save_dir)
            print(f"Directory '{save_dir}' created.")
        self.save_dir = save_dir

        self.builder = TendonModelBuilder()
        utils.load_object(self.builder, self.obj_loader,
                          object=self.obj_name,
                          ycb_object_name=self.ycb_object_name,
                          obj_rot=object_rot,
                          scale=self.scale,
                          use_simple_mesh=False,
                          is_fix=True,
                          density=object_density,
                          )
        self.builder.init_builder_tendon_variables(self.finger_num, self.finger_len, self.scale, self.requires_grad)

        self.builder.build_fem_model(
            finger_width=self.finger_width,
            finger_rot=self.finger_rot,
            obj_loader=self.obj_loader,
            finger_transform=self.finger_transform,
            is_triangle=True,
            add_connecting_cloth= not self.no_cloth,
            add_drop_cloth=False, # add cloth for dropping test
            )
        self.model = self.builder.model


        # --- MW_ADDED, for enclosed volume ---
        self.voxvol = None
        if (not self.no_voxvol) and (not self.no_cloth) and self.has_object:
            cloth_tris, solid_tris, rim_ids = prepare_vox_topology_from_model(self.model)

            cfg = VoxelFillConfig(
                voxel_size=0.002 * float(self.scale),
                pad_vox=3,
                cloth_thickness_vox=1,
                solid_thickness_vox=1,
                lid_thickness_vox=1,
                sample_step_factor=0.5,
            )

            self.voxvol = VoxelVolumeEstimator(
                cloth_tri_indices=cloth_tris,
                solid_tri_indices=solid_tris,
                rim_ids=rim_ids,
                cfg=cfg,
            )


        self.control = self.model.control(requires_grad=self.requires_grad)

        # for quick viz of the YCB object
        self.obj_local_pts = None
        if hasattr(self.obj_loader, "mesh"):
            verts = np.asarray(self.obj_loader.mesh.vertices, dtype=np.float32)
            verts = verts - verts.mean(axis=0, keepdims=True)   # same centering as shift_vs
            verts = verts * self.scale                          # same scaling as Warp shape
            verts = verts[::2]                                 # subsample for speed
            self.obj_local_pts = verts


        # store cloth ids if present
        self.cloth_ids = None
        if hasattr(self.model, "cloth_particle_ids") and self.model.cloth_particle_ids is not None:
            try:
                self.cloth_ids = self.model.cloth_particle_ids.numpy()
            except Exception:
                self.cloth_ids = None

        self.tendon_holder = TendonHolder(self.model, self.control)
        self.integrator = FEMIntegrator()

        self.log_K_warp = None
        self.kernel_seed = kernel_seed

        self.init_tendons()
        self.init_materials()

        # allocate sim states
        self.states = []
        for i in range(self.sim_substeps * self.num_frames + 1):
            self.states.append(self.model.state(requires_grad=self.requires_grad))
        self.init_particle_q = self.states[0].particle_q.numpy()[0, :]
        if self.has_object:
            self.init_body_q = self.states[0].body_q.numpy()[0, :]
            self.object_body_f = self.states[0].body_f
            self.object_q = self.states[0].body_q
        else:
            self.init_body_q = None
            self.object_body_f = None
            self.object_q = None

        if self.stage_path and is_render:
            self.renderer = TendonRenderer(self.model, self.stage_path, scaling=1.0)
        else:
            self.renderer = None

        self.use_cuda_graph = False
        if use_graph:
            self.use_cuda_graph = wp.get_device().is_cuda
        if self.use_cuda_graph:
            print("Creating CUDA graph...")
            with wp.ScopedCapture() as capture:
                self.forward()
            self.graph = capture.graph

        self.vol_logger = VolumeLogger()
        self._vox_calibrated = False


    def init_tendons(self):
        finger_waypoint_num = [len(self.builder.waypoints[i]) for i in range(self.finger_num)]
        self.tendon_holder.finger_len = self.finger_len
        self.tendon_holder.finger_num = self.finger_num
        
        # waypoints related
        self.tendon_holder.finger_waypoint_num= wp.array(
            np.hstack([np.ones(finger_waypoint_num[i])*i for i in range(self.finger_num)]), 
            dtype=wp.int32, requires_grad=False)
        self.tendon_holder.waypoints = wp.array(
            np.array(self.builder.waypoint_ids).flatten(), 
            dtype=wp.int32, requires_grad=False)
        self.tendon_holder.waypoint_pair_ids = wp.array(
            np.array(self.builder.waypoint_pair_ids).flatten(), 
            dtype=wp.int32, requires_grad=False)
        self.tendon_holder.tendon_tri_indices = wp.array(
            np.array(self.builder.waypoints_tri_indices).flatten(), 
            dtype=wp.int32, requires_grad=False)
        
        # force related
        self.tendon_holder.surface_normals = wp.array(
            np.zeros([len(self.tendon_holder.waypoints) * self.finger_num, 3]), 
            dtype=wp.vec3, requires_grad=self.requires_grad)
        self.success_flag = wp.array([1], dtype=wp.int32, requires_grad=False)

        # control related
        self.control.finger_num = self.finger_num
        self.control.waypoint_forces = wp.array(np.zeros([len(self.tendon_holder.waypoints), 3]), dtype=wp.vec3, requires_grad=self.requires_grad)
        self.control.waypoint_ids = self.tendon_holder.waypoints
        # target vel related
        self.control.vel_values = wp.array(
            np.zeros([1, 3]),
            dtype=wp.vec3, requires_grad=self.requires_grad)
        self.tendon_forces = wp.array([100.0]*self.finger_num, dtype=wp.float32, requires_grad=self.requires_grad)

        self.tendon_holder.init_tendon_variables(requires_grad=self.requires_grad)
    
    def init_materials(self):
        # print("init_K:", self.builder.init_K)
        
        self.tet_block_ids = wp.array(self.builder.tet_block_ids, dtype=wp.int32, requires_grad=False)
        self.block_num = np.max(np.array(self.builder.tet_block_ids)) + 1
        self.finger_back_ids = wp.array(self.builder.finger_back_ids, dtype=wp.int32, requires_grad=False)
        self.all_ids = wp.array(np.arange(len(self.model.particle_q)), dtype=wp.int32, requires_grad=False)

        # self.log_K_warp = wp.from_numpy(np.zeros(len(self.model.tet_materials)) + np.log(self.builder.init_K), dtype=wp.float32, requires_grad=self.requires_grad)
        K0 = self.builder.init_K  # e.g. 2.0e6 from add_finger
        self.log_K_warp = wp.from_numpy(
            np.zeros(self.block_num) + np.log(K0),
            dtype=wp.float32,
            requires_grad=self.requires_grad,
        )

        self.v = wp.array([self.builder.init_v], dtype=wp.float32, requires_grad=self.requires_grad)

        wp.launch(update_materials,
                dim=len(self.model.tet_materials),
                inputs=[self.log_K_warp, self.v, self.tet_block_ids],
                outputs=[self.model.tet_materials])

    def forward(self):
        wp.launch(update_materials,
                dim=len(self.model.tet_materials),
                inputs=[self.log_K_warp, self.v,
                        self.tet_block_ids],
                outputs=[self.model.tet_materials])
        

        # MW_ADDED, for enclosed volume, one-time rim calibration from q0
        if (self.voxvol is not None) and (not self.no_cloth) and self.has_object and (not self._vox_calibrated):
            q0 = self.states[0].particle_q.numpy()
            if q0.ndim == 3:
                q0 = q0[0]
            self.voxvol.set_open_rims(q0)

            self._vox_calibrated = True

        # NEW: voxel volume at t0 (initial state)
        if (self.voxvol is not None) and (not self.no_cloth) and self.has_object:
            q0 = self.states[0].particle_q.numpy()
            if q0.ndim == 3:
                q0 = q0[0]
            vol0, dbg0 = self.voxvol.compute(q0, return_points=False)
            self.init_voxel_volume = float(vol0)
            if self.verbose:
                print(
                    f"[vox] t0 vol={self.init_voxel_volume:.6g} "
                    f"shape={dbg0.get('shape', None)} enclosed_vox={dbg0.get('enclosed_voxels', None)}"
                )

        # compute volume every N frames
        vox_every = 100 # compute volume every 10 frames
        vox_every = max(1, vox_every)

        for frame in range(self.num_frames):

            for i in range(self.sim_substeps):
                index = i + frame * self.sim_substeps
                self.states[index].clear_forces()
                self.tendon_holder.reset()
                if i % 1 == 0: # default was % 20, (computing collisions every 20 substeps)
                    wp.sim.collide(self.model, self.states[index])
                
                self.control.update_target_vel(frame)

                force = self.tendon_forces

                self.tendon_holder.apply_force(force, self.states[index].particle_q, self.success_flag)
                self.integrator.simulate(self.model, self.states[index], self.states[index+1], self.sim_dt, self.control)

                self.object_body_f = self.states[index].body_f

            if frame == 0 or frame % (self.num_frames / 5) == 0 or frame == (self.num_frames):
                print(f"frame {frame} / {self.num_frames}: body_f:", self.object_body_f.numpy().flatten())

            # MW_ADDED, for enclosed volume
            s_end = self.states[(frame + 1) * self.sim_substeps] # after substeps, use state at end of frame

            is_last_frame = (frame == self.num_frames - 1)
            do_vol = ((frame % vox_every) == 0) or is_last_frame

            if (self.voxvol is not None) and (not self.no_cloth) and self.has_object and do_vol:

                q_np = s_end.particle_q.numpy()
                if q_np.ndim == 3:
                    q_np = q_np[0]
                elif q_np.ndim == 1:
                    q_np = q_np.reshape(-1, 3)

                vol_vox, dbg = self.voxvol.compute(q_np, return_points=is_last_frame)
                self.vol_logger.log(
                    step=(frame + 1) * self.sim_substeps - 1,
                    sim_time=(frame + 1) * self.frame_dt, # true end-of-frame time
                    frame=frame,
                    substep=self.sim_substeps - 1,
                    vol=vol_vox,
                    dbg=dbg,
                )
                if is_last_frame:
                    self.last_voxel_volume = float(vol_vox)
                    self.last_voxel_debug = dbg
                    self._last_voxel_q = q_np   # optional, for rim range prints


        self.object_q = self.states[-1].body_q
        self.object_body_f = self.states[-1].body_f # body force from last state after integration


        # MW_ADDED, for enclosed volume, save log and print
        if self.voxvol is not None:
            obj = self.ycb_object_name if self.ycb_object_name else "noobj"
            self.vol_logger.to_csv(f"logs/volume_timeseries_{obj}_f{self.finger_num}.csv")

            if self.verbose and (self.last_voxel_volume is not None):
                dbg = self.last_voxel_debug
                q_np = self._last_voxel_q
                print(f"[vox] vol={self.last_voxel_volume:.6g} shape={dbg['shape']} enclosed_vox={dbg['enclosed_voxels']}")
                dv = self.last_voxel_volume - self.init_voxel_volume
                print(f"[vox] t0={self.init_voxel_volume:.6g} t_end={self.last_voxel_volume:.6g} delta={dv:.6g}")
                print("[voxdbg]",
                    "voxel=", dbg["voxel_size"],
                    "y_bottom=", dbg["y_bottom"],
                    "y_top=", dbg["y_top"],
                    "iy_lid=", dbg["iy_lid"],
                    "blocked=", dbg["blocked_voxels"])
                print("[voxpts] blocked_pts", dbg["blocked_pts"].shape, "enclosed_pts", dbg["enclosed_pts"].shape)
                print("[rimT] top ids:", len(self.voxvol.top_rim_ids), "bottom ids:", len(self.voxvol.bottom_rim_ids))
                print("[rimT] bot_yT range:", q_np[self.voxvol.bottom_rim_ids,1].min(), q_np[self.voxvol.bottom_rim_ids,1].max())
                print("[rimT] top_yT range:", q_np[self.voxvol.top_rim_ids,1].min(), q_np[self.voxvol.top_rim_ids,1].max())

            if self.verbose and (self.last_voxel_debug is not None):
                plot_last_frame_with_voxels(self, self.last_voxel_debug)

    

    def render(self):
        if self.renderer is None:
            return

        with wp.ScopedTimer("render", print=False):
            for i in range(self.num_frames + 1):
                self.renderer.begin_frame(self.render_time)
                self.renderer.render(
                    self.states[i * self.sim_substeps], 
                    np.array(self.builder.waypoint_ids).flatten(), 
                    force_scale=0.1)
                self.renderer.end_frame()
                self.render_time += self.frame_dt


    def optimize_forces_lbfgs(self, iterations=10, learning_rate=1.0, opt_frames=10):
        """
        Safer numerical L BFGS:
        - Perturb forces in force space (delta_force), not latent space
        - Chain rule to get gradients wrt latents
        - Strong Wolfe line search for stability
        """
        max_force = 100.0
        delta_force = 1.0

        # This optimisation path assumes CUDA because FEMForceOptimization.run uses torch tensors on cuda.
        torch_device = "cuda"

        # Initialise latents from current forces via inverse sigmoid
        start_vals = self.tendon_forces.numpy() / max_force
        start_vals = np.clip(start_vals, 0.01, 0.99)
        start_latents = np.log(start_vals / (1.0 - start_vals))

        latents_param = torch.tensor(
            start_latents, dtype=torch.float32, device=torch_device, requires_grad=True
        )
        num_tendons = len(latents_param)

        optimizer = torch.optim.LBFGS(
            [latents_param],
            lr=learning_rate,
            max_iter=5,
            history_size=10,
            line_search_fn="strong_wolfe",
        )

        history = {"loss": [], "forces": []}

        def closure():
            optimizer.zero_grad()

            # Forward pass
            current_forces_pct = torch.sigmoid(latents_param)
            current_forces = max_force * current_forces_pct
            forces_np = current_forces.detach().cpu().numpy()

            loss_base = FEMForceOptimization.run(
                torch.tensor(forces_np, device=torch_device),
                self.model,
                self.states,
                self.integrator,
                self.sim_dt,
                self.control,
                self.tendon_holder,
                self.sim_substeps,
                opt_frames,
                self.builder.finger_vertex_ranges,
            )

            history["loss"].append(float(loss_base))
            history["forces"].append(forces_np.tolist())

            # Allow “evaluation only” runs
            if learning_rate == 0.0:
                latents_param.grad = torch.zeros_like(latents_param)
                return torch.tensor(loss_base, device=torch_device)

            # Numerical gradients wrt FORCE
            grads_wrt_force = np.zeros(num_tendons, dtype=np.float32)
            for t in range(num_tendons):
                test_forces_np = forces_np.copy()
                test_forces_np[t] += delta_force

                loss_new = FEMForceOptimization.run(
                    torch.tensor(test_forces_np, device=torch_device),
                    self.model,
                    self.states,
                    self.integrator,
                    self.sim_dt,
                    self.control,
                    self.tendon_holder,
                    self.sim_substeps,
                    opt_frames,
                    self.builder.finger_vertex_ranges,
                )

                grads_wrt_force[t] = (loss_new - loss_base) / delta_force

            # Chain rule: dLoss/dLatent = dLoss/dForce * dForce/dLatent
            sig = current_forces_pct.detach().cpu().numpy()
            d_force_d_latent = max_force * sig * (1.0 - sig)
            final_grads = grads_wrt_force * d_force_d_latent

            latents_param.grad = torch.from_numpy(final_grads).to(device=torch_device)

            print(f"   L BFGS Loss: {loss_base:.4f} | Grads (Latent): {final_grads}")
            return torch.tensor(loss_base, device=torch_device)

        for i in range(iterations):
            print(f"Iter {i}:")
            optimizer.step(closure)

        final_forces = max_force * torch.sigmoid(latents_param)
        self.tendon_forces = wp.from_torch(final_forces.detach())
        print(f"--- Done. Final Forces: {final_forces.detach().cpu().numpy()} ---")
        return history

    
    def plot_single_run(self, history, method_name="Optimization", save_dir="logs"):
        """
        Plots loss and forces for a single optimization run and saves data to CSV.
        
        Args:
            history: Dict containing 'loss' (list) and 'forces' (list of lists).
            method_name: Label for the plots and filename (e.g., 'Adam', 'SGD').
            save_dir: Folder to save outputs.
        """
        if not os.path.exists(save_dir):
            os.makedirs(save_dir)

        # --- 1. Save Data to CSV ---
        data = []
        num_iters = len(history['loss'])
        num_tendons = len(history['forces'][0])

        for i in range(num_iters):
            row = {
                "Iteration": i,
                "Loss": history['loss'][i]
            }
            for t in range(num_tendons):
                row[f"Force_{t}"] = history['forces'][i][t]
            data.append(row)

        df = pd.DataFrame(data)
        csv_filename = f"{method_name}_results.csv"
        df.to_csv(os.path.join(save_dir, csv_filename), index=False)
        print(f"[{method_name}] Data saved to {csv_filename}")

        # --- 2. Generate Plots ---
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))
        
        # Plot A: Loss
        ax1.plot(df["Iteration"], df["Loss"], 'b-o', linewidth=2, label="Loss")
        ax1.set_title(f"{method_name}: Loss Convergence")
        ax1.set_xlabel("Iteration")
        ax1.set_ylabel("Total Loss")
        ax1.grid(True, linestyle='--', alpha=0.7)
        ax1.legend()

        # Plot B: Forces
        # We plot every tendon individually
        colors = plt.cm.jet(np.linspace(0, 1, num_tendons))
        for t in range(num_tendons):
            ax2.plot(df["Iteration"], df[f"Force_{t}"], 
                    label=f"Tendon {t}", color=colors[t], linewidth=1.5, alpha=0.8)

        ax2.set_title(f"{method_name}: Force Trajectories")
        ax2.set_xlabel("Iteration")
        ax2.set_ylabel("Force (N)")
        ax2.grid(True, linestyle='--', alpha=0.7)
        
        # Handle legend placement if there are many tendons
        if num_tendons <= 5:
            ax2.legend(loc='best')
        else:
            ax2.legend(loc='upper right', bbox_to_anchor=(1.3, 1.0), fontsize='small')

        plt.tight_layout()
        plot_filename = f"{method_name}_plot.png"
        plt.savefig(os.path.join(save_dir, plot_filename), dpi=150)
        print(f"[{method_name}] Plots saved to {plot_filename}")
        plt.close(fig)



def print_mass_breakdown(tendon):
    """ADDED"""

    m_model = tendon.model
    print("\nMASSES IN SCENE:")
    # rigid body (YCB, coral, ...)
    if m_model.body_count:
        body_masses = m_model.body_mass.numpy()
        for i, m in enumerate(body_masses):
            print(f" body {i+1} mass: {m:.6f} kg")
    else:
        print("No rigid bodies in model.")

    # particle masses (fingers + cloth)
    inv_m = m_model.particle_inv_mass.numpy() # 1/mass
    masses = np.zeros_like(inv_m)
    mask = inv_m > 0.0
    masses[mask] = 1.0 / inv_m[mask]

    # print(f" Total particle mass (fingers + cloth): {masses.sum():.6f} kg")

    # cloth mass
    cloth_ids = getattr(tendon.builder, "drop_cloth_ids", None)
    if cloth_ids is None and getattr(tendon.builder, "cloth_particle_ids", None) is not None:
        cloth_ids = tendon.builder.cloth_particle_ids

    if cloth_ids is not None:
        cloth_mass = masses[cloth_ids].sum()
        print(f" Cloth mass: {cloth_mass:.6f} kg")
    else:
        print(" No cloth ids found on builder.")

    # finger mass (everything that is not cloth)
    all_ids = np.arange(m_model.particle_count)
    if cloth_ids is not None:
        finger_ids = np.setdiff1d(all_ids, cloth_ids)
    else:
        finger_ids = all_ids

    finger_mass = masses[finger_ids].sum()
    print(f" Finger (soft body) mass: {finger_mass:.6f} kg\n")



# MW_ADDED, for enclosed volume,
class VolumeLogger:
    def __init__(self):
        self.rows = []

    def log(self, step, sim_time, frame, substep, vol, dbg=None):
        row = {
            "step": int(step),
            "t": float(sim_time),
            "frame": int(frame),
            "substep": int(substep),
            "vol_vox": float(vol),
        }
        # optional debug columns (nice for sanity checks)
        if dbg is not None:
            row.update({
                "enclosed_voxels": int(dbg.get("enclosed_voxels", -1)),
                "blocked_voxels": int(dbg.get("blocked_voxels", -1)),
                "y_bottom": float(dbg.get("y_bottom", np.nan)),
                "y_top": float(dbg.get("y_top", np.nan)),
            })
        self.rows.append(row)

    def to_csv(self, path):
        os.makedirs(os.path.dirname(path), exist_ok=True)
        pd.DataFrame(self.rows).to_csv(path, index=False)

    
if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument("--device", type=str, default=None, help="Override the default Warp device.")
    parser.add_argument(
        "--stage_path",
        type=lambda x: None if x == "None" else str(x),
        default="sim",
        help="Path to the output USD file.",
    )
    parser.add_argument("--num_frames", type=int, default=1000, help="Total number of frames per training iteration.")
    parser.add_argument("--stiff_iters", type=int, default=3, help="Total number of sampling stiffness iterations.")
    parser.add_argument("--pose_iters", type=int, default=1000, help="Total number of pose iterations.")
    parser.add_argument("--object_name", type=str, default="006_mustard_bottle", help="Name of the object to load.")
    parser.add_argument("--object_density", type=float, default=2e0, help="Density of the object.")
    parser.add_argument("--verbose", action="store_true", help="Print out additional status messages during execution.")
    parser.add_argument("--no_init", action="store_true", help="Automatically initialize the fingers.")
    parser.add_argument("--use_graph", action="store_true", help="Use CUDA graph for forward pass.")
    parser.add_argument("--save_log", action="store_true", help="Save the logs.")
    parser.add_argument("--log_prefix", type=str, default="", help="Prefix for the log file.")
    parser.add_argument("--pose_id", type=int, default=0, help="Initial pose id from anygrasp")
    parser.add_argument("--random", action="store_true", help="Add random noise to the initial position.")
    parser.add_argument("--finger_num", type=int, default=2, help="Number of fingers.")
    parser.add_argument("--is_render", action="store_true", help="Enable USD rendering.")
    parser.add_argument("--no_cloth", action="store_true", help="Disable cloth that connects the fingers.")


    # quick vizualization related args
    parser.add_argument("--quick_viz", action="store_true", help="Quick matplotlib point cloud animation.")
    parser.add_argument("--quick_viz_save", type=str, default=None, help="Optional mp4 save path, eg output/cloth.mp4")
    parser.add_argument("--quick_viz_stride", type=int, default=20, help="Use every nth frame for quick viz.")
    parser.add_argument("--quick_viz_interval", type=int, default=20, help="Milliseconds between frames.")
    parser.add_argument("--quick_viz_every", type=int, default=1, help="Use every nth frame for quick viz.")
    parser.add_argument("--quick_viz_elev", type=float, default=30, help="Camera elevation.")
    parser.add_argument("--quick_viz_azim", type=float, default=45, help="Camera azimuth.")

    # optimizer choice
    parser.add_argument("--optimizer", type=str, default="lbfgs", choices=["lbfgs"], help="Choose the optimization method.")
    parser.add_argument("--no_force_opt", action="store_true", help="Disable force optimization.")

    # expensive logging
    parser.add_argument("--no_voxvol", action="store_true", help="Disable voxel enclosed volume computation.")


    args = parser.parse_known_args()[0]

    finger_len = 11
    finger_rot = np.pi/30
    finger_width = 0.08
    scale = 5.0
    object_rot = wp.quat_rpy(-math.pi/2, 0.0, 0.0)
    is_triangle = True

    with wp.ScopedDevice(args.device):
        # --------------------------------------------------
        # 1) OPTIONAL: optimise initial fingers on a circle
        # --------------------------------------------------
        finger_transform = None
        init_finger_q = None

        if not args.no_init and True:
            print("Running InitializeFingers to optimise initial pose...")
            init_finger = InitializeFingers(
                stage_path="femtendon_sim.usd",     # or args.stage_path, doesn’t really matter here
                finger_len=finger_len,
                finger_rot=finger_rot,
                finger_width=finger_width,
                stop_margin=0.0005,
                num_frames=30,                      # short optimisation horizon
                iterations=args.pose_iters,         # how many gradient steps
                scale=scale,
                num_envs=1,
                ycb_object_name=args.object_name,
                object_rot=object_rot,
                is_render=False,                    # no USD rendering during optimisation
                verbose=args.verbose,
                is_triangle=False,
                finger_num=args.finger_num,
                add_random=args.random,
                consider_cloth=not args.no_cloth,
            )

            finger_transform, init_finger_q = init_finger.get_initial_position()

            # freeze init_pose proxy points and attach to FEM sim for viz
            if init_finger is not None:
                init_finger.capture_proxy_points_frozen()


            # if optimisation fails, fall back gracefully
            if finger_transform is None:
                print("[WARN] InitializeFingers did not converge, falling back to default transforms.")
                finger_transform = None
        else:
            init_finger = init_finger_q
        # --------------------------------------------------
        # 2) Build the full FEM tendon sim, using the initial pose
        # --------------------------------------------------
        tendon = FEMTendon(
            stage_path=args.stage_path, 
            num_frames=args.num_frames, 
            verbose=args.verbose, 
            save_log=args.save_log,
            log_prefix=args.log_prefix,
            is_render=args.is_render,
            kernel_seed=np.random.randint(0, 10000),
            train_iters=args.pose_iters,
            object_rot=object_rot,
            object_density=args.object_density,
            ycb_object_name=args.object_name,
            use_graph=args.use_graph,
            finger_len=finger_len,
            finger_rot=finger_rot,
            finger_width=finger_width,
            scale=scale,
            finger_transform=finger_transform,
            finger_num=args.finger_num,
            requires_grad=True,
            init_finger=init_finger,
            no_cloth=args.no_cloth,
            no_voxvol=args.no_voxvol)
        
        print_mass_breakdown(tendon)
        
        if init_finger is not None and getattr(init_finger, "proxy_pts_frozen", None) is not None:
            tendon.proxy_pts_frozen = init_finger.proxy_pts_frozen

        # --- optimize forces ---
        history = None
        method_name = args.optimizer

        if not args.no_force_opt:
            print(f"--- Running optimization using: {method_name.upper()} ---")
            history = tendon.optimize_forces_lbfgs(
                iterations=1, learning_rate=1.0, opt_frames=100
            )

            # --- Automatic Plotting ---
            if history is not None:
                tendon.plot_single_run(history, method_name=method_name, save_dir="logs")
            else:
                print("No optimization run (history is None).")


        tendon.forward()
        if args.is_render:
            tendon.render()
            tendon.renderer.save()
        # ADDED
        if args.quick_viz:
            quick_visualize(
                tendon,
                stride=args.quick_viz_stride,
                interval=args.quick_viz_interval,
                save_path=args.quick_viz_save,
                elev=args.quick_viz_elev,
                azim=args.quick_viz_azim,
            )
        


