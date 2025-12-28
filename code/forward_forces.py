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

import utils
from object_loader import ObjectLoader
from integrator_euler_fem import FEMIntegrator
from tendon_model import TendonModelBuilder, TendonRenderer, TendonHolder
from init_pose import InitializeFingers

import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation, FFMpegWriter
from scipy.spatial.transform import Rotation as R

from scipy.signal import butter, filtfilt

def animate_point_clouds(point_clouds,
                         interval=50,
                         elev=30,
                         azim=45,
                         save_path=None,
                         colors=None,
                         axis_map="xyz",
                         stride=1,
                         show=True,
                         point_size=5):
    """
    point_clouds: numpy array of shape (T, N, 3)
    interval: ms between frames
    save_path: if not None, saves mp4 (requires ffmpeg)
    """
    point_clouds = np.asarray(point_clouds)
    # stride (skip frames)
    pcs  = point_clouds[::stride]

    T, N, D = pcs.shape

    map_dict = {"x": 0, "y": 1, "z": 2}
    order = [map_dict[c] for c in axis_map]
    pcs = pcs[..., order]

    fig = plt.figure()
    ax = fig.add_subplot(111, projection="3d")

    # limits
    ax.set_xlim(pcs[..., 0].min(), pcs[..., 0].max())
    ax.set_ylim(pcs[..., 1].min(), pcs[..., 1].max())
    ax.set_zlim(pcs[..., 2].min(), pcs[..., 2].max())
    ax.view_init(elev=elev, azim=azim)

    # initial scatter
    scat = ax.scatter(pcs[0, :, 0], pcs[0, :, 1], pcs[0, :, 2],
                      s=point_size)
    if colors is not None:
        scat.set_color(colors)

    def update(frame):
        P = pcs[frame]
        scat._offsets3d = (P[:, 0], P[:, 1], P[:, 2])
        if colors is not None:
            scat.set_color(colors)
        return scat,

    anim = FuncAnimation(fig, update, frames=T, interval=interval, blit=False)

    if save_path is not None:
        writer = FFMpegWriter(fps=max(1, 1000 // interval))
        anim.save(save_path, writer=writer)
        print("Saved quick viz to:", save_path)

    if save_path is None and show:
        plt.show()

    return anim

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

@wp.kernel
def sample_logk(
    kernel_seed: wp.int32,
    min_val: wp.float32, max_val: wp.float32,
    log_K: wp.array(dtype=wp.float32)):
    tid = wp.tid()
    state = wp.rand_init(kernel_seed, tid)
    log_K[tid] = wp.randf(state, min_val, max_val)

@wp.kernel
def apply_gravity(
    gravity: wp.vec3,
    body_mass: wp.array(dtype=wp.float32),
    body_f: wp.array(dtype=wp.spatial_vector)
    ):
    tid = wp.tid()
    wp.atomic_add(body_f, tid, wp.spatial_vector(0.0, 0.0, 0.0, 0.0, 1.0*gravity[1] * body_mass[tid], 0.0))

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
                 finger_num=2,
                 quick_viz=False,
                 quick_viz_stride = 1,
                 quick_viz_interval=20,
                 quick_viz_every=1,
                 quick_viz_save=None,
                 quick_viz_elev=30,
                 quick_viz_azim=45,
                 init_finger=None):
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
        self.requires_grad = False

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


        self.curr_dir = os.path.dirname(os.path.realpath(__file__))
        self.stage_path = self.curr_dir + "/../output/" + stage_path + f"{ycb_object_name}_{log_prefix}_frame{num_frames}" + ".usd"

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
                          is_fix=False,
                          density=object_density,
                          )
        self.builder.init_builder_tendon_variables(self.finger_num, self.finger_len, self.scale, self.requires_grad)
        self.builder.build_fem_model(
            finger_width=self.finger_width,
            finger_rot=self.finger_rot,
            obj_loader=self.obj_loader,
            finger_transform=self.finger_transform,
            is_triangle=True
            )
        self.model = self.builder.model
        self.control = self.model.control(requires_grad=self.requires_grad)

        # for quick viz of the YCB object
        self.obj_local_pts = None
        if hasattr(self.obj_loader, "mesh"):
            verts = np.asarray(self.obj_loader.mesh.vertices, dtype=np.float32)
            verts = verts - verts.mean(axis=0, keepdims=True)   # same centering as shift_vs
            verts = verts * self.scale                          # same scaling as Warp shape
            verts = verts[::50]                                 # subsample for speed
            self.obj_local_pts = verts

        # quick viz settings
        self.quick_viz = quick_viz
        self.quick_viz_stride = quick_viz_stride
        self.quick_viz_interval = quick_viz_interval
        self.quick_viz_every = max(1, int(quick_viz_every))
        self.quick_viz_save = quick_viz_save
        self.quick_viz_elev = quick_viz_elev
        self.quick_viz_azim = quick_viz_azim

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
        self.object_grav = np.zeros(6)
        self.log_K_list = []
        self.save_list = []
        self.mass_list = []
        self.run_name = f"{time.strftime('%Y-%m-%d_%H-%M-%S')}"
        self.file_name = self.save_dir + f"{self.run_name}.npz"

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

        # ----- fingertip particle indices (one per finger) -----
        # Here we assume the last waypoint of each finger is at/near the tip.
        tip_ids = []
        for i in range(self.finger_num):
            tip_ids.append(self.builder.waypoints[i][-1])
        self.tip_particle_ids = np.array(tip_ids, dtype=np.int32)


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

        for frame in range(self.num_frames):

            for i in range(self.sim_substeps):
                index = i + frame * self.sim_substeps
                self.states[index].clear_forces()
                self.tendon_holder.reset()
                if i % 20 == 0:
                    wp.sim.collide(self.model, self.states[index])
                
                self.control.update_target_vel(frame)

                force = self.tendon_forces

                self.tendon_holder.apply_force(force, self.states[index].particle_q, self.success_flag)
                self.integrator.simulate(self.model, self.states[index], self.states[index+1], self.sim_dt, self.control)

                self.object_body_f = self.states[index].body_f

        self.object_q = self.states[-1].body_q


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

    def quick_visualize(self):
        if not getattr(self, "quick_viz", False):
            print("quick_viz is off")
            return

        pcs = []

        # total particles
        P_total = self.model.particle_count
        all_ids = np.arange(P_total, dtype=np.int32)

        # ---- optional cloth handling ----
        cloth_ids = None
        if hasattr(self.model, "cloth_particle_ids") and self.model.cloth_particle_ids is not None:
            try:
                cloth_ids = self.model.cloth_particle_ids.numpy().astype(np.int32)
            except Exception:
                cloth_ids = None

        # everything not cloth = fingers / soft body
        if cloth_ids is not None:
            finger_ids = np.setdiff1d(all_ids, cloth_ids)
        else:
            finger_ids = all_ids

        # optional YCB object points
        obj_local = getattr(self, "obj_local_pts", None)
        n_obj = 0 if obj_local is None else len(obj_local)

        # colors
        finger_color = np.array([0.2, 0.8, 0.2])  # green
        cloth_color  = np.array([0.9, 0.4, 0.1])  # orange
        obj_color    = np.array([0.2, 0.3, 1.0])  # blue

        n_f = len(finger_ids)
        n_c = 0 if cloth_ids is None else len(cloth_ids)
        N = n_f + n_c + n_obj

        colors = np.zeros((N, 3), dtype=np.float32)
        colors[:n_f] = finger_color
        if n_c > 0:
            colors[n_f:n_f+n_c] = cloth_color
        if n_obj > 0:
            colors[n_f+n_c:] = obj_color

        # stride over frames
        stride_frames = getattr(self, "quick_viz_stride", 20)

        for f in range(0, self.num_frames + 1, stride_frames):
            state = self.states[f * self.sim_substeps]

            q = state.particle_q.numpy()
            if q.ndim == 3:
                q = q[0]  # (P,3)

            P_finger = q[finger_ids]
            P_list = [P_finger]

            if cloth_ids is not None:
                P_cloth = q[cloth_ids]
                P_list.append(P_cloth)

            # add object samples if we have them
            if obj_local is not None:
                bq = state.body_q.numpy()
                if bq.ndim == 3:
                    bq = bq[0]
                pos = bq[0, :3]
                quat = bq[0, 3:7]  # xyzw

                rotm = R.from_quat(quat).as_matrix()
                obj_world = (obj_local @ rotm.T) + pos
                P_list.append(obj_world.astype(np.float32))

            P = np.vstack(P_list)       # all 2D now
            pcs.append(P.astype(np.float32))

        pcs = np.stack(pcs, axis=0)  # (T,N,3)

        save_path = getattr(self, "quick_viz_save", None)
        show = getattr(self, "quick_viz_show", save_path is None)

        anim = animate_point_clouds(
            pcs,
            interval=getattr(self, "quick_viz_interval", 30),
            elev=getattr(self, "quick_viz_elev", 25),
            azim=getattr(self, "quick_viz_azim", 45),
            save_path=save_path,
            colors=colors,
            axis_map="xzy",   # so gravity is vertical
            stride=1,         # temporal stride already applied
            show=show,
            point_size=getattr(self, "quick_viz_point_size", 6),
        )
        return anim
    
    def set_tendon_force(self, F: float):
        """Set same tendon force on all fingers."""
        self.tendon_forces = wp.array(
            [F] * self.finger_num,
            dtype=wp.float32,
            requires_grad=self.requires_grad,
        )

    def get_tip_particle_id(self, finger_index: int = 0) -> int:
        """
        Fingertip = last waypoint particle of this finger.
        Uses builder.waypoint_ids, which stores particle indices along the tendon path.
        """
        wp_ids = self.builder.waypoint_ids[finger_index]
        tip_id = int(wp_ids[-1])
        return tip_id

    def run_and_record_tip(self, finger_index: int = 0):
        """
        Run the simulation from t=0 and record fingertip position
        at every substep. Returns np.ndarray of shape (T, 3).
        """
        tip_id = self.get_tip_particle_id(finger_index)
        tip_traj = []

        first_tip_pos = None  # for simple logging

        for frame in range(self.num_frames):
            for i in range(self.sim_substeps):
                index = i + frame * self.sim_substeps

                self.states[index].clear_forces()
                self.tendon_holder.reset()

                if i % 20 == 0:
                    wp.sim.collide(self.model, self.states[index])

                self.control.update_target_vel(frame)
                self.tendon_holder.apply_force(
                    self.tendon_forces,
                    self.states[index].particle_q,
                    self.success_flag,
                )
                self.integrator.simulate(
                    self.model,
                    self.states[index],
                    self.states[index + 1],
                    self.sim_dt,
                    self.control,
                )

                # record from UPDATED state
                q = self.states[index + 1].particle_q.numpy()
                if q.ndim == 3:
                    q = q[0]  # (P,3)

                P = q.shape[0]

                tip_pos = q[tip_id].copy()
                tip_traj.append(tip_pos)

                if first_tip_pos is None:
                    first_tip_pos = tip_pos

        tip_traj = np.array(tip_traj)

        return tip_traj
    
    def set_stiffness(self, K: float):
        """
        Set a uniform stiffness K (bulk modulus, same meaning as builder.init_K)
        for all tet blocks and update the material parameters.
        """
        logK = float(np.log(K))  # natural log, consistent with sample_logk

        # create Warp array for per-block log(K)
        self.log_K_warp = wp.from_numpy(
            np.zeros(self.block_num, dtype=np.float32) + logK,
            dtype=wp.float32,
            requires_grad=self.requires_grad,
        )

        # push to tet_materials
        wp.launch(
            update_materials,
            dim=len(self.model.tet_materials),
            inputs=[self.log_K_warp, self.v, self.tet_block_ids],
            outputs=[self.model.tet_materials],
        )
    
    def run_and_record_force(self):
        """
        Run the simulation and record the linear force acting on the object (Body 0).
        Returns: np.ndarray of shape (T, 3)
        """
        recorded_forces = []
        
        # Assuming the object is the first rigid body (index 0)
        object_body_idx = 0
        
        for frame in range(self.num_frames):
            for i in range(self.sim_substeps):
                index = i + frame * self.sim_substeps

                self.states[index].clear_forces()
                self.tendon_holder.reset()

                if i % 20 == 0:
                    wp.sim.collide(self.model, self.states[index])

                self.control.update_target_vel(frame)
                
                # Apply tendon forces
                self.tendon_holder.apply_force(
                    self.tendon_forces,
                    self.states[index].particle_q,
                    self.success_flag,
                )
                
                # Integrate physics
                self.integrator.simulate(
                    self.model,
                    self.states[index],
                    self.states[index + 1],
                    self.sim_dt,
                    self.control,
                )

                # --- EXTRACT FORCE HERE ---
                # body_f is a spatial vector array. We need to copy it to CPU.
                # Shape is (Num_Bodies, 6). 0-2 is Torque, 3-5 is Force.
                # Note: This .numpy() call creates a sync overhead, but is necessary for recording.
                current_body_f = self.states[index].body_f.numpy()
                
                # Extract linear force (indices 3,4,5) for the object (index 0)
                # If your object ID is different, change [0] to [object_id]
                linear_force = current_body_f[object_body_idx, 3:6]
                
                recorded_forces.append(linear_force.copy())

        return np.array(recorded_forces)
    
    

def get_effective_max_force(time_array, force_array, cutoff_freq=50.0):
    """
    Returns the absolute maximum force (magnitude) after filtering.
    """
    if len(time_array) < 2: return 0.0, np.zeros_like(force_array)
    dt = time_array[1] - time_array[0]
    fs = 1.0 / dt
    
    # Filter
    nyq = 0.5 * fs
    normal_cutoff = cutoff_freq / nyq
    b, a = butter(N=2, Wn=normal_cutoff, btype='low', analog=False)
    filtered_force = filtfilt(b, a, force_array)
    
    # --- CHANGED: Use Absolute Maximum ---
    # This finds the strongest point, whether it's positive (push) or negative (pull/hook)
    abs_max = np.max(np.abs(filtered_force))
    
    return abs_max, filtered_force

        


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
    parser.add_argument("--object_name", type=str, default="", help="Name of the object to load.")
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
    parser.add_argument("--plot_forces", action="store_true", help="plot forces on object.")
    parser.add_argument("--plot_positions", action="store_true", help="plot fingertip positions.")

    # quick vizualization related args
    parser.add_argument("--quick_viz", action="store_true", help="Quick matplotlib cloth point cloud animation.")
    parser.add_argument("--quick_viz_save", type=str, default=None, help="Optional mp4 save path, eg output/cloth.mp4")
    parser.add_argument("--quick_viz_stride", type=int, default=20, help="Use every nth frame for quick viz.")
    parser.add_argument("--quick_viz_interval", type=int, default=20, help="Milliseconds between frames.")
    parser.add_argument("--quick_viz_every", type=int, default=1, help="Use every nth frame for quick viz.")
    parser.add_argument("--quick_viz_elev", type=float, default=30, help="Camera elevation.")
    parser.add_argument("--quick_viz_azim", type=float, default=45, help="Camera azimuth.")


    args = parser.parse_known_args()[0]

    finger_len = 11
    finger_rot = np.pi/30
    finger_width = 0.08
    scale = 5.0
    object_rot = wp.quat_rpy(-math.pi/2, 0.0, 0.0)
    is_triangle = True

    with wp.ScopedDevice(args.device):
        finger_transform = None
        init_finger = None

        #force_values = [100.0]
        force_values = [20.0, 40.0, 60.0, 80.0, 100.0]
        #force_values = [10.0, 20.0, 30.0, 40.0, 50.0, 60.0, 70.0, 80.0, 90.0, 100.0]
        stiffness_values = [2e6]

        trajectories = {}   # (F, K) -> (T, 3)
        force_trajectories = {}  # F -> (T, 3)
        sample_dt = {}      # (F, K) -> dt

        print("Running InitializeFingers to optimise initial pose...")
        if not args.no_init:
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
            )

            finger_transform, init_finger_q = init_finger.get_initial_position()

        for K in stiffness_values:
            print(f"\n=== Using stiffness K = {K:.2e} ===")
            for F in force_values:
                print(f"   -> Running simulation with tendon force {F} N")

                tendon = FEMTendon(
                    stage_path=args.stage_path,
                    num_frames=args.num_frames,
                    verbose=args.verbose,
                    save_log=False,
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
                    quick_viz=args.quick_viz,
                    quick_viz_stride=args.quick_viz_stride,
                    quick_viz_interval=args.quick_viz_interval,
                    quick_viz_every=args.quick_viz_every,
                    quick_viz_save=args.quick_viz_save,
                    quick_viz_elev=args.quick_viz_elev,
                    quick_viz_azim=args.quick_viz_azim,
                    init_finger=init_finger,
                )

                tendon.set_stiffness(K)
                tendon.set_tendon_force(F)


                #tip_traj = tendon.run_and_record_tip(finger_index=0)  # (T, 3)

                #trajectories[(F, K)] = tip_traj
                object_forces = tendon.run_and_record_force()  # (T, 3)

                force_trajectories[(F,K)] = object_forces
                sample_dt[(F, K)] = tendon.sim_dt

                # just to confirm:
                K_current = float(np.exp(tendon.log_K_warp.numpy()[0]))
                print(f"Force {F} N used stiffness K = {K_current:.3e}")

        if args.plot_forces:
            fig, ax = plt.subplots(1, 1, figsize=(10, 6))
            
            sorted_keys = sorted(force_trajectories.keys())

            for (F, K) in sorted_keys:
                obj_f_vec = force_trajectories[(F, K)]
                if obj_f_vec.size == 0: continue
                
                # Use Force X (since we determined it's the pushing direction)
                raw_force = obj_f_vec[:, 0]
                
                # Create time array
                dt = sample_dt[(F, K)]
                t = np.arange(len(raw_force)) * dt

                # ... inside the loop ...
            
                # --- APPLY THE FILTER ---
                # We now get the 'abs_max' magnitude
                force_mag, smooth_force = get_effective_max_force(t, raw_force, cutoff_freq=50)
                
                # Calculate Ratio
                ratio = (force_mag / F) * 100.0
                
                print(f"Input: {F}N -> Max Magnitude: {force_mag:.2f}N (Efficiency: {ratio:.1f}%)")

                # Plotting
                # We plot the smooth force, but the label shows the MAGNITUDE
                p = ax.plot(t, raw_force, alpha=0.2) 
                color = p[0].get_color()
                ax.plot(t, smooth_force, color=color, linewidth=2, label=f"F={F}N (Mag: {force_mag:.1f}N)")

            ax.set_ylabel("Force X [N]")
            ax.set_xlabel("Time [s]")
            ax.set_title("Filtered Pushing Force")
            ax.legend()
            ax.grid(True)
            
            plt.savefig("filtered_max_force.png")
            print("Saved filtered_max_force.png")


        if args.plot_positions: 
            # ---------- Y-position vs time ----------
            plt.figure(figsize=(8, 4))

            for (F, K), traj in trajectories.items():
                if traj.size == 0:
                    continue
                if traj.ndim != 2 or traj.shape[1] != 3:
                    continue

                mask = np.isfinite(traj).all(axis=1)
                if not np.any(mask):
                    continue

                y = traj[mask, 1]
                n = y.size
                dt = float(sample_dt[(F, K)])
                t = np.arange(n) * dt

                label = f"F={F:.1f} N, K={K:.2e}"
                plt.plot(t, y, label=label)

            plt.xlabel("Time [s]")
            plt.ylabel("Tip Y position")
            plt.title("Fingertip Y-position over time")
            plt.grid(True)
            plt.legend()

            out_path_y = "fingertip_y_vs_time.png"
            plt.savefig(out_path_y, dpi=300, bbox_inches="tight")
            print(f"Saved Y-vs-time figure to {out_path_y}")

            # ---------- Final tip Y vs tendon force (one curve per stiffness) ----------
            from collections import defaultdict

            forces_by_K = defaultdict(list)
            final_y_by_K = defaultdict(list)

            for (F, K), traj in trajectories.items():
                if traj.size == 0:
                    continue

                mask = np.isfinite(traj).all(axis=1)
                if not np.any(mask):
                    continue

                traj_valid = traj[mask]
                y_final = traj_valid[-1, 1]

                forces_by_K[K].append(F)
                final_y_by_K[K].append(y_final)

            plt.figure(figsize=(6, 4))

            for K, forces_list in forces_by_K.items():
                forces_arr = np.array(forces_list)
                y_arr = np.array(final_y_by_K[K])

                # sort by force for a clean line
                idx = np.argsort(forces_arr)
                forces_arr = forces_arr[idx]
                y_arr = y_arr[idx]

                plt.plot(forces_arr, y_arr, "o-", lw=2, label=f"K={K:.2e}")

            plt.xlabel("Tendon force [N]")
            plt.ylabel("Final fingertip Y position")
            plt.title("Final Y-position vs Tendon Force (per stiffness)")
            plt.grid(True)
            plt.legend()

            out_path_force = "final_y_vs_force.png"
            plt.savefig(out_path_force, dpi=300, bbox_inches="tight")
            print(f"Saved final-Y-vs-force plot to {out_path_force}")

        # optional: still allow quick_viz / render for the last run if you want
        if args.is_render:
            tendon.render()
            tendon.renderer.save()
        if args.quick_viz:
            tendon.quick_visualize()

