import numpy as np
import warp as wp
import warp.sim.render
from warp.sim.model import *
import utils
import math

@wp.kernel
def calculate_surface_normals(
    particle_num: wp.int32,
    tri_indices: wp.array(dtype=wp.int32), # len=particle_num
    particle_q: wp.array(dtype=wp.vec3), # len=particle_num
    tri_particle_indices: wp.array2d(dtype=wp.int32), # len=tri_num*3
    surface_normals: wp.array(dtype=wp.vec3)):
    
    tid = wp.tid()

    if tid >= particle_num: return
    idx = tri_indices[tid]
    p0 = particle_q[tri_particle_indices[idx][0]]
    p1 = particle_q[tri_particle_indices[idx][1]]
    p2 = particle_q[tri_particle_indices[idx][2]]
    n = wp.cross(p1 - p0, p2 - p0)

    wp.atomic_add(surface_normals, tid, wp.normalize(n))  

@wp.kernel
def calculate_force(
    particle_indices: wp.array(dtype=wp.int32), # len=particle_num, all ids of particles in the waypoints, 1d array
    particle_num: wp.int32,
    finger_length: wp.int32, # the number of waypoints in each fingers
    waypoint_pair_ids: wp.array(dtype=wp.int32), # len=particle_num
    particle_q: wp.array(dtype=wp.vec3), # len=particle_num
    surface_normals: wp.array(dtype=wp.vec3), # len=particle_num
    finger_waypoint_num: wp.array(dtype=wp.int32), # finger id of each waypoint, len=len(waypoints)
    external_force: wp.array(dtype=wp.float32), # len=particle_num
    # control_particle_activations: wp.array(dtype=wp.vec3)):
    waypoint_activations: wp.array(dtype=wp.vec3),
    success: wp.array(dtype=wp.int32)):

    tid = wp.tid()

    if tid >= particle_num: return
    # potential bug if fingers are not of the same length
    finger_id = int(finger_waypoint_num[tid])
    # wp_id = int(waypoint_pair_ids[tid] + finger_id * finger_length)
    this_pid = int(particle_indices[tid])
    
    # the first contact force is 0
    if tid == (finger_id * finger_length):
        return
    # the last contact force goes inside
    if tid == ((finger_id+1) * finger_length - 1):
        f_dir = particle_q[particle_indices[tid-1]] - particle_q[this_pid]
        if wp.length(f_dir) < 1e-4:
            success[0] = 0
            return
        f_dir_norm = wp.normalize(f_dir)
        f = f_dir_norm * external_force[finger_id]
        wp.atomic_add(waypoint_activations, tid, f)
        return
    
    # find the two neighboring waypoints of this waypoint
    left_pid = int(particle_indices[tid-1])
    right_pid = int(particle_indices[tid+1])
    # find the uniform vector in both directions
    left_vec = particle_q[left_pid] - particle_q[this_pid]
    right_vec = particle_q[right_pid] - particle_q[this_pid]
    if wp.length(left_vec) < 1e-4 or wp.length(right_vec) < 1e-4:
        success[0] = 0
        return
    left_dir = wp.normalize(left_vec)
    right_dir = wp.normalize(right_vec)

    # consider the contact point is a pulley
    tendon_direction = left_dir + right_dir
    if wp.length(tendon_direction) < 1e-5:
        return

    proj_tendon = wp.dot(tendon_direction, surface_normals[tid]) * surface_normals[tid]
    tendon_f_dir = tendon_direction - proj_tendon
    
    threshold = 1e-4
    blend_factor = wp.smoothstep(0.0, threshold, wp.length(tendon_f_dir))
    f_dir = blend_factor * wp.normalize(tendon_f_dir) + (1.0 - blend_factor) * wp.vec3(0.0, 0.0, 0.0)

    f = wp.dot(external_force[finger_id] * tendon_direction, f_dir) * f_dir # should use wp.dot instead of np.dot in warp kernels
    wp.atomic_add(waypoint_activations, tid, f)

@wp.kernel
def compute_diff_length(
    particle_indices: wp.array(dtype=wp.int32),
    particle_num: wp.int32,
    finger_waypoint_num: wp.array(dtype=wp.int32),
    particle_q: wp.array(dtype=wp.vec3),
    tendon_length: wp.array(dtype=wp.float32)):
    
    tid = wp.tid()

    if tid >= (particle_num-1): return
    this_pid = particle_indices[tid]
    pair_pid = particle_indices[tid+1]
    this_finger_id = finger_waypoint_num[tid]
    pair_finger_id = finger_waypoint_num[tid+1]
    if this_finger_id != pair_finger_id:
        return

    diff = particle_q[pair_pid] - particle_q[this_pid]
    wp.atomic_add(tendon_length, this_finger_id, wp.length(diff))

@wp.kernel
def calculate_tendon(
    tendon_length: wp.array(dtype=wp.float32),
    init_tendon_length: wp.array(dtype=wp.float32),
    tendon_position: wp.array(dtype=wp.float32)):
    tid = wp.tid()
    if init_tendon_length[tid] < 0:
        init_tendon_length[tid] = tendon_length[tid]
    # tendon_position[tid] = init_tendon_length[tid] - tendon_length[tid]
    pos = init_tendon_length[tid] - tendon_length[tid]
    wp.atomic_add(tendon_position, tid, pos)

@wp.kernel
def force_pid(
    p: float,
    target_pos: wp.array(dtype=wp.vec2),
    curr_pos: wp.array(dtype=wp.float32),
    frame: wp.int32,
    force: wp.array(dtype=wp.float32)):
    tid = wp.tid()

    max_force = 100.0

    f = 1e5 * (target_pos[frame][tid] - curr_pos[tid])
    force[tid] = f
    if force[tid] > max_force:
        force[tid] = max_force
    if force[tid] < 0.0:
        force[tid] = 0.0

@wp.kernel
def find_glue_index(
    this_particle: wp.vec3,
    glue_points: wp.array(dtype=wp.vec3),
    index: wp.array(dtype=wp.int32)):
    tid = wp.tid()
    if wp.length(this_particle - glue_points[tid]) < 1e-6:
        wp.atomic_add(index, 0, tid + 1)

@wp.kernel
def target_vel(
    frame_id: wp.int32,
    target_vels: wp.array(dtype=wp.vec3)):

    tid = wp.tid()
    target_vels[tid] = wp.vec3(0.0, 0.0, 0.0)

    move_frame_id = 500
    offset_frame = 150
    # if frame_id > (move_frame_id-offset_frame)  and frame_id < move_frame_id:
    #     target_vels[tid] = wp.vec3(0.0, 3e-4, 0.0)
    # elif frame_id > move_frame_id and frame_id < (move_frame_id+offset_frame):
    #     target_vels[tid] = wp.vec3(0.0, -3e-4, 0.0)
    target_vels[tid] = wp.vec3(0.0, 0.0, 0.0)

class TendonHolder:
    def __init__(self, model, control):
        self.model = model
        self.control = control

        # tendon related variables
        self.finger_len = None
        self.finger_num = None
        self.waypoints = None
        self.finger_waypoint_num = None
        self.waypoint_pair_ids = None
        self.tendon_tri_indices = None
        self.external_force = None
        
        # tendon variables to be recalcuated
        self.surface_normals = None
        self.tendon_length = None

        # control related variables
        self.init_tendon_length = None
        self.tendon_position = None
    
    def init_tendon_variables(self, requires_grad=False):
        self.tendon_length = wp.zeros(self.finger_num, dtype=wp.float32, requires_grad=requires_grad)
        self.init_tendon_length = wp.from_numpy(np.zeros(self.finger_num)-1, dtype=wp.float32, requires_grad=requires_grad)
        self.tendon_position = wp.zeros(self.finger_num, dtype=wp.float32, requires_grad=requires_grad)

        self.control.force = wp.zeros(self.finger_num, dtype=wp.float32, requires_grad=requires_grad)

    def apply_force(self, external_force, particle_q, success_flag):
        wp.launch(calculate_surface_normals,
            dim = len(self.waypoints),
            inputs=[
                len(self.waypoints),
                self.tendon_tri_indices,
                particle_q,
                self.model.tri_indices,
                ],
            outputs=[self.surface_normals])
        wp.launch(calculate_force,
            dim = len(self.waypoints),
            inputs=[
                self.waypoints,
                len(self.waypoints),
                self.finger_len,
                self.waypoint_pair_ids,
                particle_q,
                self.surface_normals,
                self.finger_waypoint_num,
                external_force],
            outputs=[self.control.waypoint_forces,
                     success_flag])

    def get_tendon_length(self, particle_q):
        wp.launch(compute_diff_length,
            dim = len(self.waypoints),
            inputs=[
                self.waypoints,
                len(self.waypoints),
                self.finger_waypoint_num,
                particle_q,
                ],
            outputs=[self.tendon_length])
        wp.launch(calculate_tendon,
            dim = self.finger_num,
            inputs=[
                self.tendon_length,
                ],
            outputs=[
                self.init_tendon_length, 
                self.tendon_position])
        return self.tendon_position

    def print_requires_grad(self):
        for name, value in vars(self).items():
            if isinstance(value, wp.array):  # Check if it is a Warp array
                print(f"{name}: requires_grad={value.requires_grad}")
            elif isinstance(value, list):  # Check if it's a list of Warp arrays
                for i, v in enumerate(value):
                    if isinstance(v, wp.array):
                        print(f"{name}[{i}]: requires_grad={v.requires_grad}")
    
    def reset(self):
        self.control.waypoint_forces.zero_()
        if self.control.vel_values is not None:
            self.control.vel_values.zero_()
        self.surface_normals.zero_()
        self.tendon_length.zero_()
        self.tendon_position.zero_()

class TendonControl(Control):
    def __init__(self, model):
        super().__init__(model)
        self.finger_num = None
        self.pid = {'p': 5e-1, 'i': 0.0, 'd': 0.0}
        self.waypoint_ids = None
        self.waypoint_forces = None
        self.force = None
        self.target_positions = None
        # self.vel_ids = None
        self.vel_values = None
    
    def force_from_position(self, curr_pos, target_pos, frame):
        pid_p = self.pid['p']
        pid_i = self.pid['i']
        pid_d = self.pid['d']
        wp.launch(force_pid,
            dim = self.finger_num,
            inputs=[
                pid_p,
                target_pos,
                curr_pos,
                frame
                ],
            outputs=[self.force])
        return self.force
    
    def update_target_vel(self, time):
        wp.launch(target_vel,
                dim=1,
                inputs=[
                    time],
                outputs=[self.vel_values])

    def print_requires_grad(self):
        for name, value in vars(self).items():
            if isinstance(value, wp.array):  # Check if it is a Warp array
                print(f"{name}: requires_grad={value.requires_grad}")
            elif isinstance(value, list):  # Check if it's a list of Warp arrays
                for i, v in enumerate(value):
                    if isinstance(v, wp.array):
                        print(f"{name}[{i}]: requires_grad={v.requires_grad}")
    
    def reset(self):
        super().reset()
        self.waypoint_forces.zero_()
        if self.vel_values is not None:
            self.vel_values.zero_()

class TendonModel(Model):
    def __init__(self, device):
        super().__init__(device)
    
    def print_requires_grad(self):
        for name, value in vars(self).items():
            if isinstance(value, wp.array):  # Check if it is a Warp array
                print(f"{name}: requires_grad={value.requires_grad}")
            elif isinstance(value, list):  # Check if it's a list of Warp arrays
                for i, v in enumerate(value):
                    if isinstance(v, wp.array):
                        print(f"{name}[{i}]: requires_grad={v.requires_grad}")

    def control(self, requires_grad=None, clone_variables=True) -> TendonControl:
        tendon_control = TendonControl(self)
        c = super().control(requires_grad, clone_variables)
        tendon_control.__dict__.update(c.__dict__)
        return tendon_control

class TendonModelBuilder(ModelBuilder):
    # Default triangle soft mesh settings
    default_tri_ke = 100.0
    default_tri_ka = 100.0
    default_tri_kd = 10.0
    default_tri_drag = 0.0
    default_tri_lift = 0.0

    def __init__(self):
        """ MW_ADDED """
        super().__init__()
        # mainly for cloth-figner collision handling
        self.finger_tri_indices_list = [] # list of np arrays (n_k, 3)
        self.finger_tri_mats_list    = [] # list of np arrays (n_k, 5)
        self.drop_cloth_ids = None # indices of drop_cloth indices
        self.cloth_particle_ids = None # final union off all cloth particles, for cloth-finger collision

        # mainly to color attachment points in the viz
        self.attached_cloth_ids      = [] # list of np arrays, one per attached cloth patch
        self.attached_cloth_edge_ids = [] # list of lists, edge vertices of cloth that we attach to fingers


    def add_soft_grid_glue(
        self,
        pos: Vec3,
        rot: Quat,
        vel: Vec3,
        dim_x: int,
        dim_y: int,
        dim_z: int,
        cell_x: float,
        cell_y: float,
        cell_z: float,
        density: float,
        k_mu: float,
        k_lambda: float,
        k_damp: float,
        glue_points: wp.array(dtype=wp.vec3) = [],
        tri_ke: float = default_tri_ke,
        tri_ka: float = default_tri_ka,
        tri_kd: float = default_tri_kd,
        tri_drag: float = default_tri_drag,
        tri_lift: float = default_tri_lift,
    ):
        """Helper to create a rectangular tetrahedral FEM grid
        Args:
            glue_points: List of points to which the particles are glued
        """

        start_vertex = len(self.particle_q)
        sub_index = np.zeros((dim_x + 1) * (dim_y + 1) * (dim_z + 1), dtype=np.int32)
        new_count = 0

        mass = cell_x * cell_y * cell_z * density

        def grid_index(x, y, z):
            return (dim_x + 1) * (dim_y + 1) * z + (dim_x + 1) * y + x

        for z in range(dim_z + 1):
            for y in range(dim_y + 1):
                for x in range(dim_x + 1):
                    v = wp.vec3(x * cell_x, y * cell_y, z * cell_z)
                    m = mass

                    p = wp.quat_rotate(rot, v) + pos

                    this_index = wp.from_numpy(np.zeros(1)-1, dtype=wp.int32)
                    wp.launch(find_glue_index,
                        dim = len(glue_points),
                        inputs=[
                            p,
                            glue_points,
                            ],
                        outputs=[this_index])
                    this_index = int(this_index.numpy()[0])
                    if this_index != -1:
                        # glue this particle
                        sub_index[grid_index(x, y, z)] = this_index
                    else:
                        # add this new particle
                        sub_index[grid_index(x, y, z)] = new_count + start_vertex
                        self.add_particle(p, vel, m)
                        new_count += 1

        # dict of open faces
        faces = {}

        def add_face(i: int, j: int, k: int):
            key = tuple(sorted((i, j, k)))

            if key not in faces:
                faces[key] = (i, j, k)
            else:
                del faces[key]

        def add_tet(i: int, j: int, k: int, l: int):
            self.add_tetrahedron(i, j, k, l, k_mu, k_lambda, k_damp)

            add_face(i, k, j)
            add_face(j, k, l)
            add_face(i, j, l)
            add_face(i, l, k)

        for z in range(dim_z):
            for y in range(dim_y):
                for x in range(dim_x):
                    v0 = sub_index[grid_index(x, y, z)]
                    v1 = sub_index[grid_index(x + 1, y, z)]
                    v2 = sub_index[grid_index(x + 1, y, z + 1)]
                    v3 = sub_index[grid_index(x, y, z + 1)]
                    v4 = sub_index[grid_index(x, y + 1, z)]
                    v5 = sub_index[grid_index(x + 1, y + 1, z)]
                    v6 = sub_index[grid_index(x + 1, y + 1, z + 1)]
                    v7 = sub_index[grid_index(x, y + 1, z + 1)]

                    if (x & 1) ^ (y & 1) ^ (z & 1):
                        add_tet(v0, v1, v4, v3)
                        add_tet(v2, v3, v6, v1)
                        add_tet(v5, v4, v1, v6)
                        add_tet(v7, v6, v3, v4)
                        add_tet(v4, v1, v6, v3)

                    else:
                        add_tet(v1, v2, v5, v0)
                        add_tet(v3, v0, v7, v2)
                        add_tet(v4, v7, v0, v5)
                        add_tet(v6, v5, v2, v7)
                        add_tet(v5, v2, v7, v0)

        # add triangles
        for _k, v in faces.items():
            self.add_triangle(v[0], v[1], v[2], tri_ke, tri_ka, tri_kd, tri_drag, tri_lift)
    
    def init_builder_tendon_variables(self, finger_num, finger_len, scale, requires_grad):
        self.finger_num = finger_num
        self.finger_len = finger_len
        self.scale = scale
        self.requires_grad = requires_grad
        self.waypoints = [[] for _ in range(finger_num)]
        self.waypoint_ids = [[] for _ in range(finger_num)]
        self.waypoint_pair_ids = [[] for _ in range(finger_num)]
        self.waypoints_tri_indices = [[] for _ in range(finger_num)]

        self.finger_vertex_ranges = []
        self.finger_face_ranges = []
        self.finger_particle_offsets = []

        # MW_ADDED
        self.finger_back_ids_per_finger = [[] for _ in range(finger_num)]
        # for cloth-finger attachment
        self.finger_particle_ids = [[] for _ in range(finger_num)]

        # combined list for viz
        self.finger_attach_ids = [[] for _ in range(finger_num)] # list of lists, per finger
        # per-edge lists for attachment
        self.finger_left_attach_ids  = [[] for _ in range(finger_num)]
        self.finger_right_attach_ids = [[] for _ in range(finger_num)]

    def build_fem_model(self, 
                        finger_width=0.06, finger_rot=0.3,
                        h_dis_init=0.2,
                        obj_loader=None, 
                        finger_transform=None,
                        is_triangle=False,
                        add_connecting_cloth=False, #cloth that connects fingers
                        add_drop_cloth=False): # cloth for drop behavior tests
        s = self.scale
        self.finger_transform = finger_transform

        cell_dim = [2, [1, 6], 2]
        cell_size = [0.008 * s, 
                     [0.003 * s, 0.002 * s], 
                     0.01 * s]
        conn_dim = [1, 1, cell_dim[2]]
        conn_size = 0.008 * s

        finger_THK = cell_dim[1][1] * cell_size[1][1]
        finger_LEN = (cell_dim[0] * cell_size[0] * (self.finger_len // 2 + 1) + conn_dim[0] * conn_size * (self.finger_len // 2)) / s
        self.finger_back_ids = []
        self.tet_block_ids = []
        self.block_id = 0
        finger_transforms = []

        finger_height = cell_dim[2]*cell_size[2]

        h_dis = 0.0
        
        # --- helper to generate default transforms if none were provided ---
        def _default_transform(i):

            # use a radius proportional to finger length, not thickness
            R = max(finger_height, finger_LEN * 2.8)   # tweak 0.5 as you like
            plane_y = h_dis + finger_height * 10.0

            # evenly spaced in angle
            theta = 2.0 * math.pi * (i / float(self.finger_num))
            x = R * math.cos(theta)
            z = R * math.sin(theta)
            pos = np.array([x, plane_y, z])

            yaw = theta + math.pi
            rot = wp.quat_rpy(yaw, 0.0, -math.pi/2)
            return wp.transform(pos, rot)

        # if a list of transforms was passed in, use it; otherwise generate
        transforms = []
        for i in range(self.finger_num):
            if finger_transform and len(finger_transform) == self.finger_num:
                transforms.append(finger_transform[i])
            else:
                transforms.append(_default_transform(i))

        # FINAL model geometry is built by repeatedly calling add_finger
        for i in range(self.finger_num):
            # ---- record starts ----
            v_start = len(self.particle_q)
            f_start = len(self.tri_indices)

            self.add_finger(
                cell_dim, cell_size, conn_dim, conn_size,
                transform=transforms[i],
                index=i,
                is_triangle=is_triangle,
            )
            finger_transforms.append(transforms[i])

            # ---- record ends ----
            v_end = len(self.particle_q)
            f_end = len(self.tri_indices)

            # ---- store ranges/offsets ----
            self.finger_vertex_ranges.append((v_start, v_end))
            self.finger_face_ranges.append((f_start, f_end))
            self.finger_particle_offsets.append(v_start)

        #########################################################
        # add cloth grids, MW_ADDED
        #########################################################

        # --- Cloth dropping onto the scene for behavior test ---
        if add_drop_cloth:
            self.add_drop_cloth_grid(
                obj_loader=obj_loader,
                cloth_res=(32, 32)
            )

        # --- Cloth connecting between fingers ---
        # build welded membrane between the *edges* of the two fingers
        # here: right edge of finger 0 -> left edge of finger 1
        if add_connecting_cloth:
            for i in range(self.finger_num):
                j = (i + 1) % self.finger_num
                self.add_connecting_cloth(
                    finger_a=i,
                    finger_b=j,
                    edge_a="right",
                    edge_b="left",
                    dx_nominal=cell_size[0],
                    mass_per_vertex=1e-3,
                    tri_ke=1.0e2,
                    tri_ka=1.0e2,
                    tri_kd=1.0e1,
                )


        # --- Merge all cloth particles into one list for cloth-finger collisions / viz ---
        all_cloth_ids = []
        # drop cloth (if any)
        if getattr(self, "drop_cloth_ids", None) is not None:
            all_cloth_ids.append(self.drop_cloth_ids)
        # attached cloth strips (interior vertices of connecting patches)
        if self.attached_cloth_ids:
            all_cloth_ids.extend(self.attached_cloth_ids)
        if all_cloth_ids:
            # union of all cloth particles, no duplicates
            self.cloth_particle_ids = np.unique(
                np.concatenate(all_cloth_ids)
            ).astype(np.int32)
        else:
            self.cloth_particle_ids = None
        

        # finalize model (now includes fingers, cloth, YCB object)
        self.model = self.finalize(requires_grad=self.requires_grad)

        contact_scale = 0.01 # 0.01

        radii = wp.zeros(self.model.particle_count, dtype=wp.float32)
        radii.fill_(1e-3 * self.scale) # was 1e-3*s, later 2e-3
        self.model.particle_radius = radii
        self.model.ground = True
        self.model.gravity = wp.vec3(0.0, -9.8, 0.0)
        self.model.particle_kf = 1.0e1
        self.model.particle_mu = 1.0
        self.model.particle_max_velocity = 1.0e1 # original: 1.0e1, changed it to 1.0e5 for some reason
        self.model.particle_adhesion = 1.0e-3 # original: 1.0e-3, changed it to 1.0e-4 for some reason
        self.model.soft_contact_ke = 1.0e3 * contact_scale # original 1.0e3, v6: 2.0e3, normal contact stiffness between soft bodies and colliders, Think of it as: F_normal ≈ ke * penetration ###
        self.model.soft_contact_kd = 1.0e1 * contact_scale # original 1.0e1, v6: 5.0e1, normal contact damping, proportional to relative normal velocity. Controls bounciness vs “thud”. ###
        self.model.soft_contact_kf = 1.0e1 * contact_scale # tangential friction stiffness,
        self.model.soft_contact_mu = 1.0 # friction model parameters.
        self.model.soft_contact_margin = 1e-3 # original: 1e-3, v6: 3e-3,
        self.model.rigid_contact_margin = 1e-4 # max(3e-3 * self.scale, 0.25 * cloth_cell) # original: 1e-4, distance where collisions are “inflated” so they happen a bit before visual touching, also affects how soft /squishy contact feels.
        # self.model.enable_tri_collisions = True # self-collisions
        self.model.enable_finger_cloth_collisions = False # only cloth-finger collisions, disable for cloth-ycb tests

        # particles (ground contacts), seems to have no effect
        # self.model.particle_ke = 1.0e-7
        # self.model.particle_kd = 1.0e-7

        # MW_ADDED, for cloth-finger collisions
        if self.finger_tri_indices_list:
            all_tris = np.vstack(self.finger_tri_indices_list).astype(np.int32)
            all_mats = np.vstack(self.finger_tri_mats_list).astype(np.float32)
            self.model.finger_tri_indices  = wp.array(all_tris, dtype=int)
            self.model.finger_tri_materials = wp.array(all_mats, dtype=float)

        if self.cloth_particle_ids is not None:
            self.model.cloth_particle_ids = wp.array(self.cloth_particle_ids, dtype=int)

        print("\ncloth particles:", None if self.cloth_particle_ids is None else len(self.cloth_particle_ids))
        if self.finger_tri_indices_list:
            print("finger tri sets:", len(self.finger_tri_indices_list),
                "total finger tris:", np.vstack(self.finger_tri_indices_list).shape, "\n")
                
        return finger_transforms, finger_LEN, finger_THK
 
    def add_finger(self, 
                   cell_dim, cell_size, conn_dim, conn_size, 
                   transform=None,
                   index=0,
                   is_triangle=False):
        density = 5.0

        # actual
        K = 2.0e6 # young's modulus
        # K = np.exp(13.902607917785645)
        v = 0.45 # poisson's ratio, how much it wants to keep its volume during deformation.
        k_mu = K / (2 * (1 + v))
        k_lambda = K * v / ((1 + v) * (1 - 2 * v)) # Lamé parameters derived from E and v which FEM uses internally for tetrahedra
        self.init_K = K
        self.init_v = v
        k_damp = 5e-1 # bulk damping inside the FEM elements (kills oscillations)

        tri_start = len(self.tri_indices)

        self.generate_tendon_waypoints_hirose(
            cell_dim, cell_size, conn_dim, conn_size, 
            index=index) 

        particle_start_idx = len(self.particle_q)
        block_start_idx = len(self.tet_indices)

        for i in range(self.finger_len // 2 + 1):
            base_offset = i*(cell_size[0]*cell_dim[0]
                               + conn_size*conn_dim[0])

            dim_x = cell_dim[0] 
            dim_z = cell_dim[2]
            cell_x = cell_size[0]
            cell_z = cell_size[2]

            waypt0 = self.waypoints[index][2*i][1]
            waypt1 = self.waypoints[index][2*i-1][1]
            if i == 0: waypt0 = self.waypoints[index][0][1]
            y_offset = [0.0] 
            y_offset.append(cell_dim[1][0] * cell_size[1][0])
            if (waypt0 - y_offset[-1]) > 1e-3*self.scale and (cell_dim[1][1] * cell_size[1][1] - waypt0) > 1e-3*self.scale:
                y_offset.append(waypt0)
            if (waypt1 - y_offset[-1]) > 1e-3*self.scale and (cell_dim[1][1] * cell_size[1][1] - waypt1) > 1e-3*self.scale:
                y_offset.append(waypt1)
            y_offset.append(cell_dim[1][1] * cell_size[1][1])
 
            for y_idx in range(len(y_offset)-1):
                dim_y = 1
                cell_y = y_offset[y_idx+1] - y_offset[y_idx]
                this_add_func = self.add_soft_grid if y_idx == 0 else self.add_soft_grid_glue

                params = {"pos": wp.vec3(base_offset, y_offset[y_idx], 0.0),
                        "rot": wp.quat_identity(),
                        "vel": wp.vec3(0.0, 0.0, 0.0),
                        "dim_x": dim_x,
                        "dim_y": dim_y,
                        "dim_z": dim_z,
                        "cell_x": cell_x,
                        "cell_y": cell_y,
                        "cell_z": cell_z,
                        "density": density,
                        "k_mu": k_mu if y_idx < 2 else k_mu/10.0,
                        "k_lambda": k_lambda if y_idx < 2 else k_lambda/10.0,
                        "k_damp": k_damp if y_idx < 2 else k_damp*10.0,
                        "tri_ke": 1e-1, # default: 1e-1 
                        "tri_ka": 1e1, # default: 1e1 
                        "tri_kd": 1e-1 if y_idx < 2 else 1e-1*10.0} # default: 1e-1 if y_idx < 2 else 1e-1*10.0}
                if y_idx == 0:
                    params["fix_left"] = True if i == 0 else False
                else:
                    params["glue_points"] = wp.array(self.particle_q, dtype=wp.vec3, requires_grad=self.requires_grad)
                
                this_add_func(**params)

            self.tet_block_ids.extend([self.block_id for _ in range(len(self.tet_indices) - block_start_idx)])
            block_start_idx = len(self.tet_indices)
            self.block_id += 1
        
        for i in range(self.finger_len // 2):  
            base_offset = i*(cell_size[0]*cell_dim[0]
                               + conn_size*conn_dim[0])

            dim_x = conn_dim[0]
            dim_y = conn_dim[1]
            dim_z = cell_dim[2]
            cell_x = conn_size
            cell_y = cell_size[1][0]
            cell_z = cell_size[2]
            self.add_soft_grid_glue(
                pos=wp.vec3(base_offset + cell_size[0]*cell_dim[0], 0.0, 0.0),
                rot=wp.quat_identity(),
                vel=wp.vec3(0.0, 0.0, 0.0),
                dim_x=dim_x,
                dim_y=dim_y,
                dim_z=dim_z,
                cell_x=cell_x,
                cell_y=cell_y,
                cell_z=cell_z,
                density=density,
                k_mu=k_mu,
                k_lambda=k_lambda,
                k_damp=k_damp,
                glue_points=wp.array(self.particle_q, dtype=wp.vec3),
                tri_ke=1e-1,
                tri_ka=1e1,
                tri_kd=1e-1
            )

            self.tet_block_ids.extend([self.block_id for _ in range(len(self.tet_indices) - block_start_idx)])
            block_start_idx = len(self.tet_indices)
            self.block_id += 1
        particle_end_idx = len(self.particle_q)
        self.find_back_ids(particle_start_idx, 
                           particle_end_idx,
                           low_threshold=np.array(
                               [1e-2, cell_dim[1][1]*cell_size[1][1], -1e-6]),
                            high_threshold=np.array(
                                [1e3, 1e3, 1e3]))
        
        # MW_ADDED
        # remember which particles belong to this created finger
        self.finger_particle_ids[index] = list(range(particle_start_idx, particle_end_idx))

        # Choose attachment nodes in local finger coords, before transform
        # We split into the two edges, then store both and also a combined list.

        num_attach_per_edge = 18

        edge_lo_ids = self.select_cloth_finger_attachment_ids(
            finger_index=index,
            which="edge_lo",
            num_attach=num_attach_per_edge,
        )
        edge_hi_ids = self.select_cloth_finger_attachment_ids(
            finger_index=index,
            which="edge_hi",
            num_attach=num_attach_per_edge,
        )

        # Store per-edge for connecting cloth
        # (Interpretation: we will later call one "left", one "right")
        self.finger_left_attach_ids[index]  = edge_lo_ids
        self.finger_right_attach_ids[index] = edge_hi_ids

        # Combined list kept for viz/backwards compatibility
        self.finger_attach_ids[index] = edge_lo_ids + edge_hi_ids


        self.find_waypoints_tri_indices(index)

        if is_triangle:
            # reform sharp finger
            Lx = ((cell_dim[0] * cell_size[0]) * (self.finger_len // 2 + 1) + (conn_dim[0] * conn_size) * (self.finger_len // 2))
            Ly = cell_dim[1][1] * cell_size[1][1] 
            x0 = cell_dim[0] * cell_size[0]
            beta = Ly - 0.035
            
            self.particle_q, self.tet_poses, self.tri_poses, self.tri_areas = utils.sharp_points(self.particle_q,
                                self.tet_indices,
                                self.tet_poses,
                                self.tri_indices,
                                self.tri_poses,
                                self.tri_areas,
                                Lx,
                                Ly,
                                x0,
                                beta,
                                particle_start_idx,
                                particle_end_idx, 
                                is_triangle)

        # apply transforms
        self.particle_q = utils.transform_points(
            self.particle_q,
            transform,
            particle_start_idx, particle_end_idx)
        self.waypoints[index] = utils.transform_points(
            np.array(self.waypoints[index]),
            transform,
            0, len(self.waypoints[index]))
        

        # MW_ADDED
        # --- for cloth_finger collision ---
        tri_end = len(self.tri_indices)
        # slice out this finger’s triangles and their materials
        finger_tris = np.array(self.tri_indices[tri_start:tri_end], dtype=np.int32)
        finger_mats = np.array(self.tri_materials[tri_start:tri_end], dtype=np.float32)
        self.finger_tri_indices_list.append(finger_tris)
        self.finger_tri_mats_list.append(finger_mats)



    def add_drop_cloth_grid(
        self,
        obj_loader=None,
        cloth_res=(32, 32),
    ):
        """
        MW_ADDED

        Create free-falling cloth for cloth–object behavior tests.
        - Generates rectangular cloth grid using add_cloth_grid().
        - Places it above the YCB object (if obj_loader is given) or above origin.
        - Stores particle indices in self.drop_cloth_ids.
        """

        s = self.scale

        cloth_dim_x, cloth_dim_y = cloth_res # cloth grid resolution
        cloth_cell = 0.01 * s # base cell size, scaled with s

        Lx = cloth_dim_x * cloth_cell
        Ly = cloth_dim_y * cloth_cell

        # approximate vertical placement, make sure the cloth starts over the fingers/ YCB Object:
        # make sure the cloth starts over the fingers/ YCB Object:
        mid_h = 0.0
        if obj_loader is not None and hasattr(obj_loader, "mid_height"):
            mid_h = float(obj_loader.mid_height)

        object_top = 2.0 * mid_h # Top of YCB object is 2 * mid_height

        cloth_y = object_top + 0.1 * s  # margin above object,
        # cloth_y = object_top + 0.5 * s  # further above fingers for cloth - ycb fall test
        # cloth_y = mid_h + finger_height + 0.05 * s

        cloth_pos = wp.vec3(-0.5 * Lx, cloth_y, -0.5 * Ly) # center of cloth in center above YCB Object

        p_start = len(self.particle_q) # save cloth particle ids to enable cloth-finger collision

        cloth_scale = 0.01 # 0.01

        self.add_cloth_grid(
            pos=cloth_pos,
            rot=wp.quat_from_axis_angle(wp.vec3(1.0, 0.0, 0.0), math.pi * 0.5), # grid in x-z plane
            vel=wp.vec3(0.0, 0.0, 0.0),
            dim_x=cloth_dim_x,
            dim_y=cloth_dim_y,
            cell_x=cloth_cell,
            cell_y=cloth_cell,
            mass=0.05 * cloth_scale, # mass per vertex, 33x33x0.05 = 54.45 kg cloth???? v1 to v6: 0.05
            fix_left=False, # set False for fully free falling cloth
            fix_right=False,
            # internal cloth stiffness params, does not affect cloth-object interaction stiffness
            tri_ke=1.0e3 * cloth_scale, # elastic stiffness, was 1.0e3, v5: 2.0e2, v6: 1.0e2, in-plane stretch/shear stiffness of triangles. ###
            tri_ka=1.0e3 * cloth_scale, # area stiffness, was 1.0e3, v5: 2.0e2, v6: 1.0e2, area preservation (resists becoming too squashed/expanded) ###
            tri_kd=1.0e1 * cloth_scale, # damping, was 1.0e1, v5: 5.0e1, v6: 1.0e2, internal damping of the cloth, controlling how much it wiggles after being disturbed ###
        )
        
        p_end = len(self.particle_q) # save cloth particle ids to enable cloth-finger collision
        self.drop_cloth_ids = np.arange(p_start, p_end, dtype=np.int32) # capture the new particle ids, separate variable for the attached cloth




    def add_connecting_cloth(
        self,
        finger_a: int = 0,
        finger_b: int = 1,
        edge_a: str = "right",
        edge_b: str = "left",
        dx_nominal: float | None = None,
        span_subdiv: int | None = None,   # if set, overrides dx-based auto selection
        mass_per_vertex: float = 1e-3,
        tri_ke: float = 1.0e2,
        tri_ka: float = 1.0e2,
        tri_kd: float = 1.0e1,
        tri_drag: float = 0.0,
        tri_lift: float = 0.0,
        max_span_subdiv: int = 32,
    ):
        """
        MW_ADDED

        Create a welded membrane cloth between two fingers.

        - Using existing finger attachment particles (self.finger_attach_ids) as boundary vertices along the finger.
        - Creates additional interior cloth vertices between the fingers.
        - Adds triangles to form a rectangular cloth patch.
        - Stores particle indices in self.attached_cloth_ids. Interior vertices are treated as cloth

        Resolution across the gap is chosen as:
        - If span_subdiv is not None:
            use it directly (clamped to [1, max_span_subdiv]).
        - Else if dx_nominal is not None:
            use dx_nominal as target spacing across the gap.
        - Else:
            estimate dx from the actual attachment points along the finger.
        """

        # basic safety checks
        if len(self.finger_left_attach_ids) <= max(finger_a, finger_b):
            raise ValueError("finger *_attach_ids not initialized or finger index out of range")

        # pick which edge of each finger to use
        edge_ids_a = self._get_edge_ids(finger_a, edge_a)
        edge_ids_b = self._get_edge_ids(finger_b, edge_b)

        ids_a = np.array(edge_ids_a, dtype=np.int32)
        ids_b = np.array(edge_ids_b, dtype=np.int32)


        if ids_a.size == 0 or ids_b.size == 0:
            # nothing to connect
            return

        # ensure both sides use the same number of samples along the finger
        n = min(ids_a.size, ids_b.size)
        if n < 2:
            # cannot build a strip with less than 2 samples along the finger
            return

        ids_a = ids_a[:n]
        ids_b = ids_b[:n]

        # positions of boundary vertices (world space is fine for distances)
        Pa = np.array([self.particle_q[i] for i in ids_a], dtype=float)
        Pb = np.array([self.particle_q[i] for i in ids_b], dtype=float)

        # Choose span_subdiv (number of interior segments across gap)
        if span_subdiv is None:
            # spacing along the finger
            # - if dx_nominal is given: use that
            # - otherwise: measure from attachments
            dx_measured = None
            diffs = np.linalg.norm(Pa[1:] - Pa[:-1], axis=1)
            diffs = diffs[diffs > 1e-6]
            if diffs.size > 0:
                dx_measured = float(np.mean(diffs))

            if dx_nominal is not None:
                dx = float(dx_nominal)
                # if you ever want to compare:
                # print("dx_nominal:", dx, "dx_measured:", dx_measured)
            else:
                dx = dx_measured

            # average gap between the two fingers
            D = float(np.mean(np.linalg.norm(Pb - Pa, axis=1)))

            if dx is not None and dx > 1e-6:
                # isotropic-ish spacing: D / (span_subdiv + 1) ~ dx
                est = D / dx
                span_subdiv_est = int(round(est)) - 1
                span_subdiv = max(1, min(max_span_subdiv, span_subdiv_est))
            else:
                # fallback: reasonable default
                span_subdiv = 4
        else:
            # user (or caller) specified span_subdiv explicitly
            span_subdiv = max(1, min(max_span_subdiv, int(span_subdiv)))

        # Ensure all boundary vertices are wp.vec3 (add_triangle expects this)
        boundary_ids = np.unique(np.concatenate([ids_a, ids_b])).astype(np.int32)
        for pid in boundary_ids:
            p = self.particle_q[pid]
            if isinstance(p, (list, tuple, np.ndarray)):
                p_arr = np.array(p, dtype=float)
                self.particle_q[pid] = wp.vec3(float(p_arr[0]), float(p_arr[1]), float(p_arr[2]))

        # Create interior cloth vertices between finger A and finger B
        #
        # layout per "column" along finger:
        #   index 0          ...          num_across-1
        #    A boundary  --- interior --- interior --- B boundary
        num_across = span_subdiv + 2   # A boundary + interior + B boundary

        # record start index for newly created cloth vertices
        p_start = len(self.particle_q)
        num_new = 0

        # indices for all vertices in the cloth strip [length, across]
        strip_indices = np.full((n, num_across), -1, dtype=np.int32)

        # set boundary columns
        strip_indices[:, 0]             = ids_a
        strip_indices[:, num_across-1]  = ids_b

        # create interior vertices (span_subdiv per column between A and B)
        for i in range(n):
            # positions of boundary vertices
            pa = np.array(self.particle_q[ids_a[i]], dtype=float)
            pb = np.array(self.particle_q[ids_b[i]], dtype=float)

            # create span_subdiv interior points along the straight line from A to B
            for k in range(span_subdiv):
                alpha = float(k + 1) / float(span_subdiv + 1)   # in (0,1)
                pos = (1.0 - alpha) * pa + alpha * pb

                # add as a new particle (cloth vertex)
                self.add_particle(
                    wp.vec3(float(pos[0]), float(pos[1]), float(pos[2])),
                    vel=wp.vec3(0.0, 0.0, 0.0),
                    mass=mass_per_vertex
                )

                # index of this new particle
                strip_indices[i, 1 + k] = p_start + num_new
                num_new += 1

        p_end = p_start + num_new

        # store interior cloth vertex ids (one patch)
        if num_new > 0:
            interior_ids = np.arange(p_start, p_end, dtype=np.int32)
            self.attached_cloth_ids.append(interior_ids)
        else:
            # no interior vertices created -> then this strip is degenerate
            return

        # store edge vertices for viz (finger–cloth boundary)
        edge_ids = np.unique(np.concatenate([ids_a, ids_b])).astype(np.int32)
        self.attached_cloth_edge_ids.append(edge_ids)

        # Build triangles over the grid strip_indices (n along finger, num_across across gap)
        for i in range(n - 1):
            for j in range(num_across - 1):
                v00 = int(strip_indices[i,     j])
                v01 = int(strip_indices[i,     j + 1])
                v10 = int(strip_indices[i + 1, j])
                v11 = int(strip_indices[i + 1, j + 1])

                # two triangles per quad:
                #  v00 ----- v01
                #   |       |
                #   |       |
                #  v10 ----- v11
                self.add_triangle(v00, v10, v11, tri_ke, tri_ka, tri_kd, tri_drag, tri_lift)
                self.add_triangle(v00, v11, v01, tri_ke, tri_ka, tri_kd, tri_drag, tri_lift)




    def select_cloth_finger_attachment_ids(
        self,
        finger_index: int,
        which: str = "spine", # "spine", "edges"
        num_attach: int = 18,
    ):
        """
        MW_ADDED
        
        Select attachment vertices on the actual back of the finger using only local
        coordinates before the global transform.

        1) Use self.finger_particle_ids[finger_index] to get all particles of
        this finger in local coords.

        2) Find the minimal local y (back side of the finger) and keep only
        vertices with y <= y_min + tol.

        3) Sort those candidates along local x (finger length).

        4) Interpret them as 3 'rows' according to layout and select
        spine / edges / all.

        5) Subsample to num_attach points.
        """

        finger_ids = self.finger_particle_ids[finger_index]
        if not finger_ids:
            return []

        # gather local positions
        P = np.array([self.particle_q[i] for i in finger_ids])
        ys = P[:, 1]

        # back of finger = smallest local y
        y_min = float(ys.min())

        # small tolerance to include only the outermost layer
        # adjust if needed, but keep it much smaller than cell_y (~0.002–0.003)
        tol = 1e-4 * self.scale

        # indices of finger_ids that lie on the back surface (local y ≈ y_min)
        back_mask = ys <= (y_min + tol)
        cand = np.array([fid for fid, m in zip(finger_ids, back_mask) if m], dtype=np.int32)

        if cand.size == 0:
            # fallback: if something went wrong, just use all finger nodes
            cand = np.array(finger_ids, dtype=np.int32)

        # store all back-surface vertices for this finger
        self.finger_back_ids_per_finger[finger_index] = cand.tolist()

        # positions of back verts
        P_back = np.array([self.particle_q[i] for i in cand])
        xs = P_back[:, 0]   # finger length direction
        zs = P_back[:, 2]   # across finger "height"/width on back surface
        
        n = cand.size
        if n < 3:
            return cand.tolist()
        
        #  Layout: line -> longitudinal line along finger
        # group back vertices by x (columns along length)
        tol_x = 1e-6 * self.scale
        order_x = np.argsort(xs)
        cols = []
        current_col = []
        current_x = None

        for idx in order_x:
            x = xs[idx]
            if current_x is None or abs(x - current_x) <= tol_x:
                current_col.append(idx)
                if current_x is None:
                    current_x = x
            else:
                cols.append(np.array(current_col, dtype=int))
                current_col = [idx]
                current_x = x

        if current_col:
            cols.append(np.array(current_col, dtype=int))

        spine_idx = []
        edge_lo_idx = []
        edge_hi_idx = []

        for col in cols:
            if col.size == 0:
                continue

            z_col = zs[col]
            order_z = np.argsort(z_col)
            col_sorted = col[order_z]

            # edges
            edge_lo_idx.append(col_sorted[0])
            if col_sorted.size > 1:
                edge_hi_idx.append(col_sorted[-1])

            # spine = middle element of this column
            mid_index = col_sorted[col_sorted.size // 2]
            spine_idx.append(mid_index)

        spine_ids = cand[np.array(spine_idx, dtype=int)]
        edge_ids_lo = cand[np.array(edge_lo_idx, dtype=int)] if edge_lo_idx else np.array([], dtype=int)
        edge_ids_hi = cand[np.array(edge_hi_idx, dtype=int)] if edge_hi_idx else np.array([], dtype=int)

        if which == "spine":
            ids = spine_ids
        elif which == "edges":
            # both edges together
            ids = np.concatenate([edge_ids_lo, edge_ids_hi])
        elif which == "edge_lo":
            # one edge only (z-min side in local coords)
            ids = edge_ids_lo
        elif which == "edge_hi":
            # other edge only (z-max side in local coords)
            ids = edge_ids_hi
        else:  # all (spine + edges)
            ids = np.concatenate([edge_ids_lo, spine_ids, edge_ids_hi])


        # sort selected ids along finger length
        P_sel = np.array([self.particle_q[i] for i in ids])
        xs_sel = P_sel[:, 0]
        order_sel = np.argsort(xs_sel)
        ids = ids[order_sel]

        # subsample along length to num_attach points
        if num_attach is not None and num_attach > 0 and ids.size > num_attach:
            idxs = np.linspace(0, ids.size - 1, num_attach).astype(int)
            ids = ids[idxs]

        return ids.tolist()

    def _get_edge_ids(self, finger_index: int, edge: str):
        """
        MW_ADDED
        
        Helper: return the attachment ids for a given finger and edge label.
        edge: "left" or "right"
        """
        if edge == "left":
            return self.finger_left_attach_ids[finger_index]
        elif edge == "right":
            return self.finger_right_attach_ids[finger_index]
        else:
            raise ValueError(f"Unknown edge label '{edge}', expected 'left' or 'right'")


    
    def find_back_ids(self, 
                      particle_start_idx, particle_end_idx, 
                      low_threshold=np.zeros(3),
                      high_threshold=np.zeros(3)+1e5):
        for i in range(particle_start_idx, particle_end_idx):
            flag = True
            for index in range(3):
                if self.particle_q[i][index] < low_threshold[index]:
                    flag = False
                if self.particle_q[i][index] > high_threshold[index]:
                    flag = False
            if not flag: continue
            if i not in self.finger_back_ids:
                self.finger_back_ids.append(i)

    def find_waypoints_tri_indices(self, n):
        waypoint_ids = []
        waypoints_tri_indices = []
        for i in range(len(self.waypoints[n])):
            this_wp = self.waypoints[n][i]
            c_idx, face_idx = self.find_triangle_idx(
                this_wp,
                wp.array(self.particle_q).numpy(),
                wp.array(self.tri_indices).numpy())
            waypoint_ids.append(c_idx)
            waypoints_tri_indices.append(face_idx)
        self.waypoint_ids[n] = waypoint_ids
        self.waypoints_tri_indices[n] = waypoints_tri_indices

    def generate_tendon_waypoints_hirose(
            self, cell_dim, cell_size, 
            conn_dim, conn_size, 
            index=0):
        waypoints = []
        waypoint_pair_ids = []
        
        Lx = ((cell_dim[0]*cell_size[0])*(self.finger_len//2 +1) +
            (conn_dim[0]*conn_size)*(self.finger_len//2))
        Ly = cell_dim[1][1]*cell_size[1][1] 
        z_perct = 0.5* np.ones(self.finger_len)
        base_offset = cell_size[0]*cell_dim[0]

        def get_y(x):
            y = Ly * (1 - x/Lx)**2
            y += cell_size[1][0]*conn_dim[1] + 2e-3
            return y

        for i in range(self.finger_len):
            pos = [0, 0, 0]
            if (i % 2) == 0:
                # finger body block
                pos = [base_offset, 
                        get_y(base_offset), 
                        z_perct[i]*(cell_size[2]*cell_dim[2])]
                base_offset += conn_size*conn_dim[0]
                waypoint_pair_ids.append(i+1 if i < self.finger_len-1 else i)
            else:
                # connector block
                pos = [base_offset,
                        get_y(base_offset),
                        z_perct[i]*(cell_size[2]*cell_dim[2])]
                base_offset += cell_size[0]*cell_dim[0]
                waypoint_pair_ids.append(i-1)
            waypoints.append(pos) 
        self.waypoint_pair_ids[index] = waypoint_pair_ids
        
        waypoints = np.array(waypoints)
        self.waypoints[index] = waypoints.tolist()

    def find_triangle_idx(self, waypt, points, 
                          faces=None,
                          start_idx=0, end_idx=0):
        if end_idx <= start_idx:
            end_idx = points.shape[0]
        force_idx = []
        curr_dis = np.linalg.norm(points[0, :] - waypt)
        for i in range(start_idx, end_idx):
            dis = np.linalg.norm(points[i, :] - waypt)
            if dis < curr_dis:
                curr_dis = dis
                force_idx.append(i)

        candidate = None
        max_area = 0.0
        for fid in range(len(force_idx)):
            this_force_idx = force_idx[len(force_idx)-1-fid]
            for i in range(faces.shape[0]):
                if this_force_idx in faces[i, :]:
                    # return this_force_idx, i
                    # find the face that is verticle
                    face_pts = points[faces[i, :], :]
                    mid_pt = np.mean(face_pts, axis=0)
                    if np.abs(mid_pt[0] - waypt[0]) < 1e-6:
                        # return this_force_idx, i
                        this_area = np.linalg.norm(np.cross(face_pts[1, :] - face_pts[0, :], face_pts[2, :] - face_pts[0, :]))
                        if this_area > max_area:
                            max_area = this_area
                            candidate = (this_force_idx, i)
        if candidate: return candidate
        assert False, "No verticle triangle found"

    def generate_fixpoints(self, cell_dim, cell_size, 
                           curr_points, index=0,
                           start_idx=0, end_idx=0):
        offset = []
        for y_dim in cell_dim[1]:
            y_base = y_dim * cell_size[1][0]
            for yi in range(y_dim):
                for zi in range(cell_dim[2]):
                    offset.append([0.0, y_base + yi*cell_size[1][1], zi*cell_size[2]])

        self.fixpoints[index] = offset
        for i in range(len(offset)):
            c_idx, _ = self.find_triangle_idx(
                offset[i],
                np.array(curr_points),
                start_idx=start_idx, end_idx=end_idx)
            self.fixpoints_ids[index].append(c_idx)


    def finalize(self, device=None, requires_grad=False) -> TendonModel:
        tendon_model = TendonModel(device)
        m = super().finalize(device=device, requires_grad=requires_grad)
        tendon_model.__dict__.update(m.__dict__)
        return tendon_model
    





class TendonRenderer(wp.sim.render.SimRenderer):
    def render(self, state, highlight_pt_ids=[], force_arr=[], force_scale=0.1, additional_pts=None):
        # super().render(state)
        if self.skip_rendering:
            return

        if self.model.particle_count:
            particle_q = state.particle_q.numpy()

            # render particles
            self.render_points(
                "particles", particle_q, 
                radius=self.model.particle_radius.numpy(), 
                colors=(0.8, 0.3, 0.2)
            )
            if len(highlight_pt_ids) > 0:
                self.render_points(
                    "force_particles", particle_q[highlight_pt_ids, :],
                    radius=0.01,
                    colors=(1, 0, 0)
                )

            # render tris
            if self.model.tri_count:
                self.render_mesh(
                    "surface",
                    particle_q,
                    self.model.tri_indices.numpy().flatten(),
                    colors=(((0.75, 0.25, 0.0),) * len(particle_q)),
                )

            # render springs
            if self.model.spring_count:
                self.render_line_list(
                    "springs", particle_q, self.model.spring_indices.numpy().flatten(), (0.25, 0.5, 0.25), 0.02
                )
            
            if len(force_arr) > 0:
                assert len(force_arr) == len(highlight_pt_ids)
                this_force_arr = force_arr.numpy()
                for i in range(len(highlight_pt_ids)):
                    idx = highlight_pt_ids[i]
                    f_arr = this_force_arr[i] * force_scale
                    f_end = particle_q[idx] + f_arr
                    points = [particle_q[idx], f_end]
                    # print("force point:", points)
                    self.render_line_strip(
                        name=f"force_{i}", vertices=points, color=(0.25, 0.5, 0.25), radius=0.005)
            
        if additional_pts is not None:
            for i in range(len(additional_pts)):
                additional_pt = additional_pts[i].numpy()
                self.render_points(
                    f"additional_pt{i}", additional_pt, 
                    radius=0.005, colors=(0.2*i, 0.0, 1.0)
                )

        # update bodies
        if self.model.body_count:
            self.update_body_transforms(state.body_q)


