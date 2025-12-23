import os
import numpy as np
import torch
import math
import csv
import matplotlib.pyplot as plt

import warp as wp
import utils

from tendon_model import TendonModel, TendonModelBuilder, TendonRenderer, TendonHolder
from object_loader import ObjectLoader


"""
Rigid init optimisation (finger + optional cloth proxy)

1) Convert (transform_9d, transform_2d) -> joint_q
   - transform_9d: wrist SE(3) in 9D (t + first two rotation columns)
   - transform_2d: per finger prismatic "radius" values
2) Run FK to compute rigid body poses (state_in.body_q) and copy to state_out.
3) For each finger i:
   - transform local mesh vertices -> world space (curr_finger_mesh[i])
   - accumulate finger-object SDF penalty at finger vertices (utils.mesh_dis)
4) Cloth proxy penalty (optional):
   - for each neighbouring finger pair (a, b), take the precomputed “facing edge”
     vertex id lists from the attachment edges:
       ids_a = cloth_left_ids[a], ids_b = cloth_right_ids[b]
   - for each alpha in cloth_alphas, create proxy points
       p = (1-alpha) * verts_a[ids_a] + alpha * verts_b[ids_b]
     (computed inside the kernel, no intermediate buffers)
   - evaluate object mesh SDF at p and add hinge-style penalties:
       penetration within margin, and optionally being too far from the surface
"""



class ForwardKinematics(torch.autograd.Function):
    @staticmethod
    def forward(ctx, transform_9d, 
                transform_2d,
                model,
                state_in,
                state_out,
                integrator,
                sim_dt,
                curr_finger_mesh,
                finger_mesh,
                finger_body_ids,
                object_com,
                distance_param,
                finger_dis,
                cloth_dis,
                obj_geo_id,
                obj_body_id,
                consider_cloth,
                cloth_pairs,
                cloth_left_ids,
                cloth_right_ids,
                cloth_alphas,
                cloth_margin_mult,
                ):
        ctx.tape = wp.Tape()
        ctx.model = model
        ctx.state_in = state_in
        ctx.state_out = state_out
        ctx.intergrator = integrator
        ctx.sim_dt = sim_dt
        ctx.transform_9d = wp.from_torch(transform_9d)
        ctx.transform_2d = wp.from_torch(transform_2d)
        ctx.joint_q = model.joint_q

        ctx.curr_finger_mesh = curr_finger_mesh
        
        ctx.finger_mesh = finger_mesh

        ctx.finger_dis = finger_dis
        ctx.cloth_dis  = cloth_dis

        ctx.obj_geo_id = obj_geo_id
        ctx.obj_body_id = obj_body_id
        ctx.consider_cloth = consider_cloth

        ctx.cloth_pairs = cloth_pairs
        ctx.cloth_left_ids = cloth_left_ids
        ctx.cloth_right_ids = cloth_right_ids
        ctx.cloth_alphas = cloth_alphas

        ctx.cloth_margin_mult = cloth_margin_mult

        with ctx.tape:
            ctx.state_in.clear_forces()
            wp.launch(utils.transform_from11d, dim=1, 
                    inputs=[ctx.transform_9d, ctx.transform_2d],
                    outputs=[ctx.joint_q])
            
            wp.sim.eval_fk(ctx.model, ctx.joint_q, ctx.model.joint_qd, None, ctx.state_in) # computes the rigid body poses from kinematics into state_in.body_q
            
            # wp.sim.collide(ctx.model, ctx.state_in)
            # ctx.intergrator.simulate(ctx.model, ctx.state_in, ctx.state_out, ctx.sim_dt) # advances one rigid dynamics step and writes poses into state_out.body_q

            # Copy FK poses to state_out so the rest of the code stays unchanged
            ctx.state_out.body_q = wp.clone(ctx.state_in.body_q)

            for i in range(len(curr_finger_mesh)):
                wp.launch(utils.transform_points_out,
                        dim=len(ctx.curr_finger_mesh[i]),
                        inputs=[ctx.finger_mesh[i],
                                finger_body_ids[i],
                                ctx.state_out.body_q],
                        outputs=[ctx.curr_finger_mesh[i]])
            
                wp.launch(utils.mesh_dis, 
                        dim=len(ctx.curr_finger_mesh[i]),
                        inputs=[ctx.model.shape_geo, 
                                ctx.obj_geo_id,
                                ctx.model.shape_body,
                                ctx.model.shape_transform,
                                ctx.state_out.body_q,
                                ctx.curr_finger_mesh[i],
                                # ctx.model.rigid_contact_margin*1.0,
                                ctx.model.object_contact_margin*1.0,
                                1e-1,
                                1e3, 0],
                        outputs=[ctx.finger_dis])

            
            # Cloth-aware init: proxy cloth penalty

            if ctx.consider_cloth and (len(ctx.cloth_pairs) > 0):
                #k = ctx.cloth_k  # use the actual k
                margin = ctx.model.object_contact_margin * ctx.cloth_margin_mult
                beta = 50.0
                d_target = margin   # has no visible effect (compared 0.3 to 1000)
                cloth_dist_param = 0.0
                cloth_pen_param = 1e8

                for (a, b) in ctx.cloth_pairs:
                    ids_a = ctx.cloth_left_ids[a]
                    ids_b = ctx.cloth_right_ids[b]

                    k = ids_a.shape[0]

                    for alpha in ctx.cloth_alphas:
                        wp.launch(
                            utils.cloth_proxy_barrier_pair,
                            dim=k,
                            inputs=[ctx.model.shape_geo,
                                ctx.obj_geo_id,
                                ctx.model.shape_body,
                                ctx.model.shape_transform,
                                ctx.state_out.body_q,
                                ctx.curr_finger_mesh[a], ids_a,
                                ctx.curr_finger_mesh[b], ids_b,
                                float(alpha),
                                margin, beta, 
                                cloth_dist_param, cloth_pen_param, 
                                d_target,
                            ],
                            outputs=[ctx.cloth_dis],
                        )
    
        # return torch tensor
        finger_loss = wp.to_torch(ctx.finger_dis)[0]
        cloth_loss  = wp.to_torch(ctx.cloth_dis)[0]
        return torch.stack([finger_loss, cloth_loss])

    
    @staticmethod
    def backward(ctx, adj_total_dis):
        max_grad_trans = 1.0
        max_grad_rot = 1e-8

        # seed output grads (shape (1,) each)
        ctx.finger_dis.grad = wp.from_torch(adj_total_dis[0:1].contiguous())
        ctx.cloth_dis.grad  = wp.from_torch(adj_total_dis[1:2].contiguous())

        ctx.tape.backward()

        trans9d_grad = wp.to_torch(ctx.tape.gradients[ctx.transform_9d]).clone()
        trans2d_grad = wp.to_torch(ctx.tape.gradients[ctx.transform_2d]).clone()
        utils.remove_nan(trans9d_grad)
        utils.remove_nan(trans2d_grad)
        trans9d_grad.clamp_(-max_grad_trans, max_grad_trans)
        trans2d_grad.clamp_(-max_grad_trans, max_grad_trans)
        trans9d_grad[0].zero_()
        #trans9d_grad[1].zero_()
        trans9d_grad[2].zero_()
        trans9d_grad[3:].zero_()

        ctx.tape.zero()
     
        # forward() now has 25 args (2 tensors + 21 non-tensors)
        return tuple([trans9d_grad] + [trans2d_grad] + [None]*20)


class InitializeFingers:
    def __init__(self, stage_path="femtendon_sim.usd", 
                 finger_len=9, finger_rot=0.01,
                 finger_width=0.08,
                 stop_margin=0.01,
                 num_frames=30, 
                 iterations=10000,
                 scale=5.0,
                 num_envs=1,
                 ycb_object_name='',
                 object_rot=wp.quat_identity(),
                 is_render=True,
                 verbose=False,
                 is_triangle=False,
                 pose_id=0,
                 add_random=False,
                 init_height_offset=0.0,
                 post_height_offset=0.0, 
                 finger_num=2,
                 is_ood=False,
                 consider_cloth=True, # consider cloth
                 cloth_k=16,
                 cloth_alphas=(0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9),
                 cloth_margin_mult=1.0,
                 ):
        self.pose_id = pose_id
        self.verbose = verbose
        self.is_render = is_render
        self.flag = True
        fps = 100
        self.frame_dt = 1.0 / fps
        self.num_frames = num_frames
        self.train_iter = iterations
        self.num_envs = num_envs
        self.is_triangle = is_triangle
        self.add_random = add_random

        self.sim_substeps = 1
        self.sim_dt = self.frame_dt / self.sim_substeps
        self.sim_time = 0.0
        self.render_time = 0.0
        self.requires_grad = True 
        self.is_ood = is_ood

        self.torch_device = wp.device_to_torch(wp.get_device())

        curr_dir = os.path.dirname(os.path.realpath(__file__))
        if stage_path:
            self.stage_path = curr_dir + "/../output/" + stage_path

        self.obj_loader = ObjectLoader()
        self.obj_name = 'ycb'
        self.ycb_object_name = ycb_object_name
        if finger_len % 2 == 0:
            raise ValueError("finger_len should be odd number")
        if finger_rot < 0.0 or finger_rot > np.pi/4:
            raise ValueError("finger_rot should be in [0, pi/4]")
        self.finger_num = finger_num
        self.finger_len = finger_len # need to be odd number
        self.finger_rot = finger_rot
        self.finger_width = finger_width
        self.stop_margin = stop_margin
        self.scale = scale
        self.object_com = None
        self.init_height_offset = init_height_offset
        self.post_height_offset = post_height_offset

        # cloth-aware init (proxy constraint)
        self.consider_cloth = consider_cloth
        self.cloth_k = int(cloth_k)
        self.cloth_alphas = tuple(float(a) for a in cloth_alphas)
        self.cloth_margin_mult = float(cloth_margin_mult)
        
        self.integrator = wp.sim.SemiImplicitIntegrator()
        self.build_rigid_model(object_rot)
        self.object_com = wp.to_torch(self.object_com, requires_grad=True)
        self.joint_q = wp.array(self.model.joint_q.numpy(), dtype=wp.float32, requires_grad=True)
        self.transform_9d_wp= wp.zeros(9, dtype=wp.float32, requires_grad=True)
        # for arbitrary number of points (e.g. one value per finger)
        self.transform_2d_wp = wp.zeros(self.finger_num, dtype=wp.float32, requires_grad=True)

        self.log = False
        # --- History Storage ---
        self.loss_history = []
        self.radius_history = []
        self.iter_history = []


        wp.launch(utils.transform_to11d, dim=1,
                  inputs=[self.joint_q],
                  outputs=[self.transform_9d_wp, self.transform_2d_wp])
        
        self.state0 = self.model.state(requires_grad=True)
        self.state1 = self.model.state(requires_grad=True)
        self.finger_dis = wp.zeros(1, dtype=wp.float32, requires_grad=True)
        self.cloth_dis = wp.zeros(1, dtype=wp.float32, requires_grad=True)
        
        # things to optimize in torch
        self.transform_9d = wp.to_torch(self.transform_9d_wp, requires_grad=True)
        self.transform_2d = wp.to_torch(self.transform_2d_wp, requires_grad=True)

        self.optimizer = torch.optim.SGD([
            {'params': self.transform_9d, 'lr': 1e-1, 'weight_decay': 1e-5},
            {'params': self.transform_2d, 'lr': 1e-2}, 
        ])
        self.scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(self.optimizer, 'min', patience=50, factor=0.05)

        # loss
        self.loss = 0
        self.init_count = 0
        
        if stage_path and is_render:
            self.renderer = TendonRenderer(self.model, self.stage_path, scaling=1.0)
        else:
            self.renderer = None

        self.use_cuda_graph = wp.get_device().is_cuda
        self.use_cuda_graph = False
    
    def build_rigid_model(self, object_rot):
        # --- build the soft finger geometry (all fingers) ---
        soft_builder = TendonModelBuilder()
        soft_builder.init_builder_tendon_variables(
            self.finger_num, self.finger_len, self.scale, self.requires_grad
        )
        self.init_transforms, finger_actual_len, finger_THK = soft_builder.build_fem_model(
            finger_width=self.finger_width,
            finger_rot=self.finger_rot,
            obj_loader=None,
            h_dis_init=0.2,
            is_triangle=self.is_triangle,
        )
        if self.verbose:
            for i, init_trans in enumerate(self.init_transforms):
                t = wp.transform_get_translation(init_trans)
                q = wp.transform_get_rotation(init_trans)
                print(f"[InitPose] finger {i}: init transform (world): translation t={list(t)} rotation q_xyzw={list(q)}")

        # --- build per-finger meshes using recorded ranges ---
        finger_mesh = []
        self.finger_mesh = []
        self.curr_finger_mesh = []

        # REQUIRE: soft_builder.finger_vertex_ranges / finger_face_ranges are populated

        finger_vertices_np = []

        for i in range(self.finger_num):
            vs, ve = soft_builder.finger_vertex_ranges[i]
            fs, fe = soft_builder.finger_face_ranges[i]

            vertices_i = soft_builder.particle_q[vs:ve]
            faces_i    = soft_builder.tri_indices[fs:fe]

            vertices_i_np = np.asarray(vertices_i, dtype=np.float32)
            finger_vertices_np.append(vertices_i_np)

            finger_mesh.append(wp.sim.Mesh(vertices_i, faces_i))
            self.finger_mesh.append(wp.array(vertices_i, dtype=wp.vec3, requires_grad=True))
            self.curr_finger_mesh.append(wp.array(vertices_i, dtype=wp.vec3, requires_grad=True))

        
        # Cloth-aware init: build side IDs from real attachment edges
        if self.consider_cloth:
            self.cloth_left_ids = []
            self.cloth_right_ids = []

            # attachment count per edge is defined in TendonModelBuilder.add_finger()
            # (num_attach_per_edge = 18 in your tendon_model.py)
            # use that as cloth_k so buffers match perfectly
            k_attach = len(soft_builder.finger_left_attach_ids[0])
            self.cloth_k = k_attach

            for i in range(self.finger_num):
                vs, ve = soft_builder.finger_vertex_ranges[i]
                n_local = ve - vs

                left_global = np.asarray(soft_builder.finger_left_attach_ids[i], dtype=np.int32)
                right_global = np.asarray(soft_builder.finger_right_attach_ids[i], dtype=np.int32)

                left_local = left_global - vs
                right_local = right_global - vs

                # safety checks
                assert left_local.min() >= 0 and left_local.max() < n_local, (i, left_local.min(), left_local.max(), n_local)
                assert right_local.min() >= 0 and right_local.max() < n_local, (i, right_local.min(), right_local.max(), n_local)

                self.cloth_left_ids.append(wp.array(left_local, dtype=wp.int32, requires_grad=False))
                self.cloth_right_ids.append(wp.array(right_local, dtype=wp.int32, requires_grad=False))

            # neighbour pairs, same as before
            if self.finger_num == 2:
                self.cloth_pairs = [(0, 1)]
            else:
                self.cloth_pairs = [(i, (i + 1) % self.finger_num) for i in range(self.finger_num)]

        else:
            # dummies so ForwardKinematics.apply always has something to pass
            self.cloth_left_ids = []
            self.cloth_right_ids = []
            self.cloth_pairs = []
            self.cloth_edge_a = wp.zeros(1, dtype=wp.vec3, requires_grad=True)
            self.cloth_edge_b = wp.zeros(1, dtype=wp.vec3, requires_grad=True)
            self.cloth_samples = wp.zeros(1, dtype=wp.vec3, requires_grad=True)


        # --- start assembling the rigid scene used for initialization ---
        self.builder = wp.sim.ModelBuilder()
        self.builder.add_articulation()

        if self.is_ood:
            curr_dir = os.path.dirname(os.path.realpath(__file__))
            self.obj_loader.data_dir = curr_dir + "/../models/ood/"

        # object body + geometry
        self.object_com, self.obj_body_id, self.obj_geo_id = utils.load_object(
            self.builder,
            self.obj_loader,
            object=self.obj_name,
            ycb_object_name=self.ycb_object_name,
            obj_rot=object_rot,
            scale=self.scale,
            use_simple_mesh=False,
            is_fix=True,
        )
        # NOTE: Don’t combine `is_fix=False` with an extra `add_joint_fixed(parent=-1, child=obj_body_id)` here.
        # Together with the wrist free joint (also parent=-1), this creates multiple roots in the articulation,
        # which messes with FK/dynamics (and gradients) and the finger-only optimisation stops converging.

        # create finger rigid bodies
        self.finger_body_idx = []
        for _ in range(self.finger_num):
            self.finger_body_idx.append(
                self.builder.add_body(origin=wp.transform_identity())
            )

        # wrist body is free; fingers connect via prismatic joints
        wrist_body = self.builder.add_body(origin=wp.transform_identity())
        self.builder.add_joint_free(parent=-1, child=wrist_body)

        # contact/thickness-based limits (same for all prismatic joints for now)
        limit_low, limit_upp = -6 * finger_THK , 6 * finger_THK # was 6 * finger_THK
        self.limit_low, self.limit_upp = limit_low, limit_upp
        self.prismatic_limit = float(limit_upp) # store one number

        # add collision shapes for each finger
        self.finger_shape_ids = []
        for i in range(self.finger_num):
            finger_shape_id = self.builder.add_shape_mesh(
                body=self.finger_body_idx[i],
                mesh=finger_mesh[i],
                density=1e1,
                ke=1.0e2,
                kd=1.0e-5,
                kf=1e1,
                mu=1.0,
            )
            self.finger_shape_ids.append(finger_shape_id)

        # prismatic joints with **radial** axis per finger
        self.prismatic_ids = []
        for i in range(self.finger_num):
            theta = 2.0 * math.pi * i / self.finger_num
            # radial direction in xz plane
            axis = wp.vec3(-math.cos(theta), 0.0, -math.sin(theta))
            jid = self.builder.add_joint_prismatic(
                parent=wrist_body,
                child=self.finger_body_idx[i],
                axis=axis,
                limit_lower=-limit_upp,
                limit_upper= limit_upp,
            )
            self.prismatic_ids.append(jid)

        # finalize and load initial pose
        self.model = self.builder.finalize(requires_grad=True)


        shape_body = self.model.shape_body.numpy()          # shape -> body mapping
        obj_shape_ids = np.where(shape_body == self.obj_body_id)[0]
        assert len(obj_shape_ids) > 0, f"No shapes found for obj_body_id={self.obj_body_id}"

        # if the object has multiple shapes, just pick the first for now
        self.obj_shape_id = int(obj_shape_ids[0])

        # print object info
        if self.verbose:
            print(
                "[InitPose] Object IDs\n"
                f" obj_body_id    = {self.obj_body_id} (shape attached body; -1 can mean static/world)\n"
                f" obj_geo_id     = {self.obj_geo_id} (from loader; check what kernels expect)\n"
                f" obj_shape_id   = {self.obj_shape_id} (index into model.shape_* arrays)\n"
                f" num_obj_shapes = {len(obj_shape_ids)}"
            )


        self.init_circle_pose()
        #self.load_grasp_pose(self.pose_id)  # make sure this now sets 7 + finger_num DOFs

        # collision margin
        self.model.object_contact_margin = self.stop_margin * self.scale


    def capture_proxy_points_frozen(self):
        """
        Capture EXACT proxy sample points (all pairs x all alphas x cloth_k points)
        from the CURRENT curr_finger_mesh buffers.
        Result is stored as self.proxy_pts_frozen (numpy float32, shape (M,3)).
        """
        if (not self.consider_cloth) or (len(self.cloth_pairs) == 0):
            self.proxy_pts_frozen = None
            return None

        # Make sure curr_finger_mesh matches the *current* transform params
        # (this runs one forward pass which updates curr_finger_mesh)
        with torch.no_grad():
            _ = self.forward(distance_param=1e-1, use_com=True)

        pts = []
        for (a, b) in self.cloth_pairs:
            ids_a = self.cloth_left_ids[a].numpy().astype(np.int32)   # local ids into curr_finger_mesh[a]
            ids_b = self.cloth_right_ids[b].numpy().astype(np.int32)  # local ids into curr_finger_mesh[b]

            Pa = self.curr_finger_mesh[a].numpy()[ids_a]  # (k,3)
            Pb = self.curr_finger_mesh[b].numpy()[ids_b]  # (k,3)

            for alpha in self.cloth_alphas:
                pts.append((1.0 - alpha) * Pa + alpha * Pb)

        self.proxy_pts_frozen = np.vstack(pts).astype(np.float32)  # (M,3)
        return self.proxy_pts_frozen


    def sweep_R0(self, distance_param=1e-1, use_com=True):
        """
        Sweep a single shared radius value across all fingers.
        span is fraction of limit_upp around the current init value.
        """
        with torch.no_grad():
            span=0.6
            num=11
            r_init = float(self.transform_2d[0].item())

            lo = max(-self.limit_upp, r_init - span * self.limit_upp)
            hi = min(+self.limit_upp, r_init + span * self.limit_upp)
            candidates = np.linspace(lo, hi, num, dtype=np.float32)

            best_r = r_init
            best_E = float("inf")

            # keep a copy (in case you want to restore)
            r_backup = self.transform_2d.clone()

            for r in candidates:
                self.transform_2d.fill_(float(r))

                td = self.forward(distance_param=distance_param, use_com=use_com)
                finger_loss = float(td[0].item())
                cloth_loss  = float(td[1].item())

                # match your normalisation in compute_loss()
                if self.consider_cloth and len(self.cloth_pairs) > 0:
                    cloth_loss = cloth_loss / (len(self.cloth_pairs) * len(self.cloth_alphas) * self.cloth_k)

                E = finger_loss + cloth_loss

                if E < best_E:
                    best_E = E
                    best_r = float(r)

            # set best radius
            self.transform_2d.copy_(r_backup)
            self.transform_2d.fill_(best_r)

            print(
                f"[InitPose] R0 sweep: r_init={r_init:.4g} "
                f"search limits=[{lo:.4g}, {hi:.4g}] n={num} -> best_r={best_r:.4g} best_E={best_E:.4g}"
            )


    def debug_print_proxy_sdf(self, state_for_T=None, stride=10, max_points=2000):
        if state_for_T is None:
            state_for_T = self.state1

        margin = float(self.model.object_contact_margin * self.cloth_margin_mult)
        print(f"[debug sdf per pair] margin={margin:.6g}")

        for pair_id, (a, b) in enumerate(self.cloth_pairs):
            ids_a = self.cloth_left_ids[a].numpy().astype(np.int32)
            ids_b = self.cloth_right_ids[b].numpy().astype(np.int32)

            Pa = self.curr_finger_mesh[a].numpy()[ids_a]
            Pb = self.curr_finger_mesh[b].numpy()[ids_b]

            # collect all points for this pair
            pts = []
            for alpha in self.cloth_alphas:
                pts.append((1.0 - alpha) * Pa + alpha * Pb)
            pts = np.vstack(pts).astype(np.float32)

            pts_wp = wp.array(pts, dtype=wp.vec3)
            d_raw  = wp.zeros(len(pts), dtype=wp.float32)
            near_c = wp.zeros(1, dtype=wp.int32)
            neg_c  = wp.zeros(1, dtype=wp.int32)

            wp.launch(
                utils.debug_sdf_points,
                dim=len(pts),
                inputs=[
                    self.model.shape_geo,
                    int(self.obj_geo_id),
                    self.model.shape_body,
                    self.model.shape_transform,
                    state_for_T.body_q,
                    pts_wp,
                    margin,
                ],
                outputs=[d_raw, near_c, neg_c],
            )

            dr = d_raw.numpy()
            contact_margin = float(self.model.object_contact_margin)
            n_contact = np.sum(dr < contact_margin)

            status = "OK"
            if int(neg_c.numpy()[0]) > 0:
                status = "PENETRATION"
            elif dr.min() < margin:
                status = "WITHIN_MARGIN"
            print(
                f"[ProxySDF] pair {pair_id} ({a}->{b}) {status}: "
                f"min_d={dr.min():.4g} max_d={dr.max():.4g} "
                f"neg={int(neg_c.numpy()[0])} near={int(near_c.numpy()[0])} "
                f"margin={margin:.4g} contact_margin={contact_margin:.4g} contact_pts={int(n_contact)}"
            )

            # print worst few for this pair
            idx = np.argsort(np.abs(dr))[:3]
            for j in idx:
                print(f"     closest d_raw={dr[j]: .6g} p={pts[j]}")



    def init_circle_pose(self):
        # --- get object position from the model state ---
        # create a temporary state to read initial body poses
        tmp_state = self.model.state()
        bq = tmp_state.body_q.numpy()[self.obj_body_id]  # shape (7,) -> [x,y,z,qx,qy,qz,qw]
        t_gb = bq[0:3].astype(np.float32)

        # lift wrist a bit in +y if you like
        t_gb[1] += self.init_height_offset

        # simple orientation (identity)
        # adjust if you want the “open side” to face +x, etc.
        R_quat = np.array([0.0, 0.0, 0.0, 1.0], dtype=np.float32)  # x,y,z,w

        # initial radius (inside joint limits)
        R0 = -0.2* self.limit_upp  # e.g. 20% of max extension
        finger_init = [R0] * self.finger_num  # all same radius to start

        self.model.joint_q = wp.array(
            t_gb.tolist() + R_quat.tolist() + finger_init,
            dtype=wp.float32,
            requires_grad=True,
        )

        # if you really want, you can also bias left/right:
        # for i in range(self.finger_num):
        #     sign = 1.0  # or -1.0 for some fingers
        #     finger_init[i] = sign * R0


    def load_grasp_pose(self, pose_id):
        self.pose_id = pose_id
        # initialization convert
        T_kc, T_gk, T_cb, T_gb = np.eye(4), np.eye(4), np.eye(4), np.eye(4)
        path = os.path.dirname(os.path.realpath(__file__)) + "/../pose_info/"
        T_path = path + str(self.ycb_object_name)+".npz"
        data = np.load(T_path)
        total_pose = data['T_pose'].shape[0]
        self.total_pose = total_pose
        if self.pose_id == -1:
            self.pose_id = np.random.randint(0, total_pose)
        pose_id = self.pose_id % total_pose
        print("Loading pose id:", pose_id, "out of total poses:", total_pose)
        T_kc = data['T_pose'][pose_id]
        if self.add_random:
            T_kc = utils.add_random_to_pose(
                T_kc, 
                t_std=3e-3*self.scale,
                r_std=5e-3)
        
        T_gk[:3, :3] = np.array([[ 0.0, -1.0, 0.0],
                                 [-1.0, 0.0, 0.0],
                                 [0.0, 0.0, -1.0]])
        T_gk[:3, 3] = np.array([0.0, 0.0, 0.0])
        
        T_gc = np.matmul(T_kc, T_gk)
        
        T_cb[:3, :3] = np.array([[-1.0, 0.0, 0.0],
                                 [0.0, 0.0, -1.0],
                                 [0.0, -1.0, 0.0]])
        T_cb[:3, 3] = np.array([0.0, 0.5, 0.0])
        T_gb = np.matmul(T_cb, T_gc)
        
        R_gb = T_gb[:3, :3]
        t_gb = T_gb[:3, 3] * self.scale
        t_gb[1] += self.init_height_offset

        R_quat = utils.mat33_to_quat(R_gb)
        finger_init = []
        #for i in range(self.finger_num):
            # Example: left half negative, right half positive, or all zero, etc.
        finger_init = [0.0] * self.finger_num

            # if i < self.finger_num // 2:
            #     finger_init.append(-self.limit_upp)
            # else:
            #     finger_init.append(self.limit_upp)

        self.model.joint_q = wp.array(
            t_gb.tolist() + R_quat.flatten().tolist() + finger_init,
            dtype=wp.float32,
            requires_grad=True,
        )

    def reset_states(self):
        self.joint_q = wp.array(self.model.joint_q.numpy(), dtype=wp.float32, requires_grad=True)

        wp.launch(utils.transform_to11d, dim=1,
                  inputs=[self.joint_q],
                  outputs=[self.transform_9d_wp, self.transform_2d_wp])
        
        self.state0 = self.model.state(requires_grad=True)
        self.state1 = self.model.state(requires_grad=True)

        with torch.no_grad():
            self.transform_9d.copy_(wp.to_torch(self.transform_9d_wp))
            self.transform_2d.copy_(wp.to_torch(self.transform_2d_wp))

        for i in range(self.finger_num):
            self.curr_finger_mesh[i] = wp.array(self.finger_mesh[i].numpy(), dtype=wp.vec3, requires_grad=True)
        self.sim_time = 0.0
        self.render_time = 0.0

        self.finger_dis.zero_()
        self.cloth_dis.zero_()

        self.optimizer = torch.optim.SGD([
            {'params': self.transform_9d, 'lr': 1e-1, 'weight_decay': 1e-5},
            {'params': self.transform_2d, 'lr': 1e-2}, 
        ])
        self.scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(self.optimizer, 'min', patience=50, factor=0.05)
        
    def forward(self, distance_param, use_com):

        self.finger_dis.zero_()
        self.cloth_dis.zero_()

        total_dis_torch = ForwardKinematics.apply(
            self.transform_9d, 
            self.transform_2d,
            self.model,
            self.state0,
            self.state1,
            self.integrator,
            self.sim_dt,
            self.curr_finger_mesh,
            self.finger_mesh,
            self.finger_body_idx,
            self.object_com,
            distance_param,
            self.finger_dis,
            self.cloth_dis,
            self.obj_geo_id,
            self.obj_body_id,
            self.consider_cloth,
            self.cloth_pairs,
            self.cloth_left_ids,
            self.cloth_right_ids,
            self.cloth_alphas,
            self.cloth_margin_mult,
        )
        return total_dis_torch

    def compute_loss(self, total_dis_torch):

        finger_loss = total_dis_torch[0]
        cloth_loss  = total_dis_torch[1]

        if self.consider_cloth and len(self.cloth_pairs) > 0:
            cloth_loss = cloth_loss / (len(self.cloth_pairs) * len(self.cloth_alphas) * self.cloth_k)
        # store for logging
        self.finger_loss = finger_loss.detach()
        self.cloth_loss  = cloth_loss.detach()

        self.loss = finger_loss + cloth_loss


    def step(self, iter, distance_param=1.0, use_com=False):
        def closure():
            total_dis_torch = self.forward(
                                distance_param, 
                                use_com)
            self.compute_loss(total_dis_torch)
            self.loss.backward()
            return self.loss

        self.optimizer.step(closure)
        with torch.no_grad():

            for i in range(self.finger_num):
                self.transform_2d.clamp_(self.limit_low, self.limit_upp)

            # --- log current radius and loss parameters ---
            current_radius = self.transform_2d.detach().cpu().numpy().copy()
            current_loss = self.loss.item()

            # --- store history ---
            self.loss_history.append(current_loss)
            self.radius_history.append(current_radius)
            self.iter_history.append(iter)

            # log to console
            if iter % 10 == 0:
                r = current_radius
                lr9  = self.optimizer.param_groups[0]["lr"]
                lr2  = self.optimizer.param_groups[1]["lr"]
                g2   = self.transform_2d.grad.detach().cpu().numpy().copy() if self.transform_2d.grad is not None else None
                print(
                    f"[iter {iter}] loss={self.loss.item():.4g}"
                    f"  finger_loss={self.finger_loss.item():.4g} cloth_loss(norm)={self.cloth_loss.item():.4g}"
                    f"  radius params (transform_2d): {current_radius}"
                    f"  lr2={lr2:.3e}, lr9={lr9:.3e}, grad2_norm={np.linalg.norm(g2) if g2 is not None else None}"
                )

        self.state0, self.state1 = self.state1, self.state0
        self.optimizer.zero_grad()
    
    def export_and_plot(self, output_dir="opt_results", threshold=None):
        """Saves optimization history to CSV and plots graphs."""
        if not os.path.exists(output_dir):
            os.makedirs(output_dir)
        
        # 1. Save to CSV
        csv_filename = os.path.join(output_dir, f"log_{self.ycb_object_name}_pose{self.pose_id}.csv")
        print(f"Saving CSV to {csv_filename}...")
        
        with open(csv_filename, mode='w', newline='') as file:
            writer = csv.writer(file)
            # Header: Iteration, Loss, Finger_0, Finger_1, ...
            headers = ["Iteration", "Loss"] + [f"Finger_{i}_Radius" for i in range(self.finger_num)]
            writer.writerow(headers)
            
            for i in range(len(self.iter_history)):
                row = [self.iter_history[i], self.loss_history[i]]
                row.extend(self.radius_history[i]) # Add all finger radii
                writer.writerow(row)

        # 2. Plot Loss vs Iteration
        plt.figure(figsize=(10, 5))
        plt.plot(self.iter_history, self.loss_history, label='Loss', color='red', linewidth=1.5)

        # --- NEW: Add Threshold Line ---
        if threshold is not None and False:
            plt.axhline(y=threshold, color='green', linestyle='--', label=f'Convergence Threshold ({threshold})')
        # Use Log Scale so we can actually see 1e-5
        plt.yscale('log')
        plt.xlabel('Iterations')
        plt.ylabel('Loss (Log Scale)')
        plt.title(f'Optimization Loss - {self.ycb_object_name}')
        plt.grid(True, which="both", ls="-", alpha=0.2) # finer grid for log scale
        plt.legend()
        plt.savefig(os.path.join(output_dir, f"plot_loss_{self.ycb_object_name}.png"))
        plt.close()

        # 3. Plot Radius vs Iteration
        plt.figure(figsize=(10, 5))
        # Convert list of arrays to a 2D array for easier plotting [iters, fingers]
        radii_array = np.array(self.radius_history) 
        
        for i in range(self.finger_num):
            plt.plot(self.iter_history, radii_array[:, i], label=f'Finger {i}')
            
        plt.xlabel('Iterations')
        plt.ylabel('Radius (meters)')
        plt.title(f'Finger Radius Optimization - {self.ycb_object_name}')
        plt.grid(True)
        plt.legend()
        plt.savefig(os.path.join(output_dir, f"plot_radius_{self.ycb_object_name}.png"))
        plt.close()
        
        print(f"Plots saved to {output_dir}")


    def render(self):
        if self.renderer is None:
            return

        with wp.ScopedTimer("render", print=False):
            self.renderer.begin_frame(self.render_time)
            self.renderer.render(
                self.state1, 
            )
            
            self.renderer.end_frame()
            self.render_time += self.frame_dt
    
    def get_initial_position(self, init_trans=None):
        prev_loss = 1e10
        convergence_threshold = 1e-5
        loss_diff_threshold = 1e-8
        patiance = 50
        stagnant_epochs = 0

        distance_param = 0.0 # was 1e-1
        use_com = True

        # NEW: sweep initial radius
        self.sweep_R0(distance_param=distance_param, use_com=use_com)

        for i in range(self.train_iter):

            self.step(i, distance_param, use_com=use_com)
            self.scheduler.step(self.loss)
            
            if self.loss < convergence_threshold:
                stagnant_epochs += 1
                if stagnant_epochs > patiance:
                    print("Converged at iteration:", i, "loss:", self.loss)
                    break
            elif np.abs(self.loss.item() - prev_loss) < loss_diff_threshold:
                stagnant_epochs += 1
                if stagnant_epochs > patiance:
                    print("Converged (stagnation) at iteration:", i, "loss:", self.loss)
                    break
            else:
                stagnant_epochs = 0
            prev_loss = self.loss.item()

            if self.is_render: self.render()

        # --- export and plot results ---
        if self.log:
            self.export_and_plot(threshold=convergence_threshold)

        if self.loss.item() > 1.0:
            print("Warning: Loss did not converge properly. Current loss:", self.loss.item())
            self.init_count += 1
            return None, None

        jq = self.model.joint_q.numpy()
        joint_trans = wp.transform(wp.vec3(jq[0], jq[1], jq[2]), 
                                  wp.quat(jq[3], jq[4], jq[5], jq[6]))
        body_trans = []
        for i in range(self.finger_num):
            body_q = self.state1.body_q.numpy()[self.finger_body_idx[i]]
            body_trans.append(
                wp.transform(wp.vec3(body_q[0], body_q[1]+self.post_height_offset, body_q[2]),
                wp.quat(body_q[3], body_q[4], body_q[5], body_q[6])))
         
        init_trans = [wp.transform_multiply(body_trans[i], self.init_transforms[i]) for i in range(self.finger_num)]

        self.debug_print_proxy_sdf(state_for_T=self.state1, stride=10)

        # after convergence / before return
        self.capture_proxy_points_frozen() # for viz of cloth proxy points
        
        return init_trans, jq
    
    def get_fixed_position(self, joint_q):
        assert joint_q.shape[0] == self.joint_q.shape[0], "joint_q shape mismatch"
        self.joint_q = wp.array(joint_q, dtype=wp.float32, requires_grad=True)
        wp.sim.eval_fk(self.model, self.joint_q, self.model.joint_qd, None, self.state0)
        
        self.integrator.simulate(self.model, self.state0, self.state1, self.sim_dt)
        
        if self.is_render: self.render()

        jq = self.model.joint_q.numpy()
        joint_trans = wp.transform(wp.vec3(jq[0], jq[1], jq[2]), 
                                  wp.quat(jq[3], jq[4], jq[5], jq[6]))
        body_trans = []
        for i in range(self.finger_num):
            body_q = self.state1.body_q.numpy()[self.finger_body_idx[i]]
            body_trans.append(
                wp.transform(wp.vec3(body_q[0], body_q[1]+self.post_height_offset, body_q[2]),
                wp.quat(body_q[3], body_q[4], body_q[5], body_q[6])))
         
        init_trans = [wp.transform_multiply(body_trans[i], self.init_transforms[i]) for i in range(self.finger_num)]
        
        return init_trans, jq

    def print_transform(self, trans):
        print("translation:", wp.transform_get_translation(trans))
        print("rotation:", wp.transform_get_rotation(trans))

if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument("--device", type=str, default=None, help="Override the default Warp device.")
    parser.add_argument(
        "--stage_path",
        type=lambda x: None if x == "None" else str(x),
        default="femtendon_sim.usd",
        help="Path to the output USD file.",
    )
    parser.add_argument("--num_envs", type=int, default=1, help="Total number of simulated environments.")
    parser.add_argument("--pose_id", type=int, default=0, help="Initial pose id from anygrasp")
    parser.add_argument("--num_frames", type=int, default=30, help="Total number of frames per training iteration.")
    parser.add_argument("--train_iters", type=int, default=15000, help="Total number of training iterations.")
    parser.add_argument("--object_name", type=str, default="013_apple", help="Name of the object to load.")
    parser.add_argument("--verbose", action="store_true", help="Print out additional status messages during execution.")
    parser.add_argument("--random", action="store_true", help="Add randomness to the loaded pose.")
    parser.add_argument("--finger_num", type=int, default=2, help="Number of fingers.")

    args = parser.parse_known_args()[0]
    finger_transform, end_state = None, None
    with wp.ScopedDevice(args.device):
        this_pose_id = 0
        curr_dir = os.path.dirname(os.path.realpath(__file__))
        file_name = curr_dir + "/../pose_info/init_opt/" + str(args.object_name) + ".npz"
        transform_arr = []
        while True:
            tendon = InitializeFingers(stage_path=args.stage_path, 
                                    finger_len=11, finger_rot=np.pi/30,
                                    finger_width=0.08,
                                    stop_margin=0.0005,
                                    scale=5.0,
                                    num_envs=args.num_envs,
                                    is_render=False,
                                    ycb_object_name=args.object_name,
                                    object_rot=wp.quat_rpy(-np.pi/2, 0.0, 0.0),
                                    num_frames=args.num_frames, 
                                    iterations=args.train_iters,
                                    verbose=args.verbose,
                                    is_triangle=False,
                                    #    pose_id=args.pose_id,
                                    pose_id=this_pose_id,
                                    finger_num=args.finger_num,
                                    add_random=args.random)
            finger_transform, jq = tendon.get_initial_position()
            if finger_transform is None:
                finger_transform = np.zeros([tendon.finger_num, 7])

            transform_arr.append(finger_transform)

            # break
            this_pose_id += 1
            if this_pose_id >= tendon.total_pose:
                break
        np.savez(file_name, finger_transform=transform_arr)
        print("saved to:", file_name)

        if tendon.renderer:
            tendon.renderer.save()