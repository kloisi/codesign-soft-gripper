import torch
import warp as wp
import warp.optim
import numpy as np
import utils

# ---------------------------------------------------------------------------
# 1. Helper Kernels 
# ---------------------------------------------------------------------------

@wp.kernel
def gather_particles_kernel(
    full_state_q: wp.array(dtype=wp.vec3),
    start_idx: int,
    out_finger_mesh: wp.array(dtype=wp.vec3)
):
    tid = wp.tid()
    out_finger_mesh[tid] = full_state_q[start_idx + tid]

@wp.kernel
def loss_aggregation_kernel(
    dist_accum: wp.array(dtype=float),
    forces: wp.array(dtype=float),
    total_loss: wp.array(dtype=float),
    dist_weight: float,
    force_weight: float,
    num_fingers: float
):
    l_dist = dist_accum[0] / num_fingers
    l_force = float(0.0)
    for i in range(forces.shape[0]):
        val = forces[i]
        l_force += val * val
    l_force = l_force / float(forces.shape[0])
    total_loss[0] = (l_dist * dist_weight) + (l_force * force_weight)

# ---------------------------------------------------------------------------
# 2. Native Warp Optimizer Class (FIXED)
# ---------------------------------------------------------------------------

class NativeWarpOptimizer:
    @staticmethod
    def optimize(
        tendon_forces: wp.array, 
        model, 
        states, 
        integrator, 
        control, 
        tendon_holder, 
        sim_dt, 
        sim_substeps, 
        opt_frames, 
        finger_vertex_ranges,
        iterations=10, 
        learning_rate=0.01
    ):
        print(f"--- Starting Native Warp Autodiff (Iterations: {iterations}, LR: {learning_rate}) ---")

        if not tendon_forces.requires_grad:
            raise ValueError("tendon_forces array must have requires_grad=True")

        # Create Optimizer
        optimizer = wp.optim.Adam([tendon_forces], lr=learning_rate)
        
        # Allocate Buffers
        loss_warp = wp.zeros(1, dtype=float, requires_grad=True)
        dist_accum = wp.zeros(1, dtype=float, requires_grad=True)
        
        finger_meshes = []
        for start, end in finger_vertex_ranges:
            finger_meshes.append(wp.zeros(end - start, dtype=wp.vec3, requires_grad=True))

        history = {"loss": [], "forces": []}

        # --- Define Weights Here (Must match inputs to loss_aggregation_kernel) ---
        W_DIST = 1.0
        W_FORCE = 0.0  # Currently 0.0 in your code
        num_fingers = float(len(finger_vertex_ranges))

        for iter_idx in range(iterations):
            # 1. Clear Gradients & Tape
            tape = wp.Tape()
            tendon_forces.grad.zero_()
            loss_warp.zero_()
            dist_accum.zero_()
            
            # 2. Reset Simulation State
            NativeWarpOptimizer._reset_states(states, model)

            # 3. Forward Pass (Recorded)
            with tape:
                success_flag = wp.zeros(1, dtype=wp.int32)
                
                # Run Physics
                for frame in range(opt_frames):
                    for i in range(sim_substeps):
                        idx = i + frame * sim_substeps
                        if idx + 1 >= len(states): break
                        
                        curr = states[idx]
                        next = states[idx+1]
                        
                        curr.clear_forces()
                        tendon_holder.reset()
                        tendon_holder.apply_force(tendon_forces, curr.particle_q, success_flag)
                        integrator.simulate(model, curr, next, sim_dt, control)

                # Compute Loss
                final_state = states[opt_frames * sim_substeps]
                
                for i, (start, end) in enumerate(finger_vertex_ranges):
                    wp.launch(
                        kernel=gather_particles_kernel,
                        dim=end-start,
                        inputs=[final_state.particle_q, start, finger_meshes[i]]
                    )
                    wp.launch(
                        kernel=utils.mesh_dis, 
                        dim=end-start, 
                        inputs=[
                            model.shape_geo, 0, model.shape_body, model.shape_transform, 
                            final_state.body_q, finger_meshes[i], 
                            model.soft_contact_margin*1.5, 100.0, 1e3, 0, 
                            dist_accum
                        ]
                    )

                wp.launch(
                    kernel=loss_aggregation_kernel,
                    dim=1,
                    inputs=[dist_accum, tendon_forces, loss_warp, W_DIST, W_FORCE, num_fingers]
                )

            # 4. Backward Pass
            loss_val = loss_warp.numpy()[0]
            
            # --- CALCULATE COMPONENT LOSSES FOR DISPLAY ---
            # We calculate these on CPU (numpy) to print them
            current_dist_loss = (dist_accum.numpy()[0] / num_fingers) * W_DIST
            
            forces_np = tendon_forces.numpy()
            current_reg_loss = np.mean(forces_np**2) * W_FORCE
            # ---------------------------------------------

            if np.isnan(loss_val):
                print(f"!!! Iter {iter_idx}: Loss is NaN. Aborting this step.")
                continue

            tape.backward(loss_warp)
            
            # Critical Fix: Sanitize Gradients
            grad_np = tendon_forces.grad.numpy()
            
            if np.any(np.isnan(grad_np)) or np.any(np.isinf(grad_np)):
                grad_np = np.nan_to_num(grad_np, nan=0.0, posinf=0.0, neginf=0.0)
            
            grad_np = np.clip(grad_np, -1.0, 1.0)
            tendon_forces.grad = wp.from_numpy(grad_np, dtype=float, device='cuda')

            # 5. Optimizer Step
            optimizer.step([tendon_forces.grad])
            
            # 6. Clamp Forces
            curr_forces_np = tendon_forces.numpy()
            curr_forces_np = np.clip(curr_forces_np, 0.0, 100.0)
            wp.copy(tendon_forces, wp.from_numpy(curr_forces_np, dtype=float, device='cuda', requires_grad=True))
            
            # --- UPDATED PRINT STATEMENT ---
            print(f"Iter {iter_idx}: Total {loss_val:.4f} | Dist {current_dist_loss:.4f} | Reg {current_reg_loss:.4f} | Forces {curr_forces_np}")
            
            history["loss"].append(loss_val)
            history["forces"].append(curr_forces_np.tolist())

        return history

    @staticmethod
    def _reset_states(states, model):
        """Helper to reset simulation states to t=0"""
        init_q = states[0].particle_q
        init_body_q = states[0].body_q if (hasattr(model, 'has_object') and model.has_object) else None

        for f in range(len(states)):
            states[f].particle_qd.zero_()
            states[f].body_qd.zero_()
            if f > 0:
                wp.copy(states[f].particle_q, init_q)
                if init_body_q:
                    wp.copy(states[f].body_q, init_body_q)

class FEMForceOptimization(torch.autograd.Function):
    """
    A simplified wrapper for running the physics simulation purely for evaluation.
    Does NOT track gradients. Returns the loss as a simple float.
    """
    @staticmethod
    def run(forces_tensor, model, states, integrator, dt, control, tendon, substeps, frames, finger_ranges):
        # 1. Setup Warp Arrays (No Tape, No Gradients needed here)
        forces_warp = wp.from_torch(forces_tensor.detach())
        success_flag = wp.zeros(1, dtype=wp.int32, device='cuda')
        
        curr_finger_meshes = []
        for start, end in finger_ranges:
            curr_finger_meshes.append(wp.zeros(end - start, dtype=wp.vec3))
        total_dis = wp.zeros(1, dtype=wp.float32)

        # 2. Reset Physics State (Crucial logic from run_simulation)
        for f in range(frames * substeps + 1):
            if f < len(states):
                states[f].particle_qd.zero_()
                states[f].body_qd.zero_()
                if f > 0:
                    wp.copy(states[f].particle_q, states[0].particle_q)
                    if hasattr(model, 'has_object') and model.has_object:
                         wp.copy(states[f].body_q, states[0].body_q)

        # 3. Run Physics Loop
        for frame in range(frames):
            for i in range(substeps):
                idx = i + frame * substeps
                if idx + 1 >= len(states): break

                state_curr = states[idx]
                state_next = states[idx+1]
                
                state_curr.clear_forces()
                tendon.reset()
                tendon.apply_force(forces_warp, state_curr.particle_q, success_flag)
                integrator.simulate(model, state_curr, state_next, dt, control)

        # 4. Calculate Distance
        final_idx = min(frames * substeps, len(states) - 1)
        final_state = states[final_idx]
        
        for i in range(len(curr_finger_meshes)):
            start = finger_ranges[i][0]
            wp.launch(utils.copy_finger_particles, 
                      dim=len(curr_finger_meshes[i]), 
                      inputs=[final_state.particle_q, start, curr_finger_meshes[i]])
            
            wp.launch(utils.mesh_dis, 
                      dim=len(curr_finger_meshes[i]), 
                      inputs=[model.shape_geo, 
                              0, 
                              model.shape_body, 
                              model.shape_transform, 
                              final_state.body_q, 
                              curr_finger_meshes[i], 
                              model.soft_contact_margin*1.5, 
                              10.0, 1e3, 0], 
                      outputs=[total_dis])
        
        # Scale with number of fingers
        finger_num = len(finger_ranges)
        dist_loss = total_dis.numpy()[0] / finger_num

        # 5. Regularize Forces
        input_forces_np = forces_tensor.detach().cpu().numpy()
        force_loss = np.mean(input_forces_np**2) 

        # Weight tuning
        dist_scale = 100.0
        force_scale = 0.005

        term_dist = dist_loss * dist_scale
        term_force = force_loss * force_scale

        # Total loss
        total_loss = term_dist + term_force

        # Debug prints (can be commented out)
        print(f"Distance Loss: {term_dist:.4f}, Force Penalty: {term_force:.4f}, Total Loss: {total_loss:.4f}, Current Forces: {input_forces_np}")

        # Return standard Python float
        return float(total_loss)
    