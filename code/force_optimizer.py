import torch
import warp as wp
import utils

class FEMForceOptimization(torch.autograd.Function):
    """
    A simplified wrapper for running the physics simulation purely for evaluation.
    Does NOT track gradients. Returns the loss as a simple float.
    """
    @staticmethod
    def run(forces_tensor, model, states, integrator, dt, control, tendon, substeps, frames, finger_ranges):
        # 1. Setup Warp Arrays (No Tape, No Gradients needed here)
        # We detach to ensure no PyTorch graph is carried over
        forces_warp = wp.from_torch(forces_tensor.detach())
        success_flag = wp.zeros(1, dtype=wp.int32, device='cuda')
        
        # Buffer allocation (local to this run)
        curr_finger_meshes = []
        for start, end in finger_ranges:
            curr_finger_meshes.append(wp.zeros(end - start, dtype=wp.vec3))
        total_dis = wp.zeros(1, dtype=wp.float32)

        # 2. Reset Physics State (Crucial logic from run_simulation)
        #    We must reset the simulation to t=0 before running.
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
                      inputs=[model.shape_geo, 0, 0, final_state.body_q, curr_finger_meshes[i], 
                              model.soft_contact_margin, 10.0, 1e3, 0], 
                      outputs=[total_dis])

        # Return standard Python float
        return total_dis.numpy()[0]
    