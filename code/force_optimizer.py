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

        target_gap = 0.01  # target gap between fingers and object
        
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
            # wp.launch(utils.mesh_dis, 
            #         dim=len(curr_finger_meshes[i]), 
            #         inputs=[model.shape_geo, 0, 0, final_state.body_q, curr_finger_meshes[i], 
            #         target_gap, 10.0, 1e3, 0], 
            #         outputs=[total_dis])

        # Return standard Python float
        return total_dis.numpy()[0]
    
    """
    Differentiable Physics Wrapper.
    Records the simulation on a Warp Tape to calculate gradients automatically.
    """

    @staticmethod
    def forward(ctx, forces_tensor, model, states, integrator, dt, control, tendon, substeps, frames, finger_ranges):
        if not control.waypoint_forces.requires_grad:
            control.waypoint_forces = wp.zeros_like(control.waypoint_forces, requires_grad=True)
        # 1. Setup Inputs
        # Convert PyTorch tensor to Warp array, preserving the gradient tracking.
        # Note: forces_tensor must have requires_grad=True in PyTorch.
        forces_warp = wp.from_torch(forces_tensor)
        
        # Create a generic float buffer for the loss that tracks gradients
        total_dis = wp.zeros(1, dtype=wp.float32, requires_grad=True)
        
        # Buffer allocation for fingers (need to set requires_grad=True for intermediate data)
        curr_finger_meshes = []
        for start, end in finger_ranges:
            curr_finger_meshes.append(wp.zeros(end - start, dtype=wp.vec3, requires_grad=True))

        # 2. Reset Physics State
        # This part happens OUTSIDE the tape because we don't want to differentiate the reset logic.
        for f in range(frames * substeps + 1):
            if f < len(states):
                states[f].particle_qd.zero_()
                states[f].body_qd.zero_()
                if f > 0:
                    wp.copy(states[f].particle_q, states[0].particle_q)
                    if hasattr(model, 'has_object') and model.has_object:
                         wp.copy(states[f].body_q, states[0].body_q)
        
        dummy_flag = wp.zeros(1, dtype=wp.int32, device='cuda')
                         
        # 3. Start Recording on the Tape
        tape = wp.Tape()
        with tape:
            # --- Physics Loop ---
            for frame in range(frames):
                for i in range(substeps):
                    idx = i + frame * substeps
                    if idx + 1 >= len(states): break

                    state_curr = states[idx]
                    state_next = states[idx+1]
                    
                    state_curr.clear_forces()
                    tendon.reset()
                    
                    # Apply forces (This connects forces_warp to the graph)
                    # passing success_flag as None or a dummy if not needed for logic control
                    tendon.apply_force(forces_warp, state_curr.particle_q, dummy_flag)
                    
                    # Integrate (This connects state_curr to state_next)
                    integrator.simulate(model, state_curr, state_next, dt, control)

            # --- Loss Calculation ---
            # We must do this INSIDE the tape so the connection 
            # (Final State -> Distance -> Loss) is recorded.
            final_idx = min(frames * substeps, len(states) - 1)
            final_state = states[final_idx]
            
            for i in range(len(curr_finger_meshes)):
                start = finger_ranges[i][0]
                
                # Copy particles to local buffer
                wp.launch(utils.copy_finger_particles, 
                          dim=len(curr_finger_meshes[i]), 
                          inputs=[final_state.particle_q, start, curr_finger_meshes[i]])
                
                # Calculate distance
                # Ensure utils.mesh_dis uses ATOMIC ADD if it writes to total_dis, 
                # otherwise the loop will just overwrite the value each time.
                wp.launch(utils.mesh_dis, 
                          dim=len(curr_finger_meshes[i]), 
                          inputs=[model.shape_geo, 0, 0, final_state.body_q, curr_finger_meshes[i], 
                                  model.soft_contact_margin, 10.0, 1e3, 0], 
                          outputs=[total_dis])

        # 4. Save Context for Backward Pass
        ctx.tape = tape
        ctx.forces_warp = forces_warp
        ctx.total_dis = total_dis
        
        # Return scalar loss to PyTorch
        return wp.to_torch(total_dis)

    @staticmethod
    def backward(ctx, grad_output):
        # 1. Retrieve Context
        tape = ctx.tape
        forces_warp = ctx.forces_warp
        total_dis = ctx.total_dis
        
        # 2. Pass incoming gradient to the Warp loss variable
        # grad_output is the gradient from PyTorch (usually 1.0)
        if grad_output is not None:
            total_dis.grad = wp.from_torch(grad_output)
            
        # 3. Play the Tape Backward
        tape.backward()
        
        # 4. Return Gradients
        # We must return a gradient for EVERY argument in 'forward'.
        # Only 'forces_tensor' needs a gradient; return None for the rest.
        forces_grad = wp.to_torch(forces_warp.grad)
        
        return forces_grad, None, None, None, None, None, None, None, None, None