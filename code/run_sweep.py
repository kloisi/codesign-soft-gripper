import warp as wp
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import math
import os

# IMPORT YOUR EXISTING CODE
from forward import FEMTendon, InitializeFingers

def run_experiment_sweep():
    # --- 1. CONFIGURATION ---
    device = "cuda"  # or "cpu"
    
    # The variables to sweep over
    #object_list = ["013_apple", "006_mustard_bottle", "acropora_cervicornis"]
    object_list = ["acropora_cervicornis", "acropora_florida", "acropora_loripes", "acropora_millepora", "acropora_nobilis", "acropora_palmata", "acropora_sarmentosa", "acropora_tenuis", "fungia_scutaria", "goniastrea_aspera", "montipora_capitata", "platygyra_daedalea", "platygyra_lamellina", "pocillopora_meandrina"]
    finger_counts = [3,4,5,6,7,8,9]
    #finger_counts = [3,4]
    
    # Fixed parameters
    finger_len = 11
    finger_rot = np.pi/30
    finger_width = 0.08
    scale = 5.0
    object_rot = wp.quat_rpy(-math.pi/2, 0.0, 0.0)
    
    # Store results here
    results_data = []

    print(f"Starting sweep over {len(object_list)} objects and {len(finger_counts)} finger configs...")

    with wp.ScopedDevice(device):
        
        for obj_name in object_list:
            for n_fingers in finger_counts:
                print(f"\n=== Processing: {obj_name} | Fingers: {n_fingers} ===")
                
                # -------------------------------------------------
                # A. Optimize Initial Position (Radius)
                # -------------------------------------------------
                try:
                    init_finger = InitializeFingers(
                        stage_path="temp_init.usd",
                        finger_len=finger_len,
                        finger_rot=finger_rot,
                        finger_width=finger_width,
                        stop_margin=0.0005,
                        num_frames=30,
                        iterations=10000,  # Lower iterations for speed during testing
                        scale=scale,
                        num_envs=1,
                        ycb_object_name=obj_name,
                        object_rot=object_rot,
                        is_render=False,
                        verbose=False,
                        finger_num=n_fingers,
                        add_random=False,
                        consider_cloth=True
                    )
                    
                    # Get the optimized transform
                    finger_transform, _ = init_finger.get_initial_position()
                    
                    # Calculate Radius (Magnitude of position vector from center)
                    # finger_transform is usually (pos, rot). pos is a wp.vec3 or numpy array
                    if hasattr(finger_transform, 'p'): # Warp transform object
                        pos = finger_transform.p
                    else: # Tuple (pos, rot)
                        pos = finger_transform[0]
                        
                    radius = np.linalg.norm(pos)
                    print(f"   -> Optimized Radius: {radius:.4f}")

                except Exception as e:
                    print(f"   [Error] Init Pose failed: {e}")
                    continue

                # -------------------------------------------------
                # B. Optimize Forces
                # -------------------------------------------------
                try:
                    tendon = FEMTendon(
                        stage_path=f"sweep_{obj_name}_{n_fingers}.usd",
                        num_frames=1000, # frames for force opt
                        verbose=False,
                        save_log=False,
                        is_render=False, # Disable render for speed
                        train_iters=1000,
                        object_rot=object_rot,
                        ycb_object_name=obj_name,
                        finger_len=finger_len,
                        finger_rot=finger_rot,
                        finger_width=finger_width,
                        scale=scale,
                        finger_transform=finger_transform,
                        finger_num=n_fingers,
                        requires_grad=True,
                        no_cloth=False
                    )

                    # Run L-BFGS Force Optimization
                    history = tendon.optimize_forces_lbfgs(
                        iterations=2, # Keep low for testing, increase for real results
                        learning_rate=1.0, 
                        opt_frames=100
                    )
                    
                    final_loss = history['loss'][-1]
                    final_forces = history['forces'][-1]
                    avg_force = np.mean(final_forces)
                    
                    print(f"   -> Final Loss: {final_loss:.4f} | Avg Force: {avg_force:.2f}")

                except Exception as e:
                    print(f"   [Error] Force Opt failed: {e}")
                    final_loss = np.nan
                    avg_force = np.nan
                    final_forces = []

                # -------------------------------------------------
                # C. Record Data
                # -------------------------------------------------
                entry = {
                    "Object": obj_name,
                    "Num_Fingers": n_fingers,
                    "Radius": radius,
                    "Final_Loss": final_loss,
                    "Avg_Force": avg_force,
                    "All_Forces": str(final_forces) # Store as string to save in CSV easily
                }
                results_data.append(entry)

                # Clean up to free GPU memory
                del init_finger
                del tendon
                import gc
                gc.collect()

    # --- 2. SAVE RESULTS ---
    df = pd.DataFrame(results_data)
    os.makedirs("sweep_results", exist_ok=True)
    csv_path = "sweep_results/experiment_data.csv"
    df.to_csv(csv_path, index=False)
    print(f"\nSweep Complete. Data saved to {csv_path}")

    return df

def plot_results(df):
    sns.set_theme(style="whitegrid")
    
    # Create a figure with 3 subplots
    fig, axes = plt.subplots(1, 3, figsize=(30, 10))
    
    # Plot 1: Radius vs Fingers (Grouped by Object)
    sns.lineplot(data=df, x="Num_Fingers", y="Radius", hue="Object", marker="o", ax=axes[0])
    axes[0].set_title("Optimal Radius")
    axes[0].set_ylabel("Radius (sim units)")
    
    # Plot 2: Final Loss vs Fingers
    sns.lineplot(data=df, x="Num_Fingers", y="Final_Loss", hue="Object",marker="o", ax=axes[1])
    axes[1].set_title("Optimization Loss")
    axes[1].set_ylabel("Distance and Force Regularization Loss")
    
    # Plot 3: Average Force vs Fingers
    sns.lineplot(data=df, x="Num_Fingers", y="Avg_Force", hue="Object",marker="o", ax=axes[2])
    axes[2].set_title("Average Optimized Force")
    axes[2].set_ylabel("Force (N)")
    
    plt.tight_layout()
    plt.savefig("sweep_results/sweep_analysis.png")
    print("Plots saved to sweep_results/sweep_analysis.png")

if __name__ == "__main__":
    df_results = run_experiment_sweep()
    #csv_path = "sweep_results/experiment_data.csv"
    #df_results = pd.read_csv(csv_path)
    if not df_results.empty:
        plot_results(df_results)