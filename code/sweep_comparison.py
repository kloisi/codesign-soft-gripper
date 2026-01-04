import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import os

def compare_four_sweeps():
    # 1. Setup File Configuration
    # Dictionary mapping readable names to filenames
    base_folder = 'sweep_results'

    # Update the files dictionary to use the path
    files = {
        'Fixed Force 100N': os.path.join(base_folder, 'coral_sweep_fixed_force_100.csv'),
        'Fixed Force 50N':  os.path.join(base_folder, 'coral_sweep_fixed_force_50.csv'),
        'Weak Force Penalty':   os.path.join(base_folder, 'coral_sweep_0.005.csv'),
        'Strong Force Penalty': os.path.join(base_folder, 'coral_sweep_0.01.csv')
    }
    # 2. Load and Merge Data
    dfs = []
    print("Loading files...")
    for label, filename in files.items():
        try:
            df = pd.read_csv(filename)
            # Select only needed columns and rename them with the label suffix
            # e.g., 'Final_Loss' -> 'Loss_Fixed 100N'
            cols = {
                'Final_Loss': f'Loss_{label}', 
                'Avg_Force': f'Force_{label}'
            }
            df = df[['Object', 'Num_Fingers', 'Final_Loss', 'Avg_Force']].rename(columns=cols)
            dfs.append(df)
        except FileNotFoundError:
            print(f"Error: File '{filename}' not found.")
            return

    # Merge all dataframes on Object and Num_Fingers
    merged = dfs[0]
    for df_next in dfs[1:]:
        merged = pd.merge(merged, df_next, on=['Object', 'Num_Fingers'])

    # 3. Calculate Statistics
    print("\n--- Summary Statistics (Averages) ---")
    stats = []
    for label in files.keys():
        stats.append({
            'Method': label,
            'Avg Loss': merged[f'Loss_{label}'].mean(),
            'Avg Force': merged[f'Force_{label}'].mean()
        })
    stats_df = pd.DataFrame(stats).set_index('Method')
    print(stats_df)

    # 4. Identify the "Best" Method for each object
    # We compare only the Loss columns
    loss_cols = [f'Loss_{label}' for label in files.keys()]
    merged['Winner'] = merged[loss_cols].idxmin(axis=1)
    # Clean up the winner string (remove 'Loss_' prefix)
    merged['Winner'] = merged['Winner'].str.replace('Loss_', '')

    print("\n--- Win Counts (Method with Lowest Loss) ---")
    print(merged['Winner'].value_counts())

    # 5. Visualization
    # Reshape data for plotting (Long Format)
    plot_data = []
    for index, row in merged.iterrows():
        for label in files.keys():
            plot_data.append({
                'Method': label,
                'Final Loss': row[f'Loss_{label}'],
                'Average Force': row[f'Force_{label}']
            })
    plot_df = pd.DataFrame(plot_data)

    fig, axes = plt.subplots(1, 2, figsize=(15, 6))

    # Plot A: Loss Distribution
    sns.boxplot(data=plot_df, x='Method', y='Final Loss', ax=axes[0], showfliers=False)
    axes[0].set_title('Comparison of Final Loss (Lower is Better)')
    axes[0].grid(True, alpha=0.3)

    # Plot B: Force Distribution
    sns.boxplot(data=plot_df, x='Method', y='Average Force', ax=axes[1])
    axes[1].set_title('Comparison of Applied Forces (Lower is Better)')
    axes[1].set_ylabel('Force (N)')
    axes[1].grid(True, alpha=0.3)
    
    # Add a horizontal line for the fixed 50N and 100N benchmarks
    axes[1].axhline(y=50, color='r', linestyle='--', alpha=0.5, label='50N Ref')
    axes[1].axhline(y=100, color='gray', linestyle='--', alpha=0.5, label='100N Ref')
    axes[1].legend()

    plt.tight_layout()
    plt.savefig('sweeps_comparison.png')
    print("\nComparison plot saved as 'sweeps_comparison.png'")

    # 6. Optional: Save merged data
    merged.to_csv('all_sweeps_merged.csv', index=False)
    print("Merged data saved to 'all_sweeps_merged.csv'")

def compare_finger_counts(merged_df):
    """
    Analyzes the effect of finger count on the average loss across all objects.
    
    Args:
        merged_df (pd.DataFrame): The merged dataframe containing all sweep results.
                                  Must have 'Num_Fingers' and 'Loss_...' columns.
    """
    print("\n--- Finger Count Analysis ---")
    
    # 1. Identify relevant columns
    loss_cols = [col for col in merged_df.columns if 'Loss_' in col]
    
    # 2. Group by 'Num_Fingers' and calculate the mean
    # We aggregate across all objects for each finger count
    finger_stats = merged_df.groupby('Num_Fingers')[loss_cols].mean()
    
    # 3. Print the results sorted by performance (optional, usually index is fine)
    print("Average Loss by Number of Fingers:")
    print(finger_stats.round(2))
    
    # Find the optimal number of fingers for each method
    print("\nOptimal Number of Fingers (Lowest Average Loss):")
    for col in loss_cols:
        optimal_fingers = finger_stats[col].idxmin()
        min_loss = finger_stats[col].min()
        print(f"  {col.replace('Loss_', '')}: {optimal_fingers} fingers (Loss: {min_loss:.2f})")

    # 4. Visualization
    plt.figure(figsize=(10, 6))
    
    # Plot a line for each method
    markers = ['o', 's', '^', 'D']
    for i, col in enumerate(loss_cols):
        method_name = col.replace('Loss_', '')
        plt.plot(finger_stats.index, finger_stats[col], 
                 marker=markers[i % len(markers)], 
                 linewidth=2, 
                 label=method_name)
    
    plt.xlabel('Number of Fingers')
    plt.ylabel('Average Final Loss')
    plt.title('Optimal Finger Count: Average Loss vs Number of Fingers')
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.xticks(finger_stats.index) # Ensure all finger numbers are shown
    
    output_filename = 'finger_count_analysis.png'
    plt.savefig(output_filename)
    print(f"\nFinger analysis plot saved to '{output_filename}'")

def analyze_fingers_force(df):
    """
    Analyzes the effect of finger count on the average force.
    """
    print("\n--- Finger Count Analysis: Forces ---")
    
    # 1. Identify force columns
    force_cols = [col for col in df.columns if 'Force_' in col]
    
    # 2. Group by 'Num_Fingers' and calculate the mean
    finger_force_stats = df.groupby('Num_Fingers')[force_cols].mean()
    
    print("Average Force by Number of Fingers:")
    print(finger_force_stats.round(2))
    
    # 3. Find the lowest force count for each method
    print("\nOptimal Number of Fingers (Lowest Average Force):")
    for col in force_cols:
        optimal_fingers = finger_force_stats[col].idxmin()
        min_force = finger_force_stats[col].min()
        print(f"  {col.replace('Force_', '')}: {optimal_fingers} fingers (Force: {min_force:.2f} N)")

    # 4. Visualization
    plt.figure(figsize=(10, 6))
    
    markers = ['o', 's', '^', 'D']
    for i, col in enumerate(force_cols):
        method_name = col.replace('Force_', '')
        plt.plot(finger_force_stats.index, finger_force_stats[col], 
                 marker=markers[i % len(markers)], 
                 linewidth=2, 
                 label=method_name)
    
    plt.xlabel('Number of Fingers')
    plt.ylabel('Average Applied Force (N)')
    plt.title('Average Force vs Number of Fingers')
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.xticks(finger_force_stats.index)
    
    output_filename = 'finger_force_analysis.png'
    plt.savefig(output_filename)
    print(f"\nForce analysis plot saved to '{output_filename}'")

def recalculate_true_loss(file_path='all_sweeps_merged.csv'):
    # 1. Load Data
    df = pd.read_csv(file_path)
    
    # 2. Configuration based on your experiments
    # (Label, Column Suffix, Force Scale)
    configs = [
        ('Fixed 100N', 'Fixed Force 100N', 0.005),
        ('Fixed 50N',  'Fixed Force 50N',  0.005),
        ('Weak Opt',   'Weak Force Penalty', 0.005),
        ('Strong Opt', 'Strong Force Penalty', 0.01)
    ]
    
    print(f"{'Method':<15} | {'Orig Loss':<10} | {'True Dist Loss':<15}")
    print("-" * 45)
    
    results = []

    for label, suffix, weight in configs:
        loss_col = f'Loss_{suffix}'
        force_col = f'Force_{suffix}'
        
        # Calculate the Penalty Term
        # Formula: weight * mean(Force^2)
        # We approximate mean(Force^2) as Avg_Force^2 
        # (This is exact for fixed forces and very close for optimized ones)
        penalty_term = weight * (df[force_col] ** 2)
        
        # Subtract penalty to get the pure Distance Term
        dist_term = df[loss_col] - penalty_term
        
        # Save results
        df[f'True_Loss_{suffix}'] = dist_term
        results.append({
            'Method': label, 
            'Avg': dist_term.mean()
        })
        
        print(f"{label:<15} | {df[loss_col].mean():<10.2f} | {dist_term.mean():<15.2f}")

    # Optional: Save to new CSV
    df.to_csv('all_sweeps_recalculated.csv', index=False)
    print("\nDetailed results saved to 'all_sweeps_recalculated.csv'")

def compare_recalculated_sweeps(file_path='all_sweeps_recalculated.csv'):
    # 1. Load the pre-calculated data
    try:
        df = pd.read_csv(file_path)
    except FileNotFoundError:
        print(f"Error: File '{file_path}' not found. Please run the recalculation script first.")
        return

    # 2. Define the methods/suffixes to compare
    methods = [
        'Fixed Force 100N', 
        'Fixed Force 50N', 
        'Weak Force Penalty', 
        'Strong Force Penalty'
    ]

    # 3. Visualization Data Prep
    plot_data = []
    for index, row in df.iterrows():
        for method in methods:
            loss_col = f'True_Loss_{method}'
            force_col = f'Force_{method}'
            
            if loss_col in df.columns and force_col in df.columns:
                plot_data.append({
                    'Method': method,
                    'True Loss': row[loss_col],
                    'Average Force': row[force_col]
                })
    
    plot_df = pd.DataFrame(plot_data)

    # --- Switch to Dark Background Style ---
    plt.style.use('dark_background')

    # 4. Plotting
    fig, axes = plt.subplots(1, 2, figsize=(15, 6))
    fig.patch.set_facecolor('black')
    
    # High contrast palette: Cyan, Gold, Lime, Magenta
    custom_palette = ['#00FFFF', '#FFD700', '#32CD32', '#FF00FF']

    # --- Plot A: True Loss Distribution ---
    sns.boxplot(
        data=plot_df, 
        x='Method', 
        y='True Loss', 
        order=methods,
        palette=custom_palette,
        ax=axes[0], 
        showfliers=False
    )
    
    # Styling Plot A
    axes[0].set_facecolor('black')
    axes[0].set_title('Comparison of Distance Loss', fontsize=14, fontweight='bold', color='white')
    axes[0].set_ylabel('Distance Loss', fontsize=12, color='white')
    axes[0].set_xlabel('')
    axes[0].tick_params(axis='x', rotation=15, colors='white', labelsize=10)
    axes[0].tick_params(axis='y', colors='white', labelsize=10)
    axes[0].grid(True, linestyle='--', alpha=0.3, color='gray')
    for spine in axes[0].spines.values():
        spine.set_edgecolor('white')

    # --- Plot B: Force Distribution ---
    sns.boxplot(
        data=plot_df, 
        x='Method', 
        y='Average Force', 
        order=methods,
        palette=custom_palette,
        ax=axes[1]
    )
    
    # Styling Plot B
    axes[1].set_facecolor('black')
    axes[1].set_title('Comparison of Applied Forces', fontsize=14, fontweight='bold', color='white')
    axes[1].set_ylabel('Force (N)', fontsize=12, color='white')
    axes[1].set_xlabel('')
    axes[1].tick_params(axis='x', rotation=15, colors='white', labelsize=10)
    axes[1].tick_params(axis='y', colors='white', labelsize=10)
    axes[1].grid(True, linestyle='--', alpha=0.3, color='gray')
    for spine in axes[1].spines.values():
        spine.set_edgecolor('white')
    
    # Add benchmarks
    axes[1].axhline(y=50, color='#FF4500', linestyle='--', alpha=0.8, label='50N Ref') # OrangeRed
    axes[1].axhline(y=100, color='#A9A9A9', linestyle='--', alpha=0.8, label='100N Ref') # LightGray
    
    legend = axes[1].legend(facecolor='black', edgecolor='white')
    for text in legend.get_texts():
        text.set_color("white")

    plt.tight_layout()
    output_file = 'recalculated_sweeps_comparison_dark.png'
    plt.savefig(output_file, facecolor=fig.get_facecolor(), edgecolor='none')
    print(f"\nComparison plot saved as '{output_file}'")

def plot_sweeps_no_radius():
    # 1. Load data
    try:
        df = pd.read_csv('all_sweeps_recalculated.csv')
    except FileNotFoundError:
        print("Error: 'all_sweeps_recalculated.csv' not found.")
        return

    # 2. Define methods to extract
    methods = [
        'Fixed Force 100N',
        'Fixed Force 50N',
        'Weak Force Penalty',
        'Strong Force Penalty'
    ]

    # 3. Reshape to long format
    long_dfs = []
    for method in methods:
        loss_col = f'True_Loss_{method}'
        force_col = f'Force_{method}'
        
        # Check if columns exist
        if loss_col in df.columns and force_col in df.columns:
            temp_df = df[['Object', 'Num_Fingers', loss_col, force_col]].copy()
            temp_df.rename(columns={
                loss_col: 'True_Loss',
                force_col: 'Avg_Force'
            }, inplace=True)
            temp_df['Method'] = method
            long_dfs.append(temp_df)
    
    if not long_dfs:
        print("No matching columns found.")
        return

    plot_df = pd.concat(long_dfs, ignore_index=True)

    # --- Switch to Dark Background Style ---
    plt.style.use('dark_background')

    # 4. Plotting
    # Create figure with explicit black background
    fig, axes = plt.subplots(1, 2, figsize=(20, 8))
    fig.patch.set_facecolor('black')

    # Define a high-contrast palette for the dark background
    # Fixed 100N (Cyan), Fixed 50N (Gold), Weak (Lime), Strong (Magenta)
    custom_palette = ['#00FFFF', '#FFD700', '#32CD32', '#FF00FF']
    
    # --- Plot 1: True Loss vs Fingers ---
    sns.lineplot(
        data=plot_df, 
        x="Num_Fingers", 
        y="True_Loss", 
        hue="Method", 
        palette=custom_palette,
        marker="o", 
        errorbar='sd', 
        ax=axes[0], 
        linewidth=2.5
    )
    
    # Styling Plot 1
    axes[0].set_facecolor('black')
    axes[0].set_title("Distance Loss (Mean ± Std Dev)", fontsize=16, fontweight='bold', color='white')
    axes[0].set_ylabel("Distance Loss", fontsize=14, color='white')
    axes[0].set_xlabel("Number of Fingers", fontsize=14, color='white')
    axes[0].tick_params(axis='both', colors='white', labelsize=12)
    axes[0].grid(True, linestyle='--', alpha=0.3, color='gray')
    for spine in axes[0].spines.values():
        spine.set_edgecolor('white')
    
    # Legend Plot 1
    legend = axes[0].legend(facecolor='black', edgecolor='white', title='Method', title_fontsize=12)
    plt.setp(legend.get_title(), color='white')
    for text in legend.get_texts():
        text.set_color("white")


    # --- Plot 2: Average Force vs Fingers ---
    sns.lineplot(
        data=plot_df, 
        x="Num_Fingers", 
        y="Avg_Force", 
        hue="Method", 
        palette=custom_palette,
        marker="o", 
        errorbar='sd', 
        ax=axes[1], 
        linewidth=2.5
    )

    # Styling Plot 2
    axes[1].set_facecolor('black')
    axes[1].set_title("Average Force (Mean ± Std Dev)", fontsize=16, fontweight='bold', color='white')
    axes[1].set_ylabel("Force (N)", fontsize=14, color='white')
    axes[1].set_xlabel("Number of Fingers", fontsize=14, color='white')
    axes[1].tick_params(axis='both', colors='white', labelsize=12)
    axes[1].grid(True, linestyle='--', alpha=0.3, color='gray')
    for spine in axes[1].spines.values():
        spine.set_edgecolor('white')
    
    # Benchmarks for Force
    axes[1].axhline(50, color='#FF4500', linestyle='--', alpha=0.8, label='50N Ref') # OrangeRed
    axes[1].axhline(100, color='#A9A9A9', linestyle='--', alpha=0.8, label='100N Ref') # LightGray
    
    # Legend Plot 2
    legend2 = axes[1].legend(facecolor='black', edgecolor='white', title='Method', title_fontsize=12)
    plt.setp(legend2.get_title(), color='white')
    for text in legend2.get_texts():
        text.set_color("white")

    plt.tight_layout()
    output_filename = 'sweep_analysis_loss_force_dark.png'
    plt.savefig(output_filename, facecolor=fig.get_facecolor(), edgecolor='none')
    print(f"Plots saved to {output_filename}")

def save_separate_run_plots(file_path='all_sweeps_recalculated.csv'):
    # 1. Load Data
    try:
        df = pd.read_csv(file_path)
    except FileNotFoundError:
        print(f"Error: '{file_path}' not found.")
        return

    # 2. Define the methods to process
    methods = [
        'Fixed Force 100N',
        'Fixed Force 50N',
        'Weak Force Penalty',
        'Strong Force Penalty'
    ]

    # --- Switch to Dark Background Style ---
    plt.style.use('dark_background')

    # 3. Loop through each method and create a separate file
    for method in methods:
        loss_col = f'True_Loss_{method}'
        force_col = f'Force_{method}'

        # Check if columns exist
        if loss_col in df.columns and force_col in df.columns:
            # Create figure with explicit black background
            fig, axes = plt.subplots(1, 2, figsize=(18, 7))
            fig.patch.set_facecolor('black')
            
            # --- Subplot 1: True Loss ---
            # Use bright Cyan (#00FFFF) for visibility
            sns.lineplot(
                data=df, 
                x="Num_Fingers", 
                y=loss_col, 
                marker="o", 
                errorbar='sd', 
                ax=axes[0],
                color='#00FFFF', 
                linewidth=2.5
            )
            # Customize axis colors
            axes[0].set_facecolor('black')
            axes[0].set_title(f"{method}: Distance Loss (Mean ± Std Dev)", fontsize=16, fontweight='bold', color='white')
            axes[0].set_ylabel("Distance Loss", fontsize=14, color='white')
            axes[0].set_xlabel("Number of Fingers", fontsize=14, color='white')
            axes[0].tick_params(axis='both', colors='white', labelsize=12)
            axes[0].grid(True, linestyle='--', alpha=0.3, color='gray')
            for spine in axes[0].spines.values():
                spine.set_edgecolor('white')

            # --- Subplot 2: Average Force ---
            # Use bright Gold (#FFD700) for visibility
            sns.lineplot(
                data=df, 
                x="Num_Fingers", 
                y=force_col, 
                marker="o", 
                errorbar='sd', 
                ax=axes[1],
                color='#FFD700', 
                linewidth=2.5
            )
            axes[1].set_facecolor('black')
            axes[1].set_title(f"{method}: Average Force (Mean ± Std Dev)", fontsize=16, fontweight='bold', color='white')
            axes[1].set_ylabel("Force (N)", fontsize=14, color='white')
            axes[1].set_xlabel("Number of Fingers", fontsize=14, color='white')
            axes[1].tick_params(axis='both', colors='white', labelsize=12)
            axes[1].grid(True, linestyle='--', alpha=0.3, color='gray')
            for spine in axes[1].spines.values():
                spine.set_edgecolor('white')

            # Add benchmarks with bright colors
            axes[1].axhline(50, color='#FF4500', linestyle='--', alpha=0.8, label='50N Ref') # OrangeRed
            axes[1].axhline(100, color='#A9A9A9', linestyle='--', alpha=0.8, label='100N Ref') # LightGray
            
            # Legend customization
            legend = axes[1].legend(facecolor='black', edgecolor='white')
            for text in legend.get_texts():
                text.set_color("white")

            # --- Save the File ---
            safe_name = method.replace(" ", "_").lower()
            filename = f"plot_{safe_name}_dark.png"
            
            plt.suptitle(f"Analysis: {method}", fontsize=20, color='white')
            plt.tight_layout()
            # Ensure saved image retains the black background
            plt.savefig(filename, facecolor=fig.get_facecolor(), edgecolor='none')
            plt.close() 
            
            print(f"Saved plot to: {filename}")
        else:
            print(f"Skipping {method} (Columns not found)")

def plot_from_optimization_file(csv_filename):
# 1. Read the file
    try:
        df = pd.read_csv(csv_filename)
    except FileNotFoundError:
        print(f"Error: '{csv_filename}' not found.")
        return

    # 2. Setup Data
    method_name = "LBFGS"
    force_cols = [c for c in df.columns if c.startswith('Force_')]
    num_tendons = len(force_cols)

    # --- Switch to Dark Background Style ---
    plt.style.use('dark_background')

    # 3. Create Figure
    # Using the larger figsize from your design example
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(18, 7))
    fig.patch.set_facecolor('black')

    # --- Plot A: Loss (Cyan Style) ---
    # Using #00FFFF (Cyan) as requested in the design
    ax1.plot(df["Iteration"], df["Loss"], marker='o', color='#00FFFF', linewidth=2.5, label="Loss")
    
    # Styling Ax1
    ax1.set_facecolor('black')
    ax1.set_title(f"{method_name}: Loss Convergence", fontsize=16, fontweight='bold', color='white')
    ax1.set_xlabel("Iteration", fontsize=14, color='white')
    ax1.set_ylabel("Total Loss", fontsize=14, color='white')
    ax1.tick_params(axis='both', colors='white', labelsize=12)
    ax1.grid(True, linestyle='--', alpha=0.3, color='gray')
    
    # Set spines to white
    for spine in ax1.spines.values():
        spine.set_edgecolor('white')
        
    # Legend for Ax1
    legend1 = ax1.legend(facecolor='black', edgecolor='white')
    for text in legend1.get_texts():
        text.set_color("white")

    # --- Plot B: Forces (Multi-Color/Neon Style) ---
    # We generate a colormap that pops against black (using 'hsv' or 'jet' for high contrast)
    colors = ['#00FFFF', '#FFD700', '#32CD32', '#FF00FF']
    
    for i, col in enumerate(force_cols):
        t_idx = col.split('_')[1] 
        ax2.plot(df["Iteration"], df[col], 
                label=f"Tendon {t_idx}", color=colors[i], linewidth=2.0, alpha=0.9)

    # Styling Ax2
    ax2.set_facecolor('black')
    ax2.set_title(f"{method_name}: Force Trajectories", fontsize=16, fontweight='bold', color='white')
    ax2.set_xlabel("Iteration", fontsize=14, color='white')
    ax2.set_ylabel("Force (N)", fontsize=14, color='white')
    ax2.tick_params(axis='both', colors='white', labelsize=12)
    ax2.grid(True, linestyle='--', alpha=0.3, color='gray')

    # Set spines to white
    for spine in ax2.spines.values():
        spine.set_edgecolor('white')

    # Legend for Ax2 (Handling placement and dark theme)
    if num_tendons <= 5:
        legend2 = ax2.legend(loc='best', facecolor='black', edgecolor='white')
    else:
        legend2 = ax2.legend(loc='upper right', bbox_to_anchor=(1.3, 1.0), 
                             fontsize='small', facecolor='black', edgecolor='white')
    
    # Fix legend text color
    for text in legend2.get_texts():
        text.set_color("white")

    # --- Save the File ---
    plt.suptitle(f"Optimization Analysis: {method_name}", fontsize=20, color='white')
    plt.tight_layout()
    
    plot_filename = f"{method_name}_plot_dark.png"
    # Ensure saved image retains the black background
    plt.savefig(plot_filename, dpi=150, facecolor=fig.get_facecolor(), edgecolor='none')
    plt.close()
    
    print(f"Plot saved to {plot_filename}")

if __name__ == "__main__":
    #compare_four_sweeps()
    #compare_finger_counts(pd.read_csv('all_sweeps_merged.csv'))
    #analyze_fingers_force(pd.read_csv('all_sweeps_merged.csv'))
    #recalculate_true_loss('all_sweeps_merged.csv')
    compare_recalculated_sweeps('all_sweeps_recalculated.csv')
    plot_sweeps_no_radius()
    save_separate_run_plots('all_sweeps_recalculated.csv')
    plot_from_optimization_file('lbfgs_results.csv')