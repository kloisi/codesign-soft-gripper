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


if __name__ == "__main__":
    #compare_four_sweeps()
    #compare_finger_counts(pd.read_csv('all_sweeps_merged.csv'))
    analyze_fingers_force(pd.read_csv('all_sweeps_merged.csv'))