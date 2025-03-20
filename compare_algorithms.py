import wandb
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import argparse
from collections import defaultdict

def parse_args():
    parser = argparse.ArgumentParser(description="Compare SAC and Actor-Critic performance across environments")
    parser.add_argument("--sac_project", type=str, default="sac-ablation", help="Wandb project name for SAC")
    parser.add_argument("--ac_project", type=str, default="actor-critic-ablation", help="Wandb project name for Actor-Critic")
    parser.add_argument("--entity", type=str, default="stablegradients", help="Wandb entity name")
    parser.add_argument("--output_dir", type=str, default="comparison_results", help="Directory to save comparison plots")
    parser.add_argument("--metric", type=str, default="eval/avg_reward", help="Metric to compare")
    parser.add_argument("--smoothing", type=int, default=1, help="Window size for smoothing the learning curves")
    return parser.parse_args()

def get_runs_data(api, project, entity, metric):
    """Fetch run data from wandb for a specific project and metric"""
    runs = api.runs(f"{entity}/{project}")
    
    env_data = defaultdict(lambda: defaultdict(list))
    
    for run in runs:
        if run.state != "finished":
            continue
            
        # Get environment name from tags
        env_name = None
        for tag in run.tags:
            if tag in ["Reacher-v4", "Swimmer-v5", "Walker2d-v5", "InvertedPendulum-v5", "Pusher-v4"]:
                env_name = tag
                break
        
        if not env_name:
            continue
            
        # Get history data
        history = run.history(keys=[metric, "train/total_env_steps"])
        
        if history.empty or metric not in history.columns:
            continue
            
        # Record the data
        env_data[env_name]["steps"].append(history["train/total_env_steps"].values)
        env_data[env_name]["values"].append(history[metric].values)
        env_data[env_name]["run_names"].append(run.name)
    
    return env_data

def smooth_curve(y, window_size):
    """Apply smoothing to the curve using a moving average"""
    if window_size <= 1:
        return y
    
    box = np.ones(window_size) / window_size
    y_smooth = np.convolve(y, box, mode='same')
    return y_smooth

def plot_comparison(sac_data, ac_data, env_name, metric, smoothing, output_dir):
    """Create a comparison plot for a specific environment"""
    plt.figure(figsize=(12, 8))
    
    # Plot SAC runs
    for i in range(len(sac_data[env_name]["steps"])):
        steps = sac_data[env_name]["steps"][i]
        values = sac_data[env_name]["values"][i]
        
        # Ensure steps and values have the same length
        min_len = min(len(steps), len(values))
        steps = steps[:min_len]
        values = values[:min_len]
        
        # Apply smoothing
        if min_len > 0:
            smoothed_values = smooth_curve(values, smoothing)
            plt.plot(steps, smoothed_values, 'b-', alpha=0.3)
    
    # Calculate and plot SAC mean and std
    if len(sac_data[env_name]["values"]) > 0:
        # Prepare data for averaging
        max_steps = max([len(steps) for steps in sac_data[env_name]["steps"]])
        aligned_values = []
        aligned_steps = None
        
        for i in range(len(sac_data[env_name]["steps"])):
            steps = sac_data[env_name]["steps"][i]
            values = sac_data[env_name]["values"][i]
            
            min_len = min(len(steps), len(values))
            if min_len > 0:
                # Pad if needed
                padded_values = np.pad(values[:min_len], (0, max_steps - min_len), 'edge')
                aligned_values.append(padded_values)
                if aligned_steps is None:
                    aligned_steps = np.pad(steps[:min_len], (0, max_steps - min_len), 'edge')
        
        if aligned_values:
            aligned_values = np.array(aligned_values)
            mean_values = np.mean(aligned_values, axis=0)
            std_values = np.std(aligned_values, axis=0)
            
            # Apply smoothing
            mean_values_smooth = smooth_curve(mean_values, smoothing)
            
            plt.plot(aligned_steps, mean_values_smooth, 'b-', label='SAC Mean', linewidth=2)
            plt.fill_between(aligned_steps, 
                            mean_values_smooth - std_values, 
                            mean_values_smooth + std_values, 
                            color='blue', alpha=0.2)
    
    # Plot Actor-Critic runs
    for i in range(len(ac_data[env_name]["steps"])):
        steps = ac_data[env_name]["steps"][i]
        values = ac_data[env_name]["values"][i]
        
        # Ensure steps and values have the same length
        min_len = min(len(steps), len(values))
        steps = steps[:min_len]
        values = values[:min_len]
        
        # Apply smoothing
        if min_len > 0:
            smoothed_values = smooth_curve(values, smoothing)
            plt.plot(steps, smoothed_values, 'r-', alpha=0.3)
    
    # Calculate and plot Actor-Critic mean and std
    if len(ac_data[env_name]["values"]) > 0:
        # Prepare data for averaging
        max_steps = max([len(steps) for steps in ac_data[env_name]["steps"]])
        aligned_values = []
        aligned_steps = None
        
        for i in range(len(ac_data[env_name]["steps"])):
            steps = ac_data[env_name]["steps"][i]
            values = ac_data[env_name]["values"][i]
            
            min_len = min(len(steps), len(values))
            if min_len > 0:
                # Pad if needed
                padded_values = np.pad(values[:min_len], (0, max_steps - min_len), 'edge')
                aligned_values.append(padded_values)
                if aligned_steps is None:
                    aligned_steps = np.pad(steps[:min_len], (0, max_steps - min_len), 'edge')
        
        if aligned_values:
            aligned_values = np.array(aligned_values)
            mean_values = np.mean(aligned_values, axis=0)
            std_values = np.std(aligned_values, axis=0)
            
            # Apply smoothing
            mean_values_smooth = smooth_curve(mean_values, smoothing)
            
            plt.plot(aligned_steps, mean_values_smooth, 'r-', label='Actor-Critic Mean', linewidth=2)
            plt.fill_between(aligned_steps, 
                            mean_values_smooth - std_values, 
                            mean_values_smooth + std_values, 
                            color='red', alpha=0.2)
    
    # Set labels and title
    plt.title(f"SAC vs Actor-Critic on {env_name}")
    plt.xlabel("Environment Steps")
    plt.ylabel(metric)
    plt.legend()
    plt.grid(True, linestyle='--', alpha=0.7)
    
    # Save the figure
    import os
    os.makedirs(output_dir, exist_ok=True)
    plt.savefig(f"{output_dir}/{env_name}_comparison.png", dpi=300, bbox_inches='tight')
    plt.close()

def create_summary_table(sac_data, ac_data, env_names, metric, output_dir):
    """Create a summary table comparing SAC and Actor-Critic performance"""
    summary = {
        "Environment": [],
        "SAC Mean": [],
        "SAC Std": [],
        "AC Mean": [],
        "AC Std": [],
        "SAC Final Mean": [],
        "SAC Final Std": [],
        "AC Final Mean": [],
        "AC Final Std": []
    }
    
    for env_name in env_names:
        summary["Environment"].append(env_name)
        
        # Calculate SAC stats
        sac_final_values = []
        sac_avg_values = []
        
        if env_name in sac_data and len(sac_data[env_name]["values"]) > 0:
            for values in sac_data[env_name]["values"]:
                if len(values) > 0:
                    sac_final_values.append(values[-1])
                    sac_avg_values.append(np.mean(values))
        
        sac_final_mean = np.mean(sac_final_values) if sac_final_values else float('nan')
        sac_final_std = np.std(sac_final_values) if sac_final_values else float('nan')
        sac_avg_mean = np.mean(sac_avg_values) if sac_avg_values else float('nan')
        sac_avg_std = np.std(sac_avg_values) if sac_avg_values else float('nan')
        
        summary["SAC Mean"].append(sac_avg_mean)
        summary["SAC Std"].append(sac_avg_std)
        summary["SAC Final Mean"].append(sac_final_mean)
        summary["SAC Final Std"].append(sac_final_std)
        
        # Calculate Actor-Critic stats
        ac_final_values = []
        ac_avg_values = []
        
        if env_name in ac_data and len(ac_data[env_name]["values"]) > 0:
            for values in ac_data[env_name]["values"]:
                if len(values) > 0:
                    ac_final_values.append(values[-1])
                    ac_avg_values.append(np.mean(values))
        
        ac_final_mean = np.mean(ac_final_values) if ac_final_values else float('nan')
        ac_final_std = np.std(ac_final_values) if ac_final_values else float('nan')
        ac_avg_mean = np.mean(ac_avg_values) if ac_avg_values else float('nan')
        ac_avg_std = np.std(ac_avg_values) if ac_avg_values else float('nan')
        
        summary["AC Mean"].append(ac_avg_mean)
        summary["AC Std"].append(ac_avg_std)
        summary["AC Final Mean"].append(ac_final_mean)
        summary["AC Final Std"].append(ac_final_std)
    
    # Create DataFrame and save to CSV
    df = pd.DataFrame(summary)
    import os
    os.makedirs(output_dir, exist_ok=True)
    df.to_csv(f"{output_dir}/summary_comparison.csv", index=False)
    
    # Create bar plot
    plt.figure(figsize=(14, 10))
    
    # Set width of bars
    barWidth = 0.35
    
    # Set positions of the bars on X axis
    r1 = np.arange(len(env_names))
    r2 = [x + barWidth for x in r1]
    
    # Create bars
    plt.bar(r1, df["SAC Final Mean"], width=barWidth, yerr=df["SAC Final Std"], label='SAC Final', color='blue', alpha=0.7)
    plt.bar(r2, df["AC Final Mean"], width=barWidth, yerr=df["AC Final Std"], label='Actor-Critic Final', color='red', alpha=0.7)
    
    # Add labels and title
    plt.xlabel('Environment', fontweight='bold')
    plt.ylabel(f'Final {metric}', fontweight='bold')
    plt.title('SAC vs Actor-Critic Final Performance')
    plt.xticks([r + barWidth/2 for r in range(len(env_names))], env_names, rotation=45)
    plt.legend()
    
    plt.tight_layout()
    plt.savefig(f"{output_dir}/final_performance_comparison.png", dpi=300, bbox_inches='tight')
    plt.close()
    
    return df

def main():
    args = parse_args()
    
    # Initialize wandb API
    api = wandb.Api()
    
    # Get data for both algorithms
    print("Fetching SAC data...")
    sac_data = get_runs_data(api, args.sac_project, args.entity, args.metric)
    
    print("Fetching Actor-Critic data...")
    ac_data = get_runs_data(api, args.ac_project, args.entity, args.metric)
    
    # Get list of all environments
    env_names = list(set(list(sac_data.keys()) + list(ac_data.keys())))
    
    # Plot comparison for each environment
    for env_name in env_names:
        print(f"Creating comparison plot for {env_name}...")
        if env_name in sac_data or env_name in ac_data:
            plot_comparison(sac_data, ac_data, env_name, args.metric, args.smoothing, args.output_dir)
    
    # Create summary table
    print("Creating summary table...")
    summary_df = create_summary_table(sac_data, ac_data, env_names, args.metric, args.output_dir)
    
    print(f"Results saved to {args.output_dir}/")
    print("\nSummary of results:")
    print(summary_df)

if __name__ == "__main__":
    main() 