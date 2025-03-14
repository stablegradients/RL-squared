import torch
import torch.nn.functional as F
import numpy as np
import gymnasium as gym
import wandb
import argparse
import os
from datetime import datetime

from SAC_models import SAC

# Set device
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def parse_args():
    parser = argparse.ArgumentParser(description="Train SAC on Half-Cheetah with wandb logging")
    
    # Environment parameters
    parser.add_argument("--env_name", type=str, default="HalfCheetah-v5", help="Gym environment name")
    
    # Training parameters
    parser.add_argument("--seed", type=int, default=0, help="Random seed")
    parser.add_argument("--max_steps", type=int, default=1000000, help="Maximum number of environment steps")
    parser.add_argument("--max_ep_len", type=int, default=1000, help="Maximum episode length")
    parser.add_argument("--updates_per_step", type=int, default=1, help="Number of updates per environment step")
    parser.add_argument("--eval_interval", type=int, default=10000, help="Interval between evaluations")
    parser.add_argument("--eval_episodes", type=int, default=10, help="Number of episodes for evaluation")
    
    # SAC hyperparameters
    parser.add_argument("--hidden_dim", type=int, default=256, help="Hidden dimension of networks")
    parser.add_argument("--lr", type=float, default=3e-4, help="Learning rate")
    parser.add_argument("--gamma", type=float, default=0.99, help="Discount factor")
    parser.add_argument("--tau", type=float, default=0.005, help="Target network update rate")
    parser.add_argument("--alpha", type=float, default=0.2, help="Temperature parameter")
    parser.add_argument("--batch_size", type=int, default=256, help="Batch size")
    parser.add_argument("--buffer_size", type=int, default=1000000, help="Replay buffer size")
    parser.add_argument("--auto_tune_alpha", type=bool, default=True, help="Automatically tune alpha")
    
    # Wandb parameters
    parser.add_argument("--wandb_project", type=str, default="sac-test", help="Wandb project name")
    parser.add_argument("--wandb_entity", type=str, default="stablegradients", help="Wandb entity name")
    parser.add_argument("--no_wandb", action="store_true", help="Disable wandb logging")
    parser.add_argument("--tags", type=str, default=None, help="Comma-separated list of tags for the run")
    
    # Checkpoint parameters
    parser.add_argument("--save_interval", type=int, default=10000, help="Interval between saving checkpoints")
    parser.add_argument("--checkpoint_dir", type=str, default="checkpoints", help="Directory to save checkpoints")
    
    return parser.parse_args()

# Example usage
if __name__ == "__main__":
    args = parse_args()
    
    # Create checkpoint directory if it doesn't exist
    os.makedirs(args.checkpoint_dir, exist_ok=True)
    
    # Set random seeds
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    
    # Create environment
    env = gym.make(args.env_name)
    
    # Set environment seed using the newer approach
    env.action_space.seed(args.seed)
    env.observation_space.seed(args.seed)
    
    # Create a unique run name and group name for wandb
    # Group name includes all hyperparameters except seed
    group_name = f"{args.env_name}_h{args.hidden_dim}_lr{args.lr}_g{args.gamma}_t{args.tau}_a{args.alpha}_bs{args.batch_size}"
    run_name = f"{group_name}_seed{args.seed}"
    
    # Initialize wandb
    if not args.no_wandb:
        tags = args.tags.split(",") if args.tags else []
        wandb.init(
            project=args.wandb_project,
            entity=args.wandb_entity,
            config=vars(args),
            name=run_name,
            group=group_name,
            tags=tags,
            monitor_gym=True  # Auto-upload videos of agents playing
        )
    
    # Initialize and train SAC agent
    sac = SAC(
        env,
        hidden_dim=args.hidden_dim,
        lr=args.lr,
        gamma=args.gamma,
        tau=args.tau,
        alpha=args.alpha,
        batch_size=args.batch_size,
        buffer_size=args.buffer_size,
        auto_tune_alpha=args.auto_tune_alpha,
        wandb_log=not args.no_wandb
    )
    
    # Training loop with periodic checkpoints
    total_steps = 0
    while total_steps < args.max_steps:
        # Train for save_interval steps
        next_checkpoint = min(total_steps + args.save_interval, args.max_steps)
        sac.train(
            max_steps=next_checkpoint,
            max_ep_len=args.max_ep_len,
            updates_per_step=args.updates_per_step,
            eval_interval=args.eval_interval,
            eval_episodes=args.eval_episodes,
            start_steps=total_steps
        )
        
        # Save checkpoint
        checkpoint_path = os.path.join(args.checkpoint_dir, f"{run_name}_steps{next_checkpoint}.pt")
        sac.save(checkpoint_path)
        print(f"Saved checkpoint to {checkpoint_path}")
        
        total_steps = next_checkpoint
    
    # Final evaluation
    sac.evaluate(eval_episodes=args.eval_episodes, log_wandb=not args.no_wandb, total_steps=total_steps)
    
    # Close wandb run when done
    if not args.no_wandb:
        wandb.finish()