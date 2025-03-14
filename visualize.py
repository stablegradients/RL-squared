import torch
import gymnasium as gym
import argparse
import numpy as np
from SAC_models import SAC

# Set device
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def visualize_policy(model_path, episodes=5, render=True):
    """
    Visualize a trained policy
    """
    # Create environment
    env = gym.make("HalfCheetah-v5", render_mode="human" if render else None)
    
    # Initialize agent
    agent = SAC(env)
    
    # Load trained model
    agent.load(model_path)
    
    for ep in range(episodes):
        state, _ = env.reset()
        done = False
        total_reward = 0
        steps = 0
        
        while not done:
            # Select action
            action = agent.select_action(state, evaluate=True)
            
            # Take step
            state, reward, terminated, truncated, _ = env.step(action)
            done = terminated or truncated
            total_reward += reward
            steps += 1
        
        print(f"Episode {ep+1}: Reward = {total_reward}, Steps = {steps}")
    
    env.close()

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", type=str, required=True, help="Path to model checkpoint")
    parser.add_argument("--episodes", type=int, default=5, help="Number of episodes to visualize")
    parser.add_argument("--no-render", action="store_true", help="Disable rendering")
    
    args = parser.parse_args()
    
    visualize_policy(args.model, args.episodes, not args.no_render) 