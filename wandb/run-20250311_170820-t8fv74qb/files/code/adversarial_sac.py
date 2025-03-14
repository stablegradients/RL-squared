import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import numpy as np
import copy
import wandb
from torch.distributions import Normal
from utils import ReplayBuffer
from models import Actor, Critic

# Set device
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Adversarial Network - takes random noise and outputs initial states
class AdversaryNetwork(nn.Module):
    def __init__(self, noise_dim, state_dim, hidden_dim=256, state_bounds=None):
        super(AdversaryNetwork, self).__init__()
        
        self.state_dim = state_dim
        self.state_bounds = state_bounds  # Should be a tuple of (min_state, max_state)
        
        # Network architecture
        self.fc1 = nn.Linear(noise_dim, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, hidden_dim)
        self.fc3 = nn.Linear(hidden_dim, state_dim)
        
    def forward(self, noise):
        x = F.relu(self.fc1(noise))
        x = F.relu(self.fc2(x))
        # Use tanh to bound the output, then scale to the desired range
        raw_state = torch.tanh(self.fc3(x))
        
        if self.state_bounds is not None:
            min_state, max_state = self.state_bounds
            min_state = torch.FloatTensor(min_state).to(device)
            max_state = torch.FloatTensor(max_state).to(device)
            
            # Scale from [-1, 1] to [min_state, max_state]
            scaled_state = 0.5 * (raw_state + 1.0) * (max_state - min_state) + min_state
            return scaled_state
        else:
            return raw_state

# Adversarial SAC Agent
class AdversarialSAC:
    def __init__(
        self, 
        env,
        noise_dim=10,
        hidden_dim=256,
        sac_hidden_dim=256,
        lr=3e-4,
        adv_lr=1e-4,
        gamma=0.99,
        tau=0.005,
        alpha=0.2,
        batch_size=256,
        buffer_size=1000000,
        auto_tune_alpha=True,
        state_bounds=None,
        adv_episodes=5,  # Number of episodes per adversarial update
        adv_gamma=0.99,  # Discount factor for adversary's return
        wandb_log=False
    ):
        self.env = env
        self.gamma = gamma
        self.tau = tau
        self.batch_size = batch_size
        self.auto_tune_alpha = auto_tune_alpha
        self.wandb_log = wandb_log
        self.noise_dim = noise_dim
        self.adv_episodes = adv_episodes
        self.adv_gamma = adv_gamma
        
        state_dim = env.observation_space.shape[0]
        action_dim = env.action_space.shape[0]
        max_action = float(env.action_space.high[0])
        
        # If state_bounds not provided, use environment's observation space
        if state_bounds is None:
            low = env.observation_space.low
            high = env.observation_space.high
            state_bounds = (low, high)
        self.state_bounds = state_bounds
        
        # Print acceptable bounds for initial state
        print("\nAcceptable bounds for adversary's initial state setting:")
        print(f"Min state values: {state_bounds[0]}")
        print(f"Max state values: {state_bounds[1]}\n")
        
        # Initialize SAC agent (policy)
        self.sac = SAC(
            env=env,
            hidden_dim=sac_hidden_dim,
            lr=lr,
            gamma=gamma,
            tau=tau,
            alpha=alpha,
            batch_size=batch_size,
            buffer_size=buffer_size,
            auto_tune_alpha=auto_tune_alpha,
            wandb_log=False  # We'll handle wandb logging in this class
        )
        
        # Initialize adversary network
        self.adversary = AdversaryNetwork(
            noise_dim=noise_dim,
            state_dim=state_dim,
            hidden_dim=hidden_dim,
            state_bounds=state_bounds
        ).to(device)
        
        # Initialize optimizer for adversary
        self.adversary_optimizer = optim.Adam(self.adversary.parameters(), lr=adv_lr)
        
        # Buffer to store adversary experiences (noise, return)
        self.adversary_buffer = ReplayBuffer(buffer_size // 10)  # Smaller buffer for adversary
    
    def generate_initial_state(self, evaluate=False):
        # Generate random noise
        noise = torch.randn(1, self.noise_dim).to(device)
        
        if evaluate:
            with torch.no_grad():
                initial_state = self.adversary(noise)
                return initial_state.cpu().data.numpy().flatten()
        else:
            # During training, we need to keep the noise for gradient updates
            initial_state = self.adversary(noise)
            return initial_state.cpu().data.numpy().flatten(), noise
    
    def update_adversary(self, noise_batch, returns_batch):
        # Convert to tensors
        noise_batch = torch.FloatTensor(noise_batch).to(device)
        returns_batch = torch.FloatTensor(returns_batch).to(device).unsqueeze(1)
        
        # Generate initial states from noise
        initial_states = self.adversary(noise_batch)
        
        # Adversary's objective is to minimize policy returns (maximize negative returns)
        adversary_loss = -returns_batch.mean()
        
        # Optimize the adversary
        self.adversary_optimizer.zero_grad()
        adversary_loss.backward()
        self.adversary_optimizer.step()
        
        return {
            'adversary_loss': adversary_loss.item(),
            'mean_policy_return': -adversary_loss.item()
        }
    
    def train(self, max_steps=1000000, policy_episodes_per_adv=5, policy_steps_per_update=1000, 
              max_ep_len=1000, updates_per_step=1, eval_interval=10000, eval_episodes=10):
        total_steps = 0
        adv_episode = 0
        
        while total_steps < max_steps:
            # Outer loop: Adversary sets the initial state
            if self.wandb_log:
                wandb.log({'train/adversary_episode': adv_episode}, step=total_steps)
            
            # Generate initial state from adversary
            initial_state, noise = self.generate_initial_state()
            
            # Inner loop: Run policy for multiple episodes from this initial state
            policy_returns = []
            
            for _ in range(policy_episodes_per_adv):
                # Set environment to the initial state
                # Note: This is environment-dependent and might need custom implementation
                self.env.reset()
                try:
                    # Try to use set_state if available
                    self.env.set_state(initial_state)
                    state = initial_state
                except AttributeError:
                    # If set_state is not available, we'll need a different approach
                    # This is highly environment-specific and may need custom implementation
                    print("Warning: Environment does not support set_state. Using reset() instead.")
                    state, _ = self.env.reset()
                
                episode_reward = 0
                episode_steps = 0
                done = False
                
                while not done and episode_steps < max_ep_len:
                    # Select action using the policy
                    action = self.sac.select_action(state)
                    
                    # Take step in environment
                    next_state, reward, terminated, truncated, _ = self.env.step(action)
                    done = terminated or truncated
                    episode_reward += reward
                    episode_steps += 1
                    total_steps += 1
                    
                    # Add experience to SAC replay buffer
                    self.sac.replay_buffer.push(state, action, reward, next_state, done)
                    
                    # Update SAC parameters
                    if len(self.sac.replay_buffer) > self.batch_size and total_steps % updates_per_step == 0:
                        update_metrics = self.sac.update_parameters()
                        
                        # Log SAC metrics to wandb
                        if self.wandb_log:
                            metrics = {
                                'train/critic_loss': update_metrics['critic_loss'],
                                'train/actor_loss': update_metrics['actor_loss'],
                                'train/alpha_loss': update_metrics['alpha_loss'],
                                'train/alpha': update_metrics['alpha'],
                                'train/critic_grad_norm': update_metrics['critic_grad_norm'],
                                'train/actor_grad_norm': update_metrics['actor_grad_norm'],
                                'train/q1_value': update_metrics['q1_value'],
                                'train/q2_value': update_metrics['q2_value'],
                                'train/log_pi': update_metrics['log_pi'],
                                'train/total_env_steps': total_steps,
                                'train/buffer_size': len(self.sac.replay_buffer)
                            }
                            wandb.log(metrics, step=total_steps)
                    
                    # Move to next state
                    state = next_state
                    
                    # Evaluate periodically
                    if total_steps % eval_interval == 0:
                        self.evaluate(eval_episodes=eval_episodes)
                
                # Track policy return for this episode
                policy_returns.append(episode_reward)
                
                if self.wandb_log:
                    wandb.log({
                        'train/policy_episode_reward': episode_reward,
                        'train/policy_episode_length': episode_steps,
                        'train/total_env_steps': total_steps
                    }, step=total_steps)
                
                print(f"Policy Episode {_ + 1}/{policy_episodes_per_adv}: Reward: {episode_reward}, Length: {episode_steps}")
            
            # Compute discounted return for adversary
            # The adversary's goal is to minimize the policy's return
            discounted_return = 0
            for i, ret in enumerate(policy_returns):
                discounted_return += (self.adv_gamma ** i) * ret
            
            # Store adversary experience
            self.adversary_buffer.push(noise.cpu().numpy().flatten(), np.array([0.0]), -discounted_return, np.zeros_like(noise.cpu().numpy().flatten()), False)
            
            # Update adversary network
            if len(self.adversary_buffer) > self.batch_size:
                # Sample batch from adversary buffer
                noise_batch, _, return_batch, _, _ = self.adversary_buffer.sample(self.batch_size)
                
                # Update adversary
                adv_metrics = self.update_adversary(noise_batch, return_batch)
                
                if self.wandb_log:
                    wandb.log({
                        'train/adversary_loss': adv_metrics['adversary_loss'],
                        'train/mean_policy_return': adv_metrics['mean_policy_return'],
                        'train/total_env_steps': total_steps
                    }, step=total_steps)
            
            # Log adversary episode information
            print(f"Adversary Episode {adv_episode + 1}: Mean Policy Return: {np.mean(policy_returns)}")
            if self.wandb_log:
                wandb.log({
                    'train/adversary_episode': adv_episode,
                    'train/mean_policy_return': np.mean(policy_returns),
                    'train/discounted_return': discounted_return,
                    'train/total_env_steps': total_steps
                }, step=total_steps)
            
            adv_episode += 1
        
        return total_steps
    
    def evaluate(self, eval_episodes=10):
        # Evaluate both the adversary and the policy
        
        # 1. Evaluate policy on random initial states
        avg_random_reward = 0.
        for _ in range(eval_episodes):
            state, _ = self.env.reset()
            done = False
            ep_reward = 0
            
            while not done:
                action = self.sac.select_action(state, evaluate=True)
                state, reward, terminated, truncated, _ = self.env.step(action)
                done = terminated or truncated
                ep_reward += reward
                
            avg_random_reward += ep_reward
        avg_random_reward /= eval_episodes
        
        # 2. Evaluate policy on adversarial initial states
        avg_adv_reward = 0.
        for _ in range(eval_episodes):
            # Generate adversarial initial state
            adv_state = self.generate_initial_state(evaluate=True)
            
            # Set environment to adversarial state
            self.env.reset()
            try:
                self.env.set_state(adv_state)
                state = adv_state
            except AttributeError:
                state, _ = self.env.reset()
                print("Warning: Environment does not support set_state. Using reset() instead.")
            
            done = False
            ep_reward = 0
            
            while not done:
                action = self.sac.select_action(state, evaluate=True)
                state, reward, terminated, truncated, _ = self.env.step(action)
                done = terminated or truncated
                ep_reward += reward
                
            avg_adv_reward += ep_reward
        avg_adv_reward /= eval_episodes
        
        print("---------------------------------------")
        print(f"Evaluation over {eval_episodes} episodes:")
        print(f"Random initial states average reward: {avg_random_reward}")
        print(f"Adversarial initial states average reward: {avg_adv_reward}")
        print("---------------------------------------")
        
        if self.wandb_log:
            wandb.log({
                'eval/random_avg_reward': avg_random_reward,
                'eval/adversarial_avg_reward': avg_adv_reward,
                'eval/reward_gap': avg_random_reward - avg_adv_reward
            })
        
        return avg_random_reward, avg_adv_reward
    
    def save(self, filename_prefix):
        # Save policy models
        self.sac.save(f"{filename_prefix}_policy.pth")
        
        # Save adversary model
        torch.save(
            {
                'adversary': self.adversary.state_dict(),
            }, 
            f"{filename_prefix}_adversary.pth"
        )
        
        if self.wandb_log:
            wandb.save(f"{filename_prefix}_adversary.pth")
    
    def load(self, filename_prefix):
        # Load policy models
        self.sac.load(f"{filename_prefix}_policy.pth")
        
        # Load adversary model
        checkpoint = torch.load(f"{filename_prefix}_adversary.pth")
        self.adversary.load_state_dict(checkpoint['adversary'])


# Importing the original SAC class from the provided code
from SAC_models import SAC

# Main function for training a single agent
def train_agent(args):
    import gymnasium
    
    # Create environment
    env = gymnasium.make(args.env_name)
    
    # Define state bounds
    state_dim = env.observation_space.shape[0]
    default_low = env.observation_space.low
    default_high = env.observation_space.high
    
    # Limit the adversary's ability to set extreme initial states
    restricted_low = np.maximum(default_low, -args.state_bound * np.ones(state_dim))
    restricted_high = np.minimum(default_high, args.state_bound * np.ones(state_dim))
    
    state_bounds = (restricted_low, restricted_high)
    
    # Initialize wandb
    if args.use_wandb:
        run_name = f"{args.exp_name}_seed{args.seed}"
        wandb.init(
            project=args.project,
            entity=args.entity,
            group=args.group,
            name=run_name,
            config={
                "env_name": args.env_name,
                "seed": args.seed,
                "noise_dim": args.noise_dim,
                "hidden_dim": args.hidden_dim,
                "sac_hidden_dim": args.sac_hidden_dim,
                "lr": args.lr,
                "adv_lr": args.adv_lr,
                "gamma": args.gamma,
                "tau": args.tau,
                "alpha": args.alpha,
                "auto_tune_alpha": args.auto_tune_alpha,
                "state_bounds": state_bounds,
                "adv_episodes": args.adv_episodes,
                "adv_gamma": args.adv_gamma,
                "max_steps": args.max_steps,
                "policy_episodes_per_adv": args.policy_episodes_per_adv,
                "policy_steps_per_update": args.policy_steps_per_update,
                "max_ep_len": args.max_ep_len,
                "updates_per_step": args.updates_per_step,
                "eval_interval": args.eval_interval,
                "eval_episodes": args.eval_episodes,
            }
        )
    
    # Set random seeds
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(args.seed)
    
    # Create adversarial SAC agent
    agent = AdversarialSAC(
        env=env,
        noise_dim=args.noise_dim,
        hidden_dim=args.hidden_dim,
        sac_hidden_dim=args.sac_hidden_dim,
        lr=args.lr,
        adv_lr=args.adv_lr,
        gamma=args.gamma,
        tau=args.tau,
        alpha=args.alpha,
        batch_size=args.batch_size,
        buffer_size=args.buffer_size,
        auto_tune_alpha=args.auto_tune_alpha,
        state_bounds=state_bounds,
        adv_episodes=args.adv_episodes,
        adv_gamma=args.adv_gamma,
        wandb_log=args.use_wandb
    )
    
    # Train the agent
    agent.train(
        max_steps=args.max_steps,
        policy_episodes_per_adv=args.policy_episodes_per_adv,
        policy_steps_per_update=args.policy_steps_per_update,
        max_ep_len=args.max_ep_len,
        updates_per_step=args.updates_per_step,
        eval_interval=args.eval_interval,
        eval_episodes=args.eval_episodes
    )
    
    # Save the trained agent
    save_path = f"{args.save_dir}/{args.exp_name}_seed{args.seed}"
    agent.save(save_path)
    
    # Close environment
    env.close()
    
    # Close wandb
    if args.use_wandb:
        wandb.finish()
    
    return agent

# Run ablation with multiple seeds
def run_ablation(args):
    seeds = [args.seed, args.seed + 1, args.seed + 2, args.seed + 3]
    
    for seed in seeds:
        print(f"\n{'='*50}")
        print(f"Running experiment with seed {seed}")
        print(f"{'='*50}\n")
        
        # Update the seed
        args.seed = seed
        
        # Train agent with this seed
        train_agent(args)

# Example usage
if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="Adversarial SAC Training")
    
    # Environment settings
    parser.add_argument("--env_name", type=str, default="HalfCheetah-v4", help="Gymnasium environment name")
    parser.add_argument("--state_bound", type=float, default=5.0, help="Bound for adversarial state initialization (±)")
    
    # Experiment settings
    parser.add_argument("--exp_name", type=str, default="adv_sac", help="Experiment name")
    parser.add_argument("--seed", type=int, default=0, help="Random seed (for ablation, this is the starting seed)")
    parser.add_argument("--ablation", action="store_true", help="Run ablation with 4 seeds")
    
    # SAC parameters
    parser.add_argument("--noise_dim", type=int, default=10, help="Dimension of noise input for adversary")
    parser.add_argument("--hidden_dim", type=int, default=256, help="Hidden dimension for adversary network")
    parser.add_argument("--sac_hidden_dim", type=int, default=256, help="Hidden dimension for SAC networks")
    parser.add_argument("--lr", type=float, default=3e-4, help="Learning rate for SAC")
    parser.add_argument("--adv_lr", type=float, default=1e-4, help="Learning rate for adversary")
    parser.add_argument("--gamma", type=float, default=0.99, help="Discount factor for SAC")
    parser.add_argument("--tau", type=float, default=0.005, help="Soft update coefficient")
    parser.add_argument("--alpha", type=float, default=0.2, help="Initial temperature parameter")
    parser.add_argument("--auto_tune_alpha", type=bool, default=True, help="Auto-tune entropy coefficient")
    parser.add_argument("--batch_size", type=int, default=256, help="Batch size")
    parser.add_argument("--buffer_size", type=int, default=1000000, help="Replay buffer size")
    
    # Adversary parameters
    parser.add_argument("--adv_episodes", type=int, default=5, help="Number of episodes per adversarial update")
    parser.add_argument("--adv_gamma", type=float, default=0.99, help="Discount factor for adversary")
    
    # Training parameters
    parser.add_argument("--max_steps", type=int, default=1000000, help="Maximum number of environment steps")
    parser.add_argument("--policy_episodes_per_adv", type=int, default=5, help="Policy episodes per adversarial update")
    parser.add_argument("--policy_steps_per_update", type=int, default=1000, help="Environment steps per policy update")
    parser.add_argument("--max_ep_len", type=int, default=1000, help="Maximum episode length")
    parser.add_argument("--updates_per_step", type=int, default=1, help="Number of updates per environment step")
    parser.add_argument("--eval_interval", type=int, default=10000, help="Evaluation interval in steps")
    parser.add_argument("--eval_episodes", type=int, default=10, help="Number of episodes for evaluation")
    
    # Logging and saving
    parser.add_argument("--use_wandb", action="store_true", help="Use Weights & Biases for logging")
    parser.add_argument("--project", type=str, default="adversarial_sac", help="W&B project name")
    parser.add_argument("--entity", type=str, default=None, help="W&B entity name")
    parser.add_argument("--group", type=str, default=None, help="W&B group name")
    parser.add_argument("--save_dir", type=str, default="./saved_models", help="Directory to save models")
    
    args = parser.parse_args()
    
    # Create save directory if it doesn't exist
    import os
    if not os.path.exists(args.save_dir):
        os.makedirs(args.save_dir)
    
    # Run experiment(s)
    if args.ablation:
        run_ablation(args)
    else:
        train_agent(args)