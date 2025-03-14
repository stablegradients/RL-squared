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

# SAC Agent
class SAC:
    def __init__(
        self, 
        env,
        hidden_dim=256,
        lr=3e-4,
        gamma=0.99,
        tau=0.005,
        alpha=0.2,
        batch_size=256,
        buffer_size=1000000,
        auto_tune_alpha=True,
        wandb_log=False
    ):
        self.env = env
        self.gamma = gamma
        self.tau = tau
        self.batch_size = batch_size
        self.auto_tune_alpha = auto_tune_alpha
        self.wandb_log = wandb_log
        
        state_dim = env.observation_space.shape[0]
        action_dim = env.action_space.shape[0]
        max_action = float(env.action_space.high[0])
        
        # Initialize actor and critic networks
        self.actor = Actor(state_dim, action_dim, max_action, hidden_dim).to(device)
        self.critic = Critic(state_dim, action_dim, hidden_dim).to(device)
        self.critic_target = copy.deepcopy(self.critic)
        
        # Freeze target networks with respect to optimizers
        for p in self.critic_target.parameters():
            p.requires_grad = False
            
        # Initialize optimizers
        self.actor_optimizer = optim.Adam(self.actor.parameters(), lr=lr)
        self.critic_optimizer = optim.Adam(self.critic.parameters(), lr=lr)
        
        # Initialize replay buffer
        self.replay_buffer = ReplayBuffer(buffer_size)
        
        # Initialize temperature parameter alpha
        self.log_alpha = torch.zeros(1, requires_grad=True, device=device)
        self.alpha = alpha
        
        # Set up automatic alpha tuning if enabled
        if auto_tune_alpha:
            self.target_entropy = -torch.prod(torch.Tensor(env.action_space.shape).to(device)).item()
            self.alpha_optimizer = optim.Adam([self.log_alpha], lr=lr)
            self.alpha = self.log_alpha.exp().item()
    
    def select_action(self, state, evaluate=False):
        state = torch.FloatTensor(state).to(device).unsqueeze(0)
        
        if evaluate:
            with torch.no_grad():
                mean, _ = self.actor(state)
                return mean.cpu().data.numpy().flatten()
        else:
            with torch.no_grad():
                action, _ = self.actor.sample(state)
                return action.cpu().data.numpy().flatten()
    
    def update_parameters(self):
        # Sample batch from replay buffer
        state_batch, action_batch, reward_batch, next_state_batch, done_batch = self.replay_buffer.sample(self.batch_size)
        
        # Convert to tensors
        state_batch = torch.FloatTensor(state_batch).to(device)
        action_batch = torch.FloatTensor(action_batch).to(device)
        reward_batch = torch.FloatTensor(reward_batch).to(device).unsqueeze(1)
        next_state_batch = torch.FloatTensor(next_state_batch).to(device)
        done_batch = torch.FloatTensor(done_batch).to(device).unsqueeze(1)
        
        with torch.no_grad():
            next_action, next_log_pi = self.actor.sample(next_state_batch)
            next_q1_target, next_q2_target = self.critic_target(next_state_batch, next_action)
            next_q_target = torch.min(next_q1_target, next_q2_target)
            next_q_target = next_q_target - self.alpha * next_log_pi
            q_target = reward_batch + (1 - done_batch) * self.gamma * next_q_target
        
        # Get current Q estimates
        current_q1, current_q2 = self.critic(state_batch, action_batch)
        critic_loss = F.mse_loss(current_q1, q_target) + F.mse_loss(current_q2, q_target)
        
        # Optimize the critic
        self.critic_optimizer.zero_grad()
        critic_loss.backward()
        
        # Calculate critic gradient norm for logging
        critic_grad_norm = 0.0
        for param in self.critic.parameters():
            if param.grad is not None:
                param_norm = param.grad.data.norm(2)
                critic_grad_norm += param_norm.item() ** 2
        critic_grad_norm = critic_grad_norm ** 0.5
        
        self.critic_optimizer.step()
        
        # Compute actor loss
        actions_pred, log_pi = self.actor.sample(state_batch)
        q1_pred, q2_pred = self.critic(state_batch, actions_pred)
        q_pred = torch.min(q1_pred, q2_pred)
        
        actor_loss = (self.alpha * log_pi - q_pred).mean()
        
        # Optimize the actor
        self.actor_optimizer.zero_grad()
        actor_loss.backward()
        
        # Calculate actor gradient norm for logging
        actor_grad_norm = 0.0
        for param in self.actor.parameters():
            if param.grad is not None:
                param_norm = param.grad.data.norm(2)
                actor_grad_norm += param_norm.item() ** 2
        actor_grad_norm = actor_grad_norm ** 0.5
        
        self.actor_optimizer.step()
        
        # Update alpha if auto-tuning is enabled
        alpha_loss = None
        if self.auto_tune_alpha:
            alpha_loss = -(self.log_alpha * (log_pi + self.target_entropy).detach()).mean()
            
            self.alpha_optimizer.zero_grad()
            alpha_loss.backward()
            self.alpha_optimizer.step()
            
            self.alpha = self.log_alpha.exp().item()
        
        # Soft update target network
        for target_param, param in zip(self.critic_target.parameters(), self.critic.parameters()):
            target_param.data.copy_(target_param.data * (1.0 - self.tau) + param.data * self.tau)
        
        # Return losses and gradient norms for logging
        return {
            'critic_loss': critic_loss.item(),
            'actor_loss': actor_loss.item(),
            'alpha_loss': alpha_loss.item() if alpha_loss is not None else 0.0,
            'alpha': self.alpha,
            'critic_grad_norm': critic_grad_norm,
            'actor_grad_norm': actor_grad_norm,
            'q1_value': current_q1.mean().item(),
            'q2_value': current_q2.mean().item(),
            'log_pi': log_pi.mean().item()
        }
    
    def train(self, max_steps=1000000, max_ep_len=1000, updates_per_step=1, eval_interval=10000, eval_episodes=10, start_steps=0):
        total_steps = start_steps
        episode_reward = 0
        episode_steps = 0
        episode_num = 0
        episode_rewards = []
        state, _ = self.env.reset()  # Gymnasium returns (state, info)
        
        while total_steps < max_steps:
            # Select action
            if total_steps < 10000:  # Initial exploration phase
                action = self.env.action_space.sample()
            else:
                action = self.select_action(state)
            
            # Take step in environment
            next_state, reward, terminated, truncated, _ = self.env.step(action)
            done = terminated or truncated  # In Gymnasium, done = terminated OR truncated
            episode_reward += reward
            episode_steps += 1
            total_steps += 1
            
            # Add experience to replay buffer
            self.replay_buffer.push(state, action, reward, next_state, done)
            
            # Update network parameters
            update_metrics = {}
            if len(self.replay_buffer) > self.batch_size:
                for _ in range(updates_per_step):
                    update_metrics = self.update_parameters()
            
            # Log metrics to wandb
            if self.wandb_log and update_metrics:
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
                    'train/buffer_size': len(self.replay_buffer)
                }
                wandb.log(metrics, step=total_steps)
            
            # Reset environment if done or max episode length reached
            if done or episode_steps >= max_ep_len:
                if self.wandb_log:
                    wandb.log({
                        'train/episode_reward': episode_reward,
                        'train/episode_length': episode_steps,
                        'train/total_env_steps': total_steps,
                        'train/episode': episode_num
                    }, step=total_steps)
                
                print(f"Episode {episode_num + 1}: Total Steps: {total_steps}, Reward: {episode_reward}, Length: {episode_steps}")
                
                episode_rewards.append(episode_reward)
                state, _ = self.env.reset()
                episode_reward = 0
                episode_steps = 0
                episode_num += 1
            else:
                state = next_state
                
            # Evaluate the policy periodically
            if total_steps % eval_interval == 0:
                eval_reward, eval_length = self.evaluate(eval_episodes=eval_episodes, log_wandb=self.wandb_log, total_steps=total_steps)
        
        return total_steps
                
    def evaluate(self, eval_episodes=10, log_wandb=False, total_steps=0):
        avg_reward = 0.
        avg_length = 0.
        for ep in range(eval_episodes):
            state, _ = self.env.reset()
            done = False
            ep_reward = 0
            ep_length = 0
            
            while not done:
                action = self.select_action(state, evaluate=True)
                state, reward, terminated, truncated, _ = self.env.step(action)
                done = terminated or truncated
                ep_reward += reward
                ep_length += 1
                
            avg_reward += ep_reward
            avg_length += ep_length
            
        avg_reward /= eval_episodes
        avg_length /= eval_episodes
        
        print("---------------------------------------")
        print(f"Evaluation over {eval_episodes} episodes: {avg_reward}")
        print(f"Average episode length: {avg_length}")
        print("---------------------------------------")
        
        if log_wandb:
            wandb.log({
                'eval/avg_reward': avg_reward,
                'eval/avg_episode_length': avg_length
            }, step=total_steps)
        
        return avg_reward, avg_length
    
    def save(self, filename):
        torch.save(
            {
                'actor': self.actor.state_dict(),
                'critic': self.critic.state_dict(),
                'critic_target': self.critic_target.state_dict()
            }, 
            filename
        )
        
        if self.wandb_log:
            wandb.save(filename)
        
    def load(self, filename):
        checkpoint = torch.load(filename)
        self.actor.load_state_dict(checkpoint['actor'])
        self.critic.load_state_dict(checkpoint['critic'])
        self.critic_target.load_state_dict(checkpoint['critic_target']) 