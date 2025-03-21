import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.distributions import Normal
import numpy as np

LOG_SIG_MAX = 2
LOG_SIG_MIN = -20
EPSILON = 1e-6

# Value Function Network
class ValueFunction(nn.Module):
    def __init__(self, state_dim, hidden_dim=256):
        super(ValueFunction, self).__init__()
        
        self.fc1 = nn.Linear(state_dim, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, hidden_dim)
        self.v = nn.Linear(hidden_dim, 1)
        
    def forward(self, state):
        x = F.relu(self.fc1(state))
        x = F.relu(self.fc2(x))
        v = self.v(x)
        return v

# Actor Network (Policy)
class Actor(nn.Module):
    def __init__(self, state_dim, action_dim, max_action, hidden_dim=256, reparameterize=True):
        super(Actor, self).__init__()
        
        self.fc1 = nn.Linear(state_dim, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, hidden_dim)
        
        self.mean = nn.Linear(hidden_dim, action_dim)
        self.log_std = nn.Linear(hidden_dim, action_dim)
        
        self.max_action = max_action
        self.action_dim = action_dim
        self.reparameterize = reparameterize
        
    def forward(self, state):
        x = F.relu(self.fc1(state))
        x = F.relu(self.fc2(x))
        
        mean = self.mean(x)
        log_std = self.log_std(x)
        log_std = torch.clamp(log_std, min=LOG_SIG_MIN, max=LOG_SIG_MAX)
        std = torch.exp(log_std)
        
        return mean, std
    
    def sample(self, state):
        mean, std = self.forward(state)
        normal = Normal(mean, std)
        
        if self.reparameterize:
            x_t = normal.rsample()  # Reparameterization trick
        else:
            x_t = normal.sample()
            
        y_t = torch.tanh(x_t)
        action = y_t * self.max_action
        
        # Calculate log probability
        log_prob = normal.log_prob(x_t)
        
        # Apply squashing correction
        log_prob -= torch.log(self.max_action * (1 - y_t.pow(2)) + EPSILON)
        log_prob = log_prob.sum(1, keepdim=True)
        
        return action, log_prob
    
    def deterministic_action(self, state):
        mean, _ = self.forward(state)
        action = torch.tanh(mean) * self.max_action
        return action

# Critic Network (Q-function)
class Critic(nn.Module):
    def __init__(self, state_dim, action_dim, hidden_dim=256):
        super(Critic, self).__init__()
        
        # Q1 architecture
        self.fc1 = nn.Linear(state_dim + action_dim, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, hidden_dim)
        self.q1 = nn.Linear(hidden_dim, 1)
        
        # Q2 architecture (for min(Q1, Q2))
        self.fc3 = nn.Linear(state_dim + action_dim, hidden_dim)
        self.fc4 = nn.Linear(hidden_dim, hidden_dim)
        self.q2 = nn.Linear(hidden_dim, 1)
        
    def forward(self, state, action):
        sa = torch.cat([state, action], 1)
        
        # Q1
        q1 = F.relu(self.fc1(sa))
        q1 = F.relu(self.fc2(q1))
        q1 = self.q1(q1)
        
        # Q2
        q2 = F.relu(self.fc3(sa))
        q2 = F.relu(self.fc4(q2))
        q2 = self.q2(q2)
        
        return q1, q2
    
    def q1_forward(self, state, action):
        sa = torch.cat([state, action], 1)
        
        q1 = F.relu(self.fc1(sa))
        q1 = F.relu(self.fc2(q1))
        q1 = self.q1(q1)
        
        return q1

# Gaussian Mixture Model policy
class GMMPolicy(nn.Module):
    def __init__(self, state_dim, action_dim, max_action, hidden_dim=256, n_components=5, reparameterize=False):
        super(GMMPolicy, self).__init__()
        self.hidden_dim = hidden_dim
        self.n_components = n_components
        self.action_dim = action_dim
        self.max_action = max_action
        self.reparameterize = reparameterize
        
        self.fc1 = nn.Linear(state_dim, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, hidden_dim)
        
        # Output n_components means, n_components log_stds, and n_components mixture weights
        self.out_layer = nn.Linear(hidden_dim, action_dim * n_components * 2 + n_components)
        
    def forward(self, state):
        x = F.relu(self.fc1(state))
        x = F.relu(self.fc2(x))
        out = self.out_layer(x)
        
        # Split the output into means, log_stds, and weights
        batch_size = state.shape[0]
        out = out.view(batch_size, self.n_components, 2 * self.action_dim + 1)
        
        means = out[:, :, :self.action_dim]  # [batch_size, n_components, action_dim]
        log_stds = out[:, :, self.action_dim:2*self.action_dim]  # [batch_size, n_components, action_dim]
        log_stds = torch.clamp(log_stds, LOG_SIG_MIN, LOG_SIG_MAX)
        
        log_weights = out[:, :, -1]  # [batch_size, n_components]
        weights = F.softmax(log_weights, dim=1)  # [batch_size, n_components]
        
        return means, log_stds, weights
    
    def sample(self, state):
        means, log_stds, weights = self.forward(state)
        batch_size = state.shape[0]
        
        # Sample component indices based on weights
        component_indices = torch.multinomial(weights, 1).squeeze(1)  # [batch_size]
        
        # Gather the means and stds of the selected components
        batch_indices = torch.arange(batch_size).to(state.device)
        selected_means = means[batch_indices, component_indices]  # [batch_size, action_dim]
        selected_log_stds = log_stds[batch_indices, component_indices]  # [batch_size, action_dim]
        selected_stds = torch.exp(selected_log_stds)
        
        # Create normal distributions and sample
        normal = Normal(selected_means, selected_stds)
        
        if self.reparameterize:
            x_t = normal.rsample()
        else:
            x_t = normal.sample()
        
        y_t = torch.tanh(x_t)
        action = y_t * self.max_action
        
        # Log probability calculation for GMM
        log_probs = normal.log_prob(x_t)
        log_probs -= torch.log(self.max_action * (1 - y_t.pow(2)) + EPSILON)
        log_probs = log_probs.sum(1, keepdim=True)
        
        # Add the log probability of selecting the component
        log_component_prob = torch.log(weights[batch_indices, component_indices].unsqueeze(1) + EPSILON)
        log_probs += log_component_prob
        
        return action, log_probs
    
    def deterministic_action(self, state):
        means, _, weights = self.forward(state)
        
        # Select component with highest weight
        component_indices = torch.argmax(weights, dim=1)
        batch_size = state.shape[0]
        batch_indices = torch.arange(batch_size).to(state.device)
        
        # Get means of highest weighted component
        selected_means = means[batch_indices, component_indices]
        
        # Apply tanh and scaling
        action = torch.tanh(selected_means) * self.max_action
        return action