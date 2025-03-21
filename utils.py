import numpy as np
import random
import torch
from collections import deque

class ReplayBuffer:
    def __init__(self, capacity, env_spec=None):
        self.buffer = deque(maxlen=capacity)
        self.max_buffer_size = capacity
        self._size = 0
        self._bottom = 0
        self._top = 0
        
        if env_spec is not None:
            self._observation_dim = env_spec.observation_space.shape[0]
            self._action_dim = env_spec.action_space.shape[0]
            self._terminals = np.zeros(capacity, dtype=np.uint8)
            self._final_state = np.zeros(capacity, dtype=np.uint8)
        
    def push(self, state, action, reward, next_state, done, final_state=False):
        """
        Add a transition tuple to the replay buffer.
        
        Args:
            state: Current state
            action: Action taken
            reward: Reward received
            next_state: Next state
            done: Whether the episode is done
            final_state: Whether this is the final state in an episode (for episode termination handling)
        """
        experience = (state, action, reward, next_state, done)
        self.buffer.append(experience)
        
        self._top = (self._top + 1) % self.max_buffer_size
        if self._size >= self.max_buffer_size:
            self._bottom = (self._bottom + 1) % self.max_buffer_size
        else:
            self._size += 1
        
        if hasattr(self, '_final_state'):
            self._terminals[self._top - 1] = done
            self._final_state[self._top - 1] = final_state
    
    def add_path(self, observations, actions, rewards, next_observations, terminals, last_obs=None):
        """
        Add a full trajectory to the buffer.
        
        Args:
            observations: List of observations
            actions: List of actions
            rewards: List of rewards
            next_observations: List of next observations
            terminals: List of terminal indicators
            last_obs: Final observation (optional)
        """
        for t in range(len(observations)):
            self.push(
                observations[t],
                actions[t],
                rewards[t],
                next_observations[t],
                terminals[t],
                False
            )
            
        if last_obs is not None:
            # Add the final state marker
            self.push(
                last_obs,
                np.zeros_like(actions[0]),
                0.0,
                last_obs,  # Next state is the same as current for final state
                False,
                True
            )
    
    def sample(self, batch_size):
        """
        Sample a batch of transitions, avoiding invalid transitions (e.g., final states).
        
        Args:
            batch_size: Size of the batch to sample
            
        Returns:
            Tuple of (states, actions, rewards, next_states, dones)
        """
        if hasattr(self, '_final_state'):
            # Advanced sampling that avoids final states
            indices = np.zeros(batch_size, dtype=np.int64)
            count = 0
            
            while count < batch_size:
                index = np.random.randint(0, min(self._size, self.max_buffer_size))
                
                # Make sure the transition is valid: not at the end of the buffer and not a final state
                if (index + 1) % self.max_buffer_size == self._top:
                    continue
                    
                # Discard if it's a final state
                if self._final_state[index]:
                    continue
                    
                indices[count] = index
                count += 1
                
            batch = [self.buffer[i] for i in indices]
        else:
            # Simple random sampling
            batch = random.sample(self.buffer, batch_size)
            
        state, action, reward, next_state, done = map(np.stack, zip(*batch))
        return state, action, reward, next_state, done
    
    def __len__(self):
        return self._size

class EpisodeBuffer:
    """Buffer to store full episodes for algorithms that require episodic data"""
    def __init__(self, capacity):
        self.buffer = deque(maxlen=capacity)
        
    def push(self, episode):
        """
        Add an episode to the buffer
        
        Args:
            episode: Dict with keys 'states', 'actions', 'rewards', 'next_states', 'dones'
        """
        self.buffer.append(episode)
        
    def sample(self, batch_size):
        """
        Sample batch_size episodes from the buffer
        
        Returns:
            List of episode dicts
        """
        return random.sample(self.buffer, min(batch_size, len(self.buffer)))
    
    def __len__(self):
        return len(self.buffer)

def create_log_gaussian(mean, log_std, t):
    """
    Create log of Gaussian probability density
    
    Args:
        mean: Mean of the Gaussian
        log_std: Log standard deviation
        t: Value to evaluate density at
    
    Returns:
        Log probability density
    """
    quadratic = -((0.5 * (t - mean) / (log_std.exp())).pow(2))
    l = mean.shape
    log_z = log_std
    z = l[-1]
    log_z += 0.5 * np.log(2 * np.pi) * z
    log_p = quadratic.sum(dim=-1) - log_z.sum(dim=-1)
    
    return log_p

def soft_update(target, source, tau):
    """
    Soft update of target network parameters
    
    Args:
        target: Target network
        source: Source network
        tau: Interpolation parameter (0 < tau < 1)
    """
    for target_param, param in zip(target.parameters(), source.parameters()):
        target_param.data.copy_(target_param.data * (1.0 - tau) + param.data * tau)

def hard_update(target, source):
    """
    Hard update of target network parameters (tau=1)
    
    Args:
        target: Target network
        source: Source network
    """
    for target_param, param in zip(target.parameters(), source.parameters()):
        target_param.data.copy_(param.data)