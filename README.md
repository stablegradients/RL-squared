# Enhanced Soft Actor-Critic (SAC) Implementation

This repository contains an implementation of the Soft Actor-Critic (SAC) algorithm that is consistent with the original Haarnoja et al. paper. It includes all the key components from the original implementation:

## Features

- Value function network (V) along with target networks
- Dual critic networks (Q1, Q2) for reducing overestimation bias
- Support for multiple policy types:
  - Gaussian policy
  - Gaussian Mixture Model (GMM) policy
- Optional reparameterization trick
- Automatic entropy tuning with a target entropy
- Advanced replay buffer with proper terminal state handling
- WandB integration for experiment tracking

## Key Differences from RL-Squared Implementation

The original RL-Squared repository had several differences from the Haarnoja SAC implementation:

1. **Value Function**: Added separate V-function networks (missing in RL-Squared)
2. **Target Computation**: Now uses proper value function targets from the original paper
3. **Policy Types**: Added support for GMM policies
4. **Reparameterization**: Made reparameterization optional via a flag
5. **Replay Buffer**: Enhanced for better handling of terminal states
6. **Update Logic**: Properly implemented the original paper's update rules

## Files

- `models.py`: Network architectures (Actor, Critic, ValueFunction, GMMPolicy)
- `utils.py`: Replay buffers and helper functions
- `SAC_models.py`: Core SAC implementation
- `main.py`: Training script with command-line arguments
- `visualize.py`: Policy visualization script

## Usage

### Training

```bash
python main.py --env_name HalfCheetah-v5 --policy_type gaussian --reparameterize
```

### Visualization

```bash
python visualize.py --model checkpoints/model.pt --env HalfCheetah-v5 --policy_type gaussian --reparameterize
```

## Command-Line Arguments

### Main Training Arguments:

- `--env_name`: Gymnasium environment name
- `--policy_type`: Policy architecture (`gaussian` or `gmm`)
- `--reparameterize`: Use reparameterization trick (flag)
- `--hidden_dim`: Hidden layer dimensions
- `--lr`: Learning rate
- `--gamma`: Discount factor
- `--tau`: Soft update rate
- `--alpha`: Temperature parameter
- `--auto_tune_alpha`: Automatically tune alpha
- `--max_steps`: Total environment steps for training
- `--wandb_project`: WandB project name
- `--no_wandb`: Disable WandB logging

## References

1. Haarnoja, T., Zhou, A., Abbeel, P., & Levine, S. (2018). Soft actor-critic: Off-policy maximum entropy deep reinforcement learning with a stochastic actor. arXiv preprint arXiv:1801.01290.
2. Haarnoja, T., Zhou, A., Hartikainen, K., Tucker, G., Ha, S., Tan, J., ... & Levine, S. (2018). Soft actor-critic algorithms and applications. arXiv preprint arXiv:1812.05905.