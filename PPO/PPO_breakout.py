# main.py
import ale_py

import gymnasium as gym
gym.register_envs(ale_py)

import torch
from PPO_Agent import PPO_agent
from wrappers import make_atari_env


# Create the wrapped environment
env = make_atari_env("ALE/Breakout-v5")
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

# Hyperparameters (tuned for Atari)
hidden_size = 512
lr = 2.5e-4
num_steps = 128          # Steps per rollout (more frequent updates)
mini_batch_size = 32
ppo_epochs = 4
frame_stacking = 4       # This is now handled by the wrapper

# Create the PPO agent
agent = PPO_agent(
    hidden_size=hidden_size,
    lr=lr,
    num_steps=num_steps,
    mini_batch_size=mini_batch_size,
    ppo_epochs=ppo_epochs,
    device=device,
    env=env,
)

# Start the training process
agent.train()
print("Training finished.")

# Close the environment when done
env.close()