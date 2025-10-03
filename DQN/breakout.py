# train_breakout.py
import ale_py
import gymnasium as gym
gym.register_envs(ale_py)
from agent_DQN import Agent
import torch

# --- Main Training Logic ---
if __name__ == '__main__':
    # --- Environment ---
    env = gym.make("ALE/Breakout-v5", render_mode="rgb_array")
    
    # --- Hyperparameters ---
    config = {
        'target_update': 10000,
        'replay_memory_size': 60000,
        'batch_size': 32,
        'learning_rate': 0.00025,
        'epochs': 1000000,
        'gamma': 0.99,
        'epsilon_start': 1.0,
        'epsilon_end': 0.005,
        'epsilon_decay': 1000000,
        'frame_stack_size': 4
    }
    
    # --- Agent Initialization ---
    agent = Agent(
        env=env,
        replay_memory_size=config['replay_memory_size'],
        batch_size=config['batch_size'],
        target_update=config['target_update'],
        gamma=config['gamma'],
        lr=config['learning_rate'],
        epsilon_start=config['epsilon_start'],
        epsilon_end=config['epsilon_end'],
        epsilon_decay=config['epsilon_decay'],
        device='cuda' if torch.cuda.is_available() else 'cpu',
        frame_stack_size=config['frame_stack_size']
    )
    print(f"Using device: {agent.memory.device}")
    
    # --- Start Training ---
    agent.test(num_episodes=3)
    
    env.close()