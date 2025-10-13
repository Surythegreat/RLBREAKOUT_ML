from stable_baselines3.common.callbacks import BaseCallback
import matplotlib.pyplot as plt

class EpisodeRewardCallback(BaseCallback):
    """
    A custom callback that logs the total reward of each episode.
    """
    def __init__(self, verbose=0):
        super(EpisodeRewardCallback, self).__init__(verbose)
        # These variables will be initialized in _on_training_start
        self.episode_rewards = []
        self.episode_count = 0

    def _on_step(self) -> bool:
        # Check for finished episodes in all environments
        for i, done in enumerate(self.dones):
            if done:
                # The 'info' dictionary contains the final reward for the episode
                # This key is added by the Monitor wrapper, which make_atari_env uses automatically
                if 'episode' in self.locals['infos'][i]:
                    episode_reward = self.locals['infos'][i]['episode']['r']
                    self.episode_count += 1
                    
                    # Log the reward to TensorBoard
                    self.logger.record('train/episode_reward', episode_reward)
                    self.logger.record('train/episode_number', self.episode_count)
                    
                    # Dump all the logs to the writer
                    self.logger.dump(self.num_timesteps)
                    
        return True # Continue training
import os
import ale_py
import gymnasium as gym

# Import the callback class
from stable_baselines3.common.callbacks import BaseCallback

gym.register_envs(ale_py)
from stable_baselines3 import DQN
from stable_baselines3.common.env_util import make_atari_env
from stable_baselines3.common.vec_env import VecFrameStack, VecVideoRecorder

# --- 1. DEFINE THE CUSTOM CALLBACK CLASS ---
class EpisodeRewardCallback(BaseCallback):
    def __init__(self, verbose=0):
        super(EpisodeRewardCallback, self).__init__(verbose)
        self.episode_count = 0
        self.episode_rewards = [] # <-- Add a list to store rewards

    def _on_step(self) -> bool:
        for i, done in enumerate(self.locals['dones']):
            if done:
                if 'episode' in self.locals['infos'][i]:
                    episode_reward = self.locals['infos'][i]['episode']['r']
                    self.episode_rewards.append(episode_reward) # <-- Save the reward
                    self.episode_count += 1
                    self.logger.record('train/episode_reward', episode_reward)
                    self.logger.record('train/episode_number', self.episode_count)
                    self.logger.dump(self.num_timesteps)
        return True

# --- Main Training & Evaluation Logic ---
if __name__ == '__main__':
    
    # --- Hyperparameters ---
    config = {
        'policy': 'CnnPolicy',
        'learning_rate': 0.0001,
        'buffer_size': 100000,
        'learning_starts': 10000,
        'batch_size': 64,
        'tau': 1.0,
        'gamma': 0.99,
        'train_freq': 4,
        'gradient_steps': 1,
        'target_update_interval': 1000,
        'exploration_fraction': 0.1,
        'exploration_initial_eps': 1.0,
        'exploration_final_eps': 0.1,
        'verbose': 1,
        'tensorboard_log': "./dqn_breakout_tensorboard/"
    }

    # --- Environment Setup ---
    vec_env = make_atari_env("ALE/Breakout-v5", n_envs=1, seed=0)
    vec_env = VecFrameStack(vec_env, n_stack=4)

    # --- Agent Initialization ---
    model = DQN(env=vec_env, **config)
    
    # --- 2. CREATE AN INSTANCE OF THE CALLBACK ---
    reward_callback = EpisodeRewardCallback()

    # --- Start Training ---
    print("--- 🚀 Starting Training ---")
    # --- 3. PASS THE CALLBACK TO THE LEARN METHOD ---
    model.learn(total_timesteps=1000000, callback=reward_callback)
    
    # --- Save the Model ---
    model_path = "dqn_breakout_model.zip"
    model.save(model_path)
    print(f"--- ✅ Training Finished. Model saved to {model_path} ---")
    
    vec_env.close()

    # --- ADD THIS PLOTTING CODE AT THE END ---
    print("--- 📊 Generating Final Reward Plot ---")
    episodes = range(1, len(reward_callback.episode_rewards) + 1)
    
    plt.figure(figsize=(12, 6))
    plt.plot(episodes, reward_callback.episode_rewards)
    plt.title('Total Reward per Episode During Training')
    plt.xlabel('Episode')
    plt.ylabel('Total Reward')
    plt.grid(True)
    
    # Save the plot to a file
    plt.savefig("training_rewards_plot.png")
    plt.show() # Display the plot