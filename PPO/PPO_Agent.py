# PPO_Agent.py
import torch
import numpy as np
import csv
from PPO_model import ActorCritic
from PPO_utilities import compute_gae

class PPO_agent:
    def __init__(self, hidden_size, lr, num_steps, mini_batch_size, ppo_epochs, device, env):
        # HYPERPARAMETERS
        self.hidden_size      = hidden_size
        self.lr               = lr
        self.num_steps        = num_steps
        self.mini_batch_size  = mini_batch_size
        self.ppo_epochs       = ppo_epochs
        self.device           = device
        self.env              = env
        
        # The wrapped env's observation space gives the correct shape
        num_inputs = self.env.observation_space.shape[-1] 
        num_outputs = self.env.action_space.n

        self.model = ActorCritic(num_inputs, num_outputs, hidden_size).to(self.device)
        self.model.load_the_model()
        self.optimizer = torch.optim.Adam(self.model.parameters(), lr=lr, eps=1e-5)

    def ppo_iter(self, states, actions, log_probs, returns, advantage):
        batch_size = states.size(0)
        for _ in range(batch_size // self.mini_batch_size):
            rand_ids = np.random.randint(0, batch_size, self.mini_batch_size)
            yield states[rand_ids, :], actions[rand_ids], log_probs[rand_ids], returns[rand_ids, :], advantage[rand_ids, :]

    def ppo_update(self, states, actions, log_probs, returns, advantages, clip_param=0.1):
        for _ in range(self.ppo_epochs):
            for state, action, old_log_probs, return_, advantage in self.ppo_iter(states, actions, log_probs, returns, advantages):
                dist, value = self.model(state)
                entropy = dist.entropy().mean()
                new_log_probs = dist.log_prob(action)

                ratio = (new_log_probs - old_log_probs).exp()
                surr1 = ratio * advantage
                surr2 = torch.clamp(ratio, 1.0 - clip_param, 1.0 + clip_param) * advantage

                actor_loss  = -torch.min(surr1, surr2).mean()
                critic_loss = (return_ - value).pow(2).mean()
                loss = 0.5 * critic_loss + actor_loss - 0.01 * entropy

                self.optimizer.zero_grad()
                loss.backward()
                # Gradient clipping
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), 0.5)
                self.optimizer.step()

    def train(self):
        log_file = 'training_log.csv'
        with open(log_file, 'w', newline='') as f:
            writer = csv.writer(f)
            writer.writerow(['episodes', 'Average Reward'])
        
        frame_idx = 0
        all_episode_rewards = []

        # The wrappers handle the reset correctly
        state, info = self.env.reset()
        print(info)
        lives= info['lives'] if 'lives' in info else 5
        ep_reward = 0
        episodes= 0
        while frame_idx < 10_000_000: # Train for a fixed number of frames
            log_probs, values, states, actions, rewards, masks = [], [], [], [], [], []
            
            # --- Data Collection Phase ---
            for _ in range(self.num_steps):
                state = torch.FloatTensor(state).unsqueeze(0).to(self.device)
                dist, value = self.model(state)
                action = dist.sample()

                # The wrappers handle everything
                next_state, reward, terminated, truncated, info = self.env.step(action.item())
               
                if(info['lives'] < lives):
                    lives = info['lives']
                    reward -=1
                ep_reward += reward
                # Store transition
                log_probs.append(dist.log_prob(action))
                values.append(value)
                rewards.append(torch.FloatTensor([reward]).unsqueeze(1).to(self.device))
                masks.append(torch.FloatTensor([1 - terminated]).unsqueeze(1).to(self.device))
                states.append(state)
                actions.append(action)
                
                state = next_state
                frame_idx += 1
                # If an episode is actually over (not just a life lost)
                if terminated or truncated:
                    episodes += 1
                    all_episode_rewards.append(ep_reward)
                    state, _ = self.env.reset()
                    if len(all_episode_rewards) % 10 == 0:
                        avg_reward = np.mean(all_episode_rewards[-100:])
                        print(f"Frames: {frame_idx}, Avg Reward (last 100): {avg_reward:.2f},episodes: {episodes}")
                        with open(log_file, 'a', newline='') as f:
                            writer = csv.writer(f)
                            writer.writerow([episodes, avg_reward])
                        self.model.save_the_model()
                    ep_reward=0
                    state, info = self.env.reset()
                    lives= info['lives'] if 'lives' in info else 5
            
            # --- Learning Phase ---
            next_state = torch.FloatTensor(next_state).unsqueeze(0).to(self.device)
            _, next_value = self.model(next_state)
            returns = compute_gae(next_value, rewards, masks, values)

            returns = torch.cat(returns).detach()
            log_probs = torch.cat(log_probs).detach()
            values = torch.cat(values).detach()
            states = torch.cat(states)
            actions = torch.cat(actions)
            advantage = returns - values
            advantage = (advantage - advantage.mean()) / (advantage.std() + 1e-8)
            
            self.ppo_update(states, actions, log_probs, returns, advantage)