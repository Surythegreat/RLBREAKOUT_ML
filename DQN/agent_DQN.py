# agent_DQN.py

import math
import torch
import torch.nn as nn
import torch.optim as optim
import random
import numpy as np
import csv
import cv2
from collections import deque

from models import Model
from buffer import ReplayMemory
from utils import preprocess_frame

class Agent:
    """
    DQN Agent that interacts with and learns from the environment.
    """
    def __init__(self, env, replay_memory_size, batch_size, target_update,
                 gamma, lr, epsilon_start, epsilon_end, epsilon_decay,device='cpu',frame_stack_size=4):
        """
        Initializes the Agent.

        Args:
            env: The gym environment.
            replay_memory_size (int): The size of the replay memory buffer.
            batch_size (int): The size of the batch to sample from memory for training.
            target_update (int): The frequency (in epochs) to update the target network.
            gamma (float): The discount factor.
            lr (float): The learning rate for the optimizer.
            epsilon_start (float): The starting value of epsilon.
            epsilon_end (float): The final value of epsilon.
            epsilon_decay (int): The rate of epsilon decay.
        """
        self.env = env
        self.action_space_n = env.action_space.n
        self.memory = ReplayMemory(replay_memory_size,device=device)
        self.device = device
        
        # --- Hyperparameters ---
        self.BATCH_SIZE = batch_size
        self.TARGET_UPDATE = target_update
        self.GAMMA = gamma
        self.EPSILON_START = epsilon_start
        self.EPSILON_END = epsilon_end
        self.EPSILON_DECAY = epsilon_decay
        self.frame_stack_size = frame_stack_size
        
        # --- Internal State ---
        self.steps_done = 0
        self.epsilon=epsilon_start
        # --- Networks ---
        self.policy_net = Model(self.action_space_n, frame_stack_size).to(device)
        self.target_net = Model(self.action_space_n, frame_stack_size).to(device)

        # Load pre-existing model if available, otherwise initialize target_net from policy_net
        try:
            self.policy_net.load_the_model()
            print("Successfully loaded pre-trained policy network.")
        except FileNotFoundError:
            print("No pre-trained model found. Starting from scratch.")
        
        self.target_net.load_state_dict(self.policy_net.state_dict())
        self.target_net.eval()  # Target network is only for inference
        
        # --- Optimizer and Loss ---
        self.optimizer = optim.Adam(self.policy_net.parameters(), lr=lr)
        self.loss_fn = nn.SmoothL1Loss()

    def _get_action(self, state_tensor):
        """
        Selects an action using an epsilon-greedy policy.

        Args:
            state_tensor (torch.Tensor): The current state of the environment.

        Returns:
            int: The chosen action.
        """
        sample = random.random()
        # Calculate current epsilon
        self.epsilon = self.EPSILON_END + (self.EPSILON_START - self.EPSILON_END) * \
                       math.exp(-1. * self.steps_done / self.EPSILON_DECAY)
        self.steps_done += 1

        if sample > self.epsilon:
            with torch.no_grad():
                # Get Q-values from the policy network and choose the best action
                q_values = self.policy_net(state_tensor)
                return torch.argmax(q_values).item()
        else:
            # Choose a random action
            return self.env.action_space.sample()

    def _optimize_model(self):
        """
        Performs one step of optimization on the policy network.
        
        Returns:
            float or None: The loss value if training was performed, otherwise None.
        """
        if len(self.memory) < self.BATCH_SIZE:
            return None  # Not enough experiences to train yet

        state_batch, action_batch, reward_batch, next_state_batch, done_batch = self.memory.sample(self.BATCH_SIZE)


        # Get the Q-values for the actions that were actually taken
        predicted_q_values = self.policy_net(state_batch).gather(1, action_batch)

        # Get the max Q-values for the next states from the target network
        with torch.no_grad():
            next_state_q_values = self.target_net(next_state_batch).max(1)[0]
        
        # Compute the expected Q values (target)
        # If the state was terminal (done), the future reward is 0
        target_q_values = (next_state_q_values * self.GAMMA * (1 - done_batch)) + reward_batch

        # Compute loss
        loss = self.loss_fn(predicted_q_values, target_q_values.unsqueeze(1))

        # Backpropagation
        self.optimizer.zero_grad()
        loss.backward()
        # Gradient clipping can be added here if needed: torch.nn.utils.clip_grad_value_(self.policy_net.parameters(), 100)
        self.optimizer.step()
        return loss.item()

    def train(self, num_epochs):
        """
        The main training loop.

        Args:
            num_epochs (int): The total number of episodes to train for.
        """
        log_file = 'training_log.csv'
        with open(log_file, 'w', newline='') as f:
            writer = csv.writer(f)
            writer.writerow(['Epoch', 'Total Reward', 'Epsilon', 'Loss'])

        for epoch in range(num_epochs):
            observation, info = self.env.reset() 
            # Initialize the frame stack
            frame = preprocess_frame(observation)
            # Stack the first frame 4 times
            state_stack = deque([frame] * self.frame_stack_size, maxlen=self.frame_stack_size)
            
            # Convert the stack to a single tensor
            state_tensor = torch.FloatTensor(np.array(state_stack)).unsqueeze(0).to(self.device)
            
            total_epoch_reward = 0
            done = False
            last_loss = 0  # To store the last loss of the epoch
            lives = info['lives']
            while not done:
                action = self._get_action(state_tensor)
                
                observation, reward, terminated, truncated, info = self.env.step(action)
                if info['lives'] < lives:
                    lives = info['lives']
                    reward -= 1  # Penalty for losing a life
                total_epoch_reward += reward
                done = terminated or truncated

                next_frame = preprocess_frame(observation)
                next_state_stack = deque(list(state_stack)[1:] + [next_frame], maxlen=self.frame_stack_size)
                next_state_tensor = torch.FloatTensor(np.array(next_state_stack)).unsqueeze(0).to(self.device)
        
                # Store the transition in memory
                action_tensor = torch.tensor([[action]], dtype=torch.long).to(self.device)
                reward_tensor = torch.tensor([reward], dtype=torch.float32).to(self.device)
                done_tensor = torch.tensor([done], dtype=torch.float32).to(self.device)
                self.memory.push(state_tensor, action_tensor, next_state_tensor, reward_tensor, done_tensor)

                # Move to the next state
                state_stack = next_state_stack
                state_tensor = next_state_tensor

                # Perform one step of optimization
                loss = self._optimize_model()
                if self.steps_done % self.TARGET_UPDATE == 0:
                    print(f"\n--- 🎯 Updating Target Network at step {self.steps_done} --- \n")
                    self.policy_net.save_the_model()
                    self.target_net.load_state_dict(self.policy_net.state_dict())
                if loss is not None:
                    last_loss = loss

                if done:
                    break

            print(f"Epoch: {epoch}, Reward: {total_epoch_reward}, Epsilon: {self.epsilon:.4f}, Loss: {last_loss:.4f}, Memory Size: {len(self.memory)}")

            with open(log_file, 'a', newline='') as f:
                writer = csv.writer(f)
                writer.writerow([epoch, total_epoch_reward, self.epsilon, last_loss])

            # Update the target network
            
                
        print("Training finished.")

    def test(self, num_episodes):
        """
        Tests the trained agent's performance by rendering the gameplay.

        Args:
            num_episodes (int): The number of episodes to run for testing.
        """
        print("\n--- 🚀 Starting Test Phase ---")
        self.policy_net.eval()  # Set the network to evaluation mode (no gradients)

        for episode in range(num_episodes):
            observation, info = self.env.reset() 
            # Initialize the frame stack
            frame = preprocess_frame(observation)
            # Stack the first frame 4 times
            state_stack = deque([frame] * self.frame_stack_size, maxlen=self.frame_stack_size)
            
            # Convert the stack to a single tensor
            state_tensor = torch.FloatTensor(np.array(state_stack)).unsqueeze(0).to(self.device)
            
            total_reward = 0
            done = False
            
            while not done:
                # Render the environment and display it
                frame = self.env.render()
                # Gymnasium returns RGB, OpenCV expects BGR, so we convert
                frame_bgr = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)
                cv2.imshow('Breakout - Agent Gameplay', frame_bgr)
                
                # Use a purely greedy policy (no exploration)
                with torch.no_grad():
                    action = self.policy_net(state_tensor).argmax().item()

                observation, reward, terminated, truncated, info = self.env.step(action)
                
                total_reward += reward

                next_frame = preprocess_frame(observation)
                next_state_stack = deque(list(state_stack)[1:] + [next_frame], maxlen=self.frame_stack_size)
                next_state_tensor = torch.FloatTensor(np.array(next_state_stack)).unsqueeze(0).to(self.device)
        
                print("moved")
                # Move to the next state
                state_stack = next_state_stack
                state_tensor = next_state_tensor
                

                # Control frame rate and allow user to quit with 'q'
                if cv2.waitKey(25) & 0xFF == ord('q'):
                    print("Testing interrupted by user.")
                    done = True # Exit the inner loop
                    num_episodes = episode # Prevent further episodes
            
            print(f"Episode {episode + 1}: Total Reward = {total_reward}")

        self.env.close()
        cv2.destroyAllWindows()
        print("--- ✅ Test Phase Finished ---")