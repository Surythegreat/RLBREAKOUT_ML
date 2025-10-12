# In buffer.py

import torch
import random
# Make sure to import deque
from collections import deque 

class ReplayMemory:
    def __init__(self, capacity=10000, device='cpu'):
        # Use a deque with a maximum length for automatic, efficient removal of old items
        self.memory = deque(maxlen=capacity)
        self.device = device

    def insert(self, transition):
        # Move tensors to CPU for storage in RAM, not VRAM
        transition_cpu = [item.to('cpu') for item in transition]
        # Appending to a deque with maxlen will automatically discard the oldest item
        self.memory.append(transition_cpu)

    def sample(self, batch_size):
        assert self.can_sample(batch_size)

        batch = random.sample(self.memory, batch_size)
        # Unzip the batch of transitions
        states, actions, next_states, rewards, dones = zip(*batch)

        # Efficiently stack and move to the target device
        states = torch.cat(states).to(self.device)
        actions = torch.cat(actions).to(self.device)
        next_states = torch.cat(next_states).to(self.device)
        rewards = torch.cat(rewards).to(self.device)
        dones = torch.cat(dones).to(self.device)

        return states, actions, next_states, rewards, dones

    def can_sample(self, batch_size):
        # Wait until you have enough samples to avoid overfitting to initial experiences
        return len(self.memory) >= batch_size * 10

    def __len__(self):
        return len(self.memory)