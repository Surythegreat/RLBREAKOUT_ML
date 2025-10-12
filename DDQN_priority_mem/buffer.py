# In buffer.py

import torch
import random
# Make sure to import deque
from collections import deque 

class ReplayMemory:
    def __init__(self, capacity=10000, device='cpu', alpha=0.6, beta=0.4, beta_increment_per_sampling=0.001):
        self.tree = SumTree(capacity)
        self.capacity = capacity
        self.device = device
        
        # PER Hyperparameters
        self.alpha = alpha  # [0~1]: how much prioritization is used
        self.beta = beta    # [0~1]: importance-sampling, annealed to 1
        self.beta_increment_per_sampling = beta_increment_per_sampling
        self.epsilon = 1e-5 # small value to avoid 0 priority
        self.max_priority = 1.0

    def add(self, transition):
        """
        Adds a new transition to the memory.
        New transitions are given max priority to ensure they are sampled.
        """
        # Move tensors to CPU for storage
        transition_cpu = [item.to('cpu') for item in transition]
        
        self.tree.add(self.max_priority, transition_cpu)

    def sample(self, batch_size):
        """
        Samples a batch of experiences, and returns them with their indices and IS weights.
        """
        assert self.can_sample(batch_size)
        
        batch = []
        idxs = []
        segment = self.tree.total() / batch_size
        priorities = []

        # Anneal beta
        self.beta = np.min([1., self.beta + self.beta_increment_per_sampling])

        for i in range(batch_size):
            a = segment * i
            b = segment * (i + 1)
            s = random.uniform(a, b)
            (idx, p, data) = self.tree.get(s)
            
            if data is not 0: # Ensure data is not the initial np.zeros
                priorities.append(p)
                batch.append(data)
                idxs.append(idx)

        sampling_probabilities = np.array(priorities) / self.tree.total()
        
        # Calculate Importance-Sampling (IS) weights
        is_weights = np.power(self.tree.n_entries * sampling_probabilities, -self.beta)
        is_weights /= is_weights.max()
        is_weights = torch.tensor(is_weights, device=self.device, dtype=torch.float32).unsqueeze(1)

        # Unzip the batch of transitions
        states, actions, next_states, rewards, dones = zip(*batch)

        # Efficiently stack and move to the target device
        states = torch.cat(states).to(self.device)
        actions = torch.cat(actions).to(self.device)
        next_states = torch.cat(next_states).to(self.device)
        rewards = torch.cat(rewards).to(self.device)
        dones = torch.cat(dones).to(self.device)

        return states, actions, next_states, rewards, dones, idxs, is_weights

    def update_priorities(self, tree_idxs, abs_td_errors):
        """
        Update the priorities of the sampled experiences.
        """
        # Add epsilon to avoid 0 priorities
        priorities = np.power(abs_td_errors + self.epsilon, self.alpha)
        
        for idx, p in zip(tree_idxs, priorities):
            self.tree.update(idx, p)
        
        # Update max priority for new experiences
        self.max_priority = max(self.max_priority, np.max(priorities))

    def can_sample(self, batch_size):
        # Wait until you have enough samples
        return self.tree.n_entries >= batch_size

    def __len__(self):
        return self.tree.n_entries
    
# In buffer.py (add this helper class)

import numpy as np

class SumTree:
    def __init__(self, capacity):
        self.capacity = capacity
        # The tree is stored in a simple array.
        # The total size is 2 * capacity - 1
        self.tree = np.zeros(2 * capacity - 1)
        # The data (transitions) are stored in a separate array
        self.data = np.zeros(capacity, dtype=object)
        self.n_entries = 0
        self.write_ptr = 0 # Write pointer

    # When a priority is updated, propagate the change up the tree
    def _propagate(self, idx, change):
        parent = (idx - 1) // 2
        self.tree[parent] += change
        if parent != 0:
            self._propagate(parent, change)

    # Find sample on leaf node
    def _retrieve(self, idx, s):
        left_child = 2 * idx + 1
        right_child = left_child + 1

        if left_child >= len(self.tree):
            return idx

        if s <= self.tree[left_child]:
            return self._retrieve(left_child, s)
        else:
            return self._retrieve(right_child, s - self.tree[left_child])

    def total(self):
        return self.tree[0]

    # Store priority and sample
    def add(self, p, data):
        # The index in the tree where we'll store the priority
        tree_idx = self.write_ptr + self.capacity - 1

        self.data[self.write_ptr] = data
        self.update(tree_idx, p)

        self.write_ptr += 1
        if self.write_ptr >= self.capacity:
            self.write_ptr = 0

        if self.n_entries < self.capacity:
            self.n_entries += 1

    # Update priority
    def update(self, idx, p):
        change = p - self.tree[idx]
        self.tree[idx] = p
        self._propagate(idx, change)

    # Get priority and sample
    def get(self, s):
        idx = self._retrieve(0, s)
        data_idx = idx - self.capacity + 1
        return (idx, self.tree[idx], self.data[data_idx])