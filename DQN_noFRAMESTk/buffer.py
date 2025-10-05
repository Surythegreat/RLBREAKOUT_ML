import torch
import random
from collections import deque, namedtuple

# Use a namedtuple to make transitions more readable
Transition = namedtuple('Transition',
                        ('state', 'action', 'next_state', 'reward', 'done'))

class ReplayMemory:
    def __init__(self, capacity, device='cpu'):
        """
        Initializes the Replay Memory.
        Args:
            capacity (int): The maximum number of transitions to store.
        """
        self.memory = deque([], maxlen=capacity)
        self.device = device

    def push(self, *args):
        """
        Saves a transition. The arguments should be in the order defined
        by the Transition namedtuple.
        """
        self.memory.append(Transition(*args))

    def sample(self, batch_size):
        """
        Samples a random batch of transitions from memory.
        Args:
            batch_size (int): The number of transitions to sample.
            device (str): The device ('cpu' or 'cuda') to move the tensors to.

        Returns:
            A namedtuple of batched tensors.
        """
        # 1. Sample transitions from the deque
        transitions = random.sample(self.memory, batch_size)

        # 2. Transpose the batch (see https://stackoverflow.com/a/19343/3343043 for details)
        # This converts a list of Transition tuples into a single Transition tuple
        # where each field contains a list of all the corresponding values.
        # For example, batch.state will contain a tuple of all the states.
        batch = Transition(*zip(*transitions))

        # 3. Convert tuples of data into batched tensors and move to the target device
        # torch.cat stacks the tensors along a new dimension.
        states = torch.cat(batch.state).to(self.device)
        actions = torch.cat(batch.action).to(self.device)
        rewards = torch.cat(batch.reward).to(self.device)
        # For next_states, handle the case where a state is None (terminal state)
        non_final_next_states_list = [s for s in batch.next_state if s is not None]
        if non_final_next_states_list:
            non_final_next_states = torch.cat(non_final_next_states_list).to(self.device)
        else:
            non_final_next_states = torch.empty(0, *batch.state[0].shape, device=self.device) #Create an empty tensor with the correct shape

        dones = torch.cat(batch.done).to(self.device)

        return states, actions, rewards, non_final_next_states, dones


    def __len__(self):
        """Returns the current size of the memory."""
        return len(self.memory)