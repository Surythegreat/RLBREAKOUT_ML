# PPO_model.py
import torch
import torch.nn as nn
import os

def init_weights(m):
    if isinstance(m, nn.Linear):
        nn.init.orthogonal_(m.weight.data, 1.0)
        nn.init.constant_(m.bias.data, 0.0)
    elif isinstance(m, nn.Conv2d):
        nn.init.orthogonal_(m.weight.data, nn.init.calculate_gain('relu'))
        nn.init.constant_(m.bias.data, 0.0)

class ActorCritic(nn.Module):
    def __init__(self, num_inputs, num_outputs, hidden_size):
        super(ActorCritic, self).__init__()
        
        # The CNN architecture from the Nature DQN paper
        self.conv = nn.Sequential(
            nn.Conv2d(num_inputs, 32, kernel_size=8, stride=4),
            nn.ReLU(),
            nn.Conv2d(32, 64, kernel_size=4, stride=2),
            nn.ReLU(),
            nn.Conv2d(64, 64, kernel_size=3, stride=1),
            nn.ReLU(),
            nn.Flatten()
        )
        
        # Critic head: estimates the state value
        self.critic = nn.Sequential(
            nn.Linear(64 * 7 * 7, hidden_size), # 3136 = 64 * 7 * 7
            nn.ReLU(),
            nn.Linear(hidden_size, 1)
        )
        
        # Actor head: outputs action logits for the policy
        self.actor = nn.Sequential(
            nn.Linear(64 * 7 * 7, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, num_outputs),
        )
        
        # Apply orthogonal initialization
        self.apply(init_weights)
        
    def forward(self, x):
        # Normalize pixel values
        x = x / 255.0
        # Permute from (N, H, W, C) to (N, C, H, W) for PyTorch convolutions
        x = x.permute(0, 3, 1, 2)
        
        conv_out = self.conv(x)
        value = self.critic(conv_out)
        logits = self.actor(conv_out)
        
        # Use Categorical distribution for discrete action spaces
        dist = torch.distributions.Categorical(logits=logits)
        return dist, value

    def save_the_model(self, path='models/model.pt'):
        if not os.path.exists('models'):
            os.makedirs('models')
        torch.save(self.state_dict(), path)

    def load_the_model(self, path='models/model.pt'):
        if os.path.exists(path):
            self.load_state_dict(torch.load(path))
            print("Model loaded successfully.")
        else:
            print("No model found, starting from scratch.")