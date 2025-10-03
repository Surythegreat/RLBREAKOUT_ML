import torch
import os
import torch.nn as nn

# --- Model Definition (from your models.py) ---
class Model(nn.Module):
    def __init__(self, output_size=4, frame_stack_size=4):
        super(Model, self).__init__()
        self.relu = nn.ReLU()
        # Expects a 1-channel (grayscale) image
        self.conv1 = nn.Conv2d(frame_stack_size, 32, kernel_size=8, stride=4)
        self.conv2 = nn.Conv2d(32, 64, kernel_size=4, stride=2)
        self.conv3 = nn.Conv2d(64, 64, kernel_size=3, stride=1)

        self.flatten = nn.Flatten()
        self.dropout = nn.Dropout(0.2)
        
        # The input size 3136 is derived from an 84x84 input image
        # (64 channels * 7 * 7 feature map size)
        self.fc1 = nn.Linear(3136, 512)
        self.fc2 = nn.Linear(512, output_size)

    def forward(self, x):
        # Input x shape: [batch_size, 1, 84, 84]
        x = self.relu(self.conv1(x))
        x = self.relu(self.conv2(x))
        x = self.relu(self.conv3(x))
        x = self.flatten(x)
        x = self.dropout(x)
        x = self.relu(self.fc1(x))
        x = self.dropout(x)
        x = self.fc2(x)
        return x
    
    def save_the_model(self, path='models/model.pt'):
        if not os.path.exists('models'):
            os.makedirs('models')
        torch.save(self.state_dict(), path)

    def load_the_model(self, path='models/model.pt'):
        try:
            self.load_state_dict(torch.load(path))
            print("Model loaded successfully")
        except:
            print("Error loading model, starting from scratch.")