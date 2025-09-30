import os
import torch
import torch.nn as nn 

class AtariNet(nn.Module):

    def __init__(self,num_actions=4):
        super(AtariNet,self).__init__()
        self.relu = nn.ReLU()

        self.conv1 = nn.Conv2d(1,32,kernel_size=(8,8),stride=(4,4))
        self.conv2 = nn.Conv2d(32,64,kernel_size=(4,4),stride=(2,2))
        self.conv3 = nn.Conv2d(64,64,kernel_size=(3,3),stride=(1,1))

        self.flatten = nn.Flatten()

        self.dropout = nn.Dropout(0.2)

        self.action_value = nn.Linear(3136,1024)
        self.action_value2 = nn.Linear(1024,1024)
        self.action_value3 = nn.Linear(1024,num_actions)

        self.state_value = nn.Linear(3136,1024)
        self.state_value2 = nn.Linear(1024,1024)
        self.state_value3 = nn.Linear(1024,1)

    def forward(self,x):
        x=torch.tensor(x)
        x = self.relu(self.conv1(x))
        x = self.relu(self.conv2(x))
        x = self.relu(self.conv3(x))

        x = self.flatten(x)
        x = self.dropout(x)

        action_value = self.relu(self.action_value(x))
        action_value = self.dropout(action_value)
        action_value = self.relu(self.action_value2(action_value))
        action_value = self.dropout(action_value)
        action_value = self.action_value3(action_value)

        state_value = self.relu(self.state_value(x))
        state_value = self.dropout(state_value)
        state_value = self.relu(self.state_value2(state_value))
        state_value = self.dropout(state_value)
        state_value = self.relu(self.state_value3(state_value))


        q_values = state_value + (action_value - action_value.mean())

        return q_values
    
    def save_the_model(self,path='models/model.pt'):
        if not os.path.exists('models'):
            os.makedirs('models')
        torch.save(self.state_dict(),path)

    def load_the_model(self,path='models/model.pt'):
        try:
            self.load_state_dict(torch.load(path))
            print("Model loaded successfully")
        except:
            print("Error loading model")
        