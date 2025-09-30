import gymnasium as gym
import numpy as np
from PIL import Image
import torch
import os
from breakout import *
from model import *
from agent import *
print(torch.cuda.is_available())
print(torch.version.cuda)
print(torch.cuda.device_count())
print(torch.cuda.get_device_name(0) if torch.cuda.is_available() else "No GPU")
print(torch.__version__)


os.environ['KMP_DUPLICATE_LIB_OK']='TRUE'

device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

environment = DQNBreakout(device=device)
model = AtariNet(num_actions=4)

model.to(device)
print(device)
model.load_the_model()

agent = Agent(model=model,device=device,epsilon=.83,action_space=4,lr=0.0001,memory_capacity = 1000000,batch_size=64)

agent.train(env = environment,epochs= 2000000)
# for _ in range(100):
#     action = environment.action_space.sample()
#     state,reward,done,info = environment.step(action)

#     print(state.shape)