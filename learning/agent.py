

import copy
import random
import torch
import torch.optim as optim
import torch.nn.functional as F
from plot import LivePlot
import numpy as np
import time

class ReplayMemory:
    def __init__(self, capacity,device = 'cpu'):
        self.capacity = capacity
        self.memory = []
        self.position = 0
        self.device = device
        self.memory_max_report = 0

    def insert(self, transition):
        # Move tensors once here
        transition = tuple(item.detach().cpu() for item in transition)
        if len(self.memory) < self.capacity:
            self.memory.append(transition)
        else:
            self.memory[self.position] = transition
        self.position = (self.position + 1) % self.capacity

        
        

    def sample(self, batch_size=32):
        batch = random.sample(self.memory, batch_size)
        state, action, reward, next_state, done = zip(*batch)
        
        # Remove extra dimensions if needed
        state = torch.stack(state).squeeze(1).to(self.device)      # BxCxHxW
        next_state = torch.stack(next_state).squeeze(1).to(self.device)
        
        return (state,
                torch.stack(action).to(self.device).long(),
                torch.stack(reward).to(self.device),
                next_state,
                torch.stack(done).to(self.device))




    def can_sample(self, batch_size):
        return len(self.memory) >= batch_size

    def __len__(self):
        return len(self.memory)
    
class Agent:
    def __init__(self, model, device='cpu', epsilon=1.0,min_epsilon=0.1,nb_warmup = 10000, action_space=None, memory_capacity=10000, batch_size=32, gamma=0.99, lr=1e-4):
        self.model = model
        self.target_model = copy.deepcopy(model).eval()
        self.device = device
        self.action_space = action_space
        self.memory = ReplayMemory(device=device, capacity=memory_capacity)
        self.epsilon = epsilon
        self.min_epsilon = min_epsilon
        self.epsilon_decay=1 - (((epsilon - min_epsilon) / nb_warmup)*2)
        self.batch_size = batch_size
        self.model.to(self.device)
        self.target_model.to(self.device)
        self.gamma = gammamodel

        self.optimizer = torch.optim.Adam(self.model.parameters(), lr=lr)
        print("starting epsilon:",self.epsilon)
        print("epsilon decay:",self.epsilon_decay)

    def select_action(self, state):
       # In select_action
        if torch.rand(1) < self.epsilon:
            return torch.randint(self.action_space, (1,))  # shape [1], not [1,1]
        else:
            av = self.model(state).detach()
            return torch.argmax(av, dim=1)  # shape [1]

       
    def train(self,env,epochs):
        stats = {'episode':[], 'reward':[], 'length':[], 'epsilon':[],'returns':[],'avgreturns':[]}
        plotter = LivePlot()

        for episode in range(1, epochs + 1):
            obs = env.reset()
            # When creating state and next_state
            state = torch.as_tensor(obs, dtype=torch.float32)
            if state.ndim == 2:              # grayscale HxW
                state = state.unsqueeze(0)    # -> 1xHxW
            elif state.ndim == 3 and state.shape[-1] in [1,3]:  # HxWxC
                state = state.permute(2,0,1)  # -> CxHxW
            state = state.to(self.device)

            done = False
            ep_return = 0.0


            while not done:
                action = self.select_action(state)
                obs, reward, done, info = env.step(action.item())
                # next_state: do NOT add extra batch dimension
                next_state = torch.as_tensor(obs, dtype=torch.float32)
                if next_state.ndim == 2:
                    next_state = next_state.unsqueeze(0)
                elif next_state.ndim == 3 and next_state.shape[-1] in [1,3]:
                    next_state = next_state.permute(2,0,1)
                next_state = next_state.to(self.device)



                # Convert reward, done into tensors
                reward_t = torch.tensor([reward], dtype=torch.float32).to(self.device)
                done_t = torch.tensor([done], dtype=torch.float32).to(self.device)
                self.memory.insert((state, action, reward_t, next_state, done_t))                    



                if self.memory.can_sample(self.batch_size):
                    state_b,action_b,reward_b,next_state_b,done_b = self.memory.sample(self.batch_size)
                    action_b=action_b.long()
                    # print(state_b.shape)  # should be [batch_size, channels, height, width]

                    qsa_b = self.model(state_b).gather(1,action_b)
                    next_qsa_b = self.target_model(next_state_b)
                    next_qsa_b = torch.max(next_qsa_b,dim=1,keepdim=True)[0]
                    done_b = done_b.float()
                    target_b = reward_b + self.gamma * next_qsa_b * (1 - done_b)
                    loss = F.mse_loss(qsa_b,target_b)
                    self.model.zero_grad(set_to_none=True)
                    loss.backward()
                    self.optimizer.step()
                

                state = next_state
                ep_return += reward.item()

            stats['returns'].append(ep_return)
            if self.epsilon > self.min_epsilon:
                self.epsilon *= self.epsilon_decay
            print(f"Episode {episode} - Return: {ep_return:.2f} - Epsilon: {self.epsilon:.3f} - Memory Size: {len(self.memory)}")
            if(episode % 10 == 0):
                self.model.save_the_model()
                print(" ")

                average_return = np.mean(stats['returns'][-100:])

                stats['avgreturns'].append(average_return)
                stats['episode'].append(episode)
                stats['reward'].append(reward.item())
                stats['length'].append(info['lives'])
                stats['epsilon'].append(self.epsilon)

                if(len(stats['returns'])>100):
                    print(f"Episode {episode} - Return: {ep_return:.2f} - Average Return: {average_return:.2f} - Epsilon: {self.epsilon:.3f} - Memory Size: {len(self.memory)}")
                    
                else:
                    print(f"Episode {episode} - Return: {ep_return:.2f} - Epsilon: {self.epsilon:.3f} - Memory Size: {len(self.memory)}")
                    

            if episode % 100 == 0:
                self.target_model.load_state_dict(self.model.state_dict())
                plotter.update(stats)
                print("Target model updated")
            if episode % 1000 == 0:
                self.model.save_the_model(f'models/savedModel/{episode}.Pt')
                print("Model saved")
        return stats
    
    def test(self,env):

        for epoch in range(1,3):
            state = env.reset()
            done =False

            for _ in range(1000):
                time.sleep(0.01)
                action = self.select_action(state)
                state,reward,done,info = env.step(action)
                if done:
                    break

