from collections import deque
import torch
import copy
import torch.optim as optim
import torch.nn.functional as F
import time
import numpy as np
# from torch.utils.tensorboard import SummaryWriter
from datetime import datetime
import os
from buffer import ReplayMemory
import cv2
import csv




class Agent:

    def __init__(self, model, device='cpu', epsilon=1.0, min_epsilon=0.1, nb_warmup=10000, nb_actions=None, memory_capacity=10000, batch_size=32, learning_rate=0.00025,state_stack=4,epsilon_decay= 0.99964):
        self.memory = ReplayMemory(device=device, capacity=memory_capacity)
        self.model = model
        self.target_model = copy.deepcopy(model).eval()
        self.epsilon = epsilon
        self.min_epsilon = min_epsilon
        self.epsilon_decay = epsilon_decay # Find the right value to take epsilon to min_epsilon over 10000 steps
        self.batch_size = batch_size
        self.model.to(device)
        self.target_model.to(device)
        self.gamma = 0.99
        self.nb_actions = nb_actions
        self.state_stack = state_stack

        self.optimizer = optim.AdamW(model.parameters(), lr=learning_rate)

        print(f"Starting epsilon is {self.epsilon}")
        print(f"Epsilon decay is {self.epsilon_decay}")

        if not os.path.exists("./tensorboard_logdir"):
            os.makedirs("./tensorboard_logdir")

    def get_action(self, state):
        if torch.rand(1) < self.epsilon:
            return torch.randint(self.nb_actions, (1, 1))
        else:
            av = self.model(state).detach()
            return torch.argmax(av, dim=-1, keepdim=True)

    def train(self, env, epochs, batch_identifier=0):
        # stats = {'AvgReturns': [], 'EpsilonCheckpoint': []}
        returns_deque = deque(maxlen=100)
        log_file = 'training_log.csv'
        with open(log_file, 'w', newline='') as f:
            writer = csv.writer(f)
            writer.writerow(['Epoch', 'return','loss','epsilon'])
        # writera = SummaryWriter(log_dir=f"./tensorboard_logdir/{datetime.now().strftime('%Y-%m-%d')}")

        for epoch in range(1, epochs + 1):
            state = env.reset()
            # state_deque = deque([state] * self.state_stack, maxlen=self.state_stack)
            # Concatenate the frames in the deque to create the initial stacked state
            # stacked_state = torch.cat(list(state_deque), dim=1)

            done = False
            ep_return = 0
            loss=F.mse_loss(torch.tensor([0.0]),torch.tensor([0.0]))
            while not done:
                action = self.get_action(state)
                next_state, reward, done, info = env.step(action)

                # next_state_deque = deque(state_deque, maxlen=self.state_stack)
                # Add the new frame to the next state deque
                # next_state_deque.append(next_state)
                # Concatenate the frames to create the next_stacked_state
                # next_stacked_state = torch.cat(list(next_state_deque), dim=1)

                self.memory.insert([state, action, reward, done, next_state])


                # QSA = Q-value, state, action

                if self.memory.can_sample(self.batch_size):
                    state_b, action_b, reward_b, done_b, next_state_b = self.memory.sample(self.batch_size)
                    qsa_b = self.model(state_b).gather(1, action_b)
                    with torch.no_grad():
                        next_qsa_b = self.target_model(next_state_b)
                        next_qsa_b = torch.max(next_qsa_b, dim=-1, keepdim=True)[0]
                        target_b = reward_b + ~done_b * self.gamma * next_qsa_b
                    loss = F.mse_loss(qsa_b, target_b)
                    self.model.zero_grad()
                    loss.backward()
                    self.optimizer.step()


                # stacked_state = next_stacked_state
                # The deque also needs to be updated for the next iteration's copy
                # state_deque.append(next_state)
                state = next_state

                ep_return += reward.item()

            # writera.add_scalar(f'Returns: {batch_identifier}', ep_return, epoch)
            with open(log_file, 'a', newline='') as f:
                writer = csv.writer(f)
                writer.writerow([epoch,ep_return,loss.item(),self.epsilon])

            returns_deque.append(ep_return)

            if self.epsilon > self.min_epsilon:
                self.epsilon = self.epsilon * self.epsilon_decay

            if epoch % 10 == 0:
                self.model.save_the_model()
                print("")

                

                average_returns = np.mean(returns_deque)

                

                if (len(returns_deque)) > 100:
                    print(f"Epoch: {epoch} - Average Return: {average_returns} - Epsilon: {self.epsilon}, loss: {loss}")
                else:
                    print(f"Epoch: {epoch} - Episode Return: {average_returns} - Epsilon: {self.epsilon}, loss: {loss}")

            if epoch % 100 == 0:
                self.target_model.load_state_dict(self.model.state_dict())

            if epoch % 1000 == 0:
                self.model.save_the_model(f"models/model_iter_{epoch}.pt")


        # return stats

    def test(self, env):
        # self.epsilon = 0 # Usually set to 0 or a very low value for testing

        for epoch in range(1, 3):
            state = env.reset()
            state_deque = deque([state] * self.state_stack, maxlen=self.state_stack)
            stacked_state = torch.cat(list(state_deque), dim=1)
            done = False

            # Loop for the duration of a single episode
            for _ in range(1000):
                time.sleep(0.02)  # Add a small delay to make the visualization smoother
                # Render the environment to get an image frame as a NumPy array
                # Note: The render mode 'rgb_array' is standard for many Gym environments
                frame = env.render()

                # OpenCV displays images in BGR format, but Gym renders in RGB.
                # We need to convert the color channels.
                frame_bgr = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)

                # Display the frame in a window named "Agent Test"
                cv2.imshow('Agent Test', frame_bgr)

                # This line is crucial for updating the window and processing GUI events.
                # It also allows you to quit the visualization by pressing the 'q' key.
                if cv2.waitKey(1) & 0xFF == ord('q'):
                    break

                # Your agent's logic remains the same
                action = self.get_action(stacked_state)
                state, reward, done, info = env.step(action)
                state_deque.append(state)
                stacked_state = torch.cat(list(state_deque), dim=1)
                if done:
                    break

        # After all epochs, close the OpenCV window to clean up
        env.close() # It's good practice to close the environment
        cv2.destroyAllWindows()
