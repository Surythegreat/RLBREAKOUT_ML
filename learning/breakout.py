import collections
# import cv2
import gymnasium as gym
import numpy as np
from PIL import Image
import torch

class DQNBreakout(gym.Wrapper):

    def __init__(self,render_mode='rgb_array',repeat=4,device='cpu'):
        env = gym.make('BreakoutNoFrameskip-v4',render_mode=render_mode)

        super(DQNBreakout,self).__init__(env)
        self.image_shape = (84,84)
        self.repeat=repeat
        self.lives=env.ale.lives()
        self.frame_buffer=[]
        self.device = device

    def step(self,action):
        total_rewards =0
        done =False

        for i in range(self.repeat):
            observation,reward,done,truncated,info =self.env.step(action)
            frame = self.env.render()

            # Convert RGB (Gym) -> BGR (OpenCV)
            # frame_bgr = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)

            # # Show
            # cv2.imshow("Gym Render", frame_bgr)

            # # Exit with 'q'
            # if cv2.waitKey(30) & 0xFF == ord("q"):
            #     break
            total_rewards +=reward

            current_lives=info['lives']
            if(current_lives<self.lives):
                total_rewards-=1
                self.lives=current_lives
            # print("lives",self.lives,"total reward:",total_rewards)
            # print(observation)
            self.frame_buffer.append(observation)

            if done:
                break
        
        max_frame = np.max(self.frame_buffer[-2:],axis=0)
        max_frame= self.process_observation(max_frame)
        max_frame=max_frame.to(self.device)

        total_rewards= torch.tensor(total_rewards).view(1,-1).float()
        total_rewards = total_rewards.to(self.device)

        done = torch.tensor(done).view(1,-1)
        done = done.to(self.device)
        return max_frame,total_rewards,done,info
    
    def reset(self):
        self.frame_buffer=[]

        observation,_ =self.env.reset()

        self.lives=self.env.ale.lives()

        observation = self.process_observation(observation)

        return observation

    def process_observation(self,observation):
        
        img = Image.fromarray(observation)
        img = img.resize(self.image_shape)
        img = img.convert("L")
        img = np.array(img)
        img=torch.from_numpy(img)
        img = img.unsqueeze(0)
        img = img.unsqueeze(0)
        img = img/255.0

        img = img.to(self.device)

        return img

