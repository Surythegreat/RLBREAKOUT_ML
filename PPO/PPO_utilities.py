import torch
import numpy as np
from PIL import Image
def test_env(vis=False,env=None, model=None, device='cpu'):
    state = env.reset()
    if vis: env.render()
    done = False
    total_reward = 0
    while not done:
        state = torch.FloatTensor(state).unsqueeze(0).to(device)
        dist, _ = model(state)
        next_state, reward, done, _ = env.step(dist.sample().cpu().numpy()[0])
        state = next_state
        if vis: env.render()
        total_reward += reward
    return total_reward

def preprocess_frame(frame):
    """
    Converts a frame to grayscale and resizes it to 84x84.
    """
    # Convert to grayscale using the luminosity method
    gray_frame = np.dot(frame[...,:3], [0.299, 0.587, 0.114])
    
    # Use Pillow for high-quality resizing
    pil_image = Image.fromarray(gray_frame)
    resized_image = pil_image.resize((84, 84), Image.Resampling.LANCZOS)
    
    return (np.array(resized_image).astype(np.float32))

# This method calculates the Generalized Advantage Estimation (GAE).
# GAE provides a way to balance the trade-off between bias and variance in advantage estimation.
def compute_gae(next_value, rewards, masks, values, gamma=0.99, tau=0.95):
    # `values` is a list of value estimates from the critic for each step.
    # We append `next_value` (the value of the state after the last step) to bootstrap the calculation.
    values = values + [next_value]
    gae = 0
    returns = [] # This list will store the GAE-based returns (targets for the value function)

    # Iterate backwards through the collected steps
    for step in reversed(range(len(rewards))):
        # Calculate the TD error (delta): R_t + gamma * V(S_{t+1}) - V(S_t)
        # `masks` are 1 for non-terminal states and 0 for terminal states.
        delta = rewards[step] + gamma * values[step + 1] * masks[step] - values[step]
        
        # Update GAE: GAE(t) = delta_t + gamma * tau * GAE(t+1)
        # `tau` is the GAE parameter that controls the bias-variance trade-off.
        gae = delta + gamma * tau * masks[step] * gae
        
        # The return is the calculated GAE plus the value function estimate at that step.
        # This is the target value for the critic.
        returns.insert(0, gae + values[step])
        
    return returns

