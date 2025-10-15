Reinforcement Learning for Atari Breakout

This repository contains the code for training a Deep Q-Network (DQN) agent to play the classic Atari game, Breakout. The agent learns to play the game from raw pixel data by interacting with the environment and maximizing its score.
Table of Contents

    1)Project Overview

    2)Model Architecture

    3)Results

    4)Getting Started

        4.1)Prerequisites

        4.2)Installation

    5)Usage

        5.1)Training

        5.2)Evaluation

    6)Technologies Used



1)Project Overview

Breakout is a classic arcade game where the player controls a paddle at the bottom of the screen to bounce a ball and destroy bricks at the top. This project implements a Reinforcement Learning agent that learns an optimal policy for playing this game without any prior knowledge of the rules.

We use a Deep Q-Network (DQN), a type of model that combines a deep neural network with Q-learning. The neural network takes the game's screen pixels as input and outputs the expected return (Q-value) for each possible action (e.g., move left, move right). The agent uses an epsilon-greedy strategy to balance exploration and exploitation, and a replay buffer to store and sample past experiences, which stabilizes the learning process.

2)Model Architecture

The agent uses a Convolutional Neural Network (CNN) to process the game state. The architecture is inspired by the original DeepMind paper on playing Atari games:

    Input: A stack of 4 pre-processed game frames (84x84 grayscale images). Stacking frames helps the agent understand the ball's motion.

    Convolutional Layers:

        Conv1: 32 filters of size 8x8 with stride 4.

        Conv2: 64 filters of size 4x4 with stride 2.

        Conv3: 64 filters of size 3x3 with stride 1.

    Fully Connected Layers:

        Flatten layer.

        Dense layer with 512 units.

        Output layer with units corresponding to the number of possible actions in the game.

All hidden layers use the ReLU activation function.

3)Results

After training for approximately 3 million frames, the agent learns effective strategies, such as carving a tunnel through the bricks to hit them from above. The agent's performance is measured by the average reward obtained over a set of evaluation episodes.

Training Episodes
	

~35000

Avg. Reward (last 100 episodes)
	

> ~40

Max Reward
	

267

Here is a sample of the agent's performance during training:
 [![Watch the video]()](DQN/Video/episode_5.mp4)

4)Getting Started

Follow these instructions to set up the project on your local machine.

4.1)Prerequisites

    Python 3.8+

    pip package manager

    A virtual environment (recommended)

4.2)Installation

    Clone the repository:

    git clone [https://github.com/Surythegreat/RLBREAKOUT_ML.git](https://github.com/Surythegreat/RLBREAKOUT_ML.git)
    cd RLBREAKOUT_ML

    Create and activate a virtual environment:

    conda create -n newRLEnv
    conda activate newRLEnv

    Install the required packages:
    The project requires the following libraries. You can install them using the reproduce_results.ipynb notebook or manually.

    conda env create -f environment.yml



5)Usage

The primary way to interact with this project is through the provided Jupyter Notebook.

5.1)Training

To train a new agent from scratch, open and run all the cells in the reproduce_results.ipynb notebook. The training section will:

    Initialize the environment and the DQN agent.

    Run the training loop for a specified number of episodes.

    Periodically save model checkpoints to the models/ directory.

    Log the training progress.

5.2)Evaluation

To evaluate a pre-trained agent, use the evaluation section in the reproduce_results.ipynb notebook. You will need to:

    Load the weights from a saved model file (e.g., models/breakout_dqn.h5).

    Run the test function in agent_DQN file, which will render the game and record the agent's performance.

    A video of the agent playing will be generated and can be viewed directly in the folder.

6)Technologies Used

    Python: Core programming language.

    TensorFlow: For building and training the deep neural network.

    Gymnasium: The toolkit for providing the Atari Breakout environment.

    NumPy: For numerical operations and managing the replay buffer.

    OpenCV: For image pre-processing (resizing and grayscaling).

