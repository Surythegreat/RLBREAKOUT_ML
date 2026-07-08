# Reinforcement Learning for Atari Breakout 🧱🎮

<div align="center">

<img src="https://img.shields.io/badge/Research--a--thon-1st%20Place-FFD700?style=for-the-badge" alt="1st Place"/>
<img src="https://img.shields.io/badge/Python-3776AB?style=for-the-badge&logo=python&logoColor=white" alt="Python"/>
<img src="https://img.shields.io/badge/TensorFlow-FF6F00?style=for-the-badge&logo=tensorflow&logoColor=white" alt="TensorFlow"/>
<img src="https://img.shields.io/badge/OpenCV-5C3EE8?style=for-the-badge&logo=opencv&logoColor=white" alt="OpenCV"/>
<img src="https://img.shields.io/badge/NumPy-013243?style=for-the-badge&logo=numpy&logoColor=white" alt="NumPy"/>

<br/>
<b>🏆 1st Place, Research-a-thon</b> — CSAI Society, Dept. of CSE, IIT (ISM) Dhanbad
<br/>
<b>Author:</b> Suryansh Kulshreshtha · <a href="BREAKOUTDQN.pdf">Read the paper</a>

</div>

This repository contains the code for training a Deep Q-Network (DQN) agent to play the classic Atari game, Breakout. The agent learns to play from raw pixel data by interacting with the environment and maximizing its score, and the accompanying paper won 1st place at the CSAI Society's Research-a-thon.

## 📋 Table of Contents

1. [Project Overview](#project-overview)
2. [Model Architecture](#model-architecture)
3. [Results](#results)
4. [Repo Guide](#repo-guide)
5. [Getting Started](#getting-started)
6. [Usage](#usage)
7. [Technologies Used](#technologies-used)

## Project Overview

Breakout is a classic arcade game where the player controls a paddle at the bottom of the screen to bounce a ball and destroy bricks at the top. This project implements a reinforcement learning agent that learns an optimal policy for playing the game without any prior knowledge of its rules.

The agent is a Deep Q-Network (DQN) — a deep neural network combined with Q-learning. The network takes the game's screen pixels as input and outputs the expected return (Q-value) for each possible action. Training uses an epsilon-greedy strategy to balance exploration and exploitation, plus a replay buffer to store and sample past experience and stabilize learning. On top of the base DQN, the project explores **frame stacking**, **prioritized experience replay (PER)**, and a **dueling network architecture**, comparing each addition's effect on final performance.

## Model Architecture

The agent uses a Convolutional Neural Network (CNN) to process the game state, inspired by the original DeepMind DQN paper:

- **Input:** a stack of 4 preprocessed game frames (84×84 grayscale). Stacking frames lets the agent infer the ball's motion from a single observation.
- **Conv1:** 32 filters, 8×8, stride 4
- **Conv2:** 64 filters, 4×4, stride 2
- **Conv3:** 64 filters, 3×3, stride 1
- **Flatten → Dense(512) → Output layer** (one Q-value per action)

All hidden layers use ReLU activations.

## Results

**Headline run:**

| Metric | Value |
|---|---|
| Training Episodes | ~35,000 |
| Avg. Reward (last 100 episodes) | ~40 |
| Max Reward | 267 |

**Effect of each technique** (from the paper — DQN variants compared over the same training budget):

| Model | Avg. Reward / Episode | Avg. Steps / Episode | Observed Behavior |
|---|---|---|---|
| DQN | 11.3 | 165.9 | Highly spontaneous motion, frequent abrupt direction changes |
| DQN + Frame Stacking | 52.8 | 1166.5 | Smoother, continuous motion; clears rows one at a time before tunneling |
| DQN + Frame Stacking + PER | 113.8 | 1419.5 | Early tunneling behavior, continuous motion, improved stability and strategic play |

The fully-tuned agent (DQN + frame stacking + PER) learned to **tunnel** — carving a channel up one side of the brick wall so the ball bounces around behind it for a cascade of high-value hits. It's a well-documented emergent strategy from DeepMind's original Atari work, and a solid signal this agent found a genuinely good policy rather than a locally decent one.

**Demo:**

https://github.com/SudoKuder/RLBREAKOUT_ML/raw/main/DQN/Video/episode_5.mp4

## Repo Guide

This repo tracks several experiments side by side:

| Folder | What it is |
|---|---|
| `DQN/` | **Main folder** — the fully-tuned agent (frame stacking + PER) with its results and demo video. Start here. |
| `DQN_normal/` | Ablation: frame stacking, no PER |
| `DQN_noFRAMESTk/` | Ablation: no frame stacking |
| `DQN_SB/` | Attempt using the Stable-Baselines library; not currently working |
| `DDQN/` | Dueling network architecture, no PER |
| `DDQN_priority_mem/` | Dueling network architecture + PER |
| `PPO/` | Follow-up experiment with PPO, noted as future work in the paper |
| `learning/` | Early scratch/practice notebooks |

Worth calling out: the dueling architecture (`DDQN*`) actually underperformed the plain DQN + frame stacking + PER setup in this environment. That negative result — and why it happened — is discussed in the paper, and is a good talking point in itself (it shows you validated rather than just assumed a "fancier" architecture would win).

## Getting Started

### Prerequisites

- Python 3.8+
- pip
- A virtual environment (recommended)

### Installation

```bash
git clone https://github.com/SudoKuder/RLBREAKOUT_ML.git
cd RLBREAKOUT_ML

conda create -n newRLEnv
conda activate newRLEnv
conda env create -f environment.yml
```

## Usage

The primary way to interact with this project is through the provided Jupyter notebook, `reproduce_results.ipynb`.

### Training

Open and run all cells in `reproduce_results.ipynb`. This will:
- Initialize the environment and the DQN agent
- Run the training loop for the specified number of episodes
- Periodically save model checkpoints to `models/`
- Log training progress

### Evaluation

Use the evaluation section of `reproduce_results.ipynb`:
- Load weights from a saved model file (e.g. `models/breakout_dqn.h5`)
- Run the test function in `agent_DQN` to render the game and record the agent's performance
- A video of the agent playing is generated and viewable directly in the corresponding folder

## Technologies Used

- **Python** — core language
- **TensorFlow** — building and training the deep neural network
- **Gymnasium** (ALE) — the Atari Breakout environment
- **NumPy** — numerical operations and the replay buffer
- **OpenCV** — image preprocessing (resizing, grayscaling)
