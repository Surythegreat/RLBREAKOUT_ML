<div align="center">

# 🎮 Deep RL Breakout — From Pixels to Pro

### Teaching an AI to master Atari Breakout using Deep Reinforcement Learning

[![Python](https://img.shields.io/badge/Python-3.8+-3776AB?style=for-the-badge&logo=python&logoColor=white)](https://python.org)
[![TensorFlow](https://img.shields.io/badge/TensorFlow-2.x-FF6F00?style=for-the-badge&logo=tensorflow&logoColor=white)](https://tensorflow.org)
[![Gymnasium](https://img.shields.io/badge/Gymnasium-Atari-0A0A2A?style=for-the-badge)](https://gymnasium.farama.org)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg?style=for-the-badge)](LICENSE)

</div>

---

## 🎬 See It In Action

> *The agent starts completely blind. No rules. No hints. Just pixels and a score. Here's what it learned.*

<div align="center">

https://github.com/SudoKuder/RLBREAKOUT_ML/raw/main/DQN/Video/episode_5.mp4

**Episode 5 — Early training. The agent is already finding the tunnel strategy.**

</div>

---

## 🧠 What This Project Is

This project implements and benchmarks **multiple Deep Reinforcement Learning algorithms** to solve Atari Breakout — one of the canonical hard problems in RL. The agent sees only raw pixel frames. It knows nothing about the physics, the rules, or even what a ball is. It figures it out entirely through trial, error, and ~35,000 episodes of self-play.

The result? An agent that not only plays competently, but discovers the **tunnel strategy** — carving a hole through the bricks to bounce the ball behind the wall for massive score chains. This emergent behavior was famously noted in DeepMind's original DQN paper, and this agent replicates it independently.

---

## 📊 Results At a Glance

| Metric | Value |
|---|---|
| Training Episodes | ~35,000 |
| Training Frames | ~3,000,000 |
| Avg. Reward (last 100 eps) | **~40** |
| Max Reward Achieved | **267** |
| Human-level score (reference) | ~31 |

> The agent surpasses average human performance.

---

## 🏗️ Algorithms Implemented & Compared

This isn't just a DQN demo — it's a **systematic comparison** of progressively advanced RL architectures:

| Folder | Algorithm | Key Idea |
|---|---|---|
| `DQN_normal/` | Vanilla DQN | Baseline: CNN + Q-learning |
| `DQN/` | DQN + Frame Stacking | 4-frame temporal stack for motion awareness |
| `DQN_noFRAMESTk/` | DQN (no stacking) | Ablation: what happens without temporal context |
| `DQN_SB/` | DQN via Stable-Baselines3 | Framework-level implementation for benchmarking (MADE WITH GEMINI) |
| `DDQN/` | Double DQN | Decoupled action selection/evaluation to reduce overestimation |
| `DDQN_priority_mem/` | DDQN + Prioritized Replay | Sample rare/important transitions more frequently |
| `PPO/` | Proximal Policy Optimization | Policy gradient alternative — on-policy comparison(not for consideration) |

Each variant is a self-contained experiment with its own training loop, checkpoints, and video recordings.

---

## 🔬 Model Architecture

The core network is a **CNN** inspired by the original DeepMind paper (Mnih et al., 2015):

```
Input: Stack of 4 grayscale frames → (84 × 84 × 4)
        ↓
Conv1:  32 filters, 8×8 kernel, stride 4  → ReLU
        ↓
Conv2:  64 filters, 4×4 kernel, stride 2  → ReLU
        ↓
Conv3:  64 filters, 3×3 kernel, stride 1  → ReLU
        ↓
Flatten → Dense(512) → ReLU
        ↓
Output: Q-values for each action [NOOP, FIRE, RIGHT, LEFT]
```

**Key training techniques:**
- **ε-greedy exploration** — starts at 1.0, anneals to 0.1 over 1M frames
- **Experience Replay Buffer** — 100,000 transitions, uniform (DQN) or prioritized (DDQN+PER)
- **Target Network** — frozen copy updated every ~10,000 steps to stabilize learning
- **Frame preprocessing** — resize to 84×84, grayscale, normalize to [0, 1]
- **Frame stacking** — 4 consecutive frames stacked to capture ball velocity

---

## 📁 Repository Structure

```
RLBREAKOUT_ML/
├── DQN/                    # DQN with frame stacking (primary implementation)
│   └── Video/              # Recorded gameplay episodes
├── DQN_normal/             # Vanilla DQN baseline
├── DQN_noFRAMESTk/         # DQN without frame stacking (ablation)
├── DQN_SB/                 # Stable-Baselines3 DQN
├── DDQN/                   # Double DQN
├── DDQN_priority_mem/      # DDQN + Prioritized Experience Replay
├── PPO/                    # Proximal Policy Optimization
├── learning/               # Exploratory notebooks and experiments
├── BREAKOUTDQN.pdf         # Full project report
└── environment.yml         # Conda environment spec
```

---

## ⚡ Quick Start

### Prerequisites
- Python 3.8+
- Conda (recommended)
- GPU optional but strongly recommended for training from scratch

### Installation

```bash
# 1. Clone the repo
git clone https://github.com/SudoKuder/RLBREAKOUT_ML.git
cd RLBREAKOUT_ML

# 2. Create conda environment
conda env create -f environment.yml
conda activate newRLEnv
```

### Run Training

```bash
# Navigate to your algorithm of choice
cd DQN

# Launch training (edit hyperparameters inside the script)
python agent_DQN.py
```

### Evaluate a Trained Agent

```bash
# Point to a saved model checkpoint and render gameplay
python agent_DQN.py --eval --model models/breakout_dqn.h5
# Video will be saved to Video/ directory
```

---

## 🔑 Key Learnings & Observations

- **Frame stacking is non-negotiable.** The `DQN_noFRAMESTk` ablation confirms: without temporal context, the agent cannot track ball direction and plateaus at near-random performance.
- **Double DQN matters late in training.** DDQN's correction for Q-value overestimation becomes more significant as the policy matures — early training differences are minimal.
- **Prioritized Replay accelerates convergence.** DDQN+PER reaches meaningful scores in fewer frames by replaying the transitions where the agent is most "confused."
- **Emergent tunnel strategy.** Around episode 15,000+, the DQN agent independently discovers that carving through the brick wall yields exponentially higher returns — replicating the hallmark behavior from the original DeepMind paper.

---

## 📄 Full Report

A detailed writeup covering methodology, hyperparameter choices, training curves, and analysis is available:

**[📑 BREAKOUTDQN.pdf](./BREAKOUTDQN.pdf)**

---

## 🛠️ Tech Stack

| Tool | Role |
|---|---|
| Python 3.8+ | Core language |
| TensorFlow 2.x | Neural network training |
| Gymnasium (ALE) | Atari environment |
| NumPy | Replay buffer, frame ops |
| OpenCV | Frame preprocessing |
| Stable-Baselines3 | Framework baseline (DQN_SB) |
| Matplotlib | Training curve visualization |

---

## 👤 Author

**Suryansh** — [@SudoKuder](https://github.com/SudoKuder)
*B.Tech CS • IIT (ISM) Dhanbad*

---

<div align="center">

*If this helped you, drop a ⭐ — it keeps the dopamine reward signal going.*

</div>
