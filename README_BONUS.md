# 🧠 Bonus Task — Neural Network Tic-Tac-Toe Player (TensorFlow Edition)

## 🎯 Overview
This bonus module implements a **self-learning Tic-Tac-Toe AI** using **TensorFlow 2 (Keras)**.  
The agent learns from scratch with the **REINFORCE policy-gradient algorithm** and can later play against a human using a simple GUI.

| Difficulty | Opponent Type | Description |
|-------------|---------------|--------------|
| 🟢 **Easy** | Random moves | Opponent plays randomly |
| 🟡 **Medium** | Heuristic rules | Opponent uses fixed smart logic (center, corners, block, win) |
| 🔴 **Hard** | Self-play | The agent competes against itself to improve |

---

## 🧩 Folder Structure
```
bonus/
└── ttt/
    ├── env.py           # Tic-Tac-Toe environment (rules, rewards)
    ├── model.py         # TensorFlow/Keras policy network
    ├── opponents.py     # Random and heuristic opponents
    ├── train.py         # REINFORCE training loop + plots
    ├── game.py          # Tkinter GUI (human vs AI)
    └── __init__.py
```

Everything runs **on CPU** — no GPU or CUDA required.

---

## ⚙️ Setup

### 1️⃣ Create a virtual environment
```bash
python3 -m venv .venv-ttt
source .venv-ttt/bin/activate
# Windows: .venv-ttt\Scripts\activate
```

### 2️⃣ Install dependencies
Create `requirements-bonus-ttt-tf.txt`:
```txt
tensorflow-cpu==2.16.1
matplotlib==3.9.0
numpy==1.26.4
```

Then install:
```bash
pip install -r requirements-bonus-ttt-tf.txt
```

If the GUI fails due to missing Tkinter on Linux:
```bash
sudo apt update && sudo apt install -y python3-tk
```

---

## 🧠 How It Works

### 🧱 Environment (`env.py`)
- 3×3 board flattened into a 9-vector (`1 = X`, `-1 = O`, `0 = empty`).
- Detects wins, draws, and illegal moves.
- Reward system:
  - `+1` → AI win
  - `−1` → AI loss
  - `0` → Draw or ongoing

### 🧮 Model (`model.py`)
A compact **policy network**:
```
Input (9) → Dense(64, tanh) → Dense(64, tanh) → Dense(9, linear)
```
Outputs **logits** for all moves, converted to probabilities via a **masked softmax** so illegal moves are ignored.

### 🎯 Training (`train.py`)
Implements the **REINFORCE** policy gradient algorithm:

∇J(θ) = E[Σ ∇θ log πθ(a_t|s_t) G_t]

Training steps:
1. Play games and record `(state, action, reward)` each turn.
2. Compute discounted returns G_t.
3. Recompute logits under one GradientTape.
4. Apply gradient ascent on −Σ(log π × G).

Outputs:
```
bonus_out_tf/
├── ttt_policy_<difficulty>.keras
└── training_<difficulty>.png
```

### 🎨 Visualization
- **Left plot:** Policy loss across episodes  
- **Right plot:** Rolling win/draw/loss rates (smoothed)

---

## 🕹️ Play the Game (`game.py`)

Run the GUI:
```bash
python bonus/ttt/game.py
```

Interface:
- 3×3 grid buttons
- Dropdown for difficulty (`easy`, `medium`, `hard`)
- Buttons:
  - **Load Model** → load trained model from `bonus_out_tf/ttt_policy_<difficulty>.keras`
  - **Reset** → start a new game

Gameplay:
- AI = **X**  
- Human = **O**  
- Messages: “AI (X) wins!”, “Draw.”, or “You (O) win!”

---

## 🏋️ Training Commands

| Mode   | Command                                                          | Description     |
|--------|------------------------------------------------------------------|-----------------|
| Easy   | `python bonus/ttt/train.py --opponent easy --episodes 4000`      | vs random       |
| Medium | `python bonus/ttt/train.py --opponent medium --episodes 6000`    | vs heuristic    |
| Hard   | `python bonus/ttt/train.py --opponent hard --episodes 12000`     | self-play       |

Optional flags:
```
--episodes <int>   # number of games
--gamma <float>    # discount factor (default 0.99)
--lr <float>       # learning rate (default 0.002)
--log-every <int>  # log frequency (default 200)
--outdir <path>    # output folder (default bonus_out_tf)
```

Example:
```bash
python bonus/ttt/train.py --opponent medium --episodes 8000 --lr 0.001
```

---

## 📊 Sample Output
```
Training vs: easy
[  200] loss=0.4829  win/draw/lose (last 200): 68.5/24.5/7.0%
[  400] loss=0.3124  win/draw/lose (last 200): 82.0/12.0/6.0%
Saved: bonus_out_tf/ttt_policy_easy.keras
Saved plot: bonus_out_tf/training_easy.png
```

Then:
```bash
python bonus/ttt/game.py
```

---

## 🧾 Notes & Tips
- **CPU-only**, no GPU required.  
- If Tkinter is missing on Linux: `sudo apt install python3-tk`.  
- Reproducibility:
  ```python
  import numpy as np, tensorflow as tf
  np.random.seed(0)
  tf.random.set_seed(0)
  ```

---

## 🧠 Key Learning Concepts

| Concept | Explanation |
|--------|-------------|
| **Policy Gradient (REINFORCE)** | Learns a probability distribution over actions instead of fixed rules |
| **Self-Play** | Agent improves by training against itself |
| **Masked Softmax** | Ensures only legal moves are chosen |
| **Reward Shaping** | +1 (win), −1 (loss), 0 (draw) |
| **Visualization** | Makes policy improvement interpretable |

---

## ✅ Summary

| Component | Description |
|-----------|-------------|
| Framework | TensorFlow 2 (Keras API) |
| Algorithm | Policy Gradient (REINFORCE) |
| Environment | Custom Tic-Tac-Toe |
| Opponents | Random / Heuristic / Self-play |
| Output | `.keras` models + training plots |
| Interface | Tkinter GUI |
| Hardware | CPU-only |

---

## 🚀 Quickstart
```bash
# Setup
python3 -m venv .venv-ttt && source .venv-ttt/bin/activate
pip install -r requirements-bonus-ttt-tf.txt

# Train (choose difficulty)
python bonus/ttt/train.py --opponent easy --episodes 4000

# Play
python bonus/ttt/game.py
```

---

## 🏁 End Result
After training:
- You’ll have a TensorFlow policy that can **intelligently play Tic-Tac-Toe**.  
- You can **visualize its learning curve**.  
- You can **challenge it in real time** via the GUI.
