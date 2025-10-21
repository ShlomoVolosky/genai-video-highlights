# bonus/ttt/train.py
import os
import argparse
import numpy as np
import tensorflow as tf
from tensorflow.keras import optimizers
import matplotlib.pyplot as plt

from env import TicTacToeEnv
from opponents import random_move, heuristic_move
from model import build_policy

# -------- helpers --------

def select_opponent(name: str) -> str:
    n = name.lower()
    if n.startswith("easy"): return "easy"
    if n.startswith("medium"): return "medium"
    return "hard"

def masked_softmax(logits: tf.Tensor, legal_mask: tf.Tensor) -> tf.Tensor:
    """legal_mask: shape (9,) bool; returns (9,) probs."""
    minus_inf = tf.fill(tf.shape(logits), tf.constant(-1e9, dtype=logits.dtype))
    masked = tf.where(legal_mask, logits, minus_inf)
    return tf.nn.softmax(masked)

def masked_sample_np(logits_np: np.ndarray, legal_idxs: list[int]) -> int:
    masked = logits_np.copy()
    illegal = [i for i in range(9) if i not in legal_idxs]
    masked[illegal] = -1e9
    probs = tf.nn.softmax(masked).numpy()
    probs = probs / probs.sum()
    return int(np.random.choice(9, p=probs))

# -------- one episode rollout (collect trajectory) --------

def play_episode_collect(model: tf.keras.Model, opponent: str):
    """
    Rollout one game, collecting:
      obs_list: (T, 9) float32
      mask_list: (T, 9) bool
      act_list: (T,) int
      rew_list: python list of float rewards (only on agent turns)
      winner: 1 (agent X), -1 (opponent O), or None
    Agent plays X (+1). Opponent plays O (-1).
    """
    env = TicTacToeEnv(agent_mark=1)
    obs = env.reset().astype(np.float32)

    obs_list, mask_list, act_list, rew_list = [], [], [], []

    while True:
        legal = env.legal_actions()
        if env.turn == env.agent_mark:
            # Agent (X) turn
            logits = model(obs[None, :], training=False).numpy()[0]
            a = masked_sample_np(logits, legal)
            obs_next, r, done, _ = env.step(a)
            # record transition
            obs_list.append(obs.copy())
            mask = np.zeros(9, dtype=bool); mask[legal] = True
            mask_list.append(mask)
            act_list.append(a)
            rew_list.append(r)
            obs = obs_next.astype(np.float32)
            if done: break
        else:
            # Opponent (O) turn
            if opponent == "easy":
                a = random_move(env.board, legal)
            elif opponent == "medium":
                a = heuristic_move(env.board, legal, mark=-1)
            else:
                # self-play: mirror board and use same policy to pick
                mirror = (-env.board).astype(np.float32)
                logits = model(mirror[None, :], training=False).numpy()[0]
                a = masked_sample_np(logits, legal)
            obs, _, done, _ = env.step(a)
            obs = obs.astype(np.float32)
            if done: break

    return (
        np.array(obs_list, dtype=np.float32),
        np.array(mask_list, dtype=bool),
        np.array(act_list, dtype=np.int32),
        rew_list,
        env.winner,
    )

def discounted_returns(rews, gamma: float) -> np.ndarray:
    G = 0.0
    out = []
    for r in reversed(rews):
        G = r + gamma * G
        out.append(G)
    out.reverse()
    if not out:
        out = [0.0]
    return np.asarray(out, dtype=np.float32)

# -------- training loop --------

def train(args):
    os.makedirs(args.outdir, exist_ok=True)
    model = build_policy(hidden=64)
    opt = optimizers.Adam(learning_rate=args.lr)

    opponent = select_opponent(args.opponent)
    print(f"Training vs: {opponent}")

    loss_hist, win_hist, draw_hist, lose_hist = [], [], [], []

    for ep in range(1, args.episodes + 1):
        obs_arr, mask_arr, act_arr, rew_list, winner = play_episode_collect(model, opponent)

        # compute normalized returns
        returns = discounted_returns(rew_list, gamma=args.gamma)
        # normalize for variance reduction
        returns = (returns - returns.mean()) / (returns.std() + 1e-8)

        with tf.GradientTape() as tape:
            # recompute logits for each step and build REINFORCE loss
            logits = model(obs_arr, training=True)  # (T,9)
            # apply masks step-wise
            masks_tf = tf.convert_to_tensor(mask_arr, dtype=tf.bool)   # (T,9)
            masked = tf.where(masks_tf, logits, tf.fill(tf.shape(logits), tf.constant(-1e9, logits.dtype)))
            probs = tf.nn.softmax(masked, axis=-1)                     # (T,9)

            # gather chosen probs
            idx = tf.stack([tf.range(tf.shape(probs)[0]), tf.convert_to_tensor(act_arr, dtype=tf.int32)], axis=1)
            chosen_probs = tf.gather_nd(probs, idx)                    # (T,)
            logp = tf.math.log(tf.clip_by_value(chosen_probs, 1e-8, 1.0))

            G = tf.convert_to_tensor(returns, dtype=tf.float32)        # (T,)
            loss = -tf.reduce_sum(logp * G)

        grads = tape.gradient(loss, model.trainable_variables)
        opt.apply_gradients(zip(grads, model.trainable_variables))

        loss_hist.append(float(loss.numpy()))
        win_hist.append(1 if winner == 1 else 0)
        draw_hist.append(1 if winner is None else 0)
        lose_hist.append(1 if winner == -1 else 0)

        if ep % args.log_every == 0:
            w = 100 * np.mean(win_hist[-args.log_every:])
            d = 100 * np.mean(draw_hist[-args.log_every:])
            l = 100 * np.mean(lose_hist[-args.log_every:])
            print(f"[{ep:5d}] loss={loss_hist[-1]:.4f}  win/draw/lose (last {args.log_every}): {w:.1f}/{d:.1f}/{l:.1f}%")

    # save model
    model_path = os.path.join(args.outdir, f"ttt_policy_{opponent}.keras")
    model.save(model_path)
    print(f"Saved: {model_path}")

    # plots
    fig, ax = plt.subplots(1, 2, figsize=(10, 4))
    ax[0].plot(loss_hist); ax[0].set_title("Policy Loss"); ax[0].set_xlabel("Episode"); ax[0].set_ylabel("Loss")
    window = max(10, args.log_every)
    def roll(xs):
        xs = np.asarray(xs, dtype=np.float32)
        if xs.size < window: return xs
        return np.convolve(xs, np.ones(window)/window, mode="valid")
    ax[1].plot(roll(win_hist), label="win")
    ax[1].plot(roll(draw_hist), label="draw")
    ax[1].plot(roll(lose_hist), label="lose")
    ax[1].legend(); ax[1].set_title(f"Outcomes vs {opponent}")
    plt.tight_layout()
    png_path = os.path.join(args.outdir, f"training_{opponent}.png")
    plt.savefig(png_path, dpi=150)
    print(f"Saved plot: {png_path}")

if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--opponent", default="easy", choices=["easy", "medium", "hard"])
    ap.add_argument("--episodes", type=int, default=4000)
    ap.add_argument("--gamma", type=float, default=0.99)
    ap.add_argument("--lr", type=float, default=2e-3)
    ap.add_argument("--log-every", type=int, default=200)
    ap.add_argument("--outdir", default="bonus_out_tf")
    args = ap.parse_args()
    train(args)
