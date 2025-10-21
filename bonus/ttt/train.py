import os
import argparse
import numpy as np
import tensorflow as tf
from tensorflow.keras import optimizers
import matplotlib.pyplot as plt

from env import TicTacToeEnv
from opponents import random_move, heuristic_move
from model import build_policy

# sampling from masked logits
def masked_sample(logits: np.ndarray, legal):
    masked = logits.copy()
    illegal = [i for i in range(9) if i not in legal]
    masked[illegal] = -1e9
    probs = tf.nn.softmax(masked).numpy()
    probs = probs / probs.sum()
    return int(np.random.choice(9, p=probs))

def select_opponent(name: str):
    name = name.lower()
    if name.startswith("easy"): return "easy"
    if name.startswith("medium"): return "medium"
    return "hard"

def agent_action(model, obs, legal):
    logits = model(obs[None, :], training=False).numpy()[0]
    return masked_sample(logits, legal)

def play_episode(model, opponent="easy", gamma=0.99):
    env = TicTacToeEnv(agent_mark=1)
    obs = env.reset()

    logps = []
    rewards = []

    while True:
        legal = env.legal_actions()
        if env.turn == env.agent_mark:
            # agent turn
            with tf.GradientTape() as tape:
                logits = model(obs[None, :], training=True)[0]
                mask = np.full(9, False)
                mask[legal] = True
                masked = tf.where(mask, logits, tf.fill([9], -1e9))
                probs = tf.nn.softmax(masked)
                dist = tf.random.categorical(tf.math.log(probs[None, :]), 1)
                a = int(dist.numpy()[0,0])
                logp = tf.math.log(tf.clip_by_value(probs[a], 1e-8, 1.0))
            obs_next, r, done, _ = env.step(a)
            logps.append((tape, logp))
            rewards.append(r)
            obs = obs_next
            if done: break
        else:
            # opponent (O)
            if opponent == "easy":
                a = random_move(env.board, legal)
            elif opponent == "medium":
                a = heuristic_move(env.board, legal, mark=-1)
            else:
                # self-play: mirror board and use same policy
                mirror = -env.board.copy().astype(np.float32)
                a = agent_action(model, mirror, legal)
            obs, _, done, _ = env.step(a)
            if done: break

    # discounted returns
    G = 0.0
    returns = []
    for r in reversed(rewards):
        G = r + gamma * G
        returns.append(G)
    returns.reverse()
    if not returns:
        returns = [0.0]
    returns = np.array(returns, dtype=np.float32)
    # normalize
    returns = (returns - returns.mean()) / (returns.std() + 1e-8)

    return logps, returns, env.winner

def train(args):
    os.makedirs(args.outdir, exist_ok=True)
    model = build_policy(hidden=64)
    opt = optimizers.Adam(learning_rate=args.lr)

    opponent = select_opponent(args.opponent)
    print(f"Training vs: {opponent}")

    loss_hist, win_hist, draw_hist, lose_hist = [], [], [], []

    for ep in range(1, args.episodes + 1):
        logps, returns, winner = play_episode(model, opponent=opponent, gamma=args.gamma)

        with tf.GradientTape(persistent=True) as outer_tape:
            # accumulate policy gradient loss
            loss_terms = []
            for (tape, logp), G in zip(logps, returns):
                loss_terms.append(-logp * G)
            if loss_terms:
                loss = tf.reduce_sum(loss_terms)
            else:
                loss = tf.constant(0.0)

        grads = outer_tape.gradient(loss, model.trainable_variables)
        opt.apply_gradients(zip(grads, model.trainable_variables))
        del outer_tape

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
    fig, ax = plt.subplots(1, 2, figsize=(10,4))
    ax[0].plot(loss_hist); ax[0].set_title("Policy Loss"); ax[0].set_xlabel("Episode"); ax[0].set_ylabel("Loss")

    window = max(10, args.log_every)
    def roll(xs):
        if len(xs) < window: return xs
        xs = np.array(xs, dtype=np.float32)
        k = np.convolve(xs, np.ones(window)/window, mode="valid")
        return k
    ax[1].plot(roll(win_hist), label="win")
    ax[1].plot(roll(draw_hist), label="draw")
    ax[1].plot(roll(lose_hist), label="lose")
    ax[1].legend(); ax[1].set_title(f"Outcomes vs {opponent}")
    import matplotlib.pyplot as plt
    plt.tight_layout()
    png_path = os.path.join(args.outdir, f"training_{opponent}.png")
    plt.savefig(png_path, dpi=150)
    print(f"Saved plot: {png_path}")

if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--opponent", default="easy", choices=["easy","medium","hard"])
    ap.add_argument("--episodes", type=int, default=4000)
    ap.add_argument("--gamma", type=float, default=0.99)
    ap.add_argument("--lr", type=float, default=2e-3)
    ap.add_argument("--log-every", type=int, default=200)
    ap.add_argument("--outdir", default="bonus_out_tf")
    args = ap.parse_args()
    train(args)
