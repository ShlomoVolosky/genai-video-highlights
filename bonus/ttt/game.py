import os
import tkinter as tk
from tkinter import ttk, messagebox
import numpy as np
import tensorflow as tf

from env import TicTacToeEnv
from model import build_policy
from opponents import random_move, heuristic_move

def masked_choice(logits, legal):
    masked = logits.copy()
    illegal = [i for i in range(9) if i not in legal]
    masked[illegal] = -1e9
    probs = tf.nn.softmax(masked).numpy()
    probs = probs / probs.sum()
    return int(np.random.choice(9, p=probs))

class GUI:
    def __init__(self, root):
        self.root = root
        self.root.title("Tic-Tac-Toe — TensorFlow Agent")
        self.env = TicTacToeEnv(agent_mark=1)
        self.model = build_policy()  # will be overwritten on load
        self.difficulty = tk.StringVar(value="easy")

        frm = ttk.Frame(root, padding=10)
        frm.grid()

        ttk.Label(frm, text="Difficulty:").grid(column=0, row=0, sticky="w")
        ttk.OptionMenu(frm, self.difficulty, "easy", "easy", "medium", "hard").grid(column=1, row=0, sticky="w")

        ttk.Button(frm, text="Load Model", command=self.load_model).grid(column=2, row=0, padx=8)
        ttk.Button(frm, text="Reset", command=self.reset).grid(column=3, row=0)

        self.buttons = []
        grid = ttk.Frame(frm, padding=(0,10))
        grid.grid(column=0, row=1, columnspan=4)
        for r in range(3):
            for c in range(3):
                b = ttk.Button(grid, text=" ", width=4, command=lambda i=3*r+c: self.human_move(i))
                b.grid(row=r, column=c, ipadx=10, ipady=10, padx=2, pady=2)
                self.buttons.append(b)

        self.status = tk.StringVar(value="AI starts (X). You are O.")
        ttk.Label(frm, textvariable=self.status).grid(column=0, row=2, columnspan=4, sticky="w")

        self.ai_move_if_needed()

    def load_model(self):
        opponent = self.difficulty.get()
        path = f"bonus_out/ttt_policy_{opponent}.keras"
        if not os.path.exists(path):
            messagebox.showinfo("Info", f"Model not found: {path}\nTrain first with:\npython bonus/ttt/train.py --opponent {opponent}")
            return
        self.model = tf.keras.models.load_model(path)
        messagebox.showinfo("Loaded", f"Loaded model: {path}")

    def reset(self):
        self.env.reset()
        for b in self.buttons:
            b.config(text=" ", state="normal")
        self.status.set("AI starts (X). You are O.")
        self.ai_move_if_needed()

    def human_move(self, idx):
        if self.env.done: return
        if self.env.turn != -1: return
        if idx not in self.env.legal_actions(): return
        self.env.step(idx)
        self.refresh()
        self.ai_move_if_needed()

    def ai_move_if_needed(self):
        if self.env.done: return
        if self.env.turn != 1: return
        legal = self.env.legal_actions()
        if not legal: return

        # Prefer loaded model if available; else heuristic/random baseline
        try:
            logits = self.model(self.env._obs()[None, :], training=False).numpy()[0]
            a = masked_choice(logits, legal)
        except Exception:
            op = self.difficulty.get()
            a = heuristic_move(self.env.board, legal, mark=1) if op == "medium" else random_move(self.env.board, legal)
        self.env.step(a)
        self.refresh()

    def refresh(self):
        sym = {1:"X", -1:"O", 0:" "}
        for i, b in enumerate(self.buttons):
            b.config(text=sym[int(self.env.board[i])])
            b.config(state="disabled" if self.env.board[i] != 0 or self.env.done else "normal")
        if self.env.done:
            if self.env.winner == 1: self.status.set("AI (X) wins!")
            elif self.env.winner == -1: self.status.set("You (O) win!")
            else: self.status.set("Draw.")
        else:
            self.status.set("Your turn." if self.env.turn == -1 else "AI thinking...")

if __name__ == "__main__":
    root = tk.Tk()
    GUI(root)
    root.mainloop()
