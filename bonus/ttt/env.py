from __future__ import annotations
from typing import List, Tuple, Optional
import numpy as np

# Board cells: 1 = X, -1 = O, 0 = empty
# Agent is X(+1) by default.

class TicTacToeEnv:
    def __init__(self, agent_mark: int = 1):
        self.agent_mark = 1 if agent_mark not in (-1, 1) else agent_mark
        self.reset()

    def reset(self) -> np.ndarray:
        self.board = np.zeros(9, dtype=np.int8)
        self.turn = 1  # X starts
        self.done = False
        self.winner: Optional[int] = None
        return self._obs()

    def _obs(self) -> np.ndarray:
        # Observation from the agent perspective
        return self.board.copy() if self.agent_mark == 1 else -self.board.copy()

    def legal_actions(self) -> List[int]:
        return [] if self.done else [i for i in range(9) if self.board[i] == 0]

    def step(self, action: int) -> Tuple[np.ndarray, float, bool, dict]:
        if self.done:
            return self._obs(), 0.0, True, {}
        if action not in self.legal_actions():
            # illegal move → end with penalty for current player
            self.done = True
            self.winner = -self.turn
            reward = -1.0 if self.turn == self.agent_mark else 1.0
            return self._obs(), reward, True, {"illegal": True}

        self.board[action] = self.turn
        self._check_terminal()
        if self.done:
            reward = 1.0 if self.winner == self.agent_mark else (-1.0 if self.winner == -self.agent_mark else 0.0)
            return self._obs(), reward, True, {}

        self.turn *= -1
        return self._obs(), 0.0, False, {}

    def _check_terminal(self):
        lines = [
            (0,1,2),(3,4,5),(6,7,8),
            (0,3,6),(1,4,7),(2,5,8),
            (0,4,8),(2,4,6)
        ]
        for a,b,c in lines:
            s = self.board[a] + self.board[b] + self.board[c]
            if s == 3:
                self.done = True; self.winner = 1; return
            if s == -3:
                self.done = True; self.winner = -1; return
        if 0 not in self.board:
            self.done = True; self.winner = None

    def render_str(self) -> str:
        sym = {1:"X", -1:"O", 0:" "}
        rows = []
        for r in range(3):
            rows.append(" | ".join(sym[int(self.board[3*r+c])] for c in range(3)))
        return "\n---------\n".join(rows)
