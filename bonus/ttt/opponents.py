import numpy as np
from typing import List

def random_move(board: np.ndarray, legal: List[int]) -> int:
    return int(np.random.choice(legal))

def heuristic_move(board: np.ndarray, legal: List[int], mark: int) -> int:
    # 1) Win, 2) Block, 3) Center, 4) Corner, 5) Random
    lines = [
        (0,1,2),(3,4,5),(6,7,8),
        (0,3,6),(1,4,7),(2,5,8),
        (0,4,8),(2,4,6)
    ]
    # win
    for a,b,c in lines:
        idxs = [a,b,c]; vals = board[idxs]
        if (vals == mark).sum() == 2 and (vals == 0).sum() == 1:
            return idxs[int(np.where(vals == 0)[0][0])]
    # block
    for a,b,c in lines:
        idxs = [a,b,c]; vals = board[idxs]
        if (vals == -mark).sum() == 2 and (vals == 0).sum() == 1:
            return idxs[int(np.where(vals == 0)[0][0])]
    # center
    if 4 in legal: return 4
    # corners
    for i in [0,2,6,8]:
        if i in legal: return i
    # random
    return int(np.random.choice(legal))
