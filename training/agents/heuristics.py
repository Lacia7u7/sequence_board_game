
from __future__ import annotations
from typing import Optional, Dict, Any
import numpy as np

from .base_agent import BaseAgent
from ..engine.engine_core import BOARD_LAYOUT

# --- Random ---
class RandomAgent(BaseAgent):
    def __init__(self, env=None):
        self.env = env
    def reset(self, env, seat: int) -> None:
        pass
    def select_action(self, legal_mask: Optional[np.ndarray], ctx: Optional[Dict[str, Any]] = None) -> int:
        legal = np.flatnonzero(legal_mask > 0.5) if legal_mask is not None else np.arange(self.env.action_dim)
        if legal.size == 0:
            return 0
        return int(np.random.choice(legal))

# --- Blocking (prevents opponent 5-in-a-row if possible) ---
_DIRS = [(0,1),(1,0),(1,1),(1,-1)]
class BlockingAgent(BaseAgent):
    def __init__(self, env=None):
        self.env = env
    def reset(self, env, seat: int) -> None:
        pass
    def select_action(self, legal_mask: Optional[np.ndarray], ctx: Optional[Dict[str, Any]] = None) -> int:
        if legal_mask is None:
            return 0
        legal = np.flatnonzero(legal_mask > 0.5)
        if legal.size == 0:
            return 0
        if self.env is None:
            cells = legal[legal < 100]
            return int(cells[0] if cells.size else legal[0])
        state = self.env._state()
        teams = max(1, int(self.env.gconf.teams))
        cur_seat = getattr(state, "current_player", 0)
        my_team = int(cur_seat) % teams
        for a in legal:
            if a >= 100:
                continue
            r, c = divmod(int(a), 10)
            if BOARD_LAYOUT[r][c] == "BONUS":
                continue
            for dr, dc in _DIRS:
                cnt = 1
                rr, cc = r - dr, c - dc
                while 0 <= rr < 10 and 0 <= cc < 10 and (BOARD_LAYOUT[rr][cc] == "BONUS" or state.board[rr][cc] == my_team ^ 1):
                    cnt += 1; rr -= dr; cc -= dc
                rr, cc = r + dr, c + dc
                while 0 <= rr < 10 and 0 <= cc < 10 and (BOARD_LAYOUT[rr][cc] == "BONUS" or state.board[rr][cc] == my_team ^ 1):
                    cnt += 1; rr += dr; cc += dc
                if cnt >= 5:
                    return int(a)
        cells = [int(x) for x in legal if x < 100]
        return int(cells[0]) if cells else int(legal[0])

# --- Greedy (complete our own 5 if possible) ---
class GreedySequenceAgent(BaseAgent):
    def __init__(self, env=None):
        self.env = env
    def reset(self, env, seat: int) -> None:
        pass
    def select_action(self, legal_mask: Optional[np.ndarray], ctx: Optional[Dict[str, Any]] = None) -> int:
        env = self.env
        legal = (
            [i for i, m in enumerate(legal_mask.flatten()) if m > 0.5]
            if legal_mask is not None
            else range(env.action_space.n)
        )
        for act in legal:
            if act < 100:
                r = act // 10; c = act % 10
                team = env.current_player % env.game_config.teams
                for dr, dc in _DIRS:
                    cnt = 1
                    rr, cc = r - dr, c - dc
                    while 0 <= rr < 10 and 0 <= cc < 10 and (BOARD_LAYOUT[rr][cc] == "BONUS" or env.game_engine.state.board[rr][cc] == team):
                        cnt += 1; rr -= dr; cc -= dc
                    rr, cc = r + dr, c + dc
                    while 0 <= rr < 10 and 0 <= cc < 10 and (BOARD_LAYOUT[rr][cc] == "BONUS" or env.game_engine.state.board[rr][cc] == team):
                        cnt += 1; rr += dr; cc += dc
                    if cnt >= 5:
                        return int(act)
        return int(legal[0]) if hasattr(legal, "__len__") and len(legal) else 0
