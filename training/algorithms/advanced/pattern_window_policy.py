# training/algorithms/advanced/pattern_window_policy.py
from __future__ import annotations
from typing import Optional
import numpy as np
from overrides import overrides

from training.algorithms.base_policy import BasePolicy, PolicyCtx
from ...engine.engine_core import BOARD_LAYOUT

class PatternWindowPolicy(BasePolicy):
    def __init__(self, env=None):
        self.env = env
        # pattern weights (open-ness partially captured by "no-mix" rule)
        self.W = [0.0, 2.0, 8.0, 24.0, 160.0, 1e6]

    def _score_board(self, board: np.ndarray, team: int) -> float:
        opp = 1 - team
        def score_window(vals, who):
            t = sum(1 for x in vals if x == who or x == -2)
            o = sum(1 for x in vals if x == (1-who))
            if t > 0 and o > 0:
                return 0.0
            if o > 0:
                return -self.W[o]
            return self.W[t]

        tmp = np.full((10,10), -1, dtype=int)
        for r in range(10):
            for c in range(10):
                tmp[r][c] = -2 if BOARD_LAYOUT[r][c]=="BONUS" else board[r][c]

        val = 0.0
        # horiz
        for r in range(10):
            for c in range(6):
                w = [tmp[r][c+i] for i in range(5)]
                val += score_window(w, team)
        # vert
        for r in range(6):
            for c in range(10):
                w = [tmp[r+i][c] for i in range(5)]
                val += score_window(w, team)
        # diag
        for r in range(6):
            for c in range(6):
                w = [tmp[r+i][c+i] for i in range(5)]
                val += score_window(w, team)
        for r in range(6):
            for c in range(4,10):
                w = [tmp[r+i][c-i] for i in range(5)]
                val += score_window(w, team)
        return val

    def _apply(self, board: np.ndarray, r: int, c: int, me: int, opp: int) -> np.ndarray:
        b = board.copy()
        if b[r][c] == opp:
            b[r][c] = -1
        else:
            b[r][c] = me
        return b

    @overrides
    def select_action(self, legal_mask: Optional[np.ndarray], ctx: Optional[PolicyCtx] = None) -> int:
        legal = list(np.flatnonzero(legal_mask > 0.5)) if legal_mask is not None else list(range(self.env.action_space.n))
        if not legal: return 0

        board = self.env.game_engine.state.board
        teams = self.env.gconf.teams
        me = self.env.current_player % teams
        opp = (me + 1) % teams if teams > 1 else me

        cells = [a for a in legal if a < 100]
        if not cells: return legal[0]

        base = self._score_board(board, me)

        # Immediate priorities
        for a in cells:
            r,c = divmod(a,10)
            b1 = self._apply(board, r, c, me, opp)
            if self._score_board(b1, me) >= 1e6/2:
                return a

        best, best_gain = cells[0], -1e18
        for a in cells:
            r,c = divmod(a,10)
            b1 = self._apply(board, r, c, me, opp)
            gain = self._score_board(b1, me) - base
            # slight centrality tie-break
            gain -= (abs(r-4.5)+abs(c-4.5))*0.5
            if gain > best_gain:
                best_gain, best = gain, a
        return best
