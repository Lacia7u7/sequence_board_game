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
    @staticmethod
    def _cell_to_int(v: Optional[object]) -> int:
        # Normalize board cell to int code:
        #  -1 empty, 0..N-1 team ids, -2 wildcard (handled separately)
        if v is None:
            return -1
        if isinstance(v, (int, np.integer)):
            return int(v)
        return -1  # fallback: treat as empty

    @staticmethod
    def _as_array(board):
        # Accept list-of-lists or ndarray; keep None by using object dtype
        return np.asarray(board, dtype=object)

    def _score_board(self, board, team: int) -> float:
        # Accept list-of-lists or ndarray
        arr = self._as_array(board)
        H, Wd = arr.shape

        def score_window(vals, who):
            # vals contain ints: -2 wildcard, -1 empty, >=0 team ids
            t = sum(1 for x in vals if x == who or x == -2)
            # Count ANY opponent (works for >2 teams)
            o = sum(1 for x in vals if x >= 0 and x != who)
            if t > 0 and o > 0:
                return 0.0
            if o > 0:
                # penalize windows that contain only opponents (+ wildcards/empties)
                # o is number of pure-opponent chips in the 5-window
                return -self.W[min(o, 5)]
            return self.W[min(t, 5)]

        # Build tmp grid of ints (no None): -2 wildcard, -1 empty, 0.. team ids
        tmp = np.full((H, Wd), -1, dtype=int)
        for r in range(H):
            for c in range(Wd):
                if BOARD_LAYOUT[r][c] == "BONUS":
                    tmp[r, c] = -2
                else:
                    tmp[r, c] = self._cell_to_int(arr[r, c])

        val = 0.0
        # horiz
        for r in range(H):
            for c in range(Wd - 4):
                w = [tmp[r, c + i] for i in range(5)]
                val += score_window(w, team)
        # vert
        for r in range(H - 4):
            for c in range(Wd):
                w = [tmp[r + i, c] for i in range(5)]
                val += score_window(w, team)
        # diag down-right
        for r in range(H - 4):
            for c in range(Wd - 4):
                w = [tmp[r + i, c + i] for i in range(5)]
                val += score_window(w, team)
        # diag up-right
        for r in range(H - 4):
            for c in range(4, Wd):
                w = [tmp[r + i, c - i] for i in range(5)]
                val += score_window(w, team)

        return val

    def _apply(self, board, r: int, c: int, me: int, opp: int):
        b = self._as_array(board).copy()
        bc = self._cell_to_int(b[r, c])
        if bc == opp:
            b[r, c] = -1
        else:
            b[r, c] = me
        return b


    @overrides
    def select_action(self, legal_mask: Optional[np.ndarray], ctx: Optional[PolicyCtx] = None) -> int:
        legal = list(np.flatnonzero(legal_mask > 0.5)) if legal_mask is not None else list(range(self.env.action_space.n))
        if not legal: return 0

        board = self._as_array(self.env.game_engine.state.board)

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
