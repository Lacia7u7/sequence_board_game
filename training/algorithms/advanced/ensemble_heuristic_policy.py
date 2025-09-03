from __future__ import annotations
from typing import Optional
import numpy as np
from overrides import overrides

from training.algorithms.base_policy import BasePolicy, PolicyCtx
from ...engine.engine_core import BOARD_LAYOUT

class EnsembleHeuristicPolicy(BasePolicy):
    def __init__(self, env=None, w_center=1.0, w_pattern=1.0, w_fork=1.5, w_block=3.0):
        self.env = env
        self.w_center = w_center
        self.w_pattern = w_pattern
        self.w_fork = w_fork
        self.w_block = w_block

    # --- tiny helpers reused from your styles ---
    @staticmethod
    def _manhattan(r, c): return abs(r-4.5) + abs(c-4.5)

    @staticmethod
    def _cell_to_int(v) -> int:
        # Environment may use None for empty; also allow numpy ints
        if v is None:
            return -1
        if isinstance(v, (int, np.integer)):
            return int(v)
        # Fallback: treat anything else as empty
        return -1

    def _line_len(self, board, team, r, c, dr, dc):
        cnt = 1
        rr, cc = r-dr, c-dc
        while 0 <= rr < 10 and 0 <= cc < 10 and (BOARD_LAYOUT[rr][cc] == "BONUS" or self._cell_to_int(board[rr][cc]) == team):
            cnt += 1; rr -= dr; cc -= dc
        rr, cc = r+dr, c+dc
        while 0 <= rr < 10 and 0 <= cc < 10 and (BOARD_LAYOUT[rr][cc] == "BONUS" or self._cell_to_int(board[rr][cc]) == team):
            cnt += 1; rr += dr; cc += dc
        return cnt

    def _completes(self, board, team, r, c):
        for dr,dc in [(0,1),(1,0),(1,1),(1,-1)]:
            if self._line_len(board, team, r, c, dr, dc) >= 5:
                return True
        return False

    def _pattern_gain(self, board, team, r, c, opp):
        # very light-weight pattern delta
        W = [0.0, 2.0, 8.0, 24.0, 160.0, 1e6]

        def eval_board(b, who):
            # Accept list-of-lists or ndarray
            arr = np.asarray(b, dtype=object)
            H, Wd = arr.shape

            # tmp holds ints: team ids, -1 empty, -2 wildcard (BONUS)
            tmp = np.full((H, Wd), -1, dtype=int)
            for i in range(H):
                for j in range(Wd):
                    if BOARD_LAYOUT[i][j] == "BONUS":
                        tmp[i, j] = -2  # wildcard
                    else:
                        tmp[i, j] = self._cell_to_int(arr[i, j])

            def score_line(vals, who_):
                t = sum(1 for x in vals if x == who_ or x == -2)
                o = sum(1 for x in vals if x == (1 - who_))
                if t > 0 and o > 0: return 0.0
                if o > 0: return -W[o]
                return W[t]

            s = 0.0
            # horiz
            for i in range(H):
                for j in range(Wd - 4):
                    s += score_line([tmp[i, j + k] for k in range(5)], who)
            # vert
            for i in range(H - 4):
                for j in range(Wd):
                    s += score_line([tmp[i + k, j] for k in range(5)], who)
            # diag down-right
            for i in range(H - 4):
                for j in range(Wd - 4):
                    s += score_line([tmp[i + k, j + k] for k in range(5)], who)
            # diag up-right
            for i in range(H - 4):
                for j in range(4, Wd):
                    s += score_line([tmp[i + k, j - k] for k in range(5)], who)
            return s

        b0 = eval_board(board, team)

        # create a real independent copy (works for lists or arrays)
        b = np.asarray(board, dtype=object).copy()

        # Robust removal/placement even if board uses None for empty
        bc = self._cell_to_int(b[r, c])
        if bc == opp:
            b[r, c] = -1
        else:
            b[r, c] = team

        return eval_board(b, team) - b0

    def _fork_pressure(self, board, team, r, c, opp):
        forks = 0
        for dr,dc in [(0,1),(1,0),(1,1),(1,-1)]:
            L = self._line_len(board, team, r, c, dr, dc)
            if L >= 5: forks += 2
            elif L == 4: forks += 1
        return forks

    @overrides
    def select_action(self, legal_mask: Optional[np.ndarray], ctx: Optional[PolicyCtx] = None) -> int:
        legal = list(np.flatnonzero(legal_mask > 0.5)) if legal_mask is not None else list(range(self.env.action_space.n))
        if not legal: return 0
        # normalize to numpy array (object dtype so None is preserved)
        board = np.asarray(self.env.game_engine.state.board, dtype=object)
        teams = self.env.gconf.teams
        me = self.env.current_player % teams
        opp = (me + 1) % teams if teams > 1 else me

        cells = [a for a in legal if a < 100]
        if not cells: return legal[0]

        # hard tactics
        for a in cells:
            r,c = divmod(a,10)
            if self._completes(board, me, r, c): return a
        for a in cells:
            r,c = divmod(a,10)
            if self._completes(board, opp, r, c): return a

        best, best_score = cells[0], -1e18
        for a in cells:
            r,c = divmod(a,10)
            center = -self._manhattan(r,c)
            pattern = self._pattern_gain(board, me, r, c, opp)
            fork = self._fork_pressure(board, me, r, c, opp)
            block = 1.0 if self._completes(board, opp, r, c) else 0.0

            score = self.w_center*center + self.w_pattern*pattern + self.w_fork*fork + self.w_block*(1e5*block)
            if score > best_score:
                best_score, best = score, a
        return best
