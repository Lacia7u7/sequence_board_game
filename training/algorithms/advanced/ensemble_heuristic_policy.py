# training/algorithms/advanced/ensemble_heuristic_policy.py
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

    def _line_len(self, board, team, r, c, dr, dc):
        cnt = 1
        rr, cc = r-dr, c-dc
        while 0 <= rr < 10 and 0 <= cc < 10 and (BOARD_LAYOUT[rr][cc] == "BONUS" or board[rr][cc] == team):
            cnt += 1; rr -= dr; cc -= dc
        rr, cc = r+dr, c+dc
        while 0 <= rr < 10 and 0 <= cc < 10 and (BOARD_LAYOUT[rr][cc] == "BONUS" or board[rr][cc] == team):
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
            # scan 5-windows, BONUS acts as wildcard
            tmp = np.full((10,10), -1, dtype=int)
            for i in range(10):
                for j in range(10):
                    tmp[i][j] = -2 if BOARD_LAYOUT[i][j]=="BONUS" else b[i][j]
            def score_line(vals, who):
                t = sum(1 for x in vals if x == who or x == -2)
                o = sum(1 for x in vals if x == (1-who))
                if t>0 and o>0: return 0.0
                if o>0: return -W[o]
                return W[t]
            s = 0.0
            for i in range(10):
                for j in range(6):
                    s += score_line([tmp[i][j+k] for k in range(5)], who)
            for i in range(6):
                for j in range(10):
                    s += score_line([tmp[i+k][j] for k in range(5)], who)
            for i in range(6):
                for j in range(6):
                    s += score_line([tmp[i+k][j+k] for k in range(5)], who)
            for i in range(6):
                for j in range(4,10):
                    s += score_line([tmp[i+k][j-k] for k in range(5)], who)
            return s
        b0 = eval_board(board, team)
        b = board.copy()
        if b[r][c] == opp: b[r][c] = -1
        else: b[r][c] = team
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
        board = self.env.game_engine.state.board
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
