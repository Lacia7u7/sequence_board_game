# training/algorithms/advanced/fork_threat_policy.py
from __future__ import annotations
from typing import Optional
import numpy as np
from overrides import overrides

from training.algorithms.base_policy import BasePolicy, PolicyCtx
from ...engine.engine_core import BOARD_LAYOUT

DIRS = [(0,1),(1,0),(1,1),(1,-1)]

class ForkThreatPolicy(BasePolicy):
    def __init__(self, env=None):
        self.env = env

    def _in_bounds(self, r, c): return 0 <= r < 10 and 0 <= c < 10

    def _line_len(self, board, team, r, c, dr, dc):
        cnt = 1
        rr, cc = r-dr, c-dc
        while self._in_bounds(rr,cc) and (BOARD_LAYOUT[rr][cc]=="BONUS" or board[rr][cc]==team):
            cnt += 1; rr -= dr; cc -= dc
        rr, cc = r+dr, c+dc
        while self._in_bounds(rr,cc) and (BOARD_LAYOUT[rr][cc]=="BONUS" or board[rr][cc]==team):
            cnt += 1; rr += dr; cc += dc
        return cnt

    def _open_ends(self, board, team, r, c, dr, dc):
        """Count open ends around (r,c) along +/- dir; 'open' = empty or BONUS or our team via BONUS."""
        open_cnt = 0
        rr, cc = r-dr, c-dc
        if self._in_bounds(rr,cc) and (board[rr][cc] == -1 or BOARD_LAYOUT[rr][cc]=="BONUS"):
            open_cnt += 1
        rr, cc = r+dr, c+dc
        if self._in_bounds(rr,cc) and (board[rr][cc] == -1 or BOARD_LAYOUT[rr][cc]=="BONUS"):
            open_cnt += 1
        return open_cnt

    @overrides
    def select_action(self, legal_mask: Optional[np.ndarray], ctx: Optional[PolicyCtx] = None) -> int:
        legal = list(np.flatnonzero(legal_mask > 0.5)) if legal_mask is not None else list(range(self.env.action_space.n))
        if not legal:
            return 0
        board = self.env.game_engine.state.board
        teams = self.env.gconf.teams
        me = self.env.current_player % teams
        opp = (me + 1) % teams if teams > 1 else me

        cells = [a for a in legal if a < 100]
        if not cells: return legal[0]

        best, best_score = cells[0], -1e18

        # 1) Immediate win or block first
        for a in cells:
            r,c = divmod(a,10)
            if self._line_len(board, me, r, c, 0,1) >= 5 or \
               self._line_len(board, me, r, c, 1,0) >= 5 or \
               self._line_len(board, me, r, c, 1,1) >= 5 or \
               self._line_len(board, me, r, c, 1,-1) >= 5:
                return a
        for a in cells:
            r,c = divmod(a,10)
            if self._line_len(board, opp, r, c, 0,1) >= 5 or \
               self._line_len(board, opp, r, c, 1,0) >= 5 or \
               self._line_len(board, opp, r, c, 1,1) >= 5 or \
               self._line_len(board, opp, r, c, 1,-1) >= 5:
                return a

        # 2) Fork creation scoring
        for a in cells:
            r,c = divmod(a,10)
            # place unless it's a removal
            is_remove = (board[r][c] == opp)
            if is_remove:
                # Removals are useful but can't create our fork directly. Give them a smaller score.
                score = 25.0
            else:
                forks = 0
                dir_scores = 0.0
                for dr,dc in DIRS:
                    L = self._line_len(board, me, r, c, dr, dc)
                    if L >= 5:
                        forks += 2  # treat as double threat (already winning)
                    elif L == 4:
                        oe = self._open_ends(board, me, r, c, dr, dc)
                        if oe >= 1:
                            forks += 1
                    dir_scores += (L**2) + 3.0 * max(0, self._open_ends(board, me, r, c, dr, dc)-1)
                score = 1e5*forks + dir_scores - (abs(r-4.5)+abs(c-4.5))

            if score > best_score:
                best_score, best = score, a

        return best
