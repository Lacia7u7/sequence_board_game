from __future__ import annotations
from typing import Optional, List, Tuple
import numpy as np
from overrides import overrides

from ..base_policy import BasePolicy, PolicyCtx
from ...engine.engine_core import BOARD_LAYOUT

_DIRS = [(0,1),(1,0),(1,1),(1,-1)]

class BlockingPolicy(BasePolicy):
    """
    Heuristic policy that tries to block opponent 5s *if env is provided*.
    API: select_action(legal_mask) -> int
    """
    def __init__(self, env=None):
        self.env = env

    @overrides
    def select_action(self, legal_mask: Optional[np.ndarray], ctx: Optional[PolicyCtx] = None) -> int:
        if legal_mask is None:
            return 0
        legal = np.flatnonzero(legal_mask > 0.5)
        if legal.size == 0:
            return 0

        # If no env, just prefer board cell actions.
        if self.env is None:
            cells = legal[legal < 100]
            return int(cells[0] if cells.size else legal[0])

        # With env, scan legal board cells and choose any move that blocks an opponent 5.
        state = self.env._state()
        teams = max(1, int(self.env.gconf.teams))
        cur_seat = getattr(state, "current_player", 0)
        my_team = int(cur_seat) % teams

        for a in legal:
            if a >= 100:
                continue
            r, c = divmod(int(a), 10)
            # if the board cell is occupied or bonus, skip (mask should filter, but be safe)
            if BOARD_LAYOUT[r][c] == "BONUS":
                continue

            # Check if placing here would disrupt an opponent run >=5 (count satisfied around (r,c))
            for opp in range(teams):
                if opp == my_team:
                    continue
                if self._would_block(state.board, r, c, opp):
                    return int(a)

        # Fallback: prefer a board cell, else any legal
        cells = [int(x) for x in legal if x < 100]
        return int(cells[0]) if cells else int(legal[0])

    def _would_block(self, board, r: int, c: int, opp_team: int) -> bool:
        # Simulate that (r,c) becomes NOT opp-team anymore (occupied by me)
        for dr, dc in _DIRS:
            cnt = 1
            rr, cc = r - dr, c - dc
            while 0 <= rr < 10 and 0 <= cc < 10 and (BOARD_LAYOUT[rr][cc] == "BONUS" or board[rr][cc] == opp_team):
                cnt += 1; rr -= dr; cc -= dc
            rr, cc = r + dr, c + dc
            while 0 <= rr < 10 and 0 <= cc < 10 and (BOARD_LAYOUT[rr][cc] == "BONUS" or board[rr][cc] == opp_team):
                cnt += 1; rr += dr; cc += dc
            if cnt >= 5:
                return True
        return False
