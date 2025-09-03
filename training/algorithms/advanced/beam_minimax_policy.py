# training/algorithms/advanced/beam_minimax_policy.py
from __future__ import annotations
from typing import Optional, List
import numpy as np
from overrides import overrides

from training.algorithms.base_policy import BasePolicy, PolicyCtx
from ...engine.engine_core import BOARD_LAYOUT
from .pattern_window_policy import PatternWindowPolicy  # reuse scorer

class BeamMinimaxPolicy(BasePolicy):
    def __init__(self, env=None, beam_size: int = 8, alpha: float = 0.7):
        self.env = env
        self.beam_size = beam_size
        self.alpha = alpha
        self.scorer = PatternWindowPolicy(env)

    def _apply(self, board: np.ndarray, r: int, c: int, me: int, opp: int) -> np.ndarray:
        b = board.copy()
        if b[r][c] == opp: b[r][c] = -1
        else: b[r][c] = me
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

        base = self.scorer._score_board(board, me)

        # rank by immediate score
        scored = []
        for a in cells:
            r,c = divmod(a,10)
            b1 = self._apply(board, r, c, me, opp)
            s = self.scorer._score_board(b1, me) - base
            scored.append((s, a))
        scored.sort(reverse=True)
        beam = [a for _,a in scored[:self.beam_size]]

        # minimax on beam
        best_a, best_val = beam[0], -1e18
        for a in beam:
            r,c = divmod(a,10)
            b1 = self._apply(board, r, c, me, opp)
            my_gain = self.scorer._score_board(b1, me) - base

            worst = +1e18
            for b_act in beam:  # opp restricted to same beam set (cheap & good)
                rr,cc = divmod(b_act,10)
                b2 = self._apply(b1, rr, cc, opp, me)
                opp_gain = self.scorer._score_board(b2, opp) - self.scorer._score_board(board, opp)
                worst = min(worst, -opp_gain)
            val = my_gain + self.alpha * worst
            if val > best_val:
                best_val, best_a = val, a

        return best_a
