# training/algorithms/advanced/beam_minimax_policy.py
from __future__ import annotations
from typing import Optional
import numpy as np
from overrides import overrides

from training.algorithms.base_policy import BasePolicy, PolicyCtx
from .pattern_window_policy import PatternWindowPolicy  # reuse scorer


class BeamMinimaxPolicy(BasePolicy):
    def __init__(self, env=None, beam_size: int = 8, alpha: float = 0.7):
        """
        beam_size: how many top-scoring candidate moves to explore for each ply
        alpha: weight on the opponent's best reply (0=greedy, 1=full worst-case within beam)
        """
        self.env = env
        self.beam_size = max(int(beam_size), 0)
        self.alpha = float(alpha)
        self.scorer = PatternWindowPolicy(env)

    @staticmethod
    def _apply(board: np.ndarray, r: int, c: int, me: int, opp: int) -> np.ndarray:
        """
        Returns a NEW board after 'me' plays at (r, c).
        If the target cell currently holds 'opp', set to -1 (capture/block),
        else set to 'me'. (Preserves your original rule.)
        """
        b = board.copy()  # for NumPy, this is a new array with its own data buffer
        if b[r, c] == opp:
            b[r, c] = -1
        else:
            b[r, c] = me
        return b

    @overrides
    def select_action(self, legal_mask: Optional[np.ndarray], ctx: Optional[PolicyCtx] = None) -> int:
        # Resolve legal actions
        if legal_mask is not None:
            legal = list(np.flatnonzero(legal_mask > 0.5))
        else:
            # Fallback: allow up to the game action space, then filter to board size below
            legal = list(range(self.env.action_space.n))

        if not legal:
            return 0  # safe fallback

        board = np.asarray(self.env.game_engine.state.board, dtype=object)
        H, W = board.shape
        board_size = H * W

        teams = getattr(self.env.gconf, "teams", 2)
        me = self.env.current_player % teams
        opp = (me + 1) % teams if teams > 1 else me

        # Only consider in-board cell actions
        cells = [a for a in legal if 0 <= a < board_size]
        if not cells:
            # If nothing maps onto the board grid, just return the first legal action as a fallback
            return legal[0]

        base_me = self.scorer._score_board(board, me)

        # --- Build my beam by immediate delta ---
        scored_my = []
        for a in cells:
            r, c = divmod(a, W)
            b1 = self._apply(board, r, c, me, opp)
            my_gain = self.scorer._score_board(b1, me) - base_me
            scored_my.append((my_gain, a))
        scored_my.sort(reverse=True, key=lambda t: t[0])

        if self.beam_size == 0:
            # Degenerate case: act greedily by immediate score
            return scored_my[0][1]

        my_beam = [a for _, a in scored_my[: self.beam_size]]

        # --- Minimax within the beam ---
        best_a, best_val = my_beam[0], -1e18
        for a in my_beam:
            r, c = divmod(a, W)
            b1 = self._apply(board, r, c, me, opp)

            # my gain relative to the original position
            my_gain = self.scorer._score_board(b1, me) - base_me

            # Opponent considers a reply from *this* position; baseline for opp is after my move
            base_opp_after_my = self.scorer._score_board(b1, opp)

            # Cheap opponent beam: evaluate all board cells (filter to grid);
            # if you have fast access to the opponent's legal mask here, use it instead.
            opp_scored = []
            for b_idx in range(board_size):
                rr, cc = divmod(b_idx, W)
                b2 = self._apply(b1, rr, cc, opp, me)
                opp_gain = self.scorer._score_board(b2, opp) - base_opp_after_my
                opp_scored.append((opp_gain, b_idx))
            opp_scored.sort(reverse=True, key=lambda t: t[0])
            opp_beam = [idx for _, idx in opp_scored[: self.beam_size]] or [opp_scored[0][1]]

            # Worst case for me is the opponent move that maximizes their gain
            # -> my "loss" term is negative of opp_gain; we take the minimum over replies
            worst_for_me = +1e18
            for b_idx in opp_beam:
                rr, cc = divmod(b_idx, W)
                b2 = self._apply(b1, rr, cc, opp, me)
                opp_gain = self.scorer._score_board(b2, opp) - base_opp_after_my
                worst_for_me = min(worst_for_me, -opp_gain)

            val = my_gain + self.alpha * worst_for_me
            if val > best_val:
                best_val, best_a = val, a

        return best_a
