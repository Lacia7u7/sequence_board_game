# training/algorithms/advanced/threat_aware_minimax_policy.py
from __future__ import annotations
from typing import Optional, List
import numpy as np
from overrides import overrides

from training.algorithms.base_policy import BasePolicy, PolicyCtx
from ...engine.engine_core import BOARD_LAYOUT

DIRS = [(0, 1), (1, 0), (1, 1), (1, -1)]


# --------- helpers (robust to list-of-lists or numpy boards) ----------------

def _safe_copy_board(board):
    """Deep copy that works for both numpy arrays and Python list-of-lists."""
    if isinstance(board, np.ndarray):
        return board.copy()
    # list-of-lists: copy each row
    return [row[:] for row in board]


def _cell_to_int(x: object) -> int:
    """Env uses None for empty; normalize to int domain: empty=-1, BONUS is handled elsewhere."""
    return -1 if x is None else int(x)


# -----------------------------------------------------------------------------

class _BoardUtilsMixin:
    @staticmethod
    def in_bounds(r: int, c: int) -> bool:
        return 0 <= r < 10 and 0 <= c < 10

    @staticmethod
    def line_len(board: np.ndarray, team: int, r: int, c: int, dr: int, dc: int) -> int:
        """Count contiguous stones including (r,c) in direction ±(dr,dc). BONUS acts as wildcard."""
        cnt = 1
        rr, cc = r - dr, c - dc
        while 0 <= rr < 10 and 0 <= cc < 10 and (BOARD_LAYOUT[rr][cc] == "BONUS" or board[rr][cc] == team):
            cnt += 1
            rr -= dr
            cc -= dc
        rr, cc = r + dr, c + dc
        while 0 <= rr < 10 and 0 <= cc < 10 and (BOARD_LAYOUT[rr][cc] == "BONUS" or board[rr][cc] == team):
            cnt += 1
            rr += dr
            cc += dc
        return cnt

    @staticmethod
    def completes_five(board: np.ndarray, team: int, r: int, c: int) -> bool:
        for dr, dc in DIRS:
            if _BoardUtilsMixin.line_len(board, team, r, c, dr, dc) >= 5:
                return True
        return False

    @staticmethod
    def simple_eval(board: np.ndarray, team: int) -> float:
        """
        Fast 5-window scoring (team positive, opponent negative).
        Uses an int buffer with: -1 empty, team ids as-is, and -2 to mark BONUS cells (wildcard).
        """
        W = [0.0, 1.0, 4.0, 12.0, 80.0, 1e6]  # contribution for 0..5 friendly stones in a window

        def score_line(seq, who):
            t = sum(1 for x in seq if x == who or x == -2)     # count friendly or BONUS
            o = sum(1 for x in seq if x == (1 - who))          # count pure opponent stones
            if t > 0 and o > 0:
                return 0.0     # mixed window -> dead
            if o > 0:
                return -W[o]   # opponent-only window penalizes us
            return W[t]        # friendly-only window rewards us

        # Build int tmp grid (no None values)
        tmp = np.full((10, 10), -1, dtype=int)
        for r in range(10):
            for c in range(10):
                if BOARD_LAYOUT[r][c] == "BONUS":
                    tmp[r, c] = -2
                else:
                    tmp[r, c] = _cell_to_int(board[r, c])

        val = 0.0
        # horiz
        for r in range(10):
            for c in range(6):
                val += score_line(tmp[r, c:c + 5], team)
        # vert
        for r in range(6):
            for c in range(10):
                val += score_line(tmp[r:r + 5, c], team)
        # diag down-right
        for r in range(6):
            for c in range(6):
                val += score_line([tmp[r + i, c + i] for i in range(5)], team)
        # diag up-right
        for r in range(6):
            for c in range(4, 10):
                val += score_line([tmp[r + i, c - i] for i in range(5)], team)
        return val

    @staticmethod
    def apply_hypo(board: np.ndarray, r: int, c: int, actor: int, opp: int) -> np.ndarray:
        """
        Return a NEW board with the hypothetical move applied.
        - If target cell holds 'opp' => simulate removal (as if using a cut jack).
        - Else => place actor's chip.
        Works regardless of original board being ndarray or list-of-lists.
        """
        b = _safe_copy_board(board)
        if not isinstance(b, np.ndarray):
            b = np.array(b, dtype=object, copy=False)  # keep object dtype (env stores ints/None)
        if b[r][c] == opp:
            b[r][c] = -1
        else:
            b[r][c] = actor
        return b


class ThreatAwareMinimaxPolicy(BasePolicy, _BoardUtilsMixin):
    """
    2-ply lookahead with simple evaluation and hard priorities:
    - if we can make 5, do it
    - else if we can block opp's 5 by moving on that cell, do it
    - else choose argmax over (my_eval_after - alpha * opp_best_eval_after)
      (opponent replies approximated from current legal cells, skipping exact same cell)
    """
    def __init__(self, env=None, alpha: float = 0.8):
        self.env = env
        self.alpha = alpha

    @overrides
    def select_action(self, legal_mask: Optional[np.ndarray], ctx: Optional[PolicyCtx] = None) -> int:
        # Gather legal actions
        if legal_mask is None:
            legal = list(range(self.env.action_space.n))
        else:
            legal = list(np.flatnonzero(legal_mask > 0.5))
        if not legal:
            return 0

        # Normalize board into a SAFE local ndarray (no in-place mutations of env state)
        board_raw = self.env.game_engine.state.board          # likely list-of-lists with ints/None
        board = np.array(board_raw, dtype=object, copy=True)  # local working copy

        teams = self.env.gconf.teams
        me = self.env.current_player % teams
        opp = (me + 1) % teams if teams > 1 else me

        # Prefer board cells over special actions (>= 100)
        cells = [a for a in legal if a < 100]
        fallback = legal[0]

        # --- Hard tactical priorities ---
        for a in cells:
            r, c = divmod(a, 10)
            if self.completes_five(board, me, r, c):
                return a  # immediate win

        for a in cells:
            r, c = divmod(a, 10)
            if self.completes_five(board, opp, r, c):
                return a  # immediate block

        # --- Minimax (1-ply) with simple eval + threat damping ---
        base_score = self.simple_eval(board, me)
        best_act, best_val = None, -1e18

        # If no cell actions, just return fallback (e.g., burn/pass decisions out of scope here)
        search_actions = cells or legal
        for a in search_actions:
            if a >= 100:
                continue  # skip non-cell actions in this policy's lookahead
            r, c = divmod(a, 10)

            # My hypothetical move
            b1 = self.apply_hypo(board, r, c, me, opp)
            my_score = self.simple_eval(b1, me) - base_score

            # Opponent best reply (approx from current cell set; skip same cell so we don't “insta-undo”)
            worst = +1e18
            for b_act in cells:
                if b_act == a:
                    continue
                rr, cc = divmod(b_act, 10)
                b2 = self.apply_hypo(b1, rr, cc, opp, me)
                opp_score = self.simple_eval(b2, opp)
                worst = min(worst, -opp_score)  # opponent maximizes their eval; minimize our perspective

            val = my_score + self.alpha * worst
            if val > best_val:
                best_val, best_act = val, a

        return best_act if best_act is not None else (cells[0] if cells else fallback)
