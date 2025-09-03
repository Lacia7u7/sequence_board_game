from __future__ import annotations
from typing import Optional, Tuple, List
import numpy as np
from overrides import overrides

from training.algorithms.base_policy import BasePolicy, PolicyCtx
from ...engine.engine_core import BOARD_LAYOUT


class BonusProximityPolicy(BasePolicy):
    """
    Heurística centrada en BONUS:
      - Prioriza celdas adyacentes a BONUS (Chebyshev dist <= 1) y, un poco menos, dentro de radio 2.
      - Bonifica rachas que pasan por al menos un BONUS (BONUS cuenta como comodín).
      - Mantiene prioridades tácticas: ganar ya, o bloquear 5 inmediato del rival.
      - Robusta a 'None' o -1 como vacío.
    """

    def __init__(
        self,
        env=None,
        w_bonus_adj: float = 12.0,   # adyacencia directa (dist<=1)
        w_bonus_r2: float = 4.0,     # vecindario ampliado (dist<=2)
        w_bonus_line: float = 10.0,  # línea que toca un BONUS
        w_open4: float = 45.0,
        w_maxrun: float = 5.0,
        w_center_tb: float = -0.75,  # desempate suave por centralidad
    ):
        self.env = env
        self.w_bonus_adj = w_bonus_adj
        self.w_bonus_r2 = w_bonus_r2
        self.w_bonus_line = w_bonus_line
        self.w_open4 = w_open4
        self.w_maxrun = w_maxrun
        self.w_center_tb = w_center_tb

        # Dimensiones y lista de BONUS
        self.H = len(BOARD_LAYOUT)
        self.W = len(BOARD_LAYOUT[0]) if self.H > 0 else 0
        self._bonus_cells: List[Tuple[int, int]] = [
            (r, c) for r in range(self.H) for c in range(self.W) if BOARD_LAYOUT[r][c] == "BONUS"
        ]

        # Direcciones principales
        self.DIRS = [(0, 1), (1, 0), (1, 1), (1, -1)]

    # -------- Utils --------
    def _in_bounds(self, r: int, c: int) -> bool:
        return 0 <= r < self.H and 0 <= c < self.W

    @staticmethod
    def _cell_to_int(v) -> int:
        # Entornos que usan None para vacío; aceptar numpy ints también
        if v is None:
            return -1
        if isinstance(v, (int, np.integer)):
            return int(v)
        return -1

    @staticmethod
    def _manhattan(r: int, c: int, center: Tuple[float, float] = (4.5, 4.5)) -> float:
        cr, cc = center
        return abs(r - cr) + abs(c - cc)

    @staticmethod
    def _chebyshev(r1: int, c1: int, r2: int, c2: int) -> int:
        return max(abs(r1 - r2), abs(c1 - c2))

    def _near_bonus_features(self, r: int, c: int) -> Tuple[int, int]:
        """
        Devuelve (#BONUS con dist<=1, #BONUS con 1<dist<=2) usando distancia Chebyshev.
        """
        if not self._bonus_cells:
            return 0, 0
        adj = 0
        r2 = 0
        for br, bc in self._bonus_cells:
            d = self._chebyshev(r, c, br, bc)
            if d <= 1:
                adj += 1
            elif d <= 2:
                r2 += 1
        return adj, r2

    def _line_features(self, board: np.ndarray, team: int, r: int, c: int, dr: int, dc: int) -> Tuple[int, bool]:
        """
        Devuelve (longitud de racha hipotética con comodín BONUS, toca_bonus?)
        contando desde (r,c) en ±(dr,dc) si colocáramos ahí.
        """
        cnt = 1
        touches_bonus = (BOARD_LAYOUT[r][c] == "BONUS")
        # atrás
        rr, cc = r - dr, c - dc
        while self._in_bounds(rr, cc) and (BOARD_LAYOUT[rr][cc] == "BONUS" or self._cell_to_int(board[rr][cc]) == team):
            cnt += 1
            touches_bonus = touches_bonus or (BOARD_LAYOUT[rr][cc] == "BONUS")
            rr -= dr; cc -= dc
        # adelante
        rr, cc = r + dr, c + dc
        while self._in_bounds(rr, cc) and (BOARD_LAYOUT[rr][cc] == "BONUS" or self._cell_to_int(board[rr][cc]) == team):
            cnt += 1
            touches_bonus = touches_bonus or (BOARD_LAYOUT[rr][cc] == "BONUS")
            rr += dr; cc += dc
        return cnt, touches_bonus

    def _completes5(self, board: np.ndarray, team: int, r: int, c: int) -> bool:
        for dr, dc in self.DIRS:
            L, _ = self._line_features(board, team, r, c, dr, dc)
            if L >= 5:
                return True
        return False

    # -------- Scoring --------
    def _score_place(self, board: np.ndarray, team: int, r: int, c: int) -> float:
        """
        Puntuación para colocar en (r,c) ponderando cercanía a BONUS y si
        las rachas resultantes tocan algún BONUS.
        """
        lengths: List[int] = []
        touches = 0
        open4 = 0
        for dr, dc in self.DIRS:
            L, tB = self._line_features(board, team, r, c, dr, dc)
            lengths.append(L)
            touches += int(tB)
            if L >= 4:
                open4 += 1

        max_run = max(lengths) if lengths else 1
        if max_run >= 5:
            return 1e6  # victoria inmediata

        adj, rad2 = self._near_bonus_features(r, c)
        bonus_adj_score = self.w_bonus_adj * adj + self.w_bonus_r2 * rad2
        bonus_line_score = self.w_bonus_line * touches
        center_tb = self.w_center_tb * self._manhattan(r, c, center=((self.H - 1) / 2.0, (self.W - 1) / 2.0))

        return self.w_open4 * open4 + self.w_maxrun * max_run + bonus_adj_score + bonus_line_score + center_tb

    def _score_remove(self, board: np.ndarray, opp: int, r: int, c: int) -> float:
        """
        Puntuación para remover en (r,c), favoreciendo romper rachas del rival
        que pasan por BONUS y fichas rivales pegadas a BONUS.
        """
        danger = 0
        touches = 0
        for dr, dc in self.DIRS:
            # Longitud de racha rival "a través" de (r,c)
            # Usamos el mismo contador pero con opp como "equipo"
            L, tB = self._line_features(board, opp, r, c, dr, dc)
            danger += max(0, L - 3)    # castiga 4/5 rivales
            touches += int(tB)

        adj, rad2 = self._near_bonus_features(r, c)
        bonus_adj_score = self.w_bonus_adj * adj + 0.5 * self.w_bonus_r2 * rad2  # un poco menos para remover
        bonus_line_score = self.w_bonus_line * touches
        center_tb = 0.5 * self.w_center_tb * self._manhattan(r, c, center=((self.H - 1) / 2.0, (self.W - 1) / 2.0))

        return 90.0 * danger + bonus_adj_score + bonus_line_score + center_tb

    # -------- Policy --------
    @overrides
    def select_action(self, legal_mask: Optional[np.ndarray], ctx: Optional[PolicyCtx] = None) -> int:
        legal = list(np.flatnonzero(legal_mask > 0.5)) if legal_mask is not None else list(range(self.env.action_space.n))
        if not legal:
            return 0

        board = self.env.game_engine.state.board
        teams = self.env.gconf.teams
        me = self.env.current_player % teams
        opp = (me + 1) % teams if teams > 1 else me

        cells = [a for a in legal if a < self.H * self.W]
        if not cells:
            return legal[0]

        # 1) Tácticas duras: ganar ya / bloquear 5 rival
        for a in cells:
            r, c = divmod(a, self.W)
            if self._completes5(board, me, r, c):
                return a
        for a in cells:
            r, c = divmod(a, self.W)
            if self._completes5(board, opp, r, c):
                return a

        # 2) Heurística centrada en BONUS
        best, best_score = cells[0], -1e18
        for a in cells:
            r, c = divmod(a, self.W)
            cell_val = self._cell_to_int(board[r][c])

            if cell_val == opp:
                score = self._score_remove(board, opp, r, c)
            else:
                score = self._score_place(board, me, r, c)

            if score > best_score:
                best_score, best = score, a

        return best
