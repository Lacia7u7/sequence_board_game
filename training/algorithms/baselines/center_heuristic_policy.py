from __future__ import annotations
from typing import Optional, Tuple, List

import numpy as np
from overrides import overrides

from training.algorithms.base_policy import BasePolicy, PolicyCtx
from ...engine.engine_core import BOARD_LAYOUT


class CenterHeuristicPolicy(BasePolicy):
    """
    Heurística de selección de acción para Sequence:
      - 'Colocar': prioriza celdas con menor distancia Manhattan al centro
        y que maximicen potencial de secuencia (completar 5, abrir 4, mayor racha).
      - 'Remover': prioriza la casilla rival que rompa más "casi-secuencias" (rachas largas).
    Respeta legal_mask; si nada encaja, fallback al primer legal.
    """

    def __init__(self, env=None):
        self.env = env

    # --- Utilidades de tablero ---
    @staticmethod
    def _in_bounds(r: int, c: int) -> bool:
        return 0 <= r < 10 and 0 <= c < 10

    @staticmethod
    def _manhattan(r: int, c: int, center: Tuple[float, float] = (4.5, 4.5)) -> float:
        cr, cc = center
        return abs(r - cr) + abs(c - cc)

    def _count_line_len(self, board: np.ndarray, team: int, r: int, c: int, dr: int, dc: int) -> int:
        """
        Cuenta longitud de racha (propia + BONUS) en dirección (dr,dc) **suponiendo** que
        ponemos nuestra ficha en (r,c). Incluye ambas direcciones.
        """
        cnt = 1  # contamos (r,c) como nuestra ficha hipotética
        # atrás
        rr, cc = r - dr, c - dc
        while self._in_bounds(rr, cc) and (BOARD_LAYOUT[rr][cc] == "BONUS" or board[rr][cc] == team):
            cnt += 1
            rr -= dr; cc -= dc
        # adelante
        rr, cc = r + dr, c + dc
        while self._in_bounds(rr, cc) and (BOARD_LAYOUT[rr][cc] == "BONUS" or board[rr][cc] == team):
            cnt += 1
            rr += dr; cc += dc
        return cnt

    def _count_line_len_opp_through_cell(self, board: np.ndarray, opp: int, r: int, c: int, dr: int, dc: int) -> int:
        """
        Cuenta longitud de racha del rival (opp + BONUS) **pasando por (r,c)**,
        para estimar cuán valioso es remover en (r,c).
        """
        cnt = 1  # (r,c) contiene una ficha rival actualmente (por legalidad de remover)
        # atrás
        rr, cc = r - dr, c - dc
        while self._in_bounds(rr, cc) and (BOARD_LAYOUT[rr][cc] == "BONUS" or board[rr][cc] == opp):
            cnt += 1
            rr -= dr; cc -= dc
        # adelante
        rr, cc = r + dr, c + dc
        while self._in_bounds(rr, cc) and (BOARD_LAYOUT[rr][cc] == "BONUS" or board[rr][cc] == opp):
            cnt += 1
            rr += dr; cc += dc
        return cnt

    def _score_place(self, board: np.ndarray, team: int, r: int, c: int) -> float:
        """
        Puntuación para colocar en (r,c):
          - +INF si completa secuencia (>=5 en alguna dirección)
          - bonifica rachas largas y 4-en-línea
          - penaliza distancia Manhattan al centro
        """
        dirs = [(0,1), (1,0), (1,1), (1,-1)]
        lengths: List[int] = [self._count_line_len(board, team, r, c, dr, dc) for dr, dc in dirs]
        max_run = max(lengths)
        open4 = sum(1 for L in lengths if L >= 4)
        completes5 = any(L >= 5 for L in lengths)

        # pesos heurísticos
        W_SEQ5 = 1e6      # si puede completar secuencia, elegir sí o sí
        W_OPEN4 = 50.0
        W_MAXRUN = 5.0
        W_CENTER = -1.0   # penaliza distancia (más cerca del centro => mayor score)

        dist = self._manhattan(r, c)

        if completes5:
            return W_SEQ5

        score = W_OPEN4 * open4 + W_MAXRUN * max_run + W_CENTER * dist
        return score

    def _score_remove(self, board: np.ndarray, opp: int, r: int, c: int) -> float:
        """
        Puntuación para remover en (r,c):
          - suma sobre direcciones el excedente de racha rival por encima de 3 (casi-secuencias).
          - pequeño tie-break con centralidad para preferir quitar amenazas centrales.
        """
        dirs = [(0,1), (1,0), (1,1), (1,-1)]
        runs = [self._count_line_len_opp_through_cell(board, opp, r, c, dr, dc) for dr, dc in dirs]

        # Heurística: cada unidad por encima de 3 indica "casi-secuencia" (4) o algo muy peligroso (5)
        danger_units = sum(max(0, L - 3) for L in runs)

        W_DANGER = 100.0
        W_CENTER_TB = -0.25  # tie-break suave con centralidad
        dist = self._manhattan(r, c)

        return W_DANGER * danger_units + W_CENTER_TB * dist

    @overrides
    def select_action(self, legal_mask: Optional[np.ndarray], ctx: Optional[PolicyCtx] = None) -> int:
        # Fallback 100% seguro si no hay máscara:
        if legal_mask is None:
            # como en políticas baseline, si no hay mask usamos todo el espacio
            legal = list(range(self.env.action_space.n))
        else:
            flat = legal_mask.flatten()
            legal = [i for i, m in enumerate(flat) if m > 0.5]

        if not legal:
            return 0  # última línea de defensa

        # Acceso a estado actual
        board = self.env.game_engine.state.board  # (10,10) int: -1 vacío, team id en fichas
        teams = self.env.gconf.teams
        team = self.env.current_player % teams
        opp = 1 - team if teams == 2 else (team + 1) % teams  # generalización

        best_act = legal[0]
        best_score = -1e18
        fallback_nonboard: Optional[int] = None

        for act in legal:
            # Acción de tablero esperada: 0..99 => celda (r,c)
            if act < 100:
                r, c = divmod(act, 10)

                cell = board[r][c]
                # Si hay ficha rival, interpretamos como acción de "remover" (JH/JS)
                if cell == opp:
                    score = self._score_remove(board, opp, r, c)
                else:
                    # Celda vacía o BONUS -> "colocar" (carta normal o Jota wild JD/JC)
                    score = self._score_place(board, team, r, c)

                if score > best_score:
                    best_score = score
                    best_act = act
            else:
                # Guardamos por si no encontramos nada mejor (espacios extra del action space)
                if fallback_nonboard is None:
                    fallback_nonboard = act

        # Si no elegimos algo del tablero por algún motivo raro, usamos el primer legal alternativo
        if best_score <= -1e17 and fallback_nonboard is not None:
            return fallback_nonboard

        return best_act
