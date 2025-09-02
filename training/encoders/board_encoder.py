# training/encoders/board_encoder.py
from __future__ import annotations
from typing import Dict, Any, List, Optional, Tuple
import numpy as np

from ..engine.board_layout import BOARD_LAYOUT
from .probability_layers import (
    probability_cell_playable_next,
    probability_cell_playable_k_steps,
    probability_opponent_target_next,
)
from ..utils.legal_utils import normalize_legal
from ..utils.keys import LegalKey


# =========================
# Escáneres rápidos 5-en-línea
# =========================
DIRS_5: List[Tuple[int, int]] = [(0, 1), (1, 0), (1, 1), (-1, 1)]  # h, v, diag, anti-diag

def _iter_line_windows_5(H: int = 10, W: int = 10):
    """Genera todas las ventanas rectas de 5 celdas en un tablero HxW."""
    for r in range(H):
        for c in range(W):
            for dr, dc in DIRS_5:
                r_end = r + dr * 4
                c_end = c + dc * 4
                if 0 <= r_end < H and 0 <= c_end < W:
                    yield [(r + k * dr, c + k * dc) for k in range(5)]

def _is_bonus(r: int, c: int) -> bool:
    return BOARD_LAYOUT[r][c] == "BONUS"

def _threats_len4_for_team(board, team: int) -> np.ndarray:
    """
    Marca con 1 la ÚNICA celda vacía de cada ventana 5-en-línea que tenga
    exactamente 4 'cumplidas' para 'team' y 0 del rival.
    'Cumplida' = chip del team o BONUS.
    """
    H = W = 10
    out = np.zeros((H, W), dtype=np.float32)
    for coords in _iter_line_windows_5(H, W):
        cnt_team = 0
        cnt_opp = 0
        empties: List[Tuple[int, int]] = []
        for (r, c) in coords:
            chip = board[r][c]
            if chip is None and not _is_bonus(r, c):
                empties.append((r, c))
            else:
                if _is_bonus(r, c):
                    cnt_team += 1
                elif chip == team:
                    cnt_team += 1
                else:
                    cnt_opp += 1
        if cnt_opp == 0 and cnt_team == 4 and len(empties) == 1:
            r, c = empties[0]
            out[r, c] = 1.0
    return out

def _extendable_len3_for_team(board, team: int) -> np.ndarray:
    """
    Marca celdas vacías que pertenecen a ventanas con exactamente:
      - 3 'cumplidas' del team (BONUS cuenta como team),
      - 0 del rival,
      - 2 vacías.
    Útil para reconocer casillas que extienden a 5 en dos pasos (fork pressure).
    """
    H = W = 10
    out = np.zeros((H, W), dtype=np.float32)
    for coords in _iter_line_windows_5(H, W):
        cnt_team = 0
        cnt_opp = 0
        empties: List[Tuple[int, int]] = []
        for (r, c) in coords:
            chip = board[r][c]
            if chip is None and not _is_bonus(r, c):
                empties.append((r, c))
            else:
                if _is_bonus(r, c):
                    cnt_team += 1
                elif chip == team:
                    cnt_team += 1
                else:
                    cnt_opp += 1
        if cnt_opp == 0 and cnt_team == 3 and len(empties) == 2:
            for (r, c) in empties:
                out[r, c] = 1.0
    return out


# =========================
# Encoder principal
# =========================
def encode_board_state(game_state: Any,
                       current_player: int,
                       config: Dict[str, Any],
                       legal: Optional[Dict[str, List]] = None,
                       public: Optional[Dict[str, Any]] = None) -> np.ndarray:
    """
    Devuelve un tensor de observación (C, 10, 10) con los canales configurados.
    SOLO información PÚBLICA (la mano se refleja vía `legal`).
    """
    channels_cfg = (config.get("observation", {}) or {}).get("channels", {}) or {}

    channels: List[np.ndarray] = []

    # --- equipos ---
    teams = int(config.get("rules", {}).get("teams", 2))
    # Team propio/ajeno (por defecto 1v1: self=0, opp=1). Permitimos override vía 'public'.
    self_team = int((public or {}).get("self_team", 0))
    opp_team = int((public or {}).get("opp_team", 1 if teams == 2 else (self_team + 1) % max(1, teams)))

    # --- team chips (hasta 3 planos) ---
    chip_planes = [np.zeros((10, 10), dtype=np.float32) for _ in range(3)]
    for r in range(10):
        for c in range(10):
            chip = game_state.board[r][c]
            if chip is not None:
                t = int(chip)
                if 0 <= t < 3:
                    chip_planes[t][r, c] = 1.0
    if channels_cfg.get("team_chips", True):
        channels.extend(chip_planes[:teams] + [np.zeros((10, 10), dtype=np.float32)] * (3 - teams))

    # --- playable mask (propia, PÚBLICA usando legal si viene) ---
    if channels_cfg.get("playable_mask_self", True):
        plane = np.zeros((10, 10), dtype=np.float32)
        if legal:
            canon = normalize_legal(
                legal,
                board_h=10, board_w=10,
                max_hand=int(config.get("action_space", {}).get("max_hand", 7)),
                include_pass=bool(config.get("action_space", {}).get("include_pass", False)),
                union_place_remove_for_targets=True,
            )
            for (r, c) in canon.get(LegalKey.TARGETS.value, []):
                plane[r, c] = 1.0
        else:
            for r in range(10):
                for c in range(10):
                    if game_state.board[r][c] is None and BOARD_LAYOUT[r][c] != "BONUS":
                        plane[r, c] = 1.0
        channels.append(plane)

    # --- secuencias por equipo (3 planos) ---
    if channels_cfg.get("team_sequences", True):
        seq_planes = [np.zeros((10, 10), dtype=np.float32) for _ in range(3)]
        for (seq_id, seq_team, cells) in game_state.sequences_meta_cells():
            if isinstance(seq_team, int) and 0 <= seq_team < 3:
                for (r, c) in cells:
                    if 0 <= r < 10 and 0 <= c < 10:
                        seq_planes[seq_team][r, c] = 1.0
        channels.extend(seq_planes)

    # --- corner mask ---
    if channels_cfg.get("corner_mask", True):
        corner = np.zeros((10, 10), dtype=np.float32)
        for (r, c) in [(0, 0), (0, 9), (9, 0), (9, 9)]:
            corner[r, c] = 1.0
        channels.append(corner)

    # --- identidad de carta estática: 17 canales (13 rangos + 4 palos) ---
    if channels_cfg.get("static_card_17ch", True):
        ranks = ["A", "2", "3", "4", "5", "6", "7", "8", "9", "10", "J", "Q", "K"]
        suits = ["S", "H", "D", "C"]
        rank_planes = [np.zeros((10, 10), dtype=np.float32) for _ in ranks]
        suit_planes = [np.zeros((10, 10), dtype=np.float32) for _ in suits]
        for r in range(10):
            for c in range(10):
                card = BOARD_LAYOUT[r][c]
                if card == "BONUS":
                    continue
                if card[-1] in suits:
                    suit_idx = suits.index(card[-1])
                    rank_str = card[:-1]
                    if rank_str in ranks:
                        rank_idx = ranks.index(rank_str)
                        rank_planes[rank_idx][r, c] = 1.0
                        suit_planes[suit_idx][r, c] = 1.0
        channels.extend(rank_planes + suit_planes)

    # --- amenazas y estructura (ahora rellenadas con escáneres rápidos) ---
    if channels_cfg.get("threats_len4_opponent", True):
        plane = _threats_len4_for_team(game_state.board, opp_team)
        channels.append(plane)

    if channels_cfg.get("extendable_len3_self", True):
        plane = _extendable_len3_for_team(game_state.board, self_team)
        channels.append(plane)

    # --- jack remove targets (desde legal si llega) ---
    if channels_cfg.get("jack_remove_targets", True):
        jr = np.zeros((10, 10), dtype=np.float32)
        if legal:
            canon = normalize_legal(
                legal,
                board_h=10, board_w=10,
                max_hand=int(config.get("action_space", {}).get("max_hand", 7)),
                include_pass=bool(config.get("action_space", {}).get("include_pass", False)),
                union_place_remove_for_targets=False,
            )
            for (r, c) in canon.get(LegalKey.REMOVE.value, []):
                if 0 <= r < 10 and 0 <= c < 10:
                    jr[r, c] = 1.0
        channels.append(jr)

    # --- NUEVAS CAPAS ÚTILES (todas PÚBLICAS) ---

    # celdas vacías
    if channels_cfg.get("empty_cells", False):
        empty = np.zeros((10, 10), dtype=np.float32)
        for r in range(10):
            for c in range(10):
                if game_state.board[r][c] is None and BOARD_LAYOUT[r][c] != "BONUS":
                    empty[r, c] = 1.0
        channels.append(empty)

    # vulnerabilidad a Jack (self): fichas propias que no son BONUS ni están en secuencia
    if channels_cfg.get("jack_vuln_self", False):
        vuln = np.zeros((10, 10), dtype=np.float32)
        seq_protected = np.zeros((10, 10), dtype=np.float32)
        # reconstruimos protección por secuencia (aunque no se pidan planos de secuencias)
        for (seq_id, seq_team, cells) in game_state.sequences_meta_cells():
            if isinstance(seq_team, int) and seq_team == self_team:
                for (r, c) in cells:
                    seq_protected[r, c] = 1.0
        for r in range(10):
            for c in range(10):
                chip = game_state.board[r][c]
                if chip == self_team and BOARD_LAYOUT[r][c] != "BONUS" and seq_protected[r, c] < 0.5:
                    vuln[r, c] = 1.0
        channels.append(vuln)

    # indicador de turno (broadcast): para 1v1, {0,1}
    if channels_cfg.get("turn_indicator", False):
        plane = np.ones((10, 10), dtype=np.float32) * float(current_player % 2)
        channels.append(plane)

    # fracción de mazo restante (broadcast)
    deck_counts = (public or {}).get("deck_counts", {}) if public else {}
    total_remaining = int((public or {}).get("total_remaining", 0)) if public else 0
    if channels_cfg.get("deck_fraction", False):
        init_size = float(config.get("engine", {}).get("initial_deck_size", 104.0))
        frac = 0.0
        if init_size > 0:
            frac = np.clip(float(total_remaining) / init_size, 0.0, 1.0)
        channels.append(np.ones((10, 10), dtype=np.float32) * float(frac))

    # última jugada (si está disponible)
    if channels_cfg.get("last_move", False):
        lm = np.zeros((10, 10), dtype=np.float32)
        # buscamos en public o en el game_state
        last_mv = None
        if public and "last_move_cell" in public:
            last_mv = public["last_move_cell"]  # (r,c)
        elif hasattr(game_state, "last_move"):
            last_mv = getattr(game_state, "last_move")
            # soporta dict {"r":..,"c":..} o tuplas
            if isinstance(last_mv, dict):
                last_mv = (last_mv.get("r", None), last_mv.get("c", None))
        if isinstance(last_mv, (tuple, list)) and len(last_mv) == 2:
            r, c = last_mv
            if isinstance(r, int) and isinstance(c, int) and 0 <= r < 10 and 0 <= c < 10:
                lm[r, c] = 1.0
        channels.append(lm)

    # --- capas de probabilidad (PÚBLICAS) ---
    if channels_cfg.get("p_cell_playable_next", True):
        plane = np.zeros((10, 10), dtype=np.float32)
        if total_remaining > 0:
            for r in range(10):
                for c in range(10):
                    if game_state.board[r][c] is None:
                        card = BOARD_LAYOUT[r][c]
                        if card and card != "BONUS":
                            plane[r, c] = probability_cell_playable_next(deck_counts, total_remaining, card)
        channels.append(plane)

    for k in channels_cfg.get("p_cell_playable_k", [2, 3, 4, 5, 6]):
        plane = np.zeros((10, 10), dtype=np.float32)
        if total_remaining > 0:
            for r in range(10):
                for c in range(10):
                    if game_state.board[r][c] is None:
                        card = BOARD_LAYOUT[r][c]
                        if card and card != "BONUS":
                            p1 = probability_cell_playable_next(deck_counts, total_remaining, card)
                            plane[r, c] = probability_cell_playable_k_steps(p1, int(k))
        channels.append(plane)

    if channels_cfg.get("p_opponent_targets_next", True):
        plane = np.zeros((10, 10), dtype=np.float32)
        if total_remaining > 0:
            for r in range(10):
                for c in range(10):
                    if game_state.board[r][c] is None:
                        card = BOARD_LAYOUT[r][c]
                        if card and card != "BONUS":
                            plane[r, c] = probability_opponent_target_next(deck_counts, total_remaining, card)
        channels.append(plane)

    obs = np.stack(channels, axis=0).astype(np.float32)  # (C,10,10)
    return obs
