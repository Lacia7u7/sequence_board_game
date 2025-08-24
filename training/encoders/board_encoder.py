# training/encoders/board_encoder.py
from __future__ import annotations
from typing import Dict, Any, List, Optional
import numpy as np

from ..engine.board_layout import BOARD_LAYOUT
from .probability_layers import (
    probability_cell_playable_next,
    probability_cell_playable_k_steps,
    probability_opponent_target_next,
)


def encode_board_state(game_state: Any,
                       current_player: int,
                       config: Dict[str, Any],
                       legal: Optional[Dict[str, List]] = None,
                       public: Optional[Dict[str, Any]] = None) -> np.ndarray:
    """
    Returns an observation tensor (C, 10, 10) with all configured channels.
    Only PUBLIC info is encoded (hand-based masks are injected by the env via `legal`).
    """
    channels_cfg = (config.get("observation", {}) or {}).get("channels", {}) or {}

    channels: List[np.ndarray] = []

    # --- team chips (3) ---
    teams = int(config.get("rules", {}).get("teams", 2))
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

    # --- playable mask (public-safe or hand-aware if `legal` provided) ---
    if channels_cfg.get("playable_mask_self", True):
        plane = np.zeros((10, 10), dtype=np.float32)
        if legal and "targets" in legal:
            for (r, c) in legal["targets"]:
                plane[r, c] = 1.0
        else:
            for r in range(10):
                for c in range(10):
                    if game_state.board[r][c] is None and BOARD_LAYOUT[r][c] != "BONUS":
                        plane[r, c] = 1.0
        channels.append(plane)

    # --- sequences per team (3) ---
    if channels_cfg.get("team_sequences", True):
        seq_planes = [np.zeros((10, 10), dtype=np.float32) for _ in range(3)]
        for (seq_id, seq_team, cells) in game_state.sequences_meta_cells():
            if 0 <= seq_team < 3:
                for (r, c) in cells:
                    seq_planes[seq_team][r, c] = 1.0
        channels.extend(seq_planes)

    # --- corner mask ---
    if channels_cfg.get("corner_mask", True):
        corner = np.zeros((10, 10), dtype=np.float32)
        for (r, c) in [(0, 0), (0, 9), (9, 0), (9, 9)]:
            corner[r, c] = 1.0
        channels.append(corner)

    # --- static card identity: 17 channels (13 ranks + 4 suits) ---
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
                # card like "10D"
                if card[-1] in suits:
                    suit_idx = suits.index(card[-1])
                    rank_str = card[:-1]
                    if rank_str in ranks:
                        rank_idx = ranks.index(rank_str)
                        rank_planes[rank_idx][r, c] = 1.0
                        suit_planes[suit_idx][r, c] = 1.0
        channels.extend(rank_planes + suit_planes)

    # --- threat & structure channels (optional; simple placeholders can be enhanced) ---
    if channels_cfg.get("threats_len4_opponent", True):
        channels.append(np.zeros((10, 10), dtype=np.float32))  # reserved; can implement later with fast scanners
    if channels_cfg.get("extendable_len3_self", True):
        channels.append(np.zeros((10, 10), dtype=np.float32))
    if channels_cfg.get("jack_remove_targets", True):
        jr = np.zeros((10, 10), dtype=np.float32)
        # public version: any opponent chip that is not a corner
        for r in range(10):
            for c in range(10):
                if BOARD_LAYOUT[r][c] == "BONUS":
                    continue
                chip = game_state.board[r][c]
                if chip is not None and int(chip) != current_player:
                    jr[r, c] = 1.0
        channels.append(jr)

    # --- probability layers (PUBLIC-only math) ---
    deck_counts = (public or {}).get("deck_counts", {}) if public else {}
    total_remaining = int((public or {}).get("total_remaining", 0)) if public else 0
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
