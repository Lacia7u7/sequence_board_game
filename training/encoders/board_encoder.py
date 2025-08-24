import numpy as np
from typing import Dict, Any, List
from ..engine.board_layout import BOARD_LAYOUT

def encode_board_state(game_state: Any, current_player: int, config: Dict) -> np.ndarray:
    teams = config.get("rules", {}).get("teams", 2)
    chan_cfg = config.get("observation", {}).get("channels", {})
    channels: List[np.ndarray] = []

    if chan_cfg.get("team_chips", False):
        for team_idx in range(teams):
            plane = np.zeros((10,10), dtype=np.float32)
            for r in range(10):
                for c in range(10):
                    if game_state.board[r][c] == team_idx:
                        plane[r,c] = 1.0
            channels.append(plane)

    if chan_cfg.get("playable_mask_self", False):
        channels.append(np.zeros((10,10), dtype=np.float32))

    if chan_cfg.get("team_sequences", False):
        for team_idx in range(teams):
            plane = np.zeros((10,10), dtype=np.float32)
            for (r,c) in getattr(game_state, "sequence_cells", []):
                if game_state.board[r][c] == team_idx or BOARD_LAYOUT[r][c] == "BONUS":
                    plane[r,c] = 1.0
            channels.append(plane)

    if chan_cfg.get("corner_mask", False):
        plane = np.zeros((10,10), dtype=np.float32)
        for (rr,cc) in [(0,0),(0,9),(9,0),(9,9)]:
            plane[rr,cc] = 1.0
        channels.append(plane)

    if chan_cfg.get("static_card_17ch", False):
        ranks = ["A","2","3","4","5","6","7","8","9","10","J","Q","K"]
        suits = ["S","H","D","C"]
        rank_to_idx = {r:i for i,r in enumerate(ranks)}
        suit_to_idx = {s:i for i,s in enumerate(suits)}
        rank_planes = np.zeros((13,10,10), dtype=np.float32)
        suit_planes = np.zeros((4,10,10), dtype=np.float32)
        for r in range(10):
            for c in range(10):
                card = BOARD_LAYOUT[r][c]
                if not card or card == "BONUS":
                    continue
                rank = card[:-1]; suit = card[-1]
                if rank in rank_to_idx:
                    rank_planes[rank_to_idx[rank], r, c] = 1.0
                if suit in suit_to_idx:
                    suit_planes[suit_to_idx[suit], r, c] = 1.0
        channels.extend(list(rank_planes))
        channels.extend(list(suit_planes))

    if chan_cfg.get("threats_len4_opponent", False):
        plane = np.zeros((10,10), dtype=np.float32)
        current_team = current_player % teams
        opponents = [t for t in range(teams) if t != current_team]
        for r in range(10):
            for c in range(10):
                if game_state.board[r][c] is None and BOARD_LAYOUT[r][c] != "BONUS":
                    for dr, dc in [(0,1),(1,0),(1,1),(1,-1)]:
                        count = 0
                        for i in range(-4,5):
                            rr = r + dr*i; cc = c + dc*i
                            if 0 <= rr < 10 and 0 <= cc < 10 and (rr,cc) != (r,c):
                                if BOARD_LAYOUT[rr][cc] == "BONUS" or game_state.board[rr][cc] in opponents:
                                    count += 1
                                else:
                                    count = 0
                            else:
                                count = 0
                            if count >= 4:
                                plane[r][c] = 1.0; break
                        if plane[r][c] == 1.0: break
        channels.append(plane)

    if chan_cfg.get("extendable_len3_self", False):
        plane = np.zeros((10,10), dtype=np.float32)
        current_team = current_player % teams
        for r in range(10):
            for c in range(10):
                if game_state.board[r][c] is None and BOARD_LAYOUT[r][c] != "BONUS":
                    for dr, dc in [(0,1),(1,0),(1,1),(1,-1)]:
                        count = 1
                        rr, cc = r-dr, c-dc
                        while 0 <= rr < 10 and 0 <= cc < 10 and (BOARD_LAYOUT[rr][cc] == "BONUS" or game_state.board[rr][cc] == current_team):
                            count += 1; rr -= dr; cc -= dc
                        rr, cc = r+dr, c+dc
                        while 0 <= rr < 10 and 0 <= cc < 10 and (BOARD_LAYOUT[rr][cc] == "BONUS" or game_state.board[rr][cc] == current_team):
                            count += 1; rr += dr; cc += dc
                        if count == 4:
                            plane[r][c] = 1.0; break
        channels.append(plane)

    if chan_cfg.get("jack_remove_targets", False):
        plane = np.zeros((10,10), dtype=np.float32)
        current_team = current_player % teams
        allow_adv = bool(config.get("rules",{}).get("allowAdvancedJack", False))
        for r in range(10):
            for c in range(10):
                chip_team = game_state.board[r][c]
                if chip_team is not None and chip_team != current_team:
                    if BOARD_LAYOUT[r][c] == "BONUS":
                        continue
                    if (r,c) in getattr(game_state,"sequence_cells", set()) and not allow_adv:
                        continue
                    plane[r][c] = 1.0
        channels.append(plane)

    if chan_cfg.get("p_cell_playable_next", False):
        channels.append(np.zeros((10,10), dtype=np.float32))
    for _k in chan_cfg.get("p_cell_playable_k", []):
        channels.append(np.zeros((10,10), dtype=np.float32))
    if chan_cfg.get("p_opponent_targets_next", False):
        channels.append(np.zeros((10,10), dtype=np.float32))

    return np.stack(channels, axis=0) if channels else np.zeros((0,10,10), dtype=np.float32)
