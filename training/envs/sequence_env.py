from __future__ import annotations

from typing import Any, Dict, Optional, List, Tuple
import numpy as np

from ..engine.engine_core import GameEngine, EngineError, is_two_eyed_jack, is_one_eyed_jack
from ..engine.state import GameConfig
from ..engine.board_layout import BOARD_LAYOUT
from ..encoders.board_encoder import encode_board_state

from ..utils.legal_utils import normalize_legal, build_action_mask


class SequenceEnv:
    """
    Gymnasium-like single-agent env around engine_core.GameEngine with reward shaping.

    Unified action space:
      0..99                  -> board cells (row-major)
      100..100+H-1           -> discard hand index (H = max_hand)
      100+H                  -> PASS (if enabled)

    Rewards (configurable via config["rewards"]):
      - illegal:   rewards.illegal             (default -0.01)
      - win:       rewards.win                 (default +1.0)
      - loss:      rewards.loss                (default -1.0)
      - seq bonus: rewards.seq_bonus * (#new sequences for acting team)
      - shaping:   (open4 / open3 deltas)
                   + rewards.open4_self * Δself_open4
                   - rewards.open4_opp  * Δopp_open4
                   + rewards.open3_self * Δself_open3
                   - rewards.open3_opp  * Δopp_open3
      - optional per-step penalty: rewards.step_penalty (default 0.0)
    """

    metadata = {"render.modes": ["human"]}

    def __init__(self, config: Dict[str, Any]):
        self.config: Dict[str, Any] = config
        self.gconf: GameConfig = GameConfig.from_dict(config) if isinstance(config, dict) else config
        self.game_engine: GameEngine = GameEngine()
        self.current_player: int = 0

        # Action space
        self.max_hand: int = int(config.get("action_space", {}).get("max_hand", 7))
        include_pass = bool(config.get("action_space", {}).get("include_pass", False))
        self._include_pass: bool = include_pass
        self.action_dim: int = 100 + self.max_hand + (1 if include_pass else 0)

        # Episode cap
        self._step_limit: int = int(getattr(self.gconf, "episode_cap", 400))
        self._steps_elapsed: int = 0

        # Reward shaping coefficients (with defaults)
        rw = dict(config.get("rewards", {}) or {})
        self.R_ILLEGAL      = float(rw.get("illegal", -0.01))
        self.R_WIN          = float(rw.get("win", 1.0))
        self.R_LOSS         = float(rw.get("loss", -1.0))
        self.R_SEQ_BONUS    = float(rw.get("seq_bonus", 0.3))
        self.R_STEP_PENALTY = float(rw.get("step_penalty", 0.0))

        self.W_OPEN4_SELF   = float(rw.get("open4_self", 0.05))
        self.W_OPEN4_OPP    = float(rw.get("open4_opp", 0.05))
        self.W_OPEN3_SELF   = float(rw.get("open3_self", 0.01))
        self.W_OPEN3_OPP    = float(rw.get("open3_opp", 0.01))

    # ---------------- Gym-like API ----------------

    def reset(self, seed: Optional[int] = None):
        if seed is not None:
            try:
                self.game_engine.seed(int(seed))
            except Exception:
                pass

        self.game_engine.start_new(self.gconf)
        self._steps_elapsed = 0
        self.current_player = self._seat_index()

        legal = self._legal_for(self.current_player)
        obs = encode_board_state(
            self._state(),
            self.current_player,
            self.config,
            legal=legal,
            public=self._public_summary(),
        )
        info = {
            "current_player": self.current_player,
            "legal_mask": self._legal_mask(legal),
        }
        return obs, info

    def step(self, action: int):
        st_before = self._state()
        acting_player = self.current_player
        acting_team = self._player_team(acting_player)
        teams = max(1, int(self.gconf.teams))

        # Features BEFORE move (for potential-like shaping)
        seq_count_before = int(self._sequences_for_team(st_before, acting_team))
        th_self_b, th_opp_b = self._threat_summary(st_before, acting_team)

        # Build legality and mask
        legal = self._legal_for(self.current_player)
        mask = self._legal_mask(legal)

        # Illegal action → small penalty, no engine step
        if not (0 <= action < self.action_dim) or mask[action] < 0.5:
            obs = encode_board_state(
                st_before, self.current_player, self.config, legal=legal, public=self._public_summary()
            )
            info = {"current_player": self.current_player, "legal_mask": mask, "illegal": True}
            return obs, float(self.R_ILLEGAL), False, False, info

        # Map unified action → engine move
        if action < 100:
            r, c = divmod(action, 10)
            move = self._resolve_cell_action(self.current_player, r, c)
        else:
            idx = action - 100
            if self._include_pass and idx == self.max_hand:
                move = {"type": "timeout-skip", "card": None, "target": None, "removed": None}
            else:
                move = self._resolve_discard_action(self.current_player, idx)

        # Execute engine step (engine returns (state, move_record))
        try:
            st_after, move_record = self.game_engine.step(move)
        except EngineError:
            # Treat engine-level rejection as illegal; no advance guaranteed
            obs = encode_board_state(
                st_before, self.current_player, self.config, legal=legal, public=self._public_summary()
            )
            info = {"current_player": self.current_player, "legal_mask": mask, "engine_reject": True}
            return obs, float(self.R_ILLEGAL), False, False, info

        # Terminal?
        winners = self.game_engine.winner_teams() if hasattr(self.game_engine, "winner_teams") else (list(getattr(st_after, "winners", [])) if st_after else [])
        terminated = bool(winners)

        # Features AFTER move
        seq_count_after = int(self._sequences_for_team(st_after, acting_team))
        th_self_a, th_opp_a = self._threat_summary(st_after, acting_team)

        # ---------------- Reward assembly ----------------
        reward = 0.0

        # Terminal outcome
        if terminated:
            reward += self.R_WIN if acting_team in winners else self.R_LOSS

        # Sequence creation bonus (works both terminal and non-terminal)
        seq_delta = max(0, seq_count_after - seq_count_before)
        if seq_delta > 0:
            reward += self.R_SEQ_BONUS * float(seq_delta)

        # Threat shaping (potential-like deltas)
        d_open4_self = th_self_a["open4"] - th_self_b["open4"]
        d_open4_opp  = th_opp_a["open4"]  - th_opp_b["open4"]
        d_open3_self = th_self_a["open3"] - th_self_b["open3"]
        d_open3_opp  = th_opp_a["open3"]  - th_opp_b["open3"]

        reward += self.W_OPEN4_SELF * d_open4_self
        reward -= self.W_OPEN4_OPP  * d_open4_opp
        reward += self.W_OPEN3_SELF * d_open3_self
        reward -= self.W_OPEN3_OPP  * d_open3_opp

        # Optional per-step penalty to encourage shorter games
        reward += self.R_STEP_PENALTY

        # Next observation
        self.current_player = self._seat_index()
        next_legal = self._legal_for(self.current_player)
        obs = encode_board_state(
            self._state(), self.current_player, self.config, legal=next_legal, public=self._public_summary()
        )

        # Truncation via episode cap
        self._steps_elapsed += 1
        truncated = False
        if not terminated and self._step_limit > 0 and self._steps_elapsed >= self._step_limit:
            truncated = True

        info = {
            "current_player": self.current_player,
            "legal_mask": self._legal_mask(next_legal),
            "move": move_record,
            "winners": winners,
            "reward_breakdown": {
                "terminal": (self.R_WIN if terminated and acting_team in winners else (self.R_LOSS if terminated else 0.0)),
                "seq_bonus": self.R_SEQ_BONUS * float(seq_delta),
                "open4_self_delta": float(d_open4_self),
                "open4_opp_delta": float(d_open4_opp),
                "open3_self_delta": float(d_open3_self),
                "open3_opp_delta": float(d_open3_opp),
                "step_penalty": float(self.R_STEP_PENALTY),
            }
        }
        return obs, float(reward), terminated, truncated, info

    # ---------------- Helpers: legality & mapping ----------------

    def _legal_for(self, seat: int) -> Dict[str, Any]:
        try:
            legal = self.game_engine.legal_actions_for(seat)
            if isinstance(legal, dict):
                return legal
        except Exception:
            pass
        return {}

    def _legal_mask(self, legal: Dict[str, Any]) -> np.ndarray:
        canon = normalize_legal(
            legal,
            board_h=10, board_w=10,
            max_hand=self.max_hand,
            include_pass=self._include_pass,
            union_place_remove_for_targets=True,
        )
        return build_action_mask(
            self.action_dim,
            board_h=10, board_w=10,
            max_hand=self.max_hand,
            include_pass=self._include_pass,
            canon_legal=canon,
        )

    def _resolve_cell_action(self, seat: int, r: int, c: int) -> Dict[str, Any]:
        # Prefer engine-native resolver if present
        if hasattr(self.game_engine, "resolve_cell_action"):
            try:
                return self.game_engine.resolve_cell_action(seat, r, c)  # type: ignore
            except Exception:
                pass

        board_cell = self._cell(r, c)
        printed = BOARD_LAYOUT[r][c]
        my_team = self._player_team(seat)
        hand = self._hand_for(seat)

        # Opponent chip present?
        if board_cell is not None and isinstance(board_cell, int) and board_cell != my_team:
            for card in hand:
                if is_one_eyed_jack(card):
                    return {"type": "jack-remove", "card": card, "target": None, "removed": {"r": r, "c": c}}

        # Empty placement
        is_empty = (board_cell is None) and (printed != "BONUS")
        if is_empty:
            # exact printed card
            for card in hand:
                if card == printed:
                    return {"type": "play", "card": card, "target": {"r": r, "c": c}, "removed": None}
            # two-eyed jack wild
            for card in hand:
                if is_two_eyed_jack(card):
                    return {"type": "wild", "card": card, "target": {"r": r, "c": c}, "removed": None}

        # fallback to burn first discard slot if any
        disc = self._first_discard_slot(seat)
        if disc is not None:
            return {"type": "burn", "card": hand[disc], "target": None, "removed": None}

        return {"type": "timeout-skip", "card": None, "target": None, "removed": None}

    def _resolve_discard_action(self, seat: int, hand_index: int) -> Dict[str, Any]:
        if hasattr(self.game_engine, "resolve_discard_action"):
            try:
                return self.game_engine.resolve_discard_action(seat, hand_index)  # type: ignore
            except Exception:
                pass
        hand = self._hand_for(seat)
        if 0 <= hand_index < len(hand):
            return {"type": "burn", "card": hand[hand_index], "target": None, "removed": None}
        return {"type": "timeout-skip", "card": None, "target": None, "removed": None}

    # ---------------- Threat / sequence features ----------------

    def _sequences_for_team(self, st, team: int) -> int:
        seqs = getattr(st, "sequences", None)
        if isinstance(seqs, dict) and team in seqs:
            return int(seqs[team])
        # alias through GameState helper if present
        if hasattr(st, "sequences_count"):
            try:
                return int(st.sequences_count.get(team, 0))  # type: ignore
            except Exception:
                pass
        return 0

    def _threat_summary(self, st, acting_team: int) -> Tuple[Dict[str, int], Dict[str, int]]:
        """
        Count "open" windows of length 5:
          - open4: 4 satisfied cells (team chip or BONUS), 1 empty, 0 opponent chips
          - open3: 3 satisfied cells, 2 empty, 0 opponent chips
        Returns two dicts: self_features, opp_features
        """
        board: List[List[Optional[int]]] = st.board  # type: ignore
        teams = max(1, int(self.gconf.teams))

        def count_for_team(team: int) -> Dict[str, int]:
            c4 = 0
            c3 = 0
            # Horizontal
            for r in range(10):
                for c in range(6):
                    cells = [(r, c+i) for i in range(5)]
                    s, e, o = self._window_stats(board, team, cells)
                    if o == 0:
                        if s == 4 and e == 1:
                            c4 += 1
                        elif s == 3 and e == 2:
                            c3 += 1
            # Vertical
            for r in range(6):
                for c in range(10):
                    cells = [(r+i, c) for i in range(5)]
                    s, e, o = self._window_stats(board, team, cells)
                    if o == 0:
                        if s == 4 and e == 1:
                            c4 += 1
                        elif s == 3 and e == 2:
                            c3 += 1
            # Diagonal down-right
            for r in range(6):
                for c in range(6):
                    cells = [(r+i, c+i) for i in range(5)]
                    s, e, o = self._window_stats(board, team, cells)
                    if o == 0:
                        if s == 4 and e == 1:
                            c4 += 1
                        elif s == 3 and e == 2:
                            c3 += 1
            # Diagonal down-left
            for r in range(6):
                for c in range(4, 10):
                    cells = [(r+i, c-i) for i in range(5)]
                    s, e, o = self._window_stats(board, team, cells)
                    if o == 0:
                        if s == 4 and e == 1:
                            c4 += 1
                        elif s == 3 and e == 2:
                            c3 += 1
            return {"open4": c4, "open3": c3}

        self_feats = count_for_team(acting_team)

        # Opponents: sum across all teams != acting_team
        opp_feats = {"open4": 0, "open3": 0}
        for t in range(teams):
            if t == acting_team:
                continue
            f = count_for_team(t)
            opp_feats["open4"] += f["open4"]
            opp_feats["open3"] += f["open3"]

        return self_feats, opp_feats

    def _window_stats(self, board: List[List[Optional[int]]], team: int, cells: List[Tuple[int, int]]) -> Tuple[int, int, int]:
        """
        For a 5-cell window return (satisfied, empty, opponent_count) wrt 'team'.
        A cell is 'satisfied' if it is a BONUS corner or has a chip of 'team'.
        An 'opponent' cell is any non-None chip != team.
        """
        satisfied = 0
        empty = 0
        opponents = 0
        for (r, c) in cells:
            printed = BOARD_LAYOUT[r][c]
            chip = board[r][c]
            if printed == "BONUS":
                # Corners count as satisfied for both teams
                satisfied += 1
            elif chip is None:
                empty += 1
            elif chip == team:
                satisfied += 1
            else:
                opponents += 1
        return satisfied, empty, opponents

    # ---------------- Engine accessors & summaries ----------------

    def _state(self):
        return self.game_engine.state

    def _seat_index(self) -> int:
        st = self._state()
        return int(getattr(st, "current_player", getattr(st, "turn_index", 0)))

    def _player_team(self, seat: int) -> int:
        teams = max(1, int(self.gconf.teams))
        return int(seat) % teams

    def _cell(self, r: int, c: int):
        st = self._state()
        return st.board[r][c]  # type: ignore

    def _hand_for(self, seat: int) -> List[str]:
        st = self._state()
        hands = getattr(st, "hands", None)
        if isinstance(hands, list) and 0 <= seat < len(hands):
            return list(hands[seat])
        return []

    def _first_discard_slot(self, seat: int) -> Optional[int]:
        legal = self._legal_for(seat)
        disc = legal.get("discard") or legal.get("discard_slots")
        if isinstance(disc, list) and len(disc) > 0:
            try:
                return int(disc[0])
            except Exception:
                return None
        hand = self._hand_for(seat)
        return 0 if len(hand) > 0 else None

    def _public_summary(self) -> Dict[str, Any]:
        # 2 copies per card (double deck), minus discard pile, minus occupied printed cells (approx.)
        base: Dict[str, int] = {f"{r}{s}": 2 for s in "SHDC" for r in ["A", "2", "3", "4", "5", "6", "7", "8", "9", "10", "J", "Q", "K"]}
        deck = getattr(self.game_engine, "deck", None)
        discard_pile = []
        if deck is not None:
            dp = getattr(deck, "discard_pile", getattr(deck, "discardPile", None))
            if isinstance(dp, list):
                discard_pile = list(dp)
        for d in discard_pile:
            if d in base:
                base[d] = max(0, base[d] - 1)
        st = self._state()
        for rr in range(10):
            for cc in range(10):
                printed = BOARD_LAYOUT[rr][cc]
                if printed == "BONUS":
                    continue
                chip = st.board[rr][cc]
                if chip is not None and printed in base:
                    base[printed] = max(0, base[printed] - 1)
        total_remaining = int(sum(base.values()))
        return {"deck_counts": base, "total_remaining": total_remaining, "discard_pile": discard_pile}

    # ---------------- Optional ASCII render ----------------

    def render(self, mode: str = "human"):
        st = self._state()
        if st is None or getattr(st, "board", None) is None:
            print("<no board>")
            return
        out = []
        for r in range(10):
            row = []
            for c in range(10):
                cell = st.board[r][c]
                row.append("." if cell is None else str(int(cell)))
            out.append(" ".join(row))
        print("\n".join(out))
        return out
