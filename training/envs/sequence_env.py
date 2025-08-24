# training/envs/sequence_env.py
from __future__ import annotations

from typing import Any, Dict, Tuple, Optional, List
import numpy as np

from ..engine.engine_core import GameEngine
from ..engine.state import GameConfig
from ..encoders.board_encoder import encode_board_state
from ..engine.board_layout import BOARD_LAYOUT


def _is_two_eyed_jack(card: str) -> bool:
    return card in ("JC", "JD")


def _is_one_eyed_jack(card: str) -> bool:
    return card in ("JH", "JS")


class SequenceEnv:
    """
    Minimal Gymnasium-like single-agent environment wrapping the engine.

    - Unified action space: 100 board cells (0..99) + max_hand discards (+ optional PASS).
    - The agent **chooses the exact board cell**. The env deterministically resolves which
      card to play/remove based on rules and current hand:
        * If cell has opponent chip and agent holds one-eyed jack -> jack-remove on that cell.
        * Else if cell empty and equals a printed card in hand -> play that card.
        * Else if cell empty (non-BONUS) and agent holds two-eyed jack -> wild on that cell.
      If multiple matching cards exist (e.g., two copies), we pick the **first** occurrence
      in the hand (stable, deterministic).
    """

    metadata = {"render.modes": ["human"]}

    def __init__(self, config: Dict[str, Any]):
        self.config: Dict[str, Any] = config
        self.gconf: GameConfig = (
            GameConfig.from_dict(config) if isinstance(config, dict) else config
        )
        self.game_engine: GameEngine = GameEngine()
        self.current_player: int = 0

        # Configure unified action space size
        self.max_hand: int = int(config.get("action_space", {}).get("max_hand", 7))
        include_pass = bool(config.get("action_space", {}).get("include_pass", False))
        self._include_pass: bool = include_pass
        self.action_dim: int = 100 + self.max_hand + (1 if include_pass else 0)

        self._last_public: Optional[Dict[str, Any]] = None

    # ---------------- Gym-like API ----------------

    def reset(self, seed: Optional[int] = None):
        if seed is not None:
            try:
                self.game_engine.seed(int(seed))
            except Exception:
                # If engine lacks seed(), ignore.
                pass

        # Start a fresh match with config
        self.game_engine.start_new(self.gconf)

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
        # Build legality and mask first
        legal = self._legal_for(self.current_player)
        mask = self._legal_mask(legal)

        # Illegal action handling: small penalty, no transition
        if not (0 <= action < self.action_dim) or mask[action] < 0.5:
            reward = -0.01
            terminated = False
            truncated = False
            info = {"current_player": self.current_player, "legal_mask": mask}
            obs = encode_board_state(
                self._state(),
                self.current_player,
                self.config,
                legal=legal,
                public=self._public_summary(),
            )
            return obs, float(reward), terminated, truncated, info

        # Deterministic mapping from unified action → engine move dict
        if action < 100:
            r, c = divmod(action, 10)
            move = self._resolve_cell_action(self.current_player, r, c)
        else:
            idx = action - 100
            if self._include_pass and idx == self.max_hand:
                move = {"type": "timeout-skip", "card": None, "target": None, "removed": None}
            else:
                move = self._resolve_discard_action(self.current_player, idx)

        # Step the engine
        acting_player = self.current_player
        res = self.game_engine.step(move)
        # engine.step might return (state, move_rec, reward, done) OR just (reward, done)
        if isinstance(res, tuple) and len(res) == 4:
            _, _, reward, done = res
        elif isinstance(res, tuple) and len(res) == 2:
            reward, done = res
        else:
            # Fallback: treat as no reward/continue
            reward, done = 0.0, False

        # Inspect updated engine state for winners or board-full condition
        state = self._state()
        winners: List[int] = []
        if isinstance(state, dict):
            winners = list(state.get("winners", []))
        elif hasattr(state, "winners"):
            winners = list(getattr(state, "winners"))  # type: ignore

        board = getattr(state, "board", None)
        if isinstance(board, list):
            board_full = all(
                (board[r][c] is not None or BOARD_LAYOUT[r][c] == "BONUS")
                for r in range(10)
                for c in range(10)
            )
        else:
            board_full = all(
                (self._cell(r, c) is not None or BOARD_LAYOUT[r][c] == "BONUS")
                for r in range(10)
                for c in range(10)
            )

        acting_team = acting_player % self.gconf.teams
        terminated = bool(done)
        reward_val = float(reward)
        if winners:
            reward_val = 1.0 if acting_team in winners else -1.0
            terminated = True
        elif board_full:
            reward_val = 0.0
            terminated = True

        # Prepare next observation
        self.current_player = self._seat_index()
        next_legal = self._legal_for(self.current_player)
        obs = encode_board_state(
            self._state(),
            self.current_player,
            self.config,
            legal=next_legal,
            public=self._public_summary(),
        )
        info = {"current_player": self.current_player, "legal_mask": self._legal_mask(next_legal)}
        truncated = False
        return obs, reward_val, terminated, truncated, info

    # ---------------- Helpers: legality & mapping ----------------

    def _legal_for(self, seat: int) -> Dict[str, Any]:
        """Fetch legal actions from engine. Be tolerant to API shapes."""
        try:
            legal = self.game_engine.legal_actions_for(seat)
            if isinstance(legal, dict):
                return legal
        except Exception:
            pass

        # Fallback: synthesize a minimal legality dict from board & hand
        return self._synthesize_legality(seat)

    def _legal_mask(self, legal: Dict[str, Any]) -> np.ndarray:
        mask = np.zeros((self.action_dim,), dtype=np.float32)

        # Board targets (placements/removals share the same 0..99 space)
        # accepted keys: "targets" OR ("place_targets" + "remove_targets")
        targets: List[Tuple[int, int]] = []
        if "targets" in legal and isinstance(legal["targets"], list):
            targets = list(legal["targets"])
        else:
            if isinstance(legal.get("place_targets"), list):
                targets += [tuple(t) for t in legal["place_targets"]]
            if isinstance(legal.get("remove_targets"), list):
                targets += [tuple(t) for t in legal["remove_targets"]]

        for (r, c) in targets:
            if 0 <= r < 10 and 0 <= c < 10:
                mask[r * 10 + c] = 1.0

        # Discard/burn actions
        discard_slots = legal.get("discard_slots", [])
        if isinstance(discard_slots, list):
            for hand_idx in discard_slots:
                if 0 <= hand_idx < self.max_hand:
                    mask[100 + int(hand_idx)] = 1.0

        # Optional PASS
        if self._include_pass and legal.get("pass", False):
            mask[100 + self.max_hand] = 1.0

        return mask

    def _resolve_cell_action(self, seat: int, r: int, c: int) -> Dict[str, Any]:
        """
        Deterministically resolve a cell action:
        - if opponent chip at (r,c) and one-eyed jack in hand -> jack-remove
        - elif (r,c) empty and printed card in hand -> play that card
        - elif (r,c) empty, non-BONUS and two-eyed jack in hand -> wild
        - else -> illegal (we should never arrive here due to mask; fallback to no-op burn)
        """
        # Prefer engine-native resolver if present
        if hasattr(self.game_engine, "resolve_cell_action"):
            try:
                return self.game_engine.resolve_cell_action(seat, r, c)
            except Exception:
                pass  # fall back to manual

        board_cell = self._cell(r, c)
        printed = BOARD_LAYOUT[r][c]

        # Chip occupancy
        chip = None
        if isinstance(board_cell, dict):
            chip = board_cell.get("chip", None)
        elif board_cell is not None and not isinstance(board_cell, (str, int)):
            chip = getattr(board_cell, "chip", None)

        # Who owns the chip (if any)?
        chip_owner = None
        if isinstance(chip, dict):
            chip_owner = chip.get("teamIndex")
        elif chip is not None and hasattr(chip, "get"):
            chip_owner = chip.get("teamIndex")  # type: ignore

        # My hand
        hand = self._hand_for(seat)

        # Removal case (one-eyed jack)
        if chip is not None and self._is_opponent_chip(seat, chip_owner):
            for i, card in enumerate(hand):
                if _is_one_eyed_jack(card):
                    return {"type": "jack-remove", "card": card, "target": None, "removed": {"r": r, "c": c}}

        # Placement case
        is_empty = (chip is None) and (printed != "BONUS")
        if is_empty:
            # 1) try printed card
            for i, card in enumerate(hand):
                if card == printed:
                    return {"type": "play", "card": card, "target": {"r": r, "c": c}, "removed": None}
            # 2) try two-eyed jack (wild)
            for i, card in enumerate(hand):
                if _is_two_eyed_jack(card):
                    return {"type": "wild", "card": card, "target": {"r": r, "c": c}, "removed": None}

        # Fallback (should not happen if mask computed properly): choose a valid discard slot if any
        disc = self._first_discard_slot(seat)
        if disc is not None:
            return {"type": "burn", "card": hand[disc], "target": None, "removed": None}

        # Absolute last resort: timeout-skip
        return {"type": "timeout-skip", "card": None, "target": None, "removed": None}

    def _resolve_discard_action(self, seat: int, hand_index: int) -> Dict[str, Any]:
        # Prefer engine-native resolver if present
        if hasattr(self.game_engine, "resolve_discard_action"):
            try:
                return self.game_engine.resolve_discard_action(seat, hand_index)
            except Exception:
                pass

        hand = self._hand_for(seat)
        if 0 <= hand_index < len(hand):
            return {"type": "burn", "card": hand[hand_index], "target": None, "removed": None}
        # Fallback
        return {"type": "timeout-skip", "card": None, "target": None, "removed": None}

    # ---------------- Public deck summary for probability encoders ----------------

    def _public_summary(self) -> Dict[str, Any]:
        """
        Compute *public* counts:
        - Base = 2 copies of each card for a double deck.
        - Subtract discard pile (public).
        - Subtract placed chips' printed cards (public) -- crude approximation.
        """
        base: Dict[str, int] = {f"{r}{s}": 2 for s in "SHDC" for r in ["A", "2", "3", "4", "5", "6", "7", "8", "9", "10", "J", "Q", "K"]}

        # Discard pile
        discard = self._discard_pile()
        for d in discard:
            if d in base:
                base[d] = max(0, base[d] - 1)

        # Subtract for each occupied (non-BONUS) cell
        st = self._state()
        for rr in range(10):
            for cc in range(10):
                printed = BOARD_LAYOUT[rr][cc]
                if printed == "BONUS":
                    continue
                cell = st.board[rr][cc] if hasattr(st, "board") else None
                chip_present = False
                if isinstance(cell, dict):
                    chip_present = cell.get("chip") is not None
                elif cell is not None and not isinstance(cell, (str, int)):
                    chip_present = getattr(cell, "chip", None) is not None
                if chip_present and printed in base:
                    base[printed] = max(0, base[printed] - 1)

        total_remaining = int(sum(base.values()))
        return {"deck_counts": base, "total_remaining": total_remaining, "discard_pile": list(discard)}

    # ---------------- Engine accessors (robust to minor API variations) ----------------

    def _state(self):
        if hasattr(self.game_engine, "state"):
            return self.game_engine.state
        # Some engines return state via a getter
        return self.game_engine.get_state()  # type: ignore

    def _seat_index(self) -> int:
        if hasattr(self.game_engine, "current_player"):
            return int(self.game_engine.current_player)  # type: ignore
        st = self._state()
        if hasattr(st, "turn_index"):
            return int(st.turn_index)  # type: ignore
        return 0

    def _cell(self, r: int, c: int):
        st = self._state()
        if hasattr(st, "board"):
            return st.board[r][c]  # type: ignore
        return None

    def _hand_for(self, seat: int) -> List[str]:
        """
        Try several common layouts:
          - state.hands is a dict keyed by seat index (int) or seat id (str)
          - state.hands is a list indexed by seat
        """
        st = self._state()
        hands = getattr(st, "hands", None)
        if isinstance(hands, dict):
            # Try int key, then string key
            if seat in hands:
                return list(hands[seat])  # type: ignore
            if str(seat) in hands:
                return list(hands[str(seat)])  # type: ignore
            # Fallback: common seat ids like "P0","P1"
            pid = f"P{seat}"
            if pid in hands:
                return list(hands[pid])  # type: ignore
        elif isinstance(hands, list):
            if 0 <= seat < len(hands):
                return list(hands[seat])  # type: ignore
        # Last resort: engine accessor
        if hasattr(self.game_engine, "hand_for"):
            return list(self.game_engine.hand_for(seat))  # type: ignore
        return []

    def _first_discard_slot(self, seat: int) -> Optional[int]:
        legal = self._legal_for(seat)
        disc = legal.get("discard_slots", [])
        if isinstance(disc, list) and len(disc) > 0:
            return int(disc[0])
        # If not provided, but we still need a slot, pick 0 if hand non-empty
        hand = self._hand_for(seat)
        return 0 if len(hand) > 0 else None

    def _discard_pile(self) -> List[str]:
        # engine.deck may be an object or dict
        deck = getattr(self.game_engine, "deck", None)
        if isinstance(deck, dict):
            dp = deck.get("discardPile") or deck.get("discard_pile") or []
            return list(dp)
        if deck is not None:
            if hasattr(deck, "discard_pile"):
                return list(deck.discard_pile)  # type: ignore
            if hasattr(deck, "discardPile"):
                return list(deck.discardPile)  # type: ignore
        # fallback: check state
        st = self._state()
        d = getattr(st, "deck", {})
        if isinstance(d, dict):
            return list(d.get("discardPile", []))
        return []

    def _is_opponent_chip(self, my_seat: int, chip_owner_team: Optional[int]) -> bool:
        """Best-effort: if engine exposes seat→team mapping, use it; else assume seat==team for 1v1."""
        # If your engine exposes mapping, prefer that here (team_of_seat(), etc.)
        # For now assume team indices match seats in 1v1, and engine legality prevents own-removal anyway.
        return chip_owner_team is not None and int(chip_owner_team) != int(my_seat)

    # ---------------- Optional render ----------------

    def render(self, mode: str = "human"):
        st = self._state()
        board = getattr(st, "board", None)
        if board is None:
            print("<no board>")
            return
        out = []
        for r in range(10):
            row = []
            for c in range(10):
                cell = board[r][c]
                if isinstance(cell, dict):
                    chip = cell.get("chip")
                    row.append("." if chip is None else str(chip.get("teamIndex", "?")))
                else:
                    # Unknown shape
                    row.append(".")
            out.append(" ".join(row))
        print("\n".join(out))
