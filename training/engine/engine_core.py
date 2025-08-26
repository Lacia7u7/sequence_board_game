# training/engine/engine_core.py
from typing import List, Optional, Tuple, Dict, Any
from .cards import Deck, is_two_eyed_jack, is_one_eyed_jack
from .board_layout import BOARD_LAYOUT
from .state import GameConfig, GameState
from .errors import EngineError, ErrorCode

_DIRECTIONS = [(0, 1, "H"), (1, 0, "V"), (1, 1, "D1"), (1, -1, "D2")]

class GameEngine:
    def __init__(self):
        self.random_seed = None
        self.game_config: Optional[GameConfig] = None
        self.state: Optional[GameState] = None
        self.deck: Optional[Deck] = None
        self._seq_meta_index: set[frozenset[Tuple[int, int]]] = set()

    def seed(self, seed: int):
        self.random_seed = seed

    def start_new(self, config: GameConfig) -> GameState:
        self.game_config = config
        self.deck = Deck()
        self.deck.shuffle(seed=self.random_seed)

        total_players = config.teams * config.players_per_team

        # Deal hands
        hands: List[List[str]] = []
        for _ in range(total_players):
            hand = []
            for _ in range(config.hand_size):
                card = self.deck.draw()
                if card is None:
                    raise RuntimeError("Deck exhausted during initial deal")
                hand.append(card)
            hands.append(hand)

        board: List[List[Optional[int]]] = [[None for _ in range(10)] for _ in range(10)]
        seq_count: Dict[int, int] = {team_idx: 0 for team_idx in range(config.teams)}

        self.state = GameState(hands=hands, board=board, sequences=seq_count)
        self.state.current_player = 0
        self.state.turns_count = 0
        self.state.winners = []
        self.state.config = config

        # Clear sequence meta structures
        self.state.sequencesMeta = []
        self.state.sequence_cells = set()
        self._seq_meta_index.clear()
        return self.state

    def _player_team(self, player_index: int) -> int:
        return player_index % (self.game_config.teams if self.game_config else 1)

    def legal_actions_for(self, player_index: int) -> Dict[str, List]:
        if self.state is None or self.game_config is None:
            return {}
        hand = self.state.hands[player_index]
        legal_cell_actions: List[Tuple[int, int]] = []
        legal_removals: List[Tuple[int, int]] = []
        legal_discards: List[int] = []
        team = self._player_team(player_index)

        for idx, card in enumerate(hand):
            if is_two_eyed_jack(card):
                for r in range(10):
                    for c in range(10):
                        if BOARD_LAYOUT[r][c] != "BONUS" and self.state.board[r][c] is None:
                            if (r, c) not in legal_cell_actions:
                                legal_cell_actions.append((r, c))
            elif is_one_eyed_jack(card):
                allow_adv = bool(self.game_config.allowAdvancedJack)
                for r in range(10):
                    for c in range(10):
                        chip = self.state.board[r][c]
                        if chip is not None and chip != team:
                            if BOARD_LAYOUT[r][c] == "BONUS":
                                continue
                            if (r, c) in self.state.sequence_cells and not allow_adv:
                                continue
                            if (r, c) not in legal_removals:
                                legal_removals.append((r, c))
            else:
                positions = [(r, c) for r in range(10) for c in range(10) if BOARD_LAYOUT[r][c] == card]
                for (r, c) in positions:
                    if self.state.board[r][c] is None:
                        if (r, c) not in legal_cell_actions:
                            legal_cell_actions.append((r, c))

        # Discards for dead-end non-jack cards
        for idx, card in enumerate(hand):
            if is_two_eyed_jack(card) or is_one_eyed_jack(card):
                continue
            positions = [(r, c) for r in range(10) for c in range(10) if BOARD_LAYOUT[r][c] == card]
            can_place = any(self.state.board[r][c] is None for (r, c) in positions)
            if not can_place:
                legal_discards.append(idx)

        # If nothing placeable/removable, allow discarding anything
        if not legal_cell_actions and not legal_removals:
            for idx in range(len(hand)):
                if idx not in legal_discards:
                    legal_discards.append(idx)

        return {"place": legal_cell_actions, "remove": legal_removals, "discard": legal_discards}

    # ---- Sequence bookkeeping helpers --------------------------------------

    def _register_sequence_meta(self, team: int, cells: List[Tuple[int, int]], axis: str) -> bool:
        if self.state is None:
            return False
        key = frozenset((int(r), int(c)) for (r, c) in cells)
        if key in self._seq_meta_index:
            return False
        self._seq_meta_index.add(key)

        seq_id = len(self.state.sequencesMeta)
        self.state.sequencesMeta.append({
            "id": seq_id,
            "team": int(team),
            "cells": [(int(r), int(c)) for (r, c) in cells],
            "axis": axis,
        })
        # Keep legacy set in sync
        for (rr, cc) in cells:
            self.state.sequence_cells.add((int(rr), int(cc)))
        # Bump canonical per-team counter
        self.state.sequences_count[team] = self.state.sequences_count.get(team, 0) + 1
        return True

    def _recompute_sequences_full(self) -> None:
        """Rescan entire board and rebuild sequence meta/counts from scratch."""
        if self.state is None or self.game_config is None:
            return
        # Clear all sequence bookkeeping
        self.state.sequencesMeta = []
        self.state.sequence_cells = set()
        self.state.sequences_count = {team_idx: 0 for team_idx in range(self.game_config.teams)}
        self._seq_meta_index.clear()

        # For each team, scan lines in each direction; start only at line heads
        for team in range(self.game_config.teams):
            for r in range(10):
                for c in range(10):
                    if not self._cell_counts_for_team(r, c, team):
                        continue
                    for dr, dc, axis in _DIRECTIONS:
                        pr, pc = r - dr, c - dc
                        # Only start at the head of a run
                        if 0 <= pr < 10 and 0 <= pc < 10 and self._cell_counts_for_team(pr, pc, team):
                            continue
                        run = self._gather_run(r, c, dr, dc, team)
                        if len(run) >= 5:
                            self._register_sequence_meta(team, run, axis)

        # Update winners from rebuilt counts
        needed = self.game_config.win_sequences_needed
        winners = [t for t, cnt in self.state.sequences_count.items() if cnt >= needed]
        self.state.winners = winners

    # ---- Main step ----------------------------------------------------------

    def step(self, action: Dict[str, Any]):
        if self.state is None or self.game_config is None or self.deck is None:
            raise RuntimeError("Game not started")
        player = self.state.current_player
        team = self._player_team(player)
        hand = self.state.hands[player]
        move_type = action.get("type")
        card: Optional[str] = action.get("card")
        target = action.get("target")
        removed = action.get("removed")
        move_record: Dict[str, Any] = {"player": player, "team": team, "type": move_type}

        if move_type == "timeout-skip":
            self._advance_turn()
            return self.state, move_record

        if move_type == "burn":
            if card not in hand:
                raise EngineError(ErrorCode.ERR_CARD_NOT_IN_HAND, details={"card": card})
            if is_two_eyed_jack(card) or is_one_eyed_jack(card):
                raise EngineError(ErrorCode.ERR_INVALID_JACK_USE, details={"as": "burn", "card": card})
            hand.remove(card)
            self.deck.discard(card)
            new_card = self.deck.draw()
            if new_card:
                hand.append(new_card)
            self._advance_turn()
            return self.state, move_record

        if move_type in ("play", "wild"):
            if card not in hand:
                raise EngineError(ErrorCode.ERR_CARD_NOT_IN_HAND, details={"card": card})
            if target is None:
                raise EngineError(ErrorCode.ERR_TARGET_OCCUPIED)
            r = int(target["r"]); c = int(target["c"])
            if not (0 <= r < 10 and 0 <= c < 10):
                raise EngineError(ErrorCode.ERR_TARGET_OCCUPIED, details={"r": r, "c": c})
            if self.state.board[r][c] is not None:
                raise EngineError(ErrorCode.ERR_TARGET_OCCUPIED, details={"r": r, "c": c})
            if is_two_eyed_jack(card):
                if BOARD_LAYOUT[r][c] == "BONUS":
                    raise EngineError(ErrorCode.ERR_INVALID_JACK_USE, details={"reason": "wild_on_bonus", "r": r, "c": c})
            elif is_one_eyed_jack(card):
                raise EngineError(ErrorCode.ERR_INVALID_JACK_USE, details={"as": "wild/play", "card": card})
            else:
                expected = BOARD_LAYOUT[r][c]
                if expected != card:
                    raise EngineError(ErrorCode.ERR_NOT_MATCHING_CARD, details={"expected": expected, "card": card, "r": r, "c": c})

            # Place chip
            hand.remove(card)
            self.state.board[r][c] = team
            self.deck.discard(card)
            new_card = self.deck.draw()
            if new_card:
                hand.append(new_card)

            # Incremental sequence detection (fast path)
            created_sequence = False
            prev_seq = set(self.state.sequence_cells)
            for dr, dc, axis in _DIRECTIONS:
                run = self._gather_run(r, c, dr, dc, team)
                new_cells = [(rr, cc) for (rr, cc) in run if (rr, cc) not in prev_seq]
                if len(new_cells) >= 5:
                    if self._register_sequence_meta(team, run, axis):
                        created_sequence = True

            # Winners
            needed = self.game_config.win_sequences_needed
            winners: List[int] = [t for t, cnt in self.state.sequences_count.items() if cnt >= needed]
            self.state.winners = winners

            # Full board reset (no winner) — clear ALL sequence bookkeeping too
            full = all((self.state.board[i][j] is not None or BOARD_LAYOUT[i][j] == "BONUS")
                       for i in range(10) for j in range(10))
            if full and not winners and self.game_config.reset_full_board_no_winner:
                for i in range(10):
                    for j in range(10):
                        if BOARD_LAYOUT[i][j] != "BONUS":
                            self.state.board[i][j] = None
                self.state.turns_count += 1
                self.deck.cards = self.deck.discard_pile.copy()
                self.deck.discard_pile.clear()
                self.deck.burned_cards.clear()
                self.deck.shuffle(seed=self.random_seed)
                # Clear sequences (both meta and counts)
                self.state.sequencesMeta = []
                self.state.sequence_cells = set()
                self.state.sequences_count = {team_idx: 0 for team_idx in range(self.game_config.teams)}
                self._seq_meta_index.clear()

            move_record["card"] = card
            move_record["target"] = {"r": r, "c": c}
            move_record["removed"] = None
            move_record["sequence_created"] = created_sequence
            if not winners:
                self._advance_turn()
            return self.state, move_record

        if move_type == "jack-remove":
            if card not in hand:
                raise EngineError(ErrorCode.ERR_CARD_NOT_IN_HAND, details={"card": card})
            if not is_one_eyed_jack(card):
                raise EngineError(ErrorCode.ERR_INVALID_JACK_USE, details={"as": "jack-remove", "card": card})
            if removed is None:
                raise EngineError(ErrorCode.ERR_TARGET_OCCUPIED)
            rr = int(removed["r"]); cc = int(removed["c"])
            if not (0 <= rr < 10 and 0 <= cc < 10):
                raise EngineError(ErrorCode.ERR_NOT_MATCHING_CARD, details={"r": rr, "c": cc})
            if self.state.board[rr][cc] is None:
                raise EngineError(ErrorCode.ERR_NOT_MATCHING_CARD, details={"r": rr, "c": cc})
            if self.state.board[rr][cc] == team:
                raise EngineError(ErrorCode.ERR_CANNOT_REMOVE_OWN_CHIP, details={"r": rr, "c": cc})
            allow_adv = bool(self.game_config.allowAdvancedJack)
            if not allow_adv and (rr, cc) in self.state.sequence_cells:
                raise EngineError(ErrorCode.ERR_INVALID_JACK_USE, details={"reason": "cell_in_sequence", "r": rr, "c": cc})

            # Remove chip
            self.state.board[rr][cc] = None
            hand.remove(card)
            self.deck.discard(card)
            new_card = self.deck.draw()
            if new_card:
                hand.append(new_card)

            # >>> NEW: recompute sequences fully after removals (can break sequences)
            self._recompute_sequences_full()

            move_record["card"] = card
            move_record["target"] = None
            move_record["removed"] = {"r": rr, "c": cc}
            # Optional: you can mark whether any sequence was lost/changed by comparing before/after if you store a snapshot
            self._advance_turn()
            return self.state, move_record

        raise EngineError(ErrorCode.ERR_UNKNOWN_MOVE, details={"move_type": move_type})

    def _advance_turn(self):
        if self.state is None or self.game_config is None:
            return
        total_players = self.game_config.teams * self.game_config.players_per_team
        self.state.current_player = (self.state.current_player + 1) % total_players
        self.state.turns_count += 1

    def _cell_counts_for_team(self, r: int, c: int, team: int) -> bool:
        if self.state is None:
            return False
        if BOARD_LAYOUT[r][c] == "BONUS":
            return True
        chip = self.state.board[r][c]
        return (chip is not None) and (chip == team)

    def _gather_run(self, r: int, c: int, dr: int, dc: int, team: int):
        run: List[Tuple[int, int]] = []
        rr, cc = r - dr, c - dc
        while 0 <= rr < 10 and 0 <= cc < 10 and self._cell_counts_for_team(rr, cc, team):
            run.insert(0, (rr, cc))
            rr -= dr; cc -= dc
        run.append((r, c))
        rr, cc = r + dr, c + dc
        while 0 <= rr < 10 and 0 <= cc < 10 and self._cell_counts_for_team(rr, cc, team):
            run.append((rr, cc))
            rr += dr; cc += dc
        return run

    def is_terminal(self) -> bool:
        return self.state is not None and len(self.state.winners) > 0

    def winner_teams(self) -> List[int]:
        if self.state is None:
            return []
        return list(self.state.winners)

# re-export helpers
BOARD_LAYOUT = BOARD_LAYOUT
is_one_eyed_jack = is_one_eyed_jack
is_two_eyed_jack = is_two_eyed_jack
EngineError = EngineError
