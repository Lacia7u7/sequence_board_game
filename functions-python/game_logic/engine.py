"""
Game engine utilities for the Sequence online backend.

This module defines helper functions to manipulate cards, decks, hands,
validate moves and update the board and game state. The goal is to encapsulate
the rules of the game in a testable manner independent of the HTTP layer.
"""

from typing import Any, Dict, List, Optional, Tuple
import random
import time

from .errors import ErrorCode, EngineError

# Define card ranks and suits for a standard deck
RANKS = ["A", "2", "3", "4", "5", "6", "7", "8", "9", "10", "J", "Q", "K"]
SUITS = ["S", "H", "D", "C"]


def create_full_deck() -> List[str]:
    """Creates a list of 52 cards represented as strings like '10D'."""
    return [f"{rank}{suit}" for suit in SUITS for rank in RANKS]


def shuffle_deck(deck: List[str], seed: Optional[int] = None) -> None:
    """Shuffles the deck in place using an optional seed for reproducibility."""
    rng = random.Random(seed)
    rng.shuffle(deck)


def deal_hands(deck: List[str], players: int, hand_size: int) -> List[List[str]]:
    """Deals hands to players, removing cards from the deck. Returns a list of hands."""
    hands = []
    for _ in range(players):
        hand = [deck.pop() for _ in range(hand_size)]
        hands.append(hand)
    return hands


def create_empty_board(rows: int, cols: int) -> List[List[Dict[str, Any]]]:
    """Creates an empty game board with given dimensions.

    Each cell is a dict: { 'card': str or 'BONUS', 'chip': None or {teamIndex:int} }
    Corners are marked as BONUS squares.
    """
    board: List[List[Dict[str, Any]]] = []
    for r in range(rows):
        row = []
        for c in range(cols):
            cell: Dict[str, Any] = {}
            if (r == 0 and c == 0) or (r == 0 and c == cols - 1) or (r == rows - 1 and c == 0) or (r == rows - 1 and c == cols - 1):
                cell["card"] = "BONUS"
                cell["chip"] = None
            else:
                cell["card"] = ""
                cell["chip"] = None
            row.append(cell)
        board.append(row)
    return board


def apply_move_to_state(
    state: Dict[str, Any],
    seat_id: str,
    team_index: int,
    move_type: str,
    card: Optional[str],
    target: Optional[Dict[str, int]],
    removed: Optional[Dict[str, int]],
    config: Dict[str, Any],
) -> Tuple[Dict[str, Any], Dict[str, Any]]:
    """Apply a move to the state and return (new_state, move_record).

    Rules implemented:
      - JC/JD (two-eyed): wild -> must place on a free, NON-BONUS cell.
      - JH/JS (one-eyed): remove an opponent chip (cannot remove chips that are in any sequence,
        unless config.allowAdvancedJack == True).
      - On placement, for each axis (H/V/D1/D2), if the contiguous run including the new chip
        (BONUS cells count as “in-run”) has at least 5 cells not already in any sequence, we
        create a NEW sequence for that axis. We add seq meta + all cells of that full run
        (including BONUS cells) to that sequence.
      - Jacks cannot be burned; burn is card-specific (only allowed if that card has no legal move).
      - Burned card goes to discard pile (and can be reshuffled later).
    """
    import time
    from firebase_admin import firestore  # for SERVER_TIMESTAMP

    # -------- helpers for jacks --------
    def is_two_eyed_jack(c: Optional[str]) -> bool:
        return isinstance(c, str) and c in ("JC", "JD")

    def is_one_eyed_jack(c: Optional[str]) -> bool:
        return isinstance(c, str) and c in ("JH", "JS")

    # -------- robust clone --------
    new_state: Dict[str, Any] = {}
    for k, v in state.items():
        if k == "hands" and isinstance(v, dict):
            new_state[k] = {hk: list(hv) for hk, hv in v.items()}
        elif isinstance(v, dict):
            new_state[k] = dict(v)
        elif isinstance(v, list):
            new_state[k] = list(v)
        else:
            new_state[k] = v

    # -------- ensure board (2D) --------
    board = new_state.get("board")
    if board is None:
        if "boardRows" in new_state:
            board = [row.get("cells", []) for row in new_state["boardRows"]]
        else:
            raise EngineError(ErrorCode.ERR_GENERIC, details={"reason": "no_board_in_state"})
        new_state["board"] = board

    hands: Dict[str, List[str]] = new_state.get("hands", {})
    deck: Dict[str, Any] = new_state.get("deck", {}) or {}
    discard_pile: List[str] = deck.get("discardPile", [])
    burned_cards: List[str] = deck.get("burnedCards", [])  # kept for compatibility

    # -------- sequence storage --------
    seq_cells = list(new_state.get("sequenceCells", []))
    seq_meta = list(new_state.get("sequencesMeta", []))
    cells_in_any_seq = {(e.get("r"), e.get("c")) for e in seq_cells if isinstance(e, dict)}

    last_move: Dict[str, Any] = {
        "bySeatId": seat_id,
        "clientTs": int(time.time() * 1000),
        "serverTs": firestore.SERVER_TIMESTAMP,
        "valid": True,
    }

    # -------- board geometry --------
    rows = len(board)
    cols = len(board[0]) if rows else 0
    directions = [
        (0, 1, "H"),
        (1, 0, "V"),
        (1, 1, "D1"),   # ↘
        (1, -1, "D2"),  # ↙
    ]

    def in_bounds(rr: int, cc: int) -> bool:
        return 0 <= rr < rows and 0 <= cc < cols

    def qualifies_for_team(rr: int, cc: int, team: int) -> bool:
        cell = board[rr][cc]
        if cell.get("card") == "BONUS":
            return True
        chip = cell.get("chip")
        return bool(chip and chip.get("teamIndex") == team)

    def draw_card() -> Optional[str]:
        nonlocal deck, discard_pile
        cards = deck.get("cards") or []
        if not cards:
            # reshuffle whole discard (burned are also in discard by policy)
            reshuffle = list(discard_pile)
            random.shuffle(reshuffle)
            deck["cards"] = reshuffle
            discard_pile.clear()
            cards = deck["cards"]
        if cards:
            return cards.pop()
        return None

    def gather_run_including(rr: int, cc: int, dr: int, dc: int, team: int) -> List[Tuple[int, int]]:
        """Return ordered positions of full contiguous run (BONUS counts) including (rr,cc)."""
        back = []
        r0, c0 = rr - dr, cc - dc
        while in_bounds(r0, c0) and qualifies_for_team(r0, c0, team):
            back.append((r0, c0))
            r0 -= dr
            c0 -= dc
        back.reverse()
        fwd = []
        r0, c0 = rr + dr, cc + dc
        while in_bounds(r0, c0) and qualifies_for_team(r0, c0, team):
            fwd.append((r0, c0))
            r0 += dr
            c0 += dc
        return back + [(rr, cc)] + fwd

    def has_no_legal_move_for_card(card_str: str) -> bool:
        allow_adv = bool(config.get("allowAdvancedJack", False))
        # JC/JD: legal if any free, NON-BONUS cell exists
        if is_two_eyed_jack(card_str):
            for rr in range(rows):
                for cc in range(cols):
                    if board[rr][cc].get("chip") is None and board[rr][cc].get("card") != "BONUS":
                        return False
            return True
        # JH/JS: legal if any opponent chip removable (not in any sequence unless allow_adv)
        if is_one_eyed_jack(card_str):
            for rr in range(rows):
                for cc in range(cols):
                    chip = board[rr][cc].get("chip")
                    if chip and chip.get("teamIndex") != team_index:
                        if allow_adv or ((rr, cc) not in cells_in_any_seq):
                            return False
            return True
        # normal: legal if any matching empty cell
        for rr in range(rows):
            for cc in range(cols):
                if board[rr][cc].get("card") == card_str and board[rr][cc].get("chip") is None:
                    return False
        return True

    # -------- timeout-skip --------
    if move_type == "timeout-skip":
        last_move.update({"type": "timeout-skip", "card": None, "target": None, "removed": None})
        # convert & return
        new_state["boardRows"] = [{"cells": row} for row in new_state["board"]]
        new_state.pop("board", None)
        new_state["sequencesMeta"] = seq_meta
        new_state["sequenceCells"] = seq_cells
        # derive sequences count per team from meta
        counts: Dict[str, int] = {}
        for m in seq_meta:
            t = str(m.get("teamIndex"))
            counts[t] = counts.get(t, 0) + 1
        new_state["sequences"] = counts
        return new_state, last_move

    # -------- burn (card-specific) --------
    if move_type == "burn":
        if card not in hands.get(seat_id, []):
            raise EngineError(ErrorCode.ERR_CARD_NOT_IN_HAND, details={"card": card})
        if is_two_eyed_jack(card) or is_one_eyed_jack(card):
            raise EngineError(ErrorCode.ERR_INVALID_JACK_USE, details={"as": "burn", "card": card})
        if not has_no_legal_move_for_card(card):
            raise EngineError(ErrorCode.ERR_CARD_BLOCKED, details={"card": card})

        hands[seat_id].remove(card)
        discard_pile.append(card)  # burned -> discard

        new_card = draw_card()
        if new_card:
            hands[seat_id].append(new_card)

        last_move.update({"type": "burn", "card": card, "target": None, "removed": None})
        deck["discardPile"] = discard_pile
        deck["burnedCards"] = burned_cards
        deck["cards"] = deck.get("cards", [])

        # persist shapes
        new_state["boardRows"] = [{"cells": row} for row in new_state["board"]]
        new_state.pop("board", None)
        new_state["sequencesMeta"] = seq_meta
        new_state["sequenceCells"] = seq_cells
        counts: Dict[str, int] = {}
        for m in seq_meta:
            t = str(m.get("teamIndex"))
            counts[t] = counts.get(t, 0) + 1
        new_state["sequences"] = counts
        return new_state, last_move

    # -------- play / wild placement --------
    if move_type in ("play", "wild"):
        if card not in hands.get(seat_id, []):
            raise EngineError(ErrorCode.ERR_CARD_NOT_IN_HAND, details={"card": card})
        if not target:
            raise EngineError(ErrorCode.ERR_TARGET_OCCUPIED)

        r = int(target["r"])
        c = int(target["c"])
        if not in_bounds(r, c):
            raise EngineError(ErrorCode.ERR_TARGET_OCCUPIED, details={"r": r, "c": c})
        cell = board[r][c]

        if cell.get("chip") is not None:
            raise EngineError(ErrorCode.ERR_TARGET_OCCUPIED, details={"r": r, "c": c})

        if is_two_eyed_jack(card):
            # Wild must be on NON-BONUS free cell
            if cell.get("card") == "BONUS":
                raise EngineError(
                    ErrorCode.ERR_INVALID_JACK_USE,
                    details={"reason": "wild_on_bonus", "r": r, "c": c},
                )
        elif is_one_eyed_jack(card):
            raise EngineError(ErrorCode.ERR_INVALID_JACK_USE, details={"as": "wild/play", "card": card})
        else:
            # normal must match printed card
            if cell.get("card") and cell["card"] != card:
                raise EngineError(
                    ErrorCode.ERR_NOT_MATCHING_CARD,
                    details={"expected": cell.get("card"), "card": card, "r": r, "c": c},
                )

        # place chip
        hands[seat_id].remove(card)
        cell["chip"] = {"teamIndex": team_index}
        discard_pile.append(card)

        # draw replacement
        new_card = draw_card()
        if new_card:
            hands[seat_id].append(new_card)

        # For each axis (max 4), decide if a *new* sequence is formed.
        created_any = False
        now_cells_in_seq = set(cells_in_any_seq)  # snapshot for this move's checks

        for dr, dc, axis in directions:
            run = gather_run_including(r, c, dr, dc, team_index)
            # How many cells in this run are not yet in ANY sequence?
            missing = [(rr, cc) for (rr, cc) in run if (rr, cc) not in now_cells_in_seq]
            if len(missing) >= 5:
                # brand new sequence in this axis
                seq_id = f"s-{int(time.time() * 1000)}-{axis}-{team_index}"
                seq_meta.append({"seqId": seq_id, "teamIndex": team_index, "axis": axis, "length": len(run)})
                for (rr, cc) in run:
                    seq_cells.append({"seqId": seq_id, "r": rr, "c": cc})
                    now_cells_in_seq.add((rr, cc))  # so a single move can still make 2 axes, not more per axis
                created_any = True
            # else: either extension of an older line (<5 brand-new cells) or too short to form a sequence

        # derive sequences per team & winners
        counts: Dict[str, int] = {}
        for m in seq_meta:
            t = str(m.get("teamIndex"))
            counts[t] = counts.get(t, 0) + 1
        new_state["sequences"] = counts

        teams_num = int(config.get("teams", 2))
        needed = 2 if teams_num == 2 else 1
        winners: List[int] = [int(t) for t, cnt in counts.items() if int(t) < teams_num and cnt >= needed]
        new_state["winners"] = winners

        # full-board reset if no winners
        is_full = all(
            (cell2.get("chip") is not None) or (cell2.get("card") == "BONUS")
            for row2 in board for cell2 in row2
        )
        if is_full and not winners:
            for row2 in board:
                for cell2 in row2:
                    cell2["chip"] = None
            new_deck_cards = create_full_deck() + create_full_deck()
            random.shuffle(new_deck_cards)
            deck["cards"] = new_deck_cards
            discard_pile.clear()
            new_state["roundCount"] = new_state.get("roundCount", 0) + 1
            seq_cells = []
            seq_meta = []
            new_state["sequences"] = {str(i): 0 for i in range(teams_num)}

        last_move.update({
            "type": "wild" if is_two_eyed_jack(card) else "play",
            "card": card,
            "target": {"r": r, "c": c},
            "removed": None,
        })

        deck["discardPile"] = discard_pile
        deck["burnedCards"] = burned_cards
        deck["cards"] = deck.get("cards", [])

    # -------- jack-remove --------
    elif move_type == "jack-remove":
        if card not in hands.get(seat_id, []):
            raise EngineError(ErrorCode.ERR_CARD_NOT_IN_HAND, details={"card": card})
        if not is_one_eyed_jack(card):
            raise EngineError(ErrorCode.ERR_INVALID_JACK_USE, details={"as": "jack-remove", "card": card})
        if not removed:
            raise EngineError(ErrorCode.ERR_TARGET_OCCUPIED)

        rr = int(removed["r"])
        cc = int(removed["c"])
        if not in_bounds(rr, cc):
            raise EngineError(ErrorCode.ERR_NOT_MATCHING_CARD, details={"r": rr, "c": cc})
        tgt = board[rr][cc]
        chip = tgt.get("chip")

        if chip is None:
            raise EngineError(ErrorCode.ERR_NOT_MATCHING_CARD, details={"r": rr, "c": cc})
        if chip.get("teamIndex") == team_index:
            raise EngineError(ErrorCode.ERR_CANNOT_REMOVE_OWN_CHIP, details={"r": rr, "c": cc})

        allow_adv = bool(config.get("allowAdvancedJack", False))
        if not allow_adv and (rr, cc) in cells_in_any_seq:
            raise EngineError(
                ErrorCode.ERR_INVALID_JACK_USE,
                details={"reason": "cell_in_sequence", "r": rr, "c": cc},
            )

        # remove
        tgt["chip"] = None
        hands[seat_id].remove(card)
        discard_pile.append(card)

        new_card = draw_card()
        if new_card:
            hands[seat_id].append(new_card)

        last_move.update({"type": "jack-remove", "card": card, "target": None, "removed": {"r": rr, "c": cc}})
        deck["discardPile"] = discard_pile
        deck["burnedCards"] = burned_cards
        deck["cards"] = deck.get("cards", [])
    else:
        raise EngineError(ErrorCode.ERR_UNKNOWN_MOVE, details={"move_type": move_type})

    # -------- persist board & sequences --------
    new_state["boardRows"] = [{"cells": row} for row in new_state["board"]]
    new_state.pop("board", None)
    new_state["sequencesMeta"] = seq_meta
    new_state["sequenceCells"] = seq_cells

    return new_state, last_move


def compute_sequences(board: List[List[Dict[str, Any]]], return_positions: bool = False) -> Any:
    """Counts sequences (five in a row) for each team on the board."""
    teams: Dict[str, int] = {}
    seq_positions: set = set()
    rows = len(board)
    cols = len(board[0]) if rows else 0
    directions = [(0, 1), (1, 0), (1, 1), (1, -1)]
    for r in range(rows):
        for c in range(cols):
            cell = board[r][c]
            chip = cell.get("chip")
            team = chip.get("teamIndex") if chip else None
            for dr, dc in directions:
                count = 0
                coords: List[Tuple[int, int]] = []
                for i in range(5):
                    nr = r + dr * i
                    nc = c + dc * i
                    if nr < 0 or nr >= rows or nc < 0 or nc >= cols:
                        break
                    ncell = board[nr][nc]
                    if ncell.get("card") == "BONUS":
                        count += 1
                        coords.append((nr, nc))
                    elif ncell.get("chip") and ncell["chip"].get("teamIndex") == team:
                        count += 1
                        coords.append((nr, nc))
                    else:
                        break
                if count == 5 and team is not None:
                    teams[str(team)] = teams.get(str(team), 0) + 1
                    seq_positions.update(coords)
    if return_positions:
        return teams, seq_positions
    return teams


def has_no_legal_moves_in_hand(hand: List[str], board: List[List[Dict[str, Any]]]) -> bool:
    """Determines whether a hand has no legal moves on the given board.

    - Any Jack is always a legal move (wild or remove), so if hand has a J -> False.
    - Any non-J card is legal if there is at least one matching card on the board that is empty.
    """
    for card in hand:
        if card.startswith("J"):
            # With JC/JD wild and JH/JS remove, there's always a legal action with a Jack.
            return False
        for row in board:
            for cell in row:
                if cell.get("chip") is None and cell.get("card") == card:
                    return False
    return True
