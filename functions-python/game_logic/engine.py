"""
Game engine utilities for the Sequence online backend.

This module defines helper functions to manipulate cards, decks, hands,
validate moves and update the board and game state. The goal is to encapsulate
the rules of the game in a testable manner independent of the HTTP layer.
"""

import random
from typing import Any, Dict, List, Optional, Tuple

# Define card ranks and suits for a standard deck
RANKS = ["A", "2", "3", "4", "5", "6", "7", "8", "9", "10", "J", "Q", "K"]
SUITS = ["♠", "♥", "♦", "♣"]


def create_full_deck() -> List[str]:
    """Creates a list of 52 cards represented as strings like 'A♠' or '10♦'."""
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
    """Applies a move to the game state and returns the updated state and move record.

    This function performs rule validations and updates the state dictionary. It
    raises ValueError for invalid moves. It does not advance the turn; caller
    must handle turn rotation.

    The returned lastMove dict contains a summary of the move for storage.
    """
    import time
    from firebase_admin import firestore  # needed for SERVER_TIMESTAMP
    # Clone state
    new_state: Dict[str, Any] = {
        k: (v.copy() if k != "hands" else {hk: hv[:] for hk, hv in v.items()})
        for k, v in state.items()
    }
    hands = new_state.get("hands", {})
    board = new_state.get("board")
    deck = new_state.get("deck", {})
    discard_pile: List[str] = deck.get("discardPile", [])
    burned_cards: List[str] = deck.get("burnedCards", [])
    last_move: Dict[str, Any] = {
        "bySeatId": seat_id,
        "clientTs": int(time.time() * 1000),
        "serverTs": firestore.SERVER_TIMESTAMP,
        "valid": True,
    }
    # Utility to draw a card from deck
    def draw_card():
        nonlocal deck
        if not deck.get("cards") or len(deck["cards"]) == 0:
            # Reshuffle from discard pile if available, excluding burned cards
            reshuffle_cards = [c for c in discard_pile if c not in burned_cards]
            random.shuffle(reshuffle_cards)
            deck["cards"] = reshuffle_cards
            discard_pile.clear()
        if deck.get("cards"):
            return deck["cards"].pop()
        return None
    # TIMEOUT skip: record and return
    if move_type == "timeout-skip":
        last_move.update({
            "type": "timeout-skip",
            "card": None,
            "target": None,
            "removed": None,
        })
        return new_state, last_move
    # Burn: discard card intentionally when no legal moves
    if move_type == "burn":
        if card not in hands.get(seat_id, []):
            raise ValueError("ERR_CARD_BLOCKED")
        if not has_no_legal_moves(hands.get(seat_id, []), board):
            raise ValueError("ERR_NOT_MATCHING_CARD")
        hands[seat_id].remove(card)
        burned_cards.append(card)
        # Draw new card if available
        new_card = draw_card()
        if new_card:
            hands[seat_id].append(new_card)
        last_move.update({
            "type": "burn",
            "card": card,
            "target": None,
            "removed": None,
        })
        # Update deck lists
        deck["burnedCards"] = burned_cards
        deck["discardPile"] = discard_pile
        deck["cards"] = deck.get("cards", [])
        return new_state, last_move
    # For play/wild and jack-remove, target or removed is required
    if move_type in ("play", "wild"):
        if not target:
            raise ValueError("ERR_TARGET_OCCUPIED")
        r = int(target["r"])
        c = int(target["c"])
        cell = board[r][c]
        # Check occupancy
        if cell.get("chip") is not None:
            raise ValueError("ERR_TARGET_OCCUPIED")
        # Determine card type: if J, maybe wild or remove
        # Two-eyed jacks: hearts or diamonds
        is_jack = card.startswith("J")
        if move_type == "play":
            if is_jack:
                # If it's a jack, treat as play; but type should be 'wild' or 'jack-remove'
                pass
            else:
                # Validate card matches board cell card
                if cell.get("card") and cell["card"] != card:
                    raise ValueError("ERR_NOT_MATCHING_CARD")
        # Remove card from hand
        if card not in hands.get(seat_id, []):
            raise ValueError("Card not in hand")
        hands[seat_id].remove(card)
        cell["chip"] = {"teamIndex": team_index}
        # Add to discard pile
        discard_pile.append(card)
        # Draw new card
        new_card = draw_card()
        if new_card:
            hands[seat_id].append(new_card)
        # Compute sequences and winners
        seqs, in_sequence_positions = compute_sequences(board, return_positions=True)
        new_state["sequences"] = seqs
        teams_num = int(config.get("teams", 2))
        meta_required = 2 if teams_num == 2 else 1
        winners: List[int] = [int(t) for t, count in seqs.items() if count >= meta_required]
        new_state["winners"] = winners
        # Reset board if full and no winner
        is_full = all(cell2.get("chip") is not None or cell2.get("card") == "BONUS" for row in board for cell2 in row)
        if is_full and not winners:
            # Remove chips but keep card mapping
            for row in board:
                for cell2 in row:
                    cell2["chip"] = None
            # Rebuild deck: full deck minus burned cards
            new_deck_cards = create_full_deck() + create_full_deck()
            # Remove burned cards
            new_deck_cards = [c for c in new_deck_cards if c not in burned_cards]
            random.shuffle(new_deck_cards)
            deck["cards"] = new_deck_cards
            discard_pile.clear()
            new_state["roundCount"] = new_state.get("roundCount", 0) + 1
            new_state["sequences"] = {str(i): 0 for i in range(teams_num)}
        last_move.update({
            "type": move_type,
            "card": card,
            "target": target,
            "removed": None,
        })
        # Write updates
        deck["discardPile"] = discard_pile
        deck["burnedCards"] = burned_cards
        deck["cards"] = deck.get("cards", [])
        return new_state, last_move
    if move_type == "jack-remove":
        if not removed:
            raise ValueError("ERR_TARGET_OCCUPIED")
        rr = int(removed["r"])
        cc = int(removed["c"])
        cell = board[rr][cc]
        if cell.get("chip") is None:
            raise ValueError("ERR_NOT_MATCHING_CARD")
        # advanced rule: cannot remove chip that is part of sequence if allowAdvancedJack is false
        # For now we allow removal; advanced rule enforcement can be added by caller
        # Remove chip
        cell["chip"] = None
        # Remove card from hand
        if card not in hands.get(seat_id, []):
            raise ValueError("Card not in hand")
        hands[seat_id].remove(card)
        discard_pile.append(card)
        new_card = draw_card()
        if new_card:
            hands[seat_id].append(new_card)
        seqs, _ = compute_sequences(board, return_positions=True)
        new_state["sequences"] = seqs
        last_move.update({
            "type": move_type,
            "card": card,
            "target": None,
            "removed": removed,
        })
        deck["discardPile"] = discard_pile
        deck["burnedCards"] = burned_cards
        deck["cards"] = deck.get("cards", [])
        return new_state, last_move
    raise ValueError("Unknown move type")


def compute_sequences(board: List[List[Dict[str, Any]]], return_positions: bool = False) -> Any:
    """Counts sequences (five in a row) for each team on the board.

    A sequence can be horizontal, vertical or diagonal. Bonus squares count as
    belonging to any team. A cell may belong to two sequences if lines cross.
    If return_positions is True, returns a tuple of (teams, positions) where
    positions is a set of coordinate tuples that are part of sequences.
    """
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


def has_no_legal_moves(hand: List[str], board: List[List[Dict[str, Any]]]) -> bool:
    """Determines whether a hand has no legal moves on the given board.

    This simplified implementation assumes any non-J card is legal if there is at least
    one matching card on the board that is empty. Jacks are always considered legal.
    """
    for card in hand:
        if card.startswith("J"):
            return False
        # Find any cell matching card and empty
        for row in board:
            for cell in row:
                if cell.get("chip") is None and cell.get("card") == card:
                    return False
    return True
