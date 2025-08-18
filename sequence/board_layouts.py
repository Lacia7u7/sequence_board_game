"""Board layout utilities.

This module loads the official Sequence board layout from a JSON file located
under ``assets/layouts/sequence_layout.json``.  The JSON contains metadata
describing the source of the layout and the 10×10 grid itself.  Each entry
in the grid is a two‑character string like ``"6S"`` (the six of spades) or
``"W"`` to denote a free corner.  Jacks are not present on the board; their
behaviour is handled separately in ``game.py``.

If the layout file is missing or cannot be parsed, a simple fallback layout
generator can be used to create a board with each non‑Jack card appearing
exactly twice in a spiral around the edges.  The fallback ensures the game
remains playable but does not exactly match the commercial layout.
"""

from __future__ import annotations

import json
import os
from typing import List, Dict, Tuple

ROOT_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
LAYOUT_PATH = os.path.join(ROOT_DIR, "assets", "layouts", "sequence_layout.json")

def load_layout() -> List[List[str]]:
    """Load the 10×10 card layout from the JSON file.

    Returns a nested list of strings.  If the file does not exist or is
    malformed the fallback layout will be generated and returned instead.
    """
    try:
        with open(LAYOUT_PATH, "r", encoding="utf-8") as f:
            data = json.load(f)
        board = data.get("board")
        if not board or len(board) != 10 or any(len(row) != 10 for row in board):
            raise ValueError("invalid board dimensions")
        return board
    except Exception:
        # Fallback: generate a simple mirrored layout
        return _generate_fallback_layout()

def card_positions(board: List[List[str]]) -> Dict[str, List[Tuple[int, int]]]:
    """Compute the list of positions for each card value on the board.

    Returns a dictionary mapping card codes (e.g. ``"2S"``) to a list of
    coordinates ``(row, col)``.  Free corners (``"W"``) are omitted.
    """
    positions: Dict[str, List[Tuple[int, int]]] = {}
    for y, row in enumerate(board):
        for x, val in enumerate(row):
            if val == "W":
                continue
            positions.setdefault(val, []).append((y, x))
    return positions

def _generate_fallback_layout() -> List[List[str]]:
    """Generate a simple fallback layout when the official JSON is unavailable.

    This function arranges each non‑Jack card twice on the perimeter of the board
    and fills the centre with a repeating pattern.  Corners are marked with
    ``"W"`` to denote free spaces.  The fallback is deterministic but does not
    match the commercial board.
    """
    # Define ranks and suits excluding jacks
    ranks = ["A", "2", "3", "4", "5", "6", "7", "8", "9", "T", "Q", "K"]
    suits = ["S", "H", "D", "C"]
    cards = [r + s for r in ranks for s in suits if r != "J"]
    # Place cards around the perimeter twice
    board: List[List[str]] = [["" for _ in range(10)] for _ in range(10)]
    # Set corners
    for corner in [(0, 0), (0, 9), (9, 0), (9, 9)]:
        board[corner[0]][corner[1]] = "W"
    idx = 0
    # Top row
    for x in range(1, 9):
        board[0][x] = cards[idx % len(cards)]
        idx += 1
    # Right column
    for y in range(1, 9):
        board[y][9] = cards[idx % len(cards)]
        idx += 1
    # Bottom row
    for x in reversed(range(1, 9)):
        board[9][x] = cards[idx % len(cards)]
        idx += 1
    # Left column
    for y in reversed(range(1, 9)):
        board[y][0] = cards[idx % len(cards)]
        idx += 1
    # Fill interior with remaining cards
    for y in range(1, 9):
        for x in range(1, 9):
            board[y][x] = cards[idx % len(cards)]
            idx += 1
    return board

if __name__ == "__main__":
    # Simple CLI demonstration
    layout = load_layout()
    for row in layout:
        print(" ".join(f"{c:>3}" for c in row))