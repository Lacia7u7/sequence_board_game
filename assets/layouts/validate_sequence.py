"""Validate the Sequence board layout.

This script checks that the layout defined in ``sequence.json`` meets the
requirements of the official Sequence board:

* Board is 10x10
* Four corners are wild (``"W"``)
* Each non-Jack card appears exactly twice
* No jacks are present on the board
"""

from __future__ import annotations

import json
from collections import Counter
from pathlib import Path

LAYOUT_FILE = Path(__file__).with_name("sequence.json")


def validate() -> None:
    board = json.loads(LAYOUT_FILE.read_text())
    assert len(board) == 10 and all(len(row) == 10 for row in board), "Board must be 10x10"
    assert board[0][0] == board[0][9] == board[9][0] == board[9][9] == "W", "Corners must be wild"

    cards: Counter[str] = Counter()
    for row in board:
        for card in row:
            if card == "W":
                continue
            assert not card.startswith("J"), f"Jacks may not appear on the board: {card}"
            cards[card] += 1

    for card, count in cards.items():
        assert count == 2, f"Card {card} appears {count} times"
    assert sum(cards.values()) == 96, "There should be 96 non-corner cells"


if __name__ == "__main__":
    validate()
    print("Layout is valid")
