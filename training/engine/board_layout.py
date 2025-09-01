# training/engine/board_layout.py
"""
Loads the static card layout for the 10x10 Sequence board from JSON.
Each cell is a card code (e.g., '7H') or 'BONUS' for corner wild cells.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Tuple, List

__all__ = ["BOARD_LAYOUT"]

def _default_json_path() -> Path:
    # this file: training/engine/board_layout.py
    # assets:     training/assets/boards/standard_10x10.json
    return (Path(__file__).resolve().parents[1] / "assets" / "boards" / "standard_10x10.json")

def _load_board_layout_json(path: Path | None = None) -> List[List[str]]:
    src = Path(path) if path else _default_json_path()
    with src.open("r", encoding="utf-8") as f:
        data = json.load(f)

    rows = int(data.get("rows", 0))
    cols = int(data.get("cols", 0))
    cells = data.get("cells", None)

    if rows != 10 or cols != 10:
        raise ValueError(f"Board JSON must be 10x10, got {rows}x{cols} at {src}")

    if not (isinstance(cells, list) and len(cells) == 10 and all(isinstance(r, list) and len(r) == 10 for r in cells)):
        raise ValueError(f"Invalid 'cells' shape in {src}")

    return cells  # type: ignore

def _freeze(grid: List[List[str]]) -> Tuple[Tuple[str, ...], ...]:
    return tuple(tuple(cell for cell in row) for row in grid)

def _validate(grid: List[List[str]]) -> None:
    # Basic checks: corners are BONUS; 96 non-BONUS cells; each non-jack card appears exactly twice
    if grid[0][0] != "BONUS" or grid[0][9] != "BONUS" or grid[9][0] != "BONUS" or grid[9][9] != "BONUS":
        raise ValueError("Corners must be 'BONUS'")

    flattened = [cell for row in grid for cell in row if cell != "BONUS"]
    if len(flattened) != 96:
        raise ValueError(f"Expected 96 non-BONUS cells, found {len(flattened)}")

    from collections import Counter
    counts = Counter(flattened)
    # Jacks shouldn't be printed on the board; every non-jack card should appear twice
    bad = {card: n for card, n in counts.items() if card.endswith("J") or n != 2}
    if bad:
        raise ValueError(f"Card multiplicities invalid (expect each non-jack exactly twice): {bad}")

# Load, validate, and freeze
_GRID = _load_board_layout_json()
_validate(_GRID)
BOARD_LAYOUT = _freeze(_GRID)
