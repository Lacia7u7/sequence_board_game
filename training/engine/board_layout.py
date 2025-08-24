"""
Defines the static card layout for the Sequence board (10x10 grid).
Each cell has a 'card' string or 'BONUS' for corner wild cells.
"""
from .cards import RANKS, SUITS

BOARD_LAYOUT = [[None for _ in range(10)] for _ in range(10)]
# Corners
for (r,c) in [(0,0),(0,9),(9,0),(9,9)]:
    BOARD_LAYOUT[r][c] = "BONUS"

non_jack_cards = [f"{rank}{suit}" for suit in SUITS for rank in RANKS if rank != "J"]
assert len(non_jack_cards) == 48

coords = []
N = 10
layer = 0
while layer < N/2:
    r = layer
    for c in range(layer, N-layer):
        if BOARD_LAYOUT[r][c] is None:
            coords.append((r,c))
    c = N - layer - 1
    for r in range(layer+1, N-layer-1):
        if BOARD_LAYOUT[r][c] is None:
            coords.append((r,c))
    r = N - layer - 1
    for c in range(N-layer-1, layer-1, -1):
        if BOARD_LAYOUT[r][c] is None:
            coords.append((r,c))
    c = layer
    for r in range(N-layer-2, layer, -1):
        if BOARD_LAYOUT[r][c] is None:
            coords.append((r,c))
    layer += 1

assert len(coords) == 96
for i, card in enumerate(non_jack_cards):
    r, c = coords[i]
    BOARD_LAYOUT[r][c] = card
    r2, c2 = coords[48 + i]
    BOARD_LAYOUT[r2][c2] = card

BOARD_LAYOUT = tuple(tuple(cell for cell in row) for row in BOARD_LAYOUT)
