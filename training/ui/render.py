from __future__ import annotations
# training/ui/renderer.py
from __future__ import annotations
from typing import Optional, Dict, Any

BOX_H = "─"; BOX_V = "│"; BOX_TL = "┌"; BOX_TR = "┐"; BOX_BL = "└"; BOX_BR = "┘"; BOX_C = "┼"; BOX_T = "┬"; BOX_B = "┴"; BOX_L = "├"; BOX_R = "┤"

class ConsoleRenderer:
    def render_env(self, env, info: Optional[Dict[str, Any]] = None) -> None:
        st = env._state()
        print("\n=== Sequence ===")
        # board
        print(BOX_TL + (BOX_H * 39) + BOX_TR)
        for r in range(10):
            row = []
            for c in range(10):
                chip = st.board[r][c]
                row.append("." if chip is None else str(int(chip)))
            print(BOX_V, " ".join(row), BOX_V)
        print(BOX_BL + (BOX_H * 39) + BOX_BR)
        # current hand of current_player
        seat = info.get("current_player", 0) if info else 0
        hand = st.hands[seat] if 0 <= seat < len(st.hands) else []
        print(f"Seat {seat} hand:", hand)

class SequenceRenderer:
    def __init__(self, env):
        self.env = env

    def draw(self) -> None:
        st = self.env._state()
        if st is None:
            print("<no state>")
            return
        print("\n   " + " ".join(str(c) for c in range(10)))
        for r in range(10):
            row = []
            for c in range(10):
                chip = st.board[r][c]
                row.append("." if chip is None else str(int(chip)))
            print(f"{r:2d} " + " ".join(row))

    def draw_hand(self, seat: int) -> None:
        st = self.env._state()
        hand = st.hands[seat] if hasattr(st, "hands") and seat < len(st.hands) else []
        print(f"Seat {seat} hand: {', '.join(hand) if hand else '<empty>'}")
