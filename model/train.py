"""Placeholder training script for reinforcement learning agent.

This script will eventually contain Stable-Baselines3 training code.  For now
it simply loads the default board layout and prints a message."""

from __future__ import annotations

from apps.server.sequence.board_layouts import load_layout


def main() -> None:
    board = load_layout()
    print(f"Loaded layout with {len(board)} rows")


if __name__ == "__main__":
    main()
