"""Entry point for the Sequence RL project.

Running this script will launch the Tkinter graphical user interface
defined in ``sequence/ui.py``.  From there you can configure a new game,
play against human or AI opponents, or train the RL agent on the fly.

Usage:
    python main.py
"""

from sequence.ui import GameUI


def main() -> None:
    ui = GameUI()
    ui.run()


if __name__ == "__main__":
    main()