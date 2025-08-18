# Sequence RL Project

This project provides a complete implementation of the classic **Sequence** board game with a desktop user interface and a built‑in reinforcement learning agent.  The game is played on a 10×10 grid where each space (except the four corners) corresponds to a playing card from two standard 52‑card decks.  Jacks are special: two‑eyed Jacks allow a player to place a chip on any empty space, while one‑eyed Jacks remove an opponent’s chip that isn’t part of a completed sequence.  Players (or teams) compete to form rows of five chips, counting the corner spaces as wild for everyone.

## Features

* **Original card layout** – The board layout used here matches the commercial Sequence board.  Each non‑Jack card appears **exactly twice**, and the four corners are free spaces.  The layout data is stored in a JSON file under `assets/layouts/sequence_layout.json` and was extracted from an open‑source reference implementation【433063295206110†L444-L455】【433063295206110†L454-L465】.
* **Tkinter UI** – A simple GUI allows players to configure the number of teams (up to three), assign human or AI control to each team, start a game and play by clicking on their cards and the board.  Chips are displayed in blue, red and yellow according to the team.
* **Complete rules** – The engine enforces all core rules: placing chips on the two matching card spaces, using one‑eyed and two‑eyed Jacks appropriately, discarding dead cards, sequence detection (including overlapping sequences), and victory conditions based on the number of required sequences (two sequences for two teams, one sequence for three teams)【47690820291022†L80-L88】【47690820291022†L189-L206】.
* **Reinforcement learning agent** – A small convolutional neural network (implemented in pure NumPy) can be trained via self‑play to serve as a computer opponent.  The agent observes the full board state, applies an action mask to forbid illegal moves and learns via a simple policy‑gradient algorithm.  Pretrained weights can be found under `assets/weights/`.

## Running the project

Install the required Python packages from `requirements.txt`:

```bash
pip install -r requirements.txt
```

Start the GUI with:

```bash
python main.py
```

The start screen will prompt for the number of teams and whether each team is controlled by a human or the AI.  Once the game begins, click a card in your hand and then click the corresponding board cell to place a chip.  Right‑click a card to discard it if both board spaces are already occupied.  You can press **N** on your keyboard to advance the game by letting an AI player make its move, and press **T** to run a quick self‑play training session during a game.  Training plays a series of games between the learning agent and a heuristic opponent, improving the neural network and saving the updated weights to `assets/weights/agent_default.npz`.

## Directory layout

```
sequence_rl_project/
├── main.py                # Launches the UI
├── requirements.txt       # Third‑party dependencies
├── README.md              # This file
├── assets/
│   ├── layouts/
│   │   └── sequence_layout.json  # Exact card layout (with metadata)
│   ├── cards/             # Generated card images (created on first run)
│   └── weights/           # Saved neural network weights
└── sequence/
    ├── __init__.py
    ├── board_layouts.py   # Loads the official board layout
    ├── game.py            # Game rules, move validation and sequence detection
    ├── cards.py           # Procedural generation of card images
    ├── ui.py              # Tkinter based user interface
    └── rl/
        ├── __init__.py
        ├── model_numpy.py # Simple CNN implementation in NumPy
        ├── agent.py       # Reinforcement learning agent (policy + mask)
        ├── train.py       # Self‑play training routine
        └── eval.py        # Evaluate the agent against a baseline
```
