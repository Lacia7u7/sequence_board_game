# Sequence RL Project

This project provides a complete implementation of the classic **Sequence** board game with a desktop user interface and a built‑in reinforcement learning agent.  The game is played on a 10×10 grid where each space (except the four corners) corresponds to a playing card from two standard 52‑card decks.  Jacks are special: two‑eyed Jacks allow a player to place a chip on any empty space, while one‑eyed Jacks remove an opponent’s chip that isn’t part of a completed sequence.  Players (or teams) compete to form rows of five chips, counting the corner spaces as wild for everyone.

## Features

* **Original card layout** – The board layout used here matches the commercial Sequence board.  Each non‑Jack card appears **exactly twice**, and the four corners are free spaces.  The layout data is stored in a JSON file under `assets/layouts/sequence_layout.json` and was extracted from an open‑source reference implementation【433063295206110†L444-L455】【433063295206110†L454-L465】.
* **Web UI (Flask + React)** – The game can now be played in a browser.  A lightweight Flask server exposes the game state while a React front end renders the board and allows players to make moves from `http://localhost:5000`.
* **Complete rules** – The engine enforces all core rules: placing chips on the two matching card spaces, using one‑eyed and two‑eyed Jacks appropriately, discarding dead cards, sequence detection (including overlapping sequences), and victory conditions based on the number of required sequences (two sequences for two teams, one sequence for three teams)【47690820291022†L80-L88】【47690820291022†L189-L206】.
* **Reinforcement learning agent** – A small convolutional neural network (implemented in pure NumPy) can be trained via self‑play to serve as a computer opponent.  The agent observes the full board state, applies an action mask to forbid illegal moves and learns via a simple policy‑gradient algorithm.  Pretrained weights can be found under `assets/weights/`.

## Running the project

Install the required Python packages from `requirements.txt`:

```bash
pip install -r requirements.txt
```

Start the web interface with:

```bash
python app.py
```

Then open [http://localhost:5000](http://localhost:5000) in your browser to play.  The React front end displays the board and your current hand.  Click any board cell to be prompted for which card to play.

## Directory layout

```
sequence_rl_project/
├── app.py                 # Flask backend serving the web UI
├── frontend/              # React front end served as static files
├── main.py                # (legacy) Launches the Tkinter UI
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
