# Sequence Game – Training Engine & Self-Play RL

This `training/` module provides a standalone game engine for Sequence (no UI or database) and a full self-play reinforcement learning stack. It includes the game rules, observation encoders, vectorized environment, PPO-LSTM algorithm, baseline agents, and training scripts.

## Quick Start

1. **Create a virtualenv** (Python 3.10+ recommended) and install:
   ```bash
   pip install -U pip
   pip install torch gymnasium numpy
   ```

2. **Run a Smoke Test**:
   ```bash
   python -m training.scripts.train --config training/configs/tiny-smoke.json
   ```

3. **Self-Play Baselines**:
   ```bash
   python -m training.scripts.selfplay --config training/configs/default.json
   ```

## Modes

- 1v1, 2v2, 1v1v1. Controlled via `configs/*.json`.

## Notes

- No UI/Firestore — pure Python.
- Deterministic with seeds.
- Unified action space (the **agent chooses a cell**; env resolves valid card deterministically).
