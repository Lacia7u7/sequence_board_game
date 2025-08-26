#!/usr/bin/env python3
"""
Create the `training/` directory structure and empty files as requested.
"""

from pathlib import Path
import argparse
import sys

FILE_PATHS = [
    "training/README.md",
    "training/configs/default.json",
    "training/configs/tiny-smoke.json",
    "training/configs/debug-fast.json",

    "training/engine/board_layout.py",
    "training/engine/cards.py",
    "training/engine/errors.py",
    "training/engine/engine_core.py",
    "training/engine/rules.py",
    "training/engine/rng.py",
    "training/engine/state.py",

    "training/encoders/board_encoder.py",
    "training/encoders/probability_layers.py",
    "training/encoders/hand_encoder.py",
    "training/encoders/history_encoder.py",
    "training/encoders/threat_features.py",

    "training/envs/sequence_env.py",
    "training/envs/vectorized_env.py",
    "training/envs/action_mapping.py",
    "training/envs/masks.py",

    "training/algorithms/ppo_lstm/ppo_lstm_policy.py",
    "training/algorithms/ppo_lstm/storage.py",
    "training/algorithms/ppo_lstm/learner.py",
    "training/algorithms/ppo_lstm/losses.py",
    "training/algorithms/ppo_lstm/utils.py",

    "training/algorithms/baselines/random_policy.py",
    "training/algorithms/baselines/greedy_sequence_policy.py",
    "training/algorithms/baselines/blocking_policy.py",

    # `nfsp/` is listed as a skeleton; create the directory (no specific files).
    # We'll create the directory below in EXTRA_DIRS.

    "training/agents/selfplay_manager.py",
    "training/agents/opponent_pool.py",
    "training/agents/rating.py",

    "training/scripts/train.py",
    "training/scripts/eval.py",
    "training/scripts/selfplay.py",
    "training/scripts/benchmark.py",
    "training/scripts/encode_sample.py",

    "training/tests/test_engine_rules.py",
    "training/tests/test_sequences.py",
    "training/tests/test_jacks.py",
    "training/tests/test_dead_cards.py",
    "training/tests/test_masks.py",
    "training/tests/test_env_api.py",
    "training/tests/test_determinism.py",
    "training/tests/test_encodings.py",
    "training/tests/test_probability_layers.py",
    "training/tests/test_action_mapping.py",

    "training/utils/logging.py",
    "training/utils/tb_writer.py",
    "training/utils/jsonio.py",
    "training/utils/timers.py",
    "training/utils/seeding.py",
    "training/utils/replay_io.py",
]

EXTRA_DIRS = [
    "training/algorithms/nfsp",  # create empty nfsp directory (skeleton)
]


def main(root: Path):
    created = []
    skipped = []

    # Create any explicit directories first
    for d in EXTRA_DIRS:
        dirpath = root / Path(d)
        dirpath.mkdir(parents=True, exist_ok=True)
        created.append(str(dirpath))

    # Create files (ensure parent dirs exist)
    for rel in FILE_PATHS:
        path = root / Path(rel)
        path.parent.mkdir(parents=True, exist_ok=True)
        try:
            # touch will create the file if missing, otherwise update mtime
            if path.exists():
                skipped.append(str(path))
            else:
                path.touch()
                created.append(str(path))
        except OSError as e:
            print(f"ERROR creating {path}: {e}", file=sys.stderr)

    # Summary
    print("\nCreation summary:")
    if created:
        print(f"  Created ({len(created)}):")
        for p in created:
            print(f"    - {p}")
    if skipped:
        print(f"\n  Skipped (already existed) ({len(skipped)}):")
        for p in skipped:
            print(f"    - {p}")
    print("\nDone.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Create training/ tree with empty files.")
    parser.add_argument("--root", type=str, default=".",
                        help="Root directory to create the tree in (default: current dir).")
    args = parser.parse_args()
    root_path = Path(args.root).resolve()
    main(root_path)
