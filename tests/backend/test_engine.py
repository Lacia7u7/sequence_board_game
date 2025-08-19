import importlib.util
import os
import sys
import pytest

# Dynamically import the engine module from the functions-python directory.
MODULE_PATH = os.path.join(os.path.dirname(__file__), '..', '..', 'functions-python', 'game_logic', 'engine.py')
spec = importlib.util.spec_from_file_location('engine', MODULE_PATH)
engine = importlib.util.module_from_spec(spec)  # type: ignore
assert spec and spec.loader
spec.loader.exec_module(engine)

create_full_deck = engine.create_full_deck
shuffle_deck = engine.shuffle_deck
create_empty_board = engine.create_empty_board
compute_sequences = engine.compute_sequences
has_no_legal_moves = engine.has_no_legal_moves


def test_compute_sequences_diagonal():
    # Create a 5x5 board with a diagonal sequence for team 0
    board = create_empty_board(5, 5)
    # Fill diagonal with chips
    for i in range(5):
        board[i][i]["chip"] = {"teamIndex": 0}
    seqs = compute_sequences(board)
    assert seqs.get("0") == 1


def test_has_no_legal_moves():
    board = create_empty_board(3, 3)
    board[0][0]["card"] = "A♠"
    board[0][0]["chip"] = None
    # Hand with A♠ has legal move
    assert not has_no_legal_moves(["A♠"], board)
    # No legal moves when board doesn't match
    assert has_no_legal_moves(["K♠"], board)
