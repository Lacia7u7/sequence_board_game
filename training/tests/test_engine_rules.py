import pytest
from training.engine import engine_core
from training.engine.engine_core import GameEngine
from training.engine.state import GameConfig

def setup_engine_1v1():
    config = GameConfig(teams=2, players_per_team=1, hand_size=7, allowAdvancedJack=False, win_sequences_needed=2, reset_full_board_no_winner=False)
    engine = GameEngine(); engine.seed(42); engine.start_new(config); return engine

def test_normal_card_play_and_match():
    engine = setup_engine_1v1()
    state = engine.state
    card = None; pos = None
    for c in state.hands[0]:
        if c and c[-1] in "SHDC" and c[:-1] != "J":
            for r in range(10):
                for col in range(10):
                    if engine_core.BOARD_LAYOUT[r][col] == c and state.board[r][col] is None:
                        card = c; pos = (r, col); break
                if card: break
        if card: break
    assert card is not None
    move = {"type": "play", "card": card, "target": {"r": pos[0], "c": pos[1]}}
    prev_chip = state.board[pos[0]][pos[1]]
    engine.step(move)
    new_chip = engine.state.board[pos[0]][pos[1]]
    assert prev_chip is None
    assert new_chip == 0
    assert card not in engine.state.hands[0]
    assert len(engine.state.hands[0]) == 7

def test_corner_and_jack_restrictions():
    engine = setup_engine_1v1()
    engine.state.hands[0][0] = "JC"
    corner_r, corner_c = 0, 0
    move = {"type": "wild", "card": "JC", "target": {"r": corner_r, "c": corner_c}}
    with pytest.raises(Exception):
        engine.step(move)
    engine.state.hands[0][0] = "JS"
    engine.state.board[0][1] = 1
    engine.state.board[0][2] = 0
    move = {"type": "jack-remove", "card": "JS", "removed": {"r": 0, "c": 2}}
    with pytest.raises(Exception):
        engine.step(move)
    move = {"type": "jack-remove", "card": "JS", "removed": {"r": 0, "c": 1}}
    state, rec = engine.step(move)
    assert state.board[0][1] is None

def test_sequence_creation_and_win():
    config = GameConfig(teams=2, players_per_team=1, hand_size=7, allowAdvancedJack=False, win_sequences_needed=1, reset_full_board_no_winner=False)
    engine = GameEngine(); engine.seed(1); engine.start_new(config)
    for c in range(4):
        engine.state.board[0][c] = 0
    card = engine_core.BOARD_LAYOUT[0][4]
    engine.state.hands[0][0] = card
    move = {"type": "play", "card": card, "target": {"r": 0, "c": 4}}
    state, rec = engine.step(move)
    assert engine.state.sequences_count[0] >= 1
    assert 0 in engine.state.winners
