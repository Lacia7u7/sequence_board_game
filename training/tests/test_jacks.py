from training.engine.engine_core import GameEngine
from training.engine.state import GameConfig
from training.engine.cards import is_two_eyed_jack, is_one_eyed_jack

def test_jack_types():
    assert is_two_eyed_jack("JC"); assert is_two_eyed_jack("JD")
    assert not is_two_eyed_jack("JH")
    assert is_one_eyed_jack("JS")
    assert not is_one_eyed_jack("QD")

def test_one_eyed_remove_and_advanced():
    config = GameConfig(teams=2, players_per_team=1, hand_size=5, allowAdvancedJack=False, win_sequences_needed=2, reset_full_board_no_winner=False)
    engine = GameEngine(); engine.seed(5); engine.start_new(config)
    engine.state.board[2][2] = 1
    engine.state.sequence_cells.add((2,2))
    engine.state.hands[0][0] = "JS"
    move = {"type": "jack-remove", "card": "JS", "removed": {"r": 2, "c": 2}}
    try:
        engine.step(move); assert False, "Expected error"
    except Exception:
        pass
    engine.game_config.allowAdvancedJack = True
    engine.state.hands[0][0] = "JH"
    move = {"type": "jack-remove", "card": "JH", "removed": {"r": 2, "c": 2}}
    state, rec = engine.step(move)
    assert state.board[2][2] is None
