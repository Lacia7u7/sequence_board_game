from training.engine import engine_core
from training.engine.engine_core import GameEngine
from training.engine.state import GameConfig

def test_dead_card_burn():
    config = GameConfig(teams=2, players_per_team=1, hand_size=3, allowAdvancedJack=False, win_sequences_needed=2, reset_full_board_no_winner=True)
    engine = GameEngine(); engine.seed(99); engine.start_new(config)
    card = None
    for c in engine.state.hands[0]:
        if c and c[-1] in "SHDC" and c[:-1] != "J":
            card = c; break
    assert card is not None
    positions = [(r,c) for r in range(10) for c in range(10) if engine_core.BOARD_LAYOUT[r][c] == card]
    for (r,c) in positions:
        engine.state.board[r][c] = 1
    legal = engine.legal_actions_for(0)
    assert any(idx for idx in legal["discard"])
    move = {"type": "burn", "card": card}
    prev_len = len(engine.state.hands[0])
    state, rec = engine.step(move)
    assert len(state.hands[0]) == prev_len
    assert card not in state.hands[0]
