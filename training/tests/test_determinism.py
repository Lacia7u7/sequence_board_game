from training.engine.engine_core import GameEngine
from training.engine.state import GameConfig

def test_repeatable_deal():
    cfg = GameConfig(teams=2, players_per_team=1, hand_size=5, allowAdvancedJack=False, win_sequences_needed=2, reset_full_board_no_winner=False)
    e1 = GameEngine(); e1.seed(123); s1 = e1.start_new(cfg)
    e2 = GameEngine(); e2.seed(123); s2 = e2.start_new(cfg)
    assert s1.hands == s2.hands
    c1 = e1.deck.draw(); c2 = e2.deck.draw()
    assert c1 == c2
