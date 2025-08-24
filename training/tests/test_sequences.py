from training.engine.engine_core import GameEngine
from training.engine.state import GameConfig

def test_detect_all_directions():
    config = GameConfig(teams=3, players_per_team=1, hand_size=6, allowAdvancedJack=False, win_sequences_needed=1, reset_full_board_no_winner=False)
    engine = GameEngine(); engine.seed(123); engine.start_new(config)
    for c in range(5):
        engine.state.board[1][c] = 0
    engine._advance_turn()
    for c in range(5):
        assert (1, c) in engine.state.sequence_cells
    engine.state.sequence_cells.clear(); engine.state.sequences_count[0] = 0
    for r in range(5):
        engine.state.board[r][2] = 1
    engine._advance_turn()
    for r in range(5):
        assert (r, 2) in engine.state.sequence_cells
    engine.state.sequence_cells.clear(); engine.state.sequences_count[1] = 0
    for i in range(5):
        engine.state.board[i][i] = 2
    engine._advance_turn()
    for i in range(5):
        assert (i, i) in engine.state.sequence_cells
    engine = GameEngine(); engine.seed(123); engine.start_new(config)
    for i in range(5):
        engine.state.board[i][4-i] = 0
    engine._advance_turn()
    for i in range(5):
        assert (i, 4-i) in engine.state.sequence_cells
