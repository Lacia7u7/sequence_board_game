import numpy as np
from training.engine.engine_core import GameEngine
from training.engine.state import GameConfig
from training.encoders.board_encoder import encode_board_state

def test_basic_channels():
    config = {"rules":{"teams":2},"observation":{"channels":{"team_chips":True,"corner_mask":True,"static_card_17ch":True}}}
    game_cfg = GameConfig(teams=2, players_per_team=1, hand_size=1, allowAdvancedJack=False, win_sequences_needed=1, reset_full_board_no_winner=False)
    engine = GameEngine(); engine.seed(1); engine.start_new(game_cfg)
    st = engine.state
    st.board[0][1] = 0; st.board[1][1] = 1
    enc = encode_board_state(st, 0, config)
    assert enc.shape[0] == 2 + 1 + 17
    assert enc[0,0,1] == 1.0
    assert enc[1,1,1] == 1.0
    corner = enc[2]
    assert corner.sum() == 4.0
