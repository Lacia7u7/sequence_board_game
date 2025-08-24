from ...engine.engine_core import BOARD_LAYOUT
class GreedySequencePolicy:
    def __init__(self, env):
        self.env = env
    def select_action(self, obs, legal_mask):
        legal = [i for i, m in enumerate(legal_mask.flatten()) if m > 0.5] if legal_mask is not None else range(self.env.action_space.n)
        for act in legal:
            if act < 100:
                r = act // 10; c = act % 10
                team = self.env.current_player % self.env.game_config.teams
                for dr, dc in [(0,1),(1,0),(1,1),(1,-1)]:
                    cnt = 1
                    rr, cc = r-dr, c-dc
                    while 0 <= rr < 10 and 0 <= cc < 10 and (BOARD_LAYOUT[rr][cc] == "BONUS" or self.env.game_engine.state.board[rr][cc] == team):
                        cnt += 1; rr -= dr; cc -= dc
                    rr, cc = r+dr, c+dc
                    while 0 <= rr < 10 and 0 <= cc < 10 and (BOARD_LAYOUT[rr][cc] == "BONUS" or self.env.game_engine.state.board[rr][cc] == team):
                        cnt += 1; rr += dr; cc += dc
                    if cnt >= 5:
                        return act
        return legal[0] if hasattr(legal, "__len__") and len(legal) else 0
