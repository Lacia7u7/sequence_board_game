import gymnasium as gym
import numpy as np
from typing import Tuple, Optional, Any, Dict, List
from ..engine import engine_core, state as engine_state, rng as engine_rng
from ..encoders import board_encoder

class SequenceEnv(gym.Env):
    metadata = {"render_modes": ["human"]}

    def __init__(self, config: Optional[Dict] = None):
        super().__init__()
        self.config = config or {"rules":{"teams":2,"players_per_team":1,"hand_size":5,"allowAdvancedJack":False,"win_sequences_needed":2,"reset_full_board_no_winner":False},
                                 "action_space":{"max_hand":5,"include_pass":False},
                                 "observation":{"channels":{"team_chips":True,"corner_mask":True,"static_card_17ch":True}}}
        rules = self.config.get("rules", {})
        teams = rules.get("teams", 2)
        players_per_team = rules.get("players_per_team", 1)
        hand_size = rules.get("hand_size", 5)
        allow_adv = rules.get("allowAdvancedJack", False)
        win_needed = rules.get("win_sequences_needed", 2 if teams == 2 else 1)
        reset_full = rules.get("reset_full_board_no_winner", True)
        self.game_engine = engine_core.GameEngine()
        self.game_config = engine_state.GameConfig(teams, players_per_team, hand_size, allow_adv, win_needed, reset_full)

        dummy_state = engine_state.GameState(hands=[[] for _ in range(teams*players_per_team)], board=[[None]*10 for _ in range(10)])
        obs = board_encoder.encode_board_state(dummy_state, 0, self.config)
        self.observation_space = gym.spaces.Box(low=0.0, high=1.0, shape=obs.shape, dtype=np.float32)
        max_hand = self.config.get("action_space", {}).get("max_hand", hand_size)
        include_pass = self.config.get("action_space", {}).get("include_pass", False)
        self.action_space = gym.spaces.Discrete(100 + max_hand + (1 if include_pass else 0))
        self.current_player = 0
        self.last_obs: Optional[np.ndarray] = None
        self._seed = None
        self.move_records: List[Dict] = []

    def seed(self, seed: Optional[int] = None):
        if seed is not None:
            self._seed = seed
            engine_rng.set_seed(seed)

    def reset(self, seed: Optional[int] = None, options: Optional[dict] = None):
        super().reset(seed=seed)
        if seed is not None:
            self.seed(seed)
        if self._seed is not None:
            self.game_engine.seed(self._seed)
        state = self.game_engine.start_new(self.game_config)
        self.current_player = state.current_player
        obs = board_encoder.encode_board_state(state, self.current_player, self.config)
        self.last_obs = obs
        self.move_records = []
        return obs, {"current_player": self.current_player}

    def step(self, action: int) -> Tuple[np.ndarray, float, bool, bool, dict]:
        assert self.game_engine.state is not None, "Environment not reset"
        player = self.current_player
        team = player % self.game_config.teams
        hand = self.game_engine.state.hands[player]
        max_hand = self.config.get("action_space", {}).get("max_hand", self.game_config.hand_size)
        include_pass = self.config.get("action_space", {}).get("include_pass", False)
        pass_index = 100 + max_hand if include_pass else None
        move_action: Dict[str, Any] = {}

        if include_pass and action == pass_index:
            move_action = {"type": "timeout-skip"}
        elif action >= 100 and action < 100 + max_hand:
            idx = action - 100
            if idx < len(hand):
                move_action = {"type": "burn", "card": hand[idx]}
            else:
                move_action = {"type": "timeout-skip"}
        else:
            r = action // 10; c = action % 10
            if self.game_engine.state.board[r][c] is not None:
                jack_card = next((x for x in hand if engine_core.is_one_eyed_jack(x)), None)
                if jack_card:
                    move_action = {"type": "jack-remove", "card": jack_card, "removed": {"r": r, "c": c}}
                else:
                    move_action = {"type": "timeout-skip"}
            else:
                cell_card = engine_core.BOARD_LAYOUT[r][c]
                if cell_card == "BONUS":
                    move_action = {"type": "timeout-skip"}
                else:
                    target_card = next((x for x in hand if x == cell_card), None)
                    target_type = "play"
                    if target_card is None:
                        target_card = next((x for x in hand if engine_core.is_two_eyed_jack(x)), None)
                        target_type = "wild" if target_card else "timeout-skip"
                    if target_card:
                        move_action = {"type": target_type, "card": target_card, "target": {"r": r, "c": c}}
                    else:
                        move_action = {"type": "timeout-skip"}

        reward = 0.0; terminated = False; truncated = False; info = {"current_player": player}
        try:
            new_state, move_rec = self.game_engine.step(move_action)
            self.move_records.append(move_rec)
        except engine_core.EngineError as e:
            reward = -1.0; terminated = True; info["error"] = e.code.value
            obs = self.last_obs if self.last_obs is not None else self.observation_space.sample()
            return obs, reward, terminated, truncated, info

        if self.game_engine.is_terminal():
            terminated = True
            winners = self.game_engine.winner_teams()
            reward = 1.0 if team in winners else -1.0

        if not terminated:
            self.current_player = self.game_engine.state.current_player
        next_player = self.game_engine.state.current_player
        obs = board_encoder.encode_board_state(self.game_engine.state, next_player, self.config)
        self.last_obs = obs
        info["current_player"] = next_player

        max_moves = self.config.get("training", {}).get("episode_cap", 400)
        if len(self.move_records) >= max_moves and not terminated:
            truncated = True
        return obs, float(reward), terminated, truncated, info

    def render(self, mode="human"):
        if self.game_engine.state is None:
            print("Environment not started."); return
        board = self.game_engine.state.board
        print("Board:")
        for r in range(10):
            row_str = ""
            for c in range(10):
                if (r,c) in self.game_engine.state.sequence_cells:
                    if board[r][c] is None: row_str += "[*] "
                    else: row_str += f"[{board[r][c]}*] "
                else:
                    cell = board[r][c]
                    row_str += "[ ] " if cell is None else f"[{cell}] "
            print(row_str)
        print(f"Current player: {self.current_player}, Team: {self.current_player % self.game_config.teams}")
