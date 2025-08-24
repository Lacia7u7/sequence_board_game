import random
from typing import List, Callable

class SelfPlayManager:
    def __init__(self, agents: List, env, opponent_pool=None):
        self.agents = agents
        self.env = env
        self.opponent_pool = opponent_pool

    def play_episode(self, render: bool = False):
        obs, info = self.env.reset()
        total_reward = {i: 0.0 for i in range(len(self.agents))}
        while True:
            current_player = info.get("current_player", 0)
            agent = self.agents[current_player]
            mask = None
            action = agent.select_action(obs, mask) if hasattr(agent, "select_action") else agent(obs, mask)
            obs, reward, terminated, truncated, info = self.env.step(action)
            if terminated or truncated:
                break
        return total_reward
