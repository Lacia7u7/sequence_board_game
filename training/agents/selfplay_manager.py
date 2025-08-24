# training/agents/selfplay_manager.py
from typing import List, Dict, Any


class SelfPlayManager:
    """
    Simple multi-agent episode driver over a single (possibly vectorized) env.
    Accumulates per-player episodic rewards correctly.
    """
    def __init__(self, agents: List, env, opponent_pool=None):
        self.agents = agents
        self.env = env
        self.opponent_pool = opponent_pool

    def play_episode(self, render: bool = False) -> Dict[int, float]:
        obs, info = self.env.reset()
        total_reward: Dict[int, float] = {i: 0.0 for i in range(len(self.agents))}
        while True:
            current_player = int(info.get("current_player", 0))
            agent = self.agents[current_player]
            mask = info.get("legal_mask", None)
            if hasattr(agent, "select_action"):
                action = agent.select_action(obs, mask)
            else:
                action = agent(obs, mask)
            obs, reward, terminated, truncated, info = self.env.step(action)
            total_reward[current_player] += float(reward)
            if terminated or truncated:
                break
            if render and hasattr(self.env, "render"):
                self.env.render()
        return total_reward
