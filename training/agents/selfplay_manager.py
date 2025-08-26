# training/agents/selfplay_manager.py
from __future__ import annotations
from typing import Dict, Any, Tuple, Optional, List
import numpy as np
from .base_agent import BaseAgent, AgentCtx

class SelfPlayManager:
    """
    Runs a single self-play episode with one env and a list of agents.
    Constructor order matches your usage: SelfPlayManager([agent0, agent1], env, ...)
    """

    def __init__(self, agents: List[BaseAgent], env, max_steps: int = 400):
        self.env = env
        self.agents = list(agents)
        self.max_steps = int(max_steps)

    def play_episode(self, seed: Optional[int] = None, render: bool = False) -> Dict[str, Any]:
        obs, info = self.env.reset(seed=seed)
        num_players = getattr(self.env, "num_players", len(self.agents)) or len(self.agents)

        # Reset agents with seat indices
        for seat in range(num_players):
            self.agents[seat % len(self.agents)].reset(self.env, seat)

        totals = {i: 0.0 for i in range(num_players)}
        steps = 0
        terminated = False
        truncated = False
        current_seat = info.get("current_player", 0)
        winners = []

        while not (terminated or truncated):
            legal_mask = info.get("legal_mask", None)

            # Build the dynamic context
            ctx: AgentCtx = {
                "obs": obs,
                "info": info,
                "seat": current_seat,
                "env": self.env,
                "legal_mask": legal_mask,
            }

            agent = self.agents[current_seat % len(self.agents)]
            action = agent.select_action(legal_mask, ctx)

            obs, reward, terminated, truncated, info = self.env.step(action)
            totals[current_seat] = totals.get(current_seat, 0.0) + float(reward)
            steps += 1
            if render:
                self.env.render()

            current_seat = info.get("current_player", (current_seat + 1) % num_players)
            if steps >= self.max_steps and not terminated:
                truncated = True

        winners = info.get("winners", [])
        return {
            "steps": steps,
            "total_reward": totals,
            "terminated": bool(terminated),
            "truncated": bool(truncated),
            "winners": winners,
        }
