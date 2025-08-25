# training/agents/selfplay_manager.py
from __future__ import annotations
from typing import Dict, Any, Tuple, Optional, List
import numpy as np


class SelfPlayManager:
    """
    Runs a single self-play episode using N identical policies (weight sharing) or a mix from a pool.
    Exposes simple hooks for reward accumulation, move logging and early termination.
    """

    def __init__(self, env, policies, max_steps: int = 400):
        """
        env: Gym-like env with reset/step
        policies: list of policy objects; policies[i].act(obs, legal_mask) -> action (int)
        """
        self.env = env
        self.policies = policies
        self.max_steps = int(max_steps)

    def play_episode(self, seed: Optional[int] = None) -> Dict[str, Any]:
        obs, info = self.env.reset(seed=seed)
        # Track rewards by seat index if available, otherwise just one stream
        num_players = getattr(self.env, "num_players", len(self.policies)) or len(self.policies)
        total_reward = {i: 0.0 for i in range(num_players)}
        steps = 0
        moves: List[Dict[str, Any]] = []

        terminated = False
        truncated = False
        current_seat = info.get("current_player", 0)

        while not (terminated or truncated):
            legal_mask = info.get("legal_mask", None)
            # Select a policy for the current seat (simple round-robin)
            pol = self.policies[current_seat % len(self.policies)]
            action = pol.select_action(obs, legal_mask)

            obs, reward, terminated, truncated, info = self.env.step(action)

            # accumulate reward for the seat that just acted
            total_reward[current_seat] = total_reward.get(current_seat, 0.0) + float(reward)
            steps += 1
            moves.append({"seat": current_seat, "action": int(action), "reward": float(reward)})

            current_seat = info.get("current_player", (current_seat + 1) % num_players)
            if steps >= self.max_steps:
                truncated = True

        # Winner info if env reports
        winners = getattr(self.env, "winners", None)
        return {
            "steps": steps,
            "total_reward": total_reward,  # per-seat totals
            "terminated": bool(terminated),
            "truncated": bool(truncated),
            "moves": moves,
            "winners": winners,
        }
