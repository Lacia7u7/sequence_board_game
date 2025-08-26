# training/agents/human_agent.py
from __future__ import annotations
from typing import Optional, Dict, Any
import numpy as np
from base_agent import BaseAgent
from ..ui.render import ConsoleRenderer

class HumanAgent(BaseAgent):
    def __init__(self, env, **kwargs):
        self.env = env
        self.r = ConsoleRenderer()

    def reset(self, env, seat: int) -> None:
        pass

    def select_action(self, legal_mask: Optional[np.ndarray], ctx: Optional[Dict[str, Any]] = None) -> int:
        obs = ctx.get("obs") if ctx else None
        info = ctx.get("info") if ctx else {}
        self.r.render_env(self.env, info=info)  # pretty board + hands in console

        legal = np.flatnonzero(legal_mask > 0.5) if legal_mask is not None else np.arange(self.env.action_dim)
        print("\nLegal actions (integers):", legal.tolist())
        while True:
            try:
                a = int(input("Choose action id: ").strip())
                if a in legal:
                    return a
                print("Not legal. Try again.")
            except Exception:
                print("Invalid input. Try again.")
