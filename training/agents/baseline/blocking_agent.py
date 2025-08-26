# training/agents/baseline/blocking_agent.py
from __future__ import annotations
from typing import Optional, Dict, Any
import numpy as np
from ..base_agent import BaseAgent
from ...algorithms.baselines.blocking_policy import BlockingPolicy

class BlockingAgent(BaseAgent):
    def __init__(self, env, **kwargs):
        self.policy = BlockingPolicy(env)

    def reset(self, env, seat: int) -> None:
        pass

    def select_action(self, legal_mask: Optional[np.ndarray], ctx: Optional[Dict[str, Any]] = None) -> int:
        return self.policy.select_action(legal_mask)

def make_agent(env, **kwargs) -> BlockingAgent:
        return BlockingAgent(env, **kwargs)
