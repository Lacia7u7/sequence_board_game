# training/agents/advanced/threat_aware_minimax_agent.py
from __future__ import annotations
from typing import Optional
import numpy as np
from overrides import overrides

from ..base_agent import BaseAgent, AgentCtx
from ...algorithms.advanced.threat_aware_minimax_policy import ThreatAwareMinimaxPolicy

class ThreatAwareMinimaxAgent(BaseAgent):
    def __init__(self, env=None, alpha: float = 0.8):
        super().__init__(env)
        self.policy = ThreatAwareMinimaxPolicy(env=env, alpha=alpha)

    def reset(self, env, seat: int) -> None:
        self.env = env
        self.policy.env = env

    @overrides
    def select_action(self, legal_mask: Optional[np.ndarray], ctx: Optional[AgentCtx] = None) -> int:
        return self.policy.select_action(legal_mask, ctx=ctx)
