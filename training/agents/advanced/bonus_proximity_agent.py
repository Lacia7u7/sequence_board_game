from __future__ import annotations
from typing import Optional
import numpy as np
from overrides import overrides

from ..base_agent import BaseAgent, AgentCtx
from ...algorithms.advanced.bonus_proximity_policy import BonusProximityPolicy


class BonusProximityAgent(BaseAgent):
    def __init__(self, env=None, **weights):
        super().__init__(env)
        self.policy = BonusProximityPolicy(env=env, **weights)

    def reset(self, env, seat: int) -> None:
        self.env = env
        self.policy.env = env

    @overrides
    def select_action(self, legal_mask: Optional[np.ndarray], ctx: Optional[AgentCtx] = None) -> int:
        return self.policy.select_action(legal_mask, ctx=ctx)
