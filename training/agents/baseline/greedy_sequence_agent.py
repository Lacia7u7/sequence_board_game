from __future__ import annotations
from typing import Optional
import numpy as np
from overrides import overrides

from ..base_agent import BaseAgent, AgentCtx
from ...algorithms.baselines.greedy_sequence_policy import GreedySequencePolicy

class GreedySequenceAgent(BaseAgent):

    def __init__(self, env=None):
        super().__init__(env)
        self.policy = GreedySequencePolicy(env=env)

    def reset(self, env, seat: int) -> None:
        pass

    @overrides
    def select_action(self, legal_mask: Optional[np.ndarray], ctx: Optional[AgentCtx] = None) -> int:
        return self.policy.select_action(legal_mask, ctx=ctx)
