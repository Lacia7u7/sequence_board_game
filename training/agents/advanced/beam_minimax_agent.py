# training/agents/advanced/beam_minimax_agent.py
from __future__ import annotations
from typing import Optional
from overrides import overrides

from ..base_agent import BaseAgent, AgentCtx
from ...algorithms.advanced.beam_minimax_policy import BeamMinimaxPolicy

class BeamMinimaxAgent(BaseAgent):
    def __init__(self, env=None, beam_size: int = 8, alpha: float = 0.7):
        super().__init__(env)
        self.policy = BeamMinimaxPolicy(env=env, beam_size=beam_size, alpha=alpha)

    def reset(self, env, seat: int) -> None:
        self.env = env
        self.policy.env = env

    @overrides
    def select_action(self, legal_mask, ctx: Optional[AgentCtx] = None) -> int:
        return self.policy.select_action(legal_mask, ctx=ctx)
