# training/agents/advanced/pattern_window_agent.py
from __future__ import annotations
from typing import Optional
from overrides import overrides

from ..base_agent import BaseAgent, AgentCtx
from ...algorithms.advanced.pattern_window_policy import PatternWindowPolicy

class PatternWindowAgent(BaseAgent):
    def __init__(self, env=None):
        super().__init__(env)
        self.policy = PatternWindowPolicy(env=env)

    def reset(self, env, seat: int) -> None:
        self.env = env
        self.policy.env = env

    @overrides
    def select_action(self, legal_mask, ctx: Optional[AgentCtx] = None) -> int:
        return self.policy.select_action(legal_mask, ctx=ctx)
