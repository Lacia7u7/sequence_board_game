from __future__ import annotations
from typing import Optional, Dict, Any
import numpy as np
from overrides import overrides

from ..base_agent import BaseAgent
from ...algorithms.baselines.random_policy import RandomPolicy


class RandomAgent(BaseAgent):

    def __init__(self, env=None):
        super().__init__(env)
        self.policy = RandomPolicy(env=env)

    def reset(self, env, seat: int) -> None:
        pass

    @overrides
    def select_action(self, legal_mask: Optional[np.ndarray], ctx: Optional[Dict[str, Any]] = None) -> int:
        return self.policy.select_action(legal_mask,ctx)
