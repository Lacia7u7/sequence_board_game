from __future__ import annotations
from typing import Optional
import numpy as np
from overrides import overrides

from ..base_policy import BasePolicy, PolicyCtx


class RandomPolicy(BasePolicy):
    def __init__(self, env=None):
        self.env = env

    @overrides
    def select_action(self, legal_mask: Optional[np.ndarray], ctx: Optional[PolicyCtx]) -> int:
        if legal_mask is None:
            return 0
        legal = np.flatnonzero(legal_mask > 0.5)
        if legal.size == 0:
            return 0
        return int(np.random.choice(legal))
