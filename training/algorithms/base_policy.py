# training/algorithms/base_policy.py
from __future__ import annotations
from typing import Optional, Dict, Any
import numpy as np
from overrides import overrides

PolicyCtx = Dict[str, Any]  # e.g. {"obs": np.ndarray, "info": dict, "seat": int, "env": <SequenceEnv>, "legal_mask": np.ndarray}

class BasePolicy:
    """
    Superclass for simple policies that DON'T look at observations.
    Override: select_action(legal_mask) -> int
    """
    def reset(self) -> None:
        pass

    def select_action(self, legal_mask: Optional[np.ndarray], ctx: Optional[PolicyCtx] = None) -> int:
        raise NotImplementedError


class BasePolicyObs (BasePolicy):
    """
    Superclass for simple policies that DON'T look at observations.
    Override: select_action(legal_mask) -> int
    """

    def reset(self) -> None:
        pass

    @overrides
    def select_action(self, legal_mask: Optional[np.ndarray], ctx: Optional[PolicyCtx]) -> int:
        raise NotImplementedError

