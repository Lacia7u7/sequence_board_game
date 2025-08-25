# training/algorithms/baselines/random_policy.py
from __future__ import annotations
from typing import Optional
import numpy as np


class RandomPolicy:
    """
    Uniform random policy **respecting the legal action mask**.
    Unlike older versions, this includes DISCARD/PASS actions if they are legal.
    """

    def __init__(self, action_dim: int):
        self.action_dim = int(action_dim)

    def select_action(self, obs: np.ndarray, legal_mask: Optional[np.ndarray]) -> int:
        if legal_mask is None:
            return int(np.random.randint(0, self.action_dim))
        legal_mask = np.asarray(legal_mask, dtype=np.float32).reshape(-1)
        if legal_mask.shape[0] != self.action_dim:
            # fall back to uniform if shape mismatch
            return int(np.random.randint(0, self.action_dim))
        legal_indices = np.nonzero(legal_mask > 0.5)[0]
        if legal_indices.size == 0:
            return int(np.random.randint(0, self.action_dim))
        return int(np.random.choice(legal_indices))
