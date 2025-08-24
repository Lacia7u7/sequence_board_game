# training/algorithms/baselines/random_policy.py
import random
from typing import Optional, Sequence


class RandomPolicy:
    def select_action(self, obs, legal_mask: Optional[Sequence[float]]):
        """
        Mask-aware random actor:
          - If a mask is provided, sample uniformly from legal actions.
          - If no legal actions, return 0.
          - If mask is None, fallback to 0 (caller should pass mask in normal use).
        """
        if legal_mask is None:
            return 0
        legal = [i for i, m in enumerate(legal_mask) if m > 0.5]
        return random.choice(legal) if legal else 0
