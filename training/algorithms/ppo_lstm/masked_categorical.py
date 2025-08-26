from __future__ import annotations

from typing import Optional

import torch


class MaskedCategorical(torch.distributions.Categorical):

    @staticmethod
    def masked_logits(logits: torch.Tensor, mask: Optional[torch.Tensor]) -> torch.Tensor:
        if mask is None:
            return logits
        very_neg = torch.finfo(logits.dtype).min / 2
        masked = torch.where(mask > 0.5, logits, very_neg)

        # Guard: if a row has no legal actions, fall back to original logits for that row
        if masked.ndim == 2:
            illegal_all = (mask <= 0.5).all(dim=-1)
            if illegal_all.any():
                masked[illegal_all] = logits[illegal_all]
        return masked

    @classmethod
    def from_logits_and_mask(cls, logits: torch.Tensor, mask: Optional[torch.Tensor]):
        return cls(logits=cls.masked_logits(logits, mask))
