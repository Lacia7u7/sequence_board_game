# training/algorithms/ppo_lstm/policy.py
from __future__ import annotations
from typing import Optional, Tuple, Dict, Any
import torch
import torch.nn as nn


class MaskedCategorical(torch.distributions.Categorical):
    @staticmethod
    def masked_logits(logits: torch.Tensor, mask: Optional[torch.Tensor]) -> torch.Tensor:
        """
        Apply a binary mask (1=legal,0=illegal) to logits by setting illegal positions to a large negative.
        logits: (B, A), mask: (B, A)
        """
        if mask is None:
            return logits
        very_neg = torch.finfo(logits.dtype).min / 2
        return torch.where(mask > 0.5, logits, very_neg)

    @classmethod
    def from_logits_and_mask(cls, logits: torch.Tensor, mask: Optional[torch.Tensor]):
        return cls(logits=cls.masked_logits(logits, mask))


class PPORecurrentPolicy(nn.Module):
    """
    Conv trunk -> LSTM -> policy & value heads.

    Public API:
      - __init__(obs_shape=..., action_dim=..., ...)
      - get_initial_state(batch_size) -> (h0, c0)  with shape (L, B, H)
      - forward(obs, masks=..., h0=..., c0=...) -> {'logits','dist','value','h','c'}
      - act(obs=..., legal_mask=..., h0=..., c0=...) -> {'action','log_prob','value','h','c'}
      - evaluate_actions(...) handled by learner using .forward outputs
    """

    def __init__(
        self,
        obs_channels: Optional[int] = None,
        action_dim: int = 0,
        conv_channels=(64, 64, 128),
        lstm_hidden: int = 512,
        lstm_layers: int = 1,
        *,
        obs_shape: Optional[tuple] = None,
        device: Optional[torch.device] = None,   # optional; learner will .to(device)
    ):
        super().__init__()
        if obs_channels is None:
            if obs_shape is None or len(obs_shape) < 1:
                raise ValueError("Please provide obs_channels or obs_shape=(C,H,W).")
            obs_channels = int(obs_shape[0])

        self.action_dim = int(action_dim)
        self.lstm_hidden = int(lstm_hidden)
        self.lstm_layers = int(lstm_layers)

        # --- Conv trunk ---
        layers = []
        prev = obs_channels
        for ch in conv_channels:
            layers += [
                nn.Conv2d(prev, ch, kernel_size=3, padding=1, bias=False),
                nn.GroupNorm(num_groups=8 if ch >= 32 else 1, num_channels=ch),
                nn.SiLU(),
            ]
            prev = ch
        self.conv = nn.Sequential(*layers)
        conv_out_ch = prev

        self.flatten = nn.Flatten(start_dim=1)

        if obs_shape is not None and len(obs_shape) >= 3:
            H, W = int(obs_shape[1]), int(obs_shape[2])
        else:
            H, W = 10, 10
        self.spatial_dim = H * W * conv_out_ch

        # --- LSTM ---
        self.lstm = nn.LSTM(
            input_size=self.spatial_dim,
            hidden_size=self.lstm_hidden,
            num_layers=self.lstm_layers,
            batch_first=True,
        )

        # --- Heads ---
        self.pi = nn.Linear(self.lstm_hidden, self.action_dim)
        self.v  = nn.Linear(self.lstm_hidden, 1)

        # Optional early device move
        if device is not None:
            try:
                self.to(device)
            except Exception:
                pass

    # -------- utilities --------
    def init_hidden(self, batch_size: int = 1) -> Tuple[torch.Tensor, torch.Tensor]:
        dev = next(self.parameters()).device
        h0 = torch.zeros(self.lstm_layers, batch_size, self.lstm_hidden, device=dev)
        c0 = torch.zeros(self.lstm_layers, batch_size, self.lstm_hidden, device=dev)
        return h0, c0

    # Back-compat alias
    def get_initial_state(self, batch_size: int = 1) -> Tuple[torch.Tensor, torch.Tensor]:
        return self.init_hidden(batch_size)

    def _encode(self, obs: torch.Tensor) -> torch.Tensor:
        x = self.conv(obs)          # (B, C', H, W)
        x = self.flatten(x)         # (B, spatial_dim)
        return x

    def _forward_core(
        self,
        obs: torch.Tensor,
        h0: Optional[torch.Tensor] = None,
        c0: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor, Tuple[torch.Tensor, torch.Tensor]]:
        B = obs.size(0)
        if h0 is None or c0 is None:
            h0, c0 = self.init_hidden(B)

        enc = self._encode(obs).unsqueeze(1)      # (B, 1, spatial_dim)
        out, (hn, cn) = self.lstm(enc, (h0, c0))  # out: (B, 1, H)
        last = out[:, -1, :]                      # (B, H)
        logits = self.pi(last)                    # (B, A)
        value  = self.v(last)                     # (B, 1)  (KEEP 2D for learner)
        return logits, value, (hn, cn)

    # -------- public API --------
    def forward(
        self,
        obs: torch.Tensor,
        masks: Optional[torch.Tensor] = None,  # (B, A) legal mask
        h0: Optional[torch.Tensor] = None,
        c0: Optional[torch.Tensor] = None,
    ) -> Dict[str, Any]:
        logits, value, (hn, cn) = self._forward_core(obs, h0=h0, c0=c0)
        dist = MaskedCategorical.from_logits_and_mask(logits, masks)
        return {"logits": logits, "dist": dist, "value": value, "h": hn, "c": cn}

    @torch.no_grad()
    def act(
        self,
        obs: torch.Tensor,
        legal_mask: Optional[torch.Tensor] = None,    # alias
        action_mask: Optional[torch.Tensor] = None,   # alias
        h0: Optional[torch.Tensor] = None,
        c0: Optional[torch.Tensor] = None,
    ) -> Dict[str, Any]:
        mask = action_mask if action_mask is not None else legal_mask
        logits, value, (hn, cn) = self._forward_core(obs, h0=h0, c0=c0)
        dist = MaskedCategorical.from_logits_and_mask(logits, mask)
        action = dist.sample()
        log_prob = dist.log_prob(action)
        return {"action": action, "log_prob": log_prob, "value": value, "h": hn, "c": cn}
