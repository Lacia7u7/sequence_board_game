# training/algorithms/ppo_lstm/ppo_lstm_policy.py
from __future__ import annotations
from typing import Optional, Tuple, Dict, Any

import numpy as np
import torch
import torch.nn as nn

try:
    from typing import override   # py3.12+
except Exception:
    try:
        from typing_extensions import override  # py<=3.11
    except Exception:
        def override(f): return f

from training.algorithms.base_policy import BasePolicyObs, PolicyCtx
from training.algorithms.ppo_lstm.masked_categorical import MaskedCategorical


class PPORecurrentPolicy(nn.Module, BasePolicyObs):
    def __init__(
        self,
        obs_channels: Optional[int] = None,
        action_dim: int = 0,
        conv_channels=(64, 64, 128),
        lstm_hidden: int = 512,
        lstm_layers: int = 1,
        value_tanh_bound: float = 5.0,
        *,
        obs_shape: Optional[tuple] = None,
        device: Optional[torch.device] = None,
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

        # gentle init
        with torch.no_grad():
            nn.init.orthogonal_(self.pi.weight, gain=0.01)
            nn.init.zeros_(self.pi.bias)
            nn.init.orthogonal_(self.v.weight, gain=0.01)
            nn.init.zeros_(self.v.bias)

        self._value_bound = float(value_tanh_bound)

        if device is not None:
            try:
                self.to(device)
            except Exception:
                pass
        # >>> store device for action-time use
        self.device = torch.device(device) if device is not None else next(self.parameters()).device

    # --- utilities ---
    def init_hidden(self, batch_size: int = 1) -> Tuple[torch.Tensor, torch.Tensor]:
        dev = next(self.parameters()).device
        h0 = torch.zeros(self.lstm_layers, batch_size, self.lstm_hidden, device=dev)
        c0 = torch.zeros(self.lstm_layers, batch_size, self.lstm_hidden, device=dev)
        return h0, c0

    def get_initial_state(self, batch_size: int = 1) -> Tuple[torch.Tensor, torch.Tensor]:
        return self.init_hidden(batch_size)

    def _encode(self, obs: torch.Tensor) -> torch.Tensor:
        x = self.conv(obs)
        x = self.flatten(x)
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
        value_raw = self.v(last)                  # (B, 1)
        value = self._value_bound * torch.tanh(value_raw)
        return logits, value, (hn, cn)

    def forward(
        self,
        obs: torch.Tensor,
        masks: Optional[torch.Tensor] = None,
        h0: Optional[torch.Tensor] = None,
        c0: Optional[torch.Tensor] = None,
    ) -> Dict[str, Any]:
        logits, value, (hn, cn) = self._forward_core(obs, h0=h0, c0=c0)
        dist = MaskedCategorical.from_logits_and_mask(logits, masks)
        return {"logits": logits, "dist": dist, "value": value, "h": hn, "c": cn}

    @override
    @torch.no_grad()
    def select_action(self, legal_mask: Optional[np.ndarray], ctx: Optional[PolicyCtx]) -> Dict[str, Any]:
        """
        Returns:
          {
            "action":  Tensor[int64] shape [1]   (sampled or argmax if ctx["deterministic"])
            "log_prob": Tensor[float32] shape [1]
            "value":   Tensor[float32] shape [1]   (critic value for this step)
            "h":       Tensor shape [L, 1, H]
            "c":       Tensor shape [L, 1, H]
          }
        """
        if ctx is None or ("obs" not in ctx and "obs_t" not in ctx):
            raise RuntimeError("select_action requires ctx['obs'] (or ctx['obs_t']).")

        # ---- OBS ----
        obs = ctx.get("obs_t", ctx.get("obs"))
        if isinstance(obs, np.ndarray):
            obs_t = torch.from_numpy(obs).to(self.device, dtype=torch.float32)
        elif isinstance(obs, torch.Tensor):
            obs_t = obs.to(self.device, dtype=torch.float32)
        else:
            raise TypeError("ctx['obs'] must be a numpy.ndarray or torch.Tensor")
        if obs_t.ndim == 3:  # (C,H,W) -> (1,C,H,W)
            obs_t = obs_t.unsqueeze(0)

        # ---- MASK ----
        mask_t: Optional[torch.Tensor] = None
        if legal_mask is not None:
            if isinstance(legal_mask, np.ndarray):
                mask_t = torch.from_numpy(legal_mask).to(self.device, dtype=torch.float32)
            elif isinstance(legal_mask, torch.Tensor):
                mask_t = legal_mask.to(self.device, dtype=torch.float32)
            else:
                raise TypeError("legal_mask must be None, numpy.ndarray, or torch.Tensor")
            if mask_t.ndim == 1:  # (A,) -> (1,A)
                mask_t = mask_t.unsqueeze(0)

        # ---- HIDDEN ----
        h0 = ctx.get("h0")
        c0 = ctx.get("c0")
        if h0 is None or c0 is None:
            h0, c0 = self.init_hidden(batch_size=1)
        h0 = h0.to(self.device)
        c0 = c0.to(self.device)

        # ---- FORWARD ----
        logits, value, (hn, cn) = self._forward_core(obs_t, h0=h0, c0=c0)

        # Distribution (masked)
        dist = MaskedCategorical.from_logits_and_mask(logits, mask_t)

        # Deterministic or sample?
        deterministic = bool(ctx.get("deterministic", False))
        if deterministic:
            # argmax over *masked* logits
            masked_logits = logits if mask_t is None else MaskedCategorical.masked_logits(logits, mask_t)
            action = torch.argmax(masked_logits, dim=-1)  # shape [1]
            log_prob = dist.log_prob(action)
        else:
            action = dist.sample()                        # shape [1]
            log_prob = dist.log_prob(action)              # shape [1]

        # value is [1,1] — squeeze last batch dim for convenience -> [1]
        value_out = value.squeeze(0)

        return {
            "action": action,          # Tensor[int64] [1]
            "log_prob": log_prob,      # Tensor[float32] [1]
            "value": value_out,        # Tensor[float32] [1]
            "h": hn,                   # [L,1,H]
            "c": cn,                   # [L,1,H]
        }

