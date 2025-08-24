# training/algorithms/ppo_lstm/policy.py
from __future__ import annotations
from typing import Optional, Tuple
import torch
import torch.nn as nn
import torch.nn.functional as F


class MaskedCategorical(torch.distributions.Categorical):
    @staticmethod
    def masked_logits(logits: torch.Tensor, mask: Optional[torch.Tensor]) -> torch.Tensor:
        if mask is None:
            return logits
        # mask shape: (B, A)
        very_neg = torch.finfo(logits.dtype).min / 2
        return torch.where(mask > 0.5, logits, very_neg)

    @classmethod
    def from_logits_and_mask(cls, logits: torch.Tensor, mask: Optional[torch.Tensor]):
        masked = cls.masked_logits(logits, mask)
        return cls(logits=masked)


class PPORecurrentPolicy(nn.Module):
    """
    Conv trunk -> LSTM -> policy & value heads.
    Exposes act() and evaluate_actions() that accept optional action masks and hidden states.
    """
    def __init__(self, obs_channels: int, action_dim: int, conv_channels=(64, 64, 128), lstm_hidden=512, lstm_layers=1):
        super().__init__()
        c_in = obs_channels
        layers = []
        prev = c_in
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
        self.spatial_dim = 10 * 10 * conv_out_ch

        self.lstm = nn.LSTM(input_size=self.spatial_dim, hidden_size=lstm_hidden, num_layers=lstm_layers, batch_first=True)
        self.pi = nn.Linear(lstm_hidden, action_dim)
        self.v = nn.Linear(lstm_hidden, 1)

        self.lstm_hidden = lstm_hidden
        self.action_dim = action_dim

    def init_hidden(self, batch_size: int = 1) -> Tuple[torch.Tensor, torch.Tensor]:
        h0 = torch.zeros(self.lstm.num_layers, batch_size, self.lstm_hidden, device=next(self.parameters()).device)
        c0 = torch.zeros(self.lstm.num_layers, batch_size, self.lstm_hidden, device=next(self.parameters()).device)
        return h0, c0

    def _encode(self, obs: torch.Tensor) -> torch.Tensor:
        # obs: (B, C, H, W)
        x = self.conv(obs)
        x = self.flatten(x)  # (B, spatial_dim)
        return x

    def forward(self, obs: torch.Tensor, hidden_state: Optional[Tuple[torch.Tensor, torch.Tensor]] = None):
        B = obs.size(0)
        enc = self._encode(obs).unsqueeze(1)  # (B, 1, spatial_dim)
        if hidden_state is None:
            h0, c0 = self.init_hidden(B)
        else:
            h0, c0 = hidden_state
        out, (hn, cn) = self.lstm(enc, (h0, c0))  # out: (B, 1, H)
        last = out[:, -1, :]  # (B, H)
        logits = self.pi(last)
        value = self.v(last).squeeze(-1)
        return logits, value, (hn, cn)

    @torch.no_grad()
    def act(self, obs: torch.Tensor, action_mask: Optional[torch.Tensor] = None, hidden_state: Optional[Tuple[torch.Tensor, torch.Tensor]] = None):
        logits, value, (hn, cn) = self.forward(obs, hidden_state=hidden_state)
        dist = MaskedCategorical.from_logits_and_mask(logits, action_mask)
        action = dist.sample()
        log_prob = dist.log_prob(action)
        return action, log_prob, value, (hn, cn)

    def evaluate_actions(
        self,
        obs: torch.Tensor,
        actions: torch.Tensor,
        action_mask: Optional[torch.Tensor] = None,
        hidden_state: Optional[Tuple[torch.Tensor, torch.Tensor]] = None,
    ):
        logits, value, _ = self.forward(obs, hidden_state=hidden_state)
        dist = MaskedCategorical.from_logits_and_mask(logits, action_mask)
        log_probs = dist.log_prob(actions)
        entropy = dist.entropy().mean()
        return log_probs, entropy, value
