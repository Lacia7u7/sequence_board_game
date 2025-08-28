from __future__ import annotations
from typing import Optional, Dict, Any, Tuple
import numpy as np
import torch

from ..agents.base_agent import BaseAgent  # same package as your BaseAgent
from ..algorithms.ppo_lstm.ppo_lstm_policy import PPORecurrentPolicy

class PPOFrozenAgent(BaseAgent):
    """Stateless (no-grad) wrapper over PPORecurrentPolicy for opponent play.

    Loads a snapshot (state_dict) and exposes the BaseAgent API. Hidden state
    is reset every episode via `reset` and kept internally between steps.
    """
    def __init__(
        self,
        obs_shape: Tuple[int, ...],
        action_dim: int,
        conv_channels,
        lstm_hidden: int,
        lstm_layers: int = 1,
        value_tanh_bound: float = 5.0,
        device: str = "cpu",
        state_dict_path: Optional[str] = None,
    ):
        self.device = torch.device(device)
        self.policy = PPORecurrentPolicy(
            obs_shape=tuple(obs_shape),
            action_dim=int(action_dim),
            conv_channels=conv_channels,
            lstm_hidden=int(lstm_hidden),
            lstm_layers=int(lstm_layers),
            device=self.device,
            value_tanh_bound=float(value_tanh_bound),
        )
        if state_dict_path is not None:
            state = torch.load(state_dict_path, map_location=self.device)
            self.policy.load_state_dict(state, strict=True)
        self.policy.eval()
        for p in self.policy.parameters():
            p.requires_grad_(False)
        self.h, self.c = self.policy.get_initial_state(batch_size=1)

    def reset(self, env, seat: int) -> None:
        self.h, self.c = self.policy.get_initial_state(batch_size=1)

    def select_action(self, legal_mask: Optional[np.ndarray], ctx: Optional[Dict[str, Any]] = None) -> int:
        obs = ctx.get("obs") if ctx else None
        if isinstance(obs, np.ndarray):
            obs_t = torch.from_numpy(obs).unsqueeze(0).to(self.device, dtype=torch.float32)
        else:
            obs_t = obs  # already a 4D tensor (1,C,H,W)
        with torch.no_grad():
            out = self.policy.select_action(
                legal_mask=legal_mask,
                ctx={"obs": obs_t, "h0": self.h, "c0": self.c},
            )
        self.h, self.c = out["h"], out["c"]
        return int(out["action"].item())