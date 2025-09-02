from __future__ import annotations
from typing import Optional, Dict, Any, Tuple
import numpy as np
import torch
from overrides import overrides

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
        *,
        # NEW: full model configuration (ResNet/Transformer/acts/norms, etc.)
        model_cfg: Optional[Dict[str, Any]] = None,
        # NEW: deterministic sampling (useful for eval-style greedy play)
        deterministic: bool = False,
        # NEW: allow non-strict loading when arch differs slightly
        load_strict: bool = True,
    ):
        super().__init__(None)
        self.device = torch.device(device)

        # Build policy with extended config
        self.policy = PPORecurrentPolicy(
            obs_shape=tuple(obs_shape),
            action_dim=int(action_dim),
            conv_channels=conv_channels,
            lstm_hidden=int(lstm_hidden),
            lstm_layers=int(lstm_layers),
            device=self.device,
            value_tanh_bound=float(value_tanh_bound),
            model_cfg=(model_cfg or {}),          # <-- passes backbone/transformer/activation/norm
            deterministic=bool(deterministic),    # <-- enables greedy if desired
        )

        self.state_dict_path = state_dict_path

        # Load weights (respect load_strict; if False and first try fails, retry non-strict)
        if state_dict_path is not None:
            state = torch.load(state_dict_path, map_location=self.device)
            try:
                self.policy.load_state_dict(state, strict=bool(load_strict))
            except Exception as e:
                if not load_strict:
                    print(f"[PPOFrozenAgent] Non-strict load retry due to: {e}")
                    self.policy.load_state_dict(state, strict=False)
                else:
                    raise

        self.policy.eval()
        for p in self.policy.parameters():
            p.requires_grad_(False)

        self.h, self.c = self.policy.get_initial_state(batch_size=1)

        # Keep kwargs to spawn cloned agents with identical config
        self.policy_kwargs = dict(
            obs_shape=obs_shape,
            action_dim=action_dim,
            conv_channels=conv_channels,
            lstm_hidden=lstm_hidden,
            lstm_layers=lstm_layers,
            value_tanh_bound=value_tanh_bound,
            device=str(device),
            model_cfg=(model_cfg or {}),
            deterministic=bool(deterministic),
            load_strict=bool(load_strict),
            state_dict_path=state_dict_path,
        )

    @overrides
    def make_new_agent(self, env):
        # Recreate an identical frozen agent (same weights/config)
        return PPOFrozenAgent(**self.policy_kwargs)

    def reset(self, env, seat: int) -> None:
        self.h, self.c = self.policy.get_initial_state(batch_size=1)

    def select_action(self, legal_mask: Optional[np.ndarray], ctx: Optional[Dict[str, Any]] = None) -> int:
        obs = ctx.get("obs") if ctx else None
        if isinstance(obs, np.ndarray):
            obs_t = torch.from_numpy(obs).unsqueeze(0).to(self.device, dtype=torch.float32)
        else:
            # already a 4D tensor (1,C,H,W)
            obs_t = obs

        with torch.no_grad():
            out = self.policy.select_action(
                legal_mask=legal_mask,
                ctx={"obs": obs_t, "h0": self.h, "c0": self.c},
            )
        self.h, self.c = out["h"], out["c"]
        return int(out["action"].item())
