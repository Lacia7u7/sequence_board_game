# training/agents/training/ppo_lstm_agent.py
from __future__ import annotations
from typing import Optional, Dict, Any
import numpy as np
import torch
from overrides import overrides

from ..base_agent import ObsAgent
from ...algorithms.ppo_lstm.ppo_lstm_policy import PPORecurrentPolicy
from ...envs.sequence_env import SequenceEnv


class PPOLstmAgent(ObsAgent):
    def __init__(self, env: SequenceEnv , policy_path: str, deterministic: bool = True, device: str = "cpu", **kwargs):
        self.env = env
        self.device = torch.device(device)
        obs_shape = env.get_obs().shape
        action_dim = env.action_dim
        # Recreate the policy with env config sizes
        conv_channels = env.config["model"].get("conv_channels", [64,64,128])
        lstm_hidden = int(env.config["model"].get("lstm_hidden", 256))
        lstm_layers = int(env.config["model"].get("lstm_layers", 1))
        self.policy = PPORecurrentPolicy(
            obs_shape=obs_shape,
            action_dim=action_dim,
            conv_channels=conv_channels,
            lstm_hidden=lstm_hidden,
            lstm_layers=lstm_layers,
            device=self.device,
        )
        sd = torch.load(policy_path, map_location=self.device)
        self.policy.load_state_dict(sd)
        self.policy.eval()
        self.deterministic = bool(deterministic)
        self.h, self.c = self.policy.get_initial_state(batch_size=1)

    @overrides
    def reset(self, env, seat: int) -> None:
        self.h, self.c = self.policy.get_initial_state(batch_size=1)

    @torch.no_grad()
    def select_action(self, legal_mask: Optional[np.ndarray], ctx: Optional[Dict[str, Any]] = None) -> int:
        if ctx is None or "obs" not in ctx:
            raise ValueError("PPOLstmAgent requires ctx['obs'].")
        obs = ctx["obs"]
        lm = legal_mask
        obs_t = torch.from_numpy(obs).unsqueeze(0).to(self.device, dtype=torch.float32)
        lm_t = None
        if lm is not None:
            lm_t = torch.from_numpy(lm).unsqueeze(0).to(self.device, dtype=torch.float32)


        out = self.policy.select_action(
            legal_mask=lm,
            ctx={
                "obs": obs_t,  # (1, C, H, W)
                "h0": self.h,
                "c0": self.c,
                "deterministic" : self.deterministic,
            },
        )
        self.c=out["c"]
        self.h=out["h"]
        return int(out["action"])

