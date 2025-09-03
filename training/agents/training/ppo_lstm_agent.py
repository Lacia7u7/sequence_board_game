# training/agents/training/ppo_lstm_agent.py
from __future__ import annotations
from typing import Optional, Dict, Any
from collections import OrderedDict

import numpy as np
import torch
from overrides import overrides

from ..base_agent import ObsAgent
from ...algorithms.ppo_lstm.ppo_lstm_policy import PPORecurrentPolicy
from ...envs.sequence_env import SequenceEnv


class PPOLstmAgent(ObsAgent):
    """
    PPO LSTM agent updated to work with the new PPORecurrentPolicy that supports:
      - Optional ResNet backbone blocks
      - Optional Transformer grid encoder
      - Flexible loading of checkpoints (handles module. prefix, nested keys, strict=False)
    """
    def __init__(
        self,
        env: SequenceEnv,
        policy_path: str,
        deterministic: bool = True,
        device: str = "cpu",
        **kwargs,
    ):
        self.env = env
        self.device = torch.device(device)

        # --- shapes from env ---
        obs_shape = env.get_obs().shape  # (C,H,W)
        action_dim = int(env.action_dim)

        # --- model config from env (with safe fallbacks) ---
        model_cfg = dict(env.config.get("model", {}))  # may contain backbone/transformer/activation/norm/etc.
        conv_channels = model_cfg.get("conv_channels", [64, 64, 128])
        lstm_hidden = int(model_cfg.get("lstm_hidden", 256))
        lstm_layers = int(model_cfg.get("lstm_layers", 1))

        # --- build policy ---
        self.policy = PPORecurrentPolicy(
            obs_shape=obs_shape,
            action_dim=action_dim,
            conv_channels=conv_channels,
            lstm_hidden=lstm_hidden,
            lstm_layers=lstm_layers,
            device=self.device,
            model_cfg=model_cfg,               # NEW: pass through entire model cfg (resnet/transformer/etc.)
            deterministic=bool(deterministic), # NEW: policy now owns this flag internally
        )

        # --- load checkpoint robustly ---
        sd = torch.load(policy_path, map_location=self.device)
        state_dict = self._extract_state_dict(sd)
        missing, unexpected = self.policy.load_state_dict(state_dict, strict=False)
        if missing:
            # These are usually harmless when toggling optional modules; keep silent or log if you have a logger.
            pass
        if unexpected:
            # Same here—keys from older/newer heads/backbones are ignored with strict=False.
            pass

        self.policy.eval()
        self.deterministic = bool(deterministic)  # keep a local copy if your runner expects it
        self.h, self.c = self.policy.get_initial_state(batch_size=1)

    # -------- ObsAgent API --------

    @overrides
    def reset(self, env, seat: int) -> None:
        self.h, self.c = self.policy.get_initial_state(batch_size=1)

    @torch.no_grad()
    def select_action(
        self,
        legal_mask: Optional[np.ndarray],
        ctx: Optional[Dict[str, Any]] = None
    ) -> int:
        """
        Expects ctx['obs'] as a numpy array shaped (C,H,W).
        Returns a Python int action.
        """
        if ctx is None or "obs" not in ctx:
            raise ValueError("PPOLstmAgent requires ctx['obs'].")

        obs_np = ctx["obs"]
        obs_t = torch.from_numpy(obs_np).unsqueeze(0).to(self.device, dtype=torch.float32)  # (1,C,H,W)

        lm_t = None
        if legal_mask is not None:
            # policy can accept numpy, but we normalize to torch.Tensor here
            lm_t = torch.as_tensor(legal_mask, device=self.device, dtype=torch.float32)

        out = self.policy.select_action(
            legal_mask=lm_t,
            ctx={
                "obs": obs_t,   # (N,C,H,W)
                "h0": self.h,   # (L,N,H)
                "c0": self.c,   # (L,N,H)
            },
        )

        # carry over recurrent state
        self.h = out["h"]
        self.c = out["c"]

        # action is shape (N,) where N=1
        action = int(out["action"].item()) if out["action"].numel() == 1 else int(out["action"][0].item())
        return action

    # -------- helpers --------

    @staticmethod
    def _extract_state_dict(ckpt: Any) -> OrderedDict:
        """
        Accepts a variety of checkpoint layouts and returns a clean state_dict:
          - raw state dict
          - {'state_dict': ...}
          - {'policy': ...}
          - {'model_state_dict': ...}
          - keys possibly prefixed with 'module.'
        """
        state_dict = None
        if isinstance(ckpt, dict):
            # common wrappers
            for k in ("state_dict", "policy", "model_state_dict"):
                if k in ckpt and isinstance(ckpt[k], dict):
                    state_dict = ckpt[k]
                    break
        if state_dict is None:
            # assume it's already a state dict
            state_dict = ckpt

        # strip potential DataParallel prefixes
        cleaned = OrderedDict()
        for k, v in state_dict.items():
            if isinstance(k, str) and k.startswith("module."):
                cleaned[k[len("module."):]] = v
            else:
                cleaned[k] = v
        return cleaned
