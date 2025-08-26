from __future__ import annotations
from typing import Optional, Tuple, Dict, Iterator
import torch


class RolloutStorage:
    """
    PPO + LSTM rollout buffer (single or multi env).

    Stores:
      - obs[t+1]  : observation *after* step t (obs[0] set via set_initial_obs)
      - actions[t], log_probs[t], values[t], rewards[t], dones[t]
      - h_pre[t], c_pre[t] : pre-action LSTM states (L, N, Hdim)
      - h_last, c_last     : LSTM states after final env step
      - legal_masks[t]     : optional (N, A), for masked re-evaluation

    Shapes (T = rollout_length, N = num_envs):
      obs           : (T+1, N, C, H, W)
      actions       : (T,   N)
      log_probs     : (T,   N)
      values        : (T,   N)
      rewards       : (T,   N)
      dones         : (T,   N)
      h_pre/c_pre   : (T,   L, N, Hdim)
      h_last/c_last : (L,   N, Hdim)
      legal_masks   : (T,   N, A)  if action_dim provided
      returns       : (T,   N)
      advantages    : (T,   N)
    """

    def __init__(
        self,
        rollout_length: int,
        num_envs: int,
        obs_shape,
        hidden_size: int,
        num_layers: int,
        *,
        action_dim: Optional[int] = None,
        device: Optional[torch.device] = None,
    ):
        self.rollout_length = int(rollout_length)
        self.num_envs = int(num_envs)
        self.hidden_size = int(hidden_size)
        self.num_layers = int(num_layers)
        self.action_dim = int(action_dim) if action_dim is not None else None

        C, H, W = int(obs_shape[0]), int(obs_shape[1]), int(obs_shape[2])

        dev = device if device is not None else torch.device("cpu")
        f32 = torch.float32
        i64 = torch.int64

        # Buffers
        self.obs = torch.zeros((self.rollout_length + 1, self.num_envs, C, H, W), dtype=f32, device=dev)

        self.actions   = torch.zeros((self.rollout_length, self.num_envs), dtype=i64, device=dev)
        self.log_probs = torch.zeros((self.rollout_length, self.num_envs), dtype=f32, device=dev)
        self.rewards   = torch.zeros((self.rollout_length, self.num_envs), dtype=f32, device=dev)
        self.dones     = torch.zeros((self.rollout_length, self.num_envs), dtype=f32, device=dev)
        self.values    = torch.zeros((self.rollout_length, self.num_envs), dtype=f32, device=dev)

        self.h_pre = torch.zeros((self.rollout_length, self.num_layers, self.num_envs, self.hidden_size), dtype=f32, device=dev)
        self.c_pre = torch.zeros((self.rollout_length, self.num_layers, self.num_envs, self.hidden_size), dtype=f32, device=dev)

        self.h_last = torch.zeros((self.num_layers, self.num_envs, self.hidden_size), dtype=f32, device=dev)
        self.c_last = torch.zeros((self.num_layers, self.num_envs, self.hidden_size), dtype=f32, device=dev)

        if self.action_dim is not None:
            self.legal_masks = torch.zeros((self.rollout_length, self.num_envs, self.action_dim), dtype=f32, device=dev)
        else:
            self.legal_masks = None

        self.returns    = torch.zeros((self.rollout_length, self.num_envs), dtype=f32, device=dev)
        self.advantages = torch.zeros((self.rollout_length, self.num_envs), dtype=f32, device=dev)

        # Optional EV readout
        self.last_values_np  = None
        self.last_returns_np = None

        self.step = 0

    # --- device control ---
    def to(self, device: torch.device):
        for name, buf in list(self.__dict__.items()):
            if isinstance(buf, torch.Tensor):
                setattr(self, name, buf.to(device))
        return self

    # --- API ---
    def set_initial_obs(self, obs0: torch.Tensor):
        """
        obs0: (N, C, H, W) on same device/dtype
        """
        assert obs0.shape[:1] == (self.num_envs,), f"obs0 must be (N,C,H,W); got {tuple(obs0.shape)}"
        self.obs[0].copy_(obs0)

    def reset(self):
        """Full zero reset of internal buffers and step counter."""
        self.step = 0
        for name in ("obs", "actions", "log_probs", "rewards", "dones", "values",
                     "h_pre", "c_pre", "h_last", "c_last", "returns", "advantages"):
            getattr(self, name).zero_()
        if self.legal_masks is not None:
            self.legal_masks.zero_()

    def clear(self):
        """Only reset the write index; keeps buffers (useful between rollouts)."""
        self.step = 0

    def insert(
        self,
        obs_next: torch.Tensor,        # (N, C, H, W)
        actions: torch.Tensor,         # (N,)
        log_probs: torch.Tensor,       # (N,)
        values: torch.Tensor,          # (N,)
        rewards: torch.Tensor,         # (N,)
        dones: torch.Tensor,           # (N,)
        h_pre: torch.Tensor,           # (L, N, Hdim)
        c_pre: torch.Tensor,           # (L, N, Hdim)
        legal_mask: Optional[torch.Tensor] = None,  # (N, A)
    ):
        t = self.step
        assert t < self.rollout_length, "RolloutStorage overflow; call clear() or increase rollout_length"
        self.obs[t + 1].copy_(obs_next)
        self.actions[t].copy_(actions)
        self.log_probs[t].copy_(log_probs)
        self.values[t].copy_(values)
        self.rewards[t].copy_(rewards)
        self.dones[t].copy_(dones)
        self.h_pre[t].copy_(h_pre)
        self.c_pre[t].copy_(c_pre)
        if self.legal_masks is not None and legal_mask is not None:
            self.legal_masks[t].copy_(legal_mask)
        self.step += 1

    def set_last_hidden(self, h_last: torch.Tensor, c_last: torch.Tensor):
        self.h_last.copy_(h_last)
        self.c_last.copy_(c_last)

    def compute_returns_and_advantages(self, last_value: torch.Tensor, gamma: float, lam: float):
        """
        last_value: (N,) or (N,1)
        Correct GAE(λ):
          adv_t = δ_t + γλ (1-done_t) adv_{t+1}
          δ_t   = r_t + γ (1-done_t) V(s_{t+1}) - V(s_t)
          return_t = adv_t + V(s_t)
        """
        T, N = self.rollout_length, self.num_envs
        device = self.obs.device

        advantages = torch.zeros((T, N), dtype=torch.float32, device=device)
        returns = torch.zeros((T, N), dtype=torch.float32, device=device)

        lv = last_value
        if lv.ndim == 2 and lv.shape[1] == 1:
            lv = lv.squeeze(1)  # (N,)

        next_value = lv.clone()  # V(s_{T})
        next_adv = torch.zeros((N,), device=device)  # adv_{T} = 0

        for t in reversed(range(T)):
            not_done = 1.0 - self.dones[t]  # (N,)
            delta = self.rewards[t] + gamma * next_value * not_done - self.values[t]
            adv_t = delta + gamma * lam * next_adv * not_done
            advantages[t] = adv_t
            returns[t] = adv_t + self.values[t]
            next_value = self.values[t]  # V(s_t) becomes next step's V(s_{t+1})
            next_adv = adv_t

        self.advantages = advantages
        self.returns = returns

        # numpy copies for explained variance printout
        self.last_values_np = self.values.detach().flatten().cpu().numpy()
        self.last_returns_np = self.returns.detach().flatten().cpu().numpy()
        return self.advantages, self.returns

    # ---------- Minibatches for PPO ----------
    def iter_minibatches(self, minibatch_size: int, device: torch.device) -> Iterator[Dict[str, torch.Tensor]]:
        """
        Flattens (T, N) -> (B = T*N). For LSTM, we pass the *pre-action* states h_pre[t], c_pre[t]
        as the initial states for each transition in the batch (shape: L, B, Hdim).
        """
        T, N = self.rollout_length, self.num_envs
        B = T * N

        # Obs aligned with actions are obs[0:T]
        obs_flat     = self.obs[:-1].reshape(B, *self.obs.shape[2:])                 # (B, C, H, W)
        actions_flat = self.actions.reshape(B, 1)                                     # (B, 1)
        returns_flat = self.returns.reshape(B)                                        # (B,)
        adv_flat     = self.advantages.reshape(B)                                     # (B,)
        values_flat  = self.values.reshape(B)                                         # (B,)
        logp_flat    = self.log_probs.reshape(B)                                      # (B,)

        # (T, L, N, H) -> (L, B, H)
        h0 = self.h_pre.permute(1, 0, 2, 3).reshape(self.num_layers, B, self.hidden_size)
        c0 = self.c_pre.permute(1, 0, 2, 3).reshape(self.num_layers, B, self.hidden_size)

        if self.legal_masks is not None:
            masks_flat = self.legal_masks.reshape(B, self.legal_masks.shape[-1])
        else:
            masks_flat = None

        idx = torch.randperm(B, device=obs_flat.device)
        for start in range(0, B, minibatch_size):
            sl = idx[start:start + minibatch_size]
            batch: Dict[str, torch.Tensor] = {
                "obs":          obs_flat.index_select(0, sl).to(device, non_blocking=True),
                "actions":      actions_flat.index_select(0, sl).to(device, non_blocking=True),
                "returns":      returns_flat.index_select(0, sl).to(device, non_blocking=True),
                "advantages":   adv_flat.index_select(0, sl).to(device, non_blocking=True),
                "values":       values_flat.index_select(0, sl).to(device, non_blocking=True),
                "log_probs":    logp_flat.index_select(0, sl).to(device, non_blocking=True),
                "h0":           h0.index_select(1, sl).to(device, non_blocking=True),    # (L, b, H)
                "c0":           c0.index_select(1, sl).to(device, non_blocking=True),    # (L, b, H)
            }
            if masks_flat is not None:
                batch["legal_masks"] = masks_flat.index_select(0, sl).to(device, non_blocking=True)
            else:
                # If no masks stored, use ones
                A = self.action_dim if self.action_dim is not None else 1
                batch["legal_masks"] = torch.ones((sl.numel(), A), dtype=torch.float32, device=device)

            yield batch
