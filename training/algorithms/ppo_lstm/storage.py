# training/algorithms/ppo_lstm/storage.py
from __future__ import annotations
from typing import Optional, Tuple, Dict, Iterator
import torch


class RolloutStorage:
    """Stores rollouts for PPO with LSTM.

    Besides the usual observations, actions, rewards etc. we also keep the
    legal-action mask for each environment step so that the learner can
    reconstruct the masked policy distribution during training.
    """
<<<<<<< Updated upstream

    def __init__(
        self,
        rollout_length: int,
        num_envs: int,
        obs_shape,
        action_dim: int,
        hidden_size: int,
        num_layers: int,
    ):
        self.rollout_length = rollout_length
        self.num_envs = num_envs
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.action_dim = action_dim

        self.obs = torch.zeros((rollout_length + 1, num_envs) + tuple(obs_shape), dtype=torch.float32)
        self.actions = torch.zeros((rollout_length, num_envs), dtype=torch.long)
        self.log_probs = torch.zeros((rollout_length, num_envs), dtype=torch.float32)
        self.rewards = torch.zeros((rollout_length, num_envs), dtype=torch.float32)
        self.dones = torch.zeros((rollout_length, num_envs), dtype=torch.float32)
        self.values = torch.zeros((rollout_length, num_envs), dtype=torch.float32)
        # mask of legal actions used at each step (t, env, action_dim)
        self.action_masks = torch.zeros(
            (rollout_length, num_envs, action_dim), dtype=torch.float32
        )
=======
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
>>>>>>> Stashed changes

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
<<<<<<< Updated upstream
        self.obs = self.obs.to(device)
        self.actions = self.actions.to(device)
        self.log_probs = self.log_probs.to(device)
        self.rewards = self.rewards.to(device)
        self.dones = self.dones.to(device)
        self.values = self.values.to(device)
        self.action_masks = self.action_masks.to(device)
        self.h_pre = self.h_pre.to(device)
        self.c_pre = self.c_pre.to(device)
        self.h_last = self.h_last.to(device)
        self.c_last = self.c_last.to(device)
        self.returns = self.returns.to(device)
=======
        for name, buf in self.__dict__.items():
            if isinstance(buf, torch.Tensor):
                setattr(self, name, buf.to(device))
>>>>>>> Stashed changes
        return self

    # --- API ---
    def set_initial_obs(self, obs0: torch.Tensor):
        """
        obs0: (N, C, H, W) on same device/dtype
        """
        assert obs0.shape[:1] == (self.num_envs,), f"obs0 must be (N,C,H,W); got {tuple(obs0.shape)}"
        self.obs[0].copy_(obs0)

    def reset(self):
        """Reset internal buffers and step counter.

        RolloutStorage accumulates data over a fixed-length buffer. When
        starting a new rollout we want to clear any stale values from the
        previous iteration and reset the write index.  This helper zeros all
        stored tensors and sets ``step`` back to ``0`` so the next call to
        :meth:`insert` starts writing from the beginning.
        """
        self.step = 0
        self.obs.zero_()
        self.actions.zero_()
        self.log_probs.zero_()
        self.rewards.zero_()
        self.dones.zero_()
        self.values.zero_()
        self.h_pre.zero_()
        self.c_pre.zero_()
        self.h_last.zero_()
        self.c_last.zero_()
        self.returns.zero_()

    def insert(
        self,
<<<<<<< Updated upstream
        obs_next: torch.Tensor,
        actions: torch.Tensor,
        log_probs: torch.Tensor,
        values: torch.Tensor,
        rewards: torch.Tensor,
        dones: torch.Tensor,
        h_pre: torch.Tensor,
        c_pre: torch.Tensor,
        action_masks: torch.Tensor,
    ):
        # obs_next is observation after env.step
        self.obs[self.step + 1].copy_(obs_next)
        self.actions[self.step].copy_(actions)
        self.log_probs[self.step].copy_(log_probs)
        self.values[self.step].copy_(values)
        self.rewards[self.step].copy_(rewards)
        self.dones[self.step].copy_(dones)
        self.h_pre[self.step].copy_(h_pre)
        self.c_pre[self.step].copy_(c_pre)
        self.action_masks[self.step].copy_(action_masks)
        self.step = (self.step + 1) % self.rollout_length
=======
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
>>>>>>> Stashed changes

    def set_last_hidden(self, h_last: torch.Tensor, c_last: torch.Tensor):
        self.h_last.copy_(h_last)
        self.c_last.copy_(c_last)

    def compute_returns_and_advantages(self, last_value: torch.Tensor, gamma: float, lam: float):
        """
        last_value: (N,)
        """
        T, N = self.rollout_length, self.num_envs
        device = self.obs.device

        advantages = torch.zeros((T, N), dtype=torch.float32, device=device)
        returns = torch.zeros((T + 1, N), dtype=torch.float32, device=device)
        returns[-1] = last_value  # bootstrap

        for t in reversed(range(T)):
            not_done = 1.0 - self.dones[t]
            delta = self.rewards[t] + gamma * returns[t + 1] * not_done - self.values[t]
            if t < T - 1:
                advantages[t] = delta + gamma * lam * advantages[t + 1] * not_done
            else:
                advantages[t] = delta
            returns[t] = advantages[t] + self.values[t]

        self.advantages = advantages
        self.returns = returns[:-1]

        # numpy copies for explained variance printout
        self.last_values_np  = self.values.detach().flatten().cpu().numpy()
        self.last_returns_np = self.returns.detach().flatten().cpu().numpy()

        return self.advantages, self.returns

    def clear(self):
        self.step = 0

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
