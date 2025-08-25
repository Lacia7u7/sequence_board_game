# training/algorithms/ppo_lstm/storage.py
from __future__ import annotations
import torch
from typing import Tuple


class RolloutStorage:
    """
    Stores rollouts for PPO with LSTM.
    We store the PRE-action hidden state (h_pre, c_pre) at each step,
    and the final hidden state after the last env step for bootstrapping.
    """
    def __init__(self, rollout_length: int, num_envs: int, obs_shape, hidden_size: int, num_layers: int):
        self.rollout_length = rollout_length
        self.num_envs = num_envs
        self.hidden_size = hidden_size
        self.num_layers = num_layers

        self.obs = torch.zeros((rollout_length + 1, num_envs) + tuple(obs_shape), dtype=torch.float32)
        self.actions = torch.zeros((rollout_length, num_envs), dtype=torch.long)
        self.log_probs = torch.zeros((rollout_length, num_envs), dtype=torch.float32)
        self.rewards = torch.zeros((rollout_length, num_envs), dtype=torch.float32)
        self.dones = torch.zeros((rollout_length, num_envs), dtype=torch.float32)
        self.values = torch.zeros((rollout_length, num_envs), dtype=torch.float32)

        # pre-action hidden states at each step
        self.h_pre = torch.zeros((rollout_length, num_layers, num_envs, hidden_size), dtype=torch.float32)
        self.c_pre = torch.zeros((rollout_length, num_layers, num_envs, hidden_size), dtype=torch.float32)

        # final hidden state for bootstrap
        self.h_last = torch.zeros((num_layers, num_envs, hidden_size), dtype=torch.float32)
        self.c_last = torch.zeros((num_layers, num_envs, hidden_size), dtype=torch.float32)

        self.step = 0
        self.returns = torch.zeros((rollout_length, num_envs), dtype=torch.float32)

    def to(self, device: torch.device):
        self.obs = self.obs.to(device)
        self.actions = self.actions.to(device)
        self.log_probs = self.log_probs.to(device)
        self.rewards = self.rewards.to(device)
        self.dones = self.dones.to(device)
        self.values = self.values.to(device)
        self.h_pre = self.h_pre.to(device)
        self.c_pre = self.c_pre.to(device)
        self.h_last = self.h_last.to(device)
        self.c_last = self.c_last.to(device)
        self.returns = self.returns.to(device)
        return self

    def set_initial_obs(self, obs0: torch.Tensor):
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
        obs_next: torch.Tensor,
        actions: torch.Tensor,
        log_probs: torch.Tensor,
        values: torch.Tensor,
        rewards: torch.Tensor,
        dones: torch.Tensor,
        h_pre: torch.Tensor,
        c_pre: torch.Tensor,
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
        self.step = (self.step + 1) % self.rollout_length

    def set_last_hidden(self, h_last: torch.Tensor, c_last: torch.Tensor):
        self.h_last.copy_(h_last)
        self.c_last.copy_(c_last)

    def compute_returns_and_advantages(self, last_value: torch.Tensor, gamma: float, lam: float):
        T, N = self.rollout_length, self.num_envs
        advantages = torch.zeros((T, N), dtype=torch.float32, device=self.obs.device)
        returns = torch.zeros((T + 1, N), dtype=torch.float32, device=self.obs.device)
        returns[-1] = last_value
        for t in reversed(range(T)):
            mask = 1.0 - self.dones[t]
            delta = self.rewards[t] + gamma * returns[t + 1] * mask - self.values[t]
            if t < T - 1:
                advantages[t] = delta + gamma * lam * advantages[t + 1] * mask
            else:
                advantages[t] = delta
            returns[t] = advantages[t] + self.values[t]
        self.returns = returns[:-1]
        return advantages, self.returns
