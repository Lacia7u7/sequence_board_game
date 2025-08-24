import torch

class RolloutStorage:
    def __init__(self, rollout_length: int, num_envs: int, obs_shape, hidden_size: int):
        self.rollout_length = rollout_length
        self.num_envs = num_envs
        self.obs = torch.zeros((rollout_length + 1, num_envs) + obs_shape)
        self.actions = torch.zeros((rollout_length, num_envs), dtype=torch.long)
        self.log_probs = torch.zeros((rollout_length, num_envs))
        self.rewards = torch.zeros((rollout_length, num_envs))
        self.dones = torch.zeros((rollout_length, num_envs))
        self.values = torch.zeros((rollout_length, num_envs))
        self.hxs = torch.zeros((rollout_length + 1, 1, num_envs, hidden_size))
        self.cxs = torch.zeros((rollout_length + 1, 1, num_envs, hidden_size))
        self.step = 0
        self.returns = torch.zeros((rollout_length, num_envs))

    def insert(self, obs, actions, log_probs, values, rewards, dones, hxs, cxs):
        self.obs[self.step + 1].copy_(obs)
        self.actions[self.step].copy_(actions)
        self.log_probs[self.step].copy_(log_probs)
        self.values[self.step].copy_(values)
        self.rewards[self.step].copy_(rewards)
        self.dones[self.step].copy_(dones)
        self.hxs[self.step + 1].copy_(hxs)
        self.cxs[self.step + 1].copy_(cxs)
        self.step = (self.step + 1) % self.rollout_length

    def compute_returns_and_advantages(self, last_value, gamma, lam):
        T, N = self.rollout_length, self.num_envs
        advantages = torch.zeros((T, N))
        returns = torch.zeros((T + 1, N))
        returns[-1] = last_value
        for t in reversed(range(T)):
            mask = 1.0 - self.dones[t]
            delta = self.rewards[t] + gamma * returns[t+1] * mask - self.values[t]
            if t < T - 1:
                advantages[t] = delta + gamma * lam * advantages[t+1] * mask
            else:
                advantages[t] = delta
            returns[t] = advantages[t] + self.values[t]
        self.returns = returns[:-1]
        return advantages, self.returns
