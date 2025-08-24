import torch
import torch.nn.functional as F
from .storage import RolloutStorage

class PPOLearner:
    def __init__(self, policy, optimizer, clip_eps=0.2, value_coef=0.5, entropy_coef=0.01, max_grad_norm=1.0):
        self.policy = policy
        self.optimizer = optimizer
        self.clip_eps = clip_eps
        self.value_coef = value_coef
        self.entropy_coef = entropy_coef
        self.max_grad_norm = max_grad_norm

    def update(self, storage: RolloutStorage):
        advantages = storage.returns - storage.values
        advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8 + 1e-12)
        batch_obs = storage.obs[:-1].reshape(-1, *storage.obs.size()[2:])
        batch_actions = storage.actions.reshape(-1)
        batch_log_probs = storage.log_probs.reshape(-1)
        batch_adv = advantages.reshape(-1)
        batch_returns = storage.returns.reshape(-1)

        new_log_probs, entropy, values = self.policy.evaluate_actions(batch_obs, batch_actions, None)
        ratio = torch.exp(new_log_probs - batch_log_probs)
        surr1 = ratio * batch_adv
        surr2 = torch.clamp(ratio, 1.0 - self.clip_eps, 1.0 + self.clip_eps) * batch_adv
        policy_loss = -torch.min(surr1, surr2).mean()
        value_loss = F.mse_loss(values, batch_returns)
        entropy_loss = -entropy.mean()
        loss = policy_loss + self.value_coef * value_loss + self.entropy_coef * entropy_loss

        self.optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(self.policy.parameters(), self.max_grad_norm)
        self.optimizer.step()

        return policy_loss.item(), value_loss.item(), entropy.mean().item()
