# training/algorithms/ppo_lstm/learner.py
from __future__ import annotations
from dataclasses import dataclass
from typing import Dict, Any, Tuple
import math
import torch
import torch.nn as nn
import torch.nn.functional as F


@dataclass
class PPOConfig:
    gamma: float = 0.997
    gae_lambda: float = 0.95
    clip_eps: float = 0.2
    entropy_coef: float = 0.015
    value_coef: float = 0.5
    max_grad_norm: float = 1.0
    lr: float = 2.5e-4
    epochs: int = 3
    minibatch_size: int = 2048
    amp: bool = True


class PPOLearner:
    """
    Minimal but solid recurrent PPO learner that **uses the stored LSTM states**
    for logprob/value evaluation. Expects a `storage` object that yields
    minibatches with (obs, actions, returns, advantages, masks, h0, c0).
    """

    def __init__(self, policy: nn.Module, cfg: PPOConfig, device: torch.device):
        self.policy = policy.to(device)
        self.cfg = cfg
        self.device = device
        self.optimizer = torch.optim.AdamW(self.policy.parameters(), lr=cfg.lr, eps=1e-8, betas=(0.9, 0.999))
        self.scaler = torch.cuda.amp.GradScaler(enabled=(cfg.amp and device.type == "cuda"))

    def _evaluate_minibatch(self, obs, actions, masks, h0, c0) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Run a recurrent forward pass with provided initial states to get:
        - log_probs of taken actions
        - entropy of policy distribution
        - value estimates
        """
        # policy.forward must accept (obs, masks, h0, c0) and return dict
        out = self.policy(obs, masks=masks, h0=h0, c0=c0)
        dist = out["dist"]         # Categorical with .log_prob/.entropy
        values = out["value"]      # [B, 1]
        log_probs = dist.log_prob(actions.squeeze(-1))  # [B]
        entropy = dist.entropy().mean()
        return log_probs, entropy, values.squeeze(-1)

<<<<<<< Updated upstream
        obs = storage.obs[:-1].reshape(B, *storage.obs.size()[2:]).to(device)
        actions = storage.actions.reshape(B).to(device)
        old_log_probs = storage.log_probs.reshape(B).to(device)
        returns = storage.returns.reshape(B).to(device)
        adv = advantages.reshape(B).to(device)
        action_masks = storage.action_masks.reshape(B, storage.action_dim).to(device)
=======
    def update(self, storage) -> Dict[str, float]:
        """
        One PPO update over data in `storage` (already collected).
        Returns a dict of scalar logs.
        """
        total_loss_acc = 0.0
        policy_loss_acc = 0.0
        value_loss_acc = 0.0
        entropy_acc = 0.0
        n_minibatches = 0
>>>>>>> Stashed changes

        for _ in range(self.cfg.epochs):
            for batch in storage.iter_minibatches(self.cfg.minibatch_size, device=self.device):
                obs      = batch["obs"]          # [B, C, H, W]
                actions  = batch["actions"]      # [B, 1]
                returns  = batch["returns"]      # [B]
                advs     = batch["advantages"]   # [B]
                masks    = batch["legal_masks"]  # [B, A]
                h0       = batch["h0"]           # [num_layers, B, hidden]
                c0       = batch["c0"]

                advs = (advs - advs.mean()) / (advs.std(unbiased=False) + 1e-8)

                with torch.cuda.amp.autocast(enabled=(self.cfg.amp and self.device.type == "cuda")):
                    logp, entropy, values = self._evaluate_minibatch(obs, actions, masks, h0, c0)
                    old_logp = batch["log_probs"]  # from rollout collection
                    ratio = (logp - old_logp).exp()

<<<<<<< Updated upstream
        for start in range(0, B, mb):
            end = start + mb
            mb_idx = idx[start:end]
            mb_obs = obs[mb_idx]
            mb_actions = actions[mb_idx]
            mb_old_log = old_log_probs[mb_idx]
            mb_adv = adv[mb_idx]
            mb_ret = returns[mb_idx]
            mb_h = h_pre[:, mb_idx, :]
            mb_c = c_pre[:, mb_idx, :]
            mb_masks = action_masks[mb_idx]

            with torch.cuda.amp.autocast(enabled=self.scaler.is_enabled()):
                new_log, entropy, values = self.policy.evaluate_actions(
                    mb_obs, mb_actions, action_mask=mb_masks, hidden_state=(mb_h, mb_c)
                )
                ratio = (new_log - mb_old_log).exp()
                surr1 = ratio * mb_adv
                surr2 = torch.clamp(ratio, 1.0 - self.cfg.clip_eps, 1.0 + self.cfg.clip_eps) * mb_adv
                policy_loss = -torch.min(surr1, surr2).mean()
                value_loss = F.mse_loss(values, mb_ret)
                loss = policy_loss + self.cfg.value_coef * value_loss - self.cfg.entropy_coef * entropy
=======
                    # policy (clipped surrogate)
                    surr1 = ratio * advs
                    surr2 = torch.clamp(ratio, 1.0 - self.cfg.clip_eps, 1.0 + self.cfg.clip_eps) * advs
                    policy_loss = -torch.min(surr1, surr2).mean()

                    # value loss (clipped)
                    values_clipped = batch["values"] + torch.clamp(values - batch["values"], -self.cfg.clip_eps, self.cfg.clip_eps)
                    value_losses = (values - returns).pow(2)
                    value_losses_clipped = (values_clipped - returns).pow(2)
                    value_loss = 0.5 * torch.max(value_losses, value_losses_clipped).mean()
>>>>>>> Stashed changes

                    loss = policy_loss + self.cfg.value_coef * value_loss - self.cfg.entropy_coef * entropy

                self.optimizer.zero_grad(set_to_none=True)
                self.scaler.scale(loss).backward()
                nn.utils.clip_grad_norm_(self.policy.parameters(), self.cfg.max_grad_norm)
                self.scaler.step(self.optimizer)
                self.scaler.update()

                total_loss_acc  += float(loss.detach().cpu())
                policy_loss_acc += float(policy_loss.detach().cpu())
                value_loss_acc  += float(value_loss.detach().cpu())
                entropy_acc     += float(entropy.detach().cpu())
                n_minibatches   += 1

        if n_minibatches == 0:
            return {"loss/total": 0.0, "loss/policy": 0.0, "loss/value": 0.0, "loss/entropy": 0.0}

        return {
            "loss/total":   total_loss_acc / n_minibatches,
            "loss/policy":  policy_loss_acc / n_minibatches,
            "loss/value":   value_loss_acc / n_minibatches,
            "loss/entropy": entropy_acc / n_minibatches,
            "optim/lr":     float(self.optimizer.param_groups[0]["lr"]),
        }
