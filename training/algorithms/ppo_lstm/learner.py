# training/algorithms/ppo_lstm/learner.py
from __future__ import annotations
import torch
import torch.nn.functional as F
from dataclasses import dataclass


@dataclass
class PPOConfig:
    lr: float = 2.5e-4
    clip_eps: float = 0.2
    value_coef: float = 0.5
    entropy_coef: float = 0.015
    max_grad_norm: float = 1.0
    epochs: int = 2
    minibatch_size: int = 4096
    amp: bool = True


class PPOLearner:
    def __init__(self, policy, cfg: PPOConfig):
        self.policy = policy
        self.cfg = cfg
        self.opt = torch.optim.AdamW(policy.parameters(), lr=cfg.lr)
        self.scaler = torch.cuda.amp.GradScaler(enabled=cfg.amp and torch.cuda.is_available())

    def update(self, storage, advantages, device):
        T, N = storage.rollout_length, storage.num_envs
        B = T * N

        obs = storage.obs[:-1].reshape(B, *storage.obs.size()[2:]).to(device)
        actions = storage.actions.reshape(B).to(device)
        old_log_probs = storage.log_probs.reshape(B).to(device)
        returns = storage.returns.reshape(B).to(device)
        adv = advantages.reshape(B).to(device)
        action_masks = storage.action_masks.reshape(B, storage.action_dim).to(device)

        # normalize advantages
        adv = (adv - adv.mean()) / (adv.std() + 1e-8)

        # reshape pre hidden states into sequence of length 1
        h_pre = storage.h_pre.permute(1, 0, 2, 3).reshape(storage.num_layers, B, storage.hidden_size).to(device)
        c_pre = storage.c_pre.permute(1, 0, 2, 3).reshape(storage.num_layers, B, storage.hidden_size).to(device)

        idx = torch.randperm(B, device=device)
        mb = self.cfg.minibatch_size
        total_loss, total_pi, total_v, total_ent = 0.0, 0.0, 0.0, 0.0

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

            self.opt.zero_grad(set_to_none=True)
            self.scaler.scale(loss).backward()
            torch.nn.utils.clip_grad_norm_(self.policy.parameters(), self.cfg.max_grad_norm)
            self.scaler.step(self.opt)
            self.scaler.update()

            total_loss += float(loss.detach())
            total_pi += float(policy_loss.detach())
            total_v += float(value_loss.detach())
            total_ent += float(entropy.detach())

        steps = max(1, (B + mb - 1) // mb)
        logs = {
            "loss/total": total_loss / steps,
            "loss/policy": total_pi / steps,
            "loss/value": total_v / steps,
            "loss/entropy": total_ent / steps,
        }
        return logs
