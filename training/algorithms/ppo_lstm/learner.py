from __future__ import annotations
from dataclasses import dataclass
from typing import Dict, Any, Tuple, Optional
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
    minibatches with (obs, actions, returns, advantages, values, log_probs, legal_masks, h0, c0).
    """

    def __init__(self, policy: nn.Module, cfg: PPOConfig, device: torch.device):
        self.policy = policy.to(device)
        self.cfg = cfg
        self.device = device
        self.optimizer = torch.optim.AdamW(self.policy.parameters(), lr=cfg.lr, eps=1e-8, betas=(0.9, 0.999))
        self.scaler = torch.cuda.amp.GradScaler(enabled=(cfg.amp and device.type == "cuda"))

    def _forward_eval(
        self,
        obs: torch.Tensor,
        actions: torch.Tensor,
        masks: Optional[torch.Tensor],
        h0: torch.Tensor,
        c0: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Returns:
          log_probs [B], entropy [], values [B]
        Tries policy.evaluate_actions first (if available), otherwise uses forward->dist.
        """
        # Preferred explicit API if available
        if hasattr(self.policy, "evaluate_actions"):
            out = self.policy.evaluate_actions(
                obs, actions.squeeze(-1),
                action_mask=masks,
                hidden_state=(h0, c0)
            )
            # Support both tuple and dict forms
            if isinstance(out, tuple) and len(out) == 3:
                new_log, entropy, values = out
            elif isinstance(out, dict):
                new_log = out["log_prob"]
                entropy = out.get("entropy", out["dist"].entropy().mean())
                values = out["value"].squeeze(-1)
            else:
                raise RuntimeError("Unknown return type from policy.evaluate_actions")
            return new_log, entropy, values

        # Generic path via forward()
        kwargs = {"obs": obs, "h0": h0, "c0": c0}
        if masks is not None:
            kwargs["legal_mask"] = masks
        out = self.policy.forward(**kwargs)
        if "dist" not in out or "value" not in out:
            raise RuntimeError("policy.forward must return dict with 'dist' and 'value'")
        dist = out["dist"]
        values = out["value"].squeeze(-1)
        log_probs = dist.log_prob(actions.squeeze(-1))
        entropy = dist.entropy().mean()
        return log_probs, entropy, values

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

        for _ in range(self.cfg.epochs):
            for batch in storage.iter_minibatches(self.cfg.minibatch_size, device=self.device):
                obs      = batch["obs"]          # [B, C, H, W]
                actions  = batch["actions"]      # [B, 1]
                returns  = batch["returns"]      # [B]
                advs     = batch["advantages"]   # [B]
                masks    = batch["legal_masks"]  # [B, A]
                h0       = batch["h0"]           # [num_layers, B, hidden]
                c0       = batch["c0"]
                old_logp = batch["log_probs"]    # [B]
                old_values = batch["values"]     # [B]

                # Normalize advantages
                advs = (advs - advs.mean()) / (advs.std(unbiased=False) + 1e-8)

                with torch.cuda.amp.autocast(enabled=(self.cfg.amp and self.device.type == "cuda")):
                    new_logp, entropy, values = self._forward_eval(obs, actions, masks, h0, c0)
                    ratio = (new_logp - old_logp).exp()

                    # policy (clipped surrogate)
                    surr1 = ratio * advs
                    surr2 = torch.clamp(ratio, 1.0 - self.cfg.clip_eps, 1.0 + self.cfg.clip_eps) * advs
                    policy_loss = -torch.min(surr1, surr2).mean()

                    # value loss (clipped)
                    values_clipped = old_values + torch.clamp(values - old_values, -self.cfg.clip_eps, self.cfg.clip_eps)
                    value_losses = (values - returns).pow(2)
                    value_losses_clipped = (values_clipped - returns).pow(2)
                    value_loss = 0.5 * torch.max(value_losses, value_losses_clipped).mean()

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
