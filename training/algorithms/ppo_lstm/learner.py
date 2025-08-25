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
        Prefer policy.evaluate_actions() when available; otherwise call forward()
        with a robust mask-arg compatibility shim.
        """
        # 1) Preferred explicit API if available
        if hasattr(self.policy, "evaluate_actions"):
            # Try a few common kwarg names for the mask argument.
            eval_kwargs_base = dict()
            # Some policies expect a tuple for hidden state
            if "hidden_state" in self.policy.evaluate_actions.__code__.co_varnames:
                eval_kwargs_base["hidden_state"] = (h0, c0)

            def _try_eval_with(name: Optional[str]):
                kw = dict(eval_kwargs_base)
                if name is not None and masks is not None:
                    kw[name] = masks
                return self.policy.evaluate_actions(obs, actions.squeeze(-1), **kw)

            for mask_key in ("action_mask", "masks", "legal_mask", None):
                try:
                    out = _try_eval_with(mask_key)
                    # normalize returns
                    if isinstance(out, tuple) and len(out) == 3:
                        new_log, entropy, values = out
                        values = values.squeeze(-1) if values.ndim == 2 and values.shape[-1] == 1 else values
                        return new_log, entropy, values
                    elif isinstance(out, dict):
                        new_log = out.get("log_prob")
                        dist = out.get("dist")
                        entropy = out.get("entropy", dist.entropy().mean() if dist is not None else None)
                        values = out.get("value")
                        if new_log is None or values is None:
                            raise RuntimeError("evaluate_actions must return log_prob and value (or dist).")
                        values = values.squeeze(-1) if values.ndim == 2 and values.shape[-1] == 1 else values
                        if entropy is None and dist is not None:
                            entropy = dist.entropy().mean()
                        return new_log, entropy, values
                except TypeError:
                    pass  # try next mask key

            # If evaluate_actions exists but didn’t match any signature, fall through to forward.

        # 2) Generic path via forward()
        # Many policies name the mask kwarg differently; try a few.
        base = {"obs": obs, "h0": h0, "c0": c0}

        def _call_forward(mask_key: Optional[str]):
            kw = dict(base)
            if mask_key is not None and masks is not None:
                kw[mask_key] = masks
            return self.policy.forward(**kw)

        last_err = None
        for mask_key in ("masks", "legal_mask", "action_mask", None):
            try:
                out = _call_forward(mask_key)
                if not isinstance(out, dict) or "value" not in out:
                    raise RuntimeError("policy.forward must return a dict with at least 'value' (and ideally 'dist').")
                dist = out.get("dist", None)
                values = out["value"]
                values = values.squeeze(-1) if values.ndim == 2 and values.shape[-1] == 1 else values
                if dist is None:
                    # If the policy doesn’t return a dist, we can’t compute log_probs/entropy;
                    # in that rare case, raise with a helpful message.
                    raise RuntimeError("policy.forward did not return 'dist'; cannot compute log_prob/entropy for PPO.")
                log_probs = dist.log_prob(actions.squeeze(-1))
                entropy = dist.entropy().mean()
                return log_probs, entropy, values
            except TypeError as e:
                last_err = e
                continue

        # If we got here, none of the signatures matched.
        raise TypeError(f"PPORecurrentPolicy.forward/evaluate_actions mask kwarg mismatch. "
                        f"Tried keys: masks, legal_mask, action_mask. Last error: {last_err}")


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
