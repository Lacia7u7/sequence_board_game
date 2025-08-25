# training/scripts/train.py
from __future__ import annotations
import argparse
import json
import os
from typing import Any, Dict, Tuple

import numpy as np
import torch

from ..envs.sequence_env import SequenceEnv
from ..algorithms.ppo_lstm.policy import PPORecurrentPolicy
from ..algorithms.ppo_lstm.learner import PPOLearner, PPOConfig
from ..algorithms.ppo_lstm import storage as ppo_storage


def load_config(path: str) -> Dict[str, Any]:
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, required=True, help="Path to JSON config.")
    parser.add_argument("--override", type=str, default=None, help='JSON string of dot-path overrides e.g. \'{"training.lr":1e-4}\'')
    args = parser.parse_args()

    cfg = load_config(args.config)
    if args.override:
        overrides = json.loads(args.override)
        for k, v in overrides.items():
            parts = k.split(".")
            node = cfg
            for p in parts[:-1]:
                node = node.setdefault(p, {})
            node[parts[-1]] = v

    env = SequenceEnv(cfg)
    obs, info = env.reset()
    obs_channels = int(obs.shape[0])
    action_dim = int(env.action_dim)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    model_cfg = cfg.get("model", {})
    policy = PPORecurrentPolicy(
        obs_channels=obs_channels,
        action_dim=action_dim,
        conv_channels=tuple(model_cfg.get("conv_channels", [64, 64, 128])),
        lstm_hidden=int(model_cfg.get("lstm_hidden", 512)),
        lstm_layers=int(model_cfg.get("lstm_layers", 1)),
    ).to(device)

    train_cfg = cfg.get("training", {})
    ppo_cfg = PPOConfig(
        lr=float(train_cfg.get("lr", 2.5e-4)),
        clip_eps=float(train_cfg.get("clip_eps", 0.2)),
        value_coef=float(train_cfg.get("value_coef", 0.5)),
        entropy_coef=float(train_cfg.get("entropy_coef", 0.015)),
        max_grad_norm=float(train_cfg.get("max_grad_norm", 1.0)),
        epochs=int(train_cfg.get("epochs", 2)),
        minibatch_size=int(train_cfg.get("minibatch_size", 4096)),
        amp=bool(train_cfg.get("amp", True)),
    )

    rollout_length = int(train_cfg.get("rollout_length", 64))
    total_updates = int(train_cfg.get("total_updates", 10))

    storage = ppo_storage.RolloutStorage(
        rollout_length=rollout_length,
        num_envs=1,
        obs_shape=obs.shape,
        action_dim=action_dim,
        hidden_size=policy.lstm_hidden,
        num_layers=policy.lstm.num_layers,
    ).to(device)

    # Initialize
    obs_t = torch.tensor(obs, dtype=torch.float32, device=device).unsqueeze(0)  # (1,C,H,W)
    hidden = policy.init_hidden(batch_size=1)

    learner = PPOLearner(policy, ppo_cfg)

    for update in range(total_updates):
        # Start a new rollout by clearing old data and setting the initial observation
        storage.reset()
        storage.set_initial_obs(obs_t)
        for t in range(rollout_length):
            mask = torch.tensor(info.get("legal_mask"), dtype=torch.float32, device=device).unsqueeze(0)
            h_pre, c_pre = hidden

            with torch.no_grad(), torch.cuda.amp.autocast(enabled=learner.scaler.is_enabled()):
                action, log_prob, value, new_hidden = policy.act(obs_t, action_mask=mask, hidden_state=hidden)

            action_int = int(action.item())
            obs_next, reward, terminated, truncated, info = env.step(action_int)

            obs_next_t = torch.tensor(obs_next, dtype=torch.float32, device=device).unsqueeze(0)
            reward_t = torch.tensor([reward], dtype=torch.float32, device=device)
            done_t = torch.tensor([1.0 if (terminated or truncated) else 0.0], dtype=torch.float32, device=device)

            storage.insert(
                obs_next=obs_next_t,
                actions=action.detach().cpu(),
                log_probs=log_prob.detach().cpu(),
                values=value.detach().cpu(),
                rewards=reward_t.detach().cpu(),
                dones=done_t.detach().cpu(),
                h_pre=h_pre.detach().cpu(),
                c_pre=c_pre.detach().cpu(),
                action_masks=mask.detach().cpu(),
            )

            obs_t = obs_next_t
            hidden = new_hidden

            if terminated or truncated:
                obs, info = env.reset()
                obs_t = torch.tensor(obs, dtype=torch.float32, device=device).unsqueeze(0)
                hidden = policy.init_hidden(batch_size=1)

        # Bootstrap with final hidden state & last obs
        storage.set_last_hidden(hidden[0].detach().cpu(), hidden[1].detach().cpu())
        with torch.no_grad():
            last_logits, last_value, _ = policy(obs_t, hidden_state=hidden)
            last_value = last_value.detach().cpu()
        advantages, _returns = storage.compute_returns_and_advantages(last_value, float(train_cfg.get("gamma", 0.997)), float(train_cfg.get("gae_lambda", 0.95)))

        logs = learner.update(storage, advantages, device)
        print(f"update {update+1}/{total_updates} | " + " | ".join(f"{k}:{v:.4f}" for k, v in logs.items()))

    # Save final checkpoint
    os.makedirs(cfg.get("logging", {}).get("logdir", "runs"), exist_ok=True)
    torch.save(policy.state_dict(), os.path.join(cfg.get("logging", {}).get("logdir", "runs"), "policy_final.pt"))


if __name__ == "__main__":
    main()
