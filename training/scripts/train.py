# training/scripts/train.py
from __future__ import annotations
import argparse
import json
import math
import os
import time
from typing import Dict, Any

import numpy as np
import torch

from ..utils.jsonio import load_json, deep_update
from ..utils.seeding import set_all_seeds
from ..utils.logging import LoggingMux
from ..envs.sequence_env import SequenceEnv
from ..algorithms.ppo_lstm.learner import PPOLearner, PPOConfig
from ..algorithms.ppo_lstm.policy import PPORecurrentPolicy
from ..algorithms.ppo_lstm.storage import RolloutStorage


def explained_variance(y_pred: np.ndarray, y_true: np.ndarray) -> float:
    var_y = float(np.var(y_true)) if len(y_true) else 0.0
    if var_y < 1e-8:
        return float("nan")
    return 1.0 - float(np.var(y_true - y_pred)) / (var_y + 1e-8)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, required=True)
    parser.add_argument("--override", type=str, default=None)
    args = parser.parse_args()

    cfg = load_json(args.config)
    if args.override:
        cfg = deep_update(cfg, json.loads(args.override))

    seed = int(cfg.get("training", {}).get("seed", 123))
    set_all_seeds(seed)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # ---- Build single env and get first obs ----
    env = SequenceEnv(cfg)
    obs_np, info = env.reset(seed=seed)
    obs_shape = tuple(obs_np.shape)  # (C,H,W)
    action_dim = env.action_dim

    # ---- Policy ----
    policy = PPORecurrentPolicy(
        obs_shape=obs_shape,
        action_dim=action_dim,
        conv_channels=cfg["model"]["conv_channels"],
        lstm_hidden=int(cfg["model"]["lstm_hidden"]),
        lstm_layers=int(cfg["model"].get("lstm_layers", 1)),
        device=device,
    )

    # ---- Storage (single env) ----
    num_envs = 1  # single SequenceEnv instance
    rollout_len = int(cfg["training"]["rollout_length"])
    storage = RolloutStorage(
        rollout_length=rollout_len,
        num_envs=num_envs,
        obs_shape=obs_shape,
        hidden_size=int(cfg["model"]["lstm_hidden"]),
        num_layers=int(cfg["model"].get("lstm_layers", 1)),
        action_dim=action_dim,
        device=device,
    )

    # Initial obs into storage
    obs_t = torch.from_numpy(obs_np).unsqueeze(0).to(device, dtype=torch.float32)  # (1,C,H,W)
    storage.set_initial_obs(obs_t)

    # ---- Learner ----
    learner_cfg = PPOConfig(
        gamma=float(cfg["training"]["gamma"]),
        gae_lambda=float(cfg["training"]["gae_lambda"]),
        clip_eps=float(cfg["training"]["clip_eps"]),
        entropy_coef=float(cfg["training"]["entropy_coef"]),
        value_coef=float(cfg["training"]["value_coef"]),
        max_grad_norm=float(cfg["training"]["max_grad_norm"]),
        lr=float(cfg["training"]["lr"]),
        epochs=int(cfg["training"].get("epochs", 3)),
        minibatch_size=int(cfg["training"].get("minibatch_size", 2048)),
        amp=bool(cfg["training"].get("amp", True)),
    )
    learner = PPOLearner(policy, learner_cfg, device=device)

    # ---- Logging ----
    log = LoggingMux(cfg)
    log.hparams(
        {
            "algo": "ppo_lstm",
            "seed": seed,
            "num_envs": num_envs,
            "rollout_length": rollout_len,
            "gamma": learner_cfg.gamma,
            "gae_lambda": learner_cfg.gae_lambda,
            "clip_eps": learner_cfg.clip_eps,
            "entropy_coef": learner_cfg.entropy_coef,
            "value_coef": learner_cfg.value_coef,
            "lr": learner_cfg.lr,
            "device": str(device),
        },
        {"hparams/created": 1.0},
    )

    total_updates = int(cfg["training"]["total_updates"])

    # LSTM hidden state for the single env
    h, c = policy.get_initial_state(batch_size=num_envs)

    global_step = 0
    t0 = time.time()

    for update_idx in range(1, total_updates + 1):
        # -------- collect rollout --------
        storage.clear()
        storage.set_initial_obs(obs_t)  # ensure obs[0] matches current env state

        for t in range(rollout_len):
            legal_mask_arr = info.get("legal_mask", None)
            if legal_mask_arr is not None:
                legal_mask_np = np.asarray(legal_mask_arr, dtype=np.float32)
                legal_mask_t = torch.from_numpy(legal_mask_np).unsqueeze(0).to(device, dtype=torch.float32)
            else:
                legal_mask_t = None

            with torch.no_grad():
                out = policy.act(
                    obs=obs_t,                 # (1,C,H,W)
                    legal_mask=legal_mask_t,   # (1,A) or None
                    h0=h, c0=c,
                )
            action = int(out["action"].item())
            logp = out["log_prob"].detach().cpu()
            value = out["value"].detach().cpu()

            # keep PRE-action state for storage
            h_pre = h.clone()
            c_pre = c.clone()
            # advance hidden state to the post-action one
            h, c = out["h"], out["c"]

            # env step
            next_obs_np, reward, terminated, truncated, info = env.step(action)
            done_flag = bool(terminated or truncated)

            # tensors for storage (N=1)
            next_obs_t = torch.from_numpy(next_obs_np).unsqueeze(0).to(device, dtype=torch.float32)
            act_t = torch.tensor([action], device=device, dtype=torch.long)
            logp_t = logp.view(1)
            value_t = value.view(1)
            rew_t = torch.tensor([reward], device=device, dtype=torch.float32)
            done_t = torch.tensor([1.0 if done_flag else 0.0], device=device, dtype=torch.float32)

            storage.insert(
                obs_next=next_obs_t,
                actions=act_t,
                log_probs=logp_t,
                values=value_t,
                rewards=rew_t,
                dones=done_t,
                h_pre=h_pre,   # (L,N,H)
                c_pre=c_pre,   # (L,N,H)
                legal_mask=legal_mask_t,  # (1,A) or None
            )

            obs_t = next_obs_t
            global_step += 1

            if done_flag:
                # episode end; reset env and hidden
                obs_np, info = env.reset()
                obs_t = torch.from_numpy(obs_np).unsqueeze(0).to(device, dtype=torch.float32)
                h, c = policy.get_initial_state(batch_size=num_envs)

        # After rollout, get bootstrap value on last obs and set last hidden state
        with torch.no_grad():
            out_val = policy.forward(obs=obs_t, h0=h, c0=c)
            last_value = out_val["value"].detach()  # (1,1) or (1,)
        storage.set_last_hidden(h_last=h, c_last=c)

        # GAE + returns
        storage.compute_returns_and_advantages(
            last_value=last_value, gamma=learner_cfg.gamma, lam=learner_cfg.gae_lambda
        )

        # -------- PPO update --------
        stats = learner.update(storage)

        fps = global_step / max(1e-6, (time.time() - t0))
        ev = float("nan")
        if storage.last_values_np is not None and storage.last_returns_np is not None:
            v_pred = np.asarray(storage.last_values_np, dtype=np.float32)
            v_true = np.asarray(storage.last_returns_np, dtype=np.float32)
            if v_true.size:
                ev = explained_variance(v_pred, v_true)

        print(
            f"update {update_idx}/{total_updates} | "
            f"loss/total:{stats.get('loss/total', 0.0):.4f} | "
            f"loss/policy:{stats.get('loss/policy', 0.0):.4f} | "
            f"loss/value:{stats.get('loss/value', 0.0):.4f} | "
            f"loss/entropy:{stats.get('loss/entropy', 0.0):.4f} | "
            f"fps:{fps:.1f}"
        )

        # TB + CSV + JSONL
        log.scalars(
            "loss",
            {
                "total":   float(stats.get("loss/total", 0.0)),
                "policy":  float(stats.get("loss/policy", 0.0)),
                "value":   float(stats.get("loss/value", 0.0)),
                "entropy": float(stats.get("loss/entropy", 0.0)),
            },
            step=update_idx,
        )
        log.scalar("perf/fps", float(fps), step=update_idx)
        if not math.isnan(ev):
            log.scalar("value/explained_variance", float(ev), step=update_idx)
        if "optim/lr" in stats:
            log.scalar("optim/lr", float(stats["optim/lr"]), step=update_idx)
        log.flush()

    # Save checkpoint into the active run dir
    run_dir = log.run_dir
    os.makedirs(run_dir, exist_ok=True)
    torch.save(policy.state_dict(), os.path.join(run_dir, "policy_final.pt"))
    log.close()


if __name__ == "__main__":
    main()
