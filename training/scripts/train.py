# training/scripts/train.py
from __future__ import annotations
import argparse
import json
import math
import os
import time
from typing import List

import numpy as np
import torch

from training.agents.baseline.center_heuristic_agent import CenterHeuristicAgent
from training.agents.opponent_pool import OpponentPool
from training.utils.agent_win_meter import AgentWinMeter
from ..utils.jsonio import load_json, deep_update
from ..utils.seeding import set_all_seeds, set_seeds_from_cfg
from ..utils.logging import LoggingMux
from ..envs.sequence_env import SequenceEnv
from ..algorithms.ppo_lstm.learner import PPOLearner, PPOConfig
from ..algorithms.ppo_lstm.ppo_lstm_policy import PPORecurrentPolicy
from ..algorithms.ppo_lstm.storage import RolloutStorage


def explained_variance(y_pred: np.ndarray, y_true: np.ndarray) -> float:
    var_y = float(np.var(y_true)) if len(y_true) else 0.0
    if var_y < 1e-8:
        return float("nan")
    return 1.0 - float(np.var(y_true - y_pred)) / (var_y + 1e-8)

def linear_schedule(start: float, end: float, progress: float) -> float:
    """progress in [0,1]"""
    p = min(max(progress, 0.0), 1.0)
    return start + (end - start) * p


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, required=True)
    parser.add_argument("--override", type=str, default=None)
    args = parser.parse_args()

    cfg = load_json(args.config)
    if args.override:
        cfg = deep_update(cfg, json.loads(args.override))

    seed = set_seeds_from_cfg(cfg, "training")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # ---- Build multiple envs ----
    num_envs = int(cfg["training"].get("num_envs", 1))
    assert num_envs >= 1, "num_envs must be >= 1"
    envs: List[SequenceEnv] = [SequenceEnv(cfg) for _ in range(num_envs)]

    # Reset all envs with distinct seeds
    obs_list, info_list = [], []
    for i, e in enumerate(envs):
        obs_i, info_i = e.reset(seed=seed + 1009 * i)
        obs_list.append(obs_i)
        info_list.append(info_i)

    obs_np = np.stack(obs_list, axis=0)   # (N, C, H, W) — not yet aligned to learner turn
    obs_shape = tuple(obs_list[0].shape)
    action_dim = envs[0].action_dim

    # ---- Policy (learner) ----
    policy = PPORecurrentPolicy(
        obs_shape=obs_shape,
        action_dim=action_dim,
        conv_channels=cfg["model"]["conv_channels"],
        lstm_hidden=int(cfg["model"]["lstm_hidden"]),
        lstm_layers=int(cfg["model"].get("lstm_layers", 1)),
        device=device,
        model_cfg=cfg.get("model", {}),
        value_tanh_bound=float(cfg["model"].get("value_tanh_bound", 5.0)),
        deterministic=bool(cfg.get("evaluation", {}).get("deterministic_policy", False)),
    )

    # ---- Storage (agent-turn transitions) ----
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

    # --- Optionally resume current policy from the latest run ---
    resume_flag = bool(cfg["training"].get("resume_from_latest_run", True))
    if resume_flag:
        latest_path = os.path.join(LoggingMux.get_run_dir(cfg), "policy_final.pt")
        try:
            state_dict = torch.load(latest_path, map_location=device)
            strict = bool(cfg["training"].get("resume_load_strict", False))
            policy.load_state_dict(state_dict, strict=strict)
            print(f"[resume] loaded latest runt weights into policy: {latest_path} (strict={strict})")
        except Exception as e:
            print(f"[resume] failed to load latest run weights from '{latest_path}': {e}")


    # Per-env LSTM state for learner
    h, c = policy.get_initial_state(batch_size=num_envs)  # (layers, N, H)

    # ---- Learner / PPO ----
    learner_cfg = PPOConfig(
        gamma=float(cfg["training"]["gamma"]),
        gae_lambda=float(cfg["training"]["gae_lambda"]),
        clip_eps=float(cfg["training"]["clip_eps"]),
        entropy_coef=float(cfg["training"]["entropy_coef"]),
        value_coef=float(cfg["training"]["value_coef"]),
        max_grad_norm=float(cfg["training"]["max_grad_norm"]),
        lr=float(cfg["training"]["lr"]),
        epochs=int(cfg["training"].get("epochs", 3)),
        minibatch_size=int(cfg["training"].get("minibatch_size", 256)),
        amp=bool(cfg["training"].get("amp", True)),
    )
    learner = PPOLearner(policy, learner_cfg, device=device)

    # ---- Logging ----
    log = LoggingMux(cfg)

    ent_start = float(cfg["training"].get("entropy_coef", 0.03))
    ent_end = float(cfg["training"].get("entropy_coef_final", ent_start))

    lr_start = float(cfg["training"].get("lr", 1.5e-4))
    lr_end = float(cfg["training"].get("lr_final", lr_start))

    # ---- Opponent pool + snapshots/heuristics ----
    try:
        from ..agents.opponent_pool import OpponentPool
    except Exception:
        from training.agents.opponent_pool import OpponentPool

    from ..utils.snapshot_manager import SnapshotManager
    from ..agents.ppo_frozen_agent import PPOFrozenAgent
    from ..agents.baseline.random_agent import RandomAgent
    from ..agents.baseline.blocking_agent import BlockingAgent
    from ..agents.baseline.greedy_sequence_agent import GreedySequenceAgent

    snapshot_every = int(cfg["training"].get("snapshot_every", 0))
    max_snaps = int(cfg["training"].get("max_snapshots", 0))
    pool_probs = cfg["training"].get("pool_probabilities", {"current": 0.0, "snapshots": 0.7, "heuristics": 0.3})

    # Build pool candidates
    heuristics = [GreedySequenceAgent(),CenterHeuristicAgent()]
    snapman = SnapshotManager(log.run_dir, max_keep=max_snaps) if snapshot_every > 0 else None
    snapshots = []  # preloaded snapshots (if resuming)
    if snapman is not None:
        policy_kwargs = dict(
            obs_shape=obs_shape,
            action_dim=action_dim,
            conv_channels=cfg["model"]["conv_channels"],
            lstm_hidden=int(cfg["model"]["lstm_hidden"]),
            lstm_layers=int(cfg["model"].get("lstm_layers", 1)),
            value_tanh_bound=float(cfg["model"].get("value_tanh_bound", 5.0)),
            device=str(device),
            model_cfg=cfg["model"]
        )
        for path in snapman.all():
            try:
                snapshots.append(PPOFrozenAgent(state_dict_path=path, **policy_kwargs))
            except Exception:
                pass

    aligned_info, aligned_obs, pool = OpponentPool.create(envs, policy, pool_probs, heuristics, snapshots)
    meter = AgentWinMeter()  # NEW

    obs_t = torch.from_numpy(np.stack(aligned_obs, axis=0)).to(device, dtype=torch.float32)  # (N,C,H,W)
    info_list = aligned_info
    storage.set_initial_obs(obs_t)

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
            "resume_from_latest_run": resume_flag,
        },
        {"hparams/created": 1.0},
    )

    total_updates = int(cfg["training"]["total_updates"])

    global_step = 0
    t0 = time.time()

    for update_idx in range(1, total_updates + 1):
        # progreso 0.0 en el 1er update, 1.0 en el último
        progress = (update_idx - 1) / max(1, total_updates - 1)

        # entropy y LR lineales
        learner.cfg.entropy_coef = linear_schedule(ent_start, ent_end, progress)
        lr_now = linear_schedule(lr_start, lr_end, progress)
        for g in learner.optimizer.param_groups:
            g["lr"] = lr_now

        # (opcional) loggea para verlo en TB
        log.scalar("sched/entropy_coef", float(learner.cfg.entropy_coef), step=update_idx)
        log.scalar("sched/lr", float(lr_now), step=update_idx)

        # -------- Collect rollout over *learner turns* --------
        storage.clear()
        storage.set_initial_obs(obs_t)

        for t in range(rollout_len):
            # Build legal mask batch at the learner's decision points
            legal_masks_np = []
            for info_i in info_list:
                lm = info_i.get("legal_mask", None)
                if lm is None:
                    lm = np.ones((action_dim,), dtype=np.float32)
                legal_masks_np.append(np.asarray(lm, dtype=np.float32))
            legal_masks_np = np.stack(legal_masks_np, axis=0)  # (N, A)

            with torch.no_grad():
                out = policy.select_action(
                    legal_mask=legal_masks_np,
                    ctx={"obs": obs_t, "h0": h, "c0": c},
                )

            # Actions/logp/value tensors
            actions = out["action"]
            if isinstance(actions, torch.Tensor):
                if actions.dim() > 1:
                    actions = actions.squeeze(-1)
                actions_np = actions.detach().cpu().numpy().astype(np.int64).reshape(-1)
            else:
                actions_np = np.asarray(actions, dtype=np.int64).reshape(-1)

            logp = out["log_prob"].view(-1).detach().cpu().to(device)
            value = out["value"].view(-1).detach().cpu().to(device)

            # keep PRE-action LSTM for storage; advance to learner's post-action state
            h_pre = h.clone()
            c_pre = c.clone()
            h, c = out["h"], out["c"]

            # Step each env with the learner's action, then fast-forward opponents to the next learner turn
            next_obs_list, next_info_list, reward_list, done_list = [], [], [], []
            for i, e in enumerate(envs):
                # 1) learner action
                _, rew_i, term_i, trunc_i, _ = e.step(int(actions_np[i]))
                done_after_own = bool(term_i or trunc_i)

                # reward base del learner (solo su jugada)
                reward_i = float(rew_i)

                # 2) si no terminó en nuestra jugada, avanzar con oponentes
                next_obs_i = None
                next_info_i = None
                rolled_terminal = False
                if not done_after_own:
                    next_obs_i, next_info_i, rolled_terminal = pool.skipTo(policy, e, i)

                # ¿terminó el episodio en total?
                done_i = done_after_own or bool(rolled_terminal)

                # 3) Si terminó por jugada del oponente, añade outcome al reward del learner
                if (not done_after_own) and rolled_terminal:
                    # quién ganó
                    try:
                        winners = list(e.game_engine.winner_teams())
                    except Exception:
                        winners = list(getattr(getattr(e, "game_engine", None), "state", {}).get("winners", [])) or []
                    learner_team = 0  # asiento 0 -> team 0 en tu config
                    if winners:
                        reward_i += (e.R_WIN if learner_team in winners else e.R_LOSS)

                # 4) Si el env terminó, resetea y alinea para mantener el batch lleno
                if done_i:
                    # contabiliza win/loss antes del reset
                    try:
                        winners = list(e.game_engine.winner_teams())
                    except Exception:
                        winners = list(getattr(getattr(e, "game_engine", None), "state", {}).get("winners", [])) or []
                    learner_team = 0
                    learner_won = (learner_team in winners)
                    opp_classes = pool.get_env_classes(i, filter=[str(type(policy).__name__)])
                    meter.update(opp_classes, learner_won)

                    # reset de hidden para ese env en la policy
                    hi, ci = policy.get_initial_state(batch_size=1)
                    h[:, i:i+1, :] = hi
                    c[:, i:i+1, :] = ci

                    # reset del env + re-sampleo de oponentes y realineación al siguiente turno del learner
                    e.reset()                       # nuevo episodio
                    pool.on_env_reset(i, e)         # reasignar seats para el episodio nuevo
                    aligned_obs_i, aligned_info_i, _ = pool.skipTo(policy, e, i)

                    # usar esta obs como "next_obs" del paso actual (aunque done=1)
                    next_obs_list.append(aligned_obs_i)
                    next_info_list.append(aligned_info_i)
                else:
                    # episodio sigue: usamos lo que devolvió skipTo
                    next_obs_list.append(next_obs_i)
                    next_info_list.append(next_info_i)

                reward_list.append(reward_i)
                done_list.append(1.0 if done_i else 0.0)

            # Pack tensors for storage at agent-turn sampling frequency
            obs_t = torch.from_numpy(np.stack(next_obs_list, axis=0)).to(device, dtype=torch.float32)
            act_t = torch.from_numpy(actions_np).to(device, dtype=torch.long)
            rew_t = torch.tensor(reward_list, device=device, dtype=torch.float32)
            done_t = torch.tensor(done_list, device=device, dtype=torch.float32)
            legal_mask_t = torch.from_numpy(legal_masks_np).to(device, dtype=torch.float32)

            storage.insert(
                obs_next=obs_t,
                actions=act_t,
                log_probs=logp,
                values=value,
                rewards=rew_t,
                dones=done_t,
                h_pre=h_pre,
                c_pre=c_pre,
                legal_mask=legal_mask_t,
            )

            info_list = next_info_list
            global_step += num_envs  # learner decisions taken

        # Bootstrap value on the *next learner-turn* obs and set last hidden
        with torch.no_grad():
            out_val = policy.forward(obs=obs_t, h0=h, c0=c)
            last_value = out_val["value"].detach().view(-1)
        storage.set_last_hidden(h_last=h, c_last=c)

        # GAE + returns over agent-turn horizon
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
            f"fps:{fps:.1f} | ",
            "[eval]", meter.short_text()
        )

        # TB/CSV/JSONL
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
        meter.log_to(log, update_idx)
        log.scalar("perf/fps", float(fps), step=update_idx)

        if not math.isnan(ev):
            log.scalar("value/explained_variance", float(ev), step=update_idx)
        if "optim/lr" in stats:
            log.scalar("optim/lr", float(stats["optim/lr"]), step=update_idx)
        log.flush()

        # --- Optional: periodic snapshot & add to pool ---
        if snapman is not None and snapshot_every > 0 and ((update_idx - 1) % snapshot_every) == 0:
            snap_path = snapman.save(policy, update_idx)
            try:
                # Load as a frozen opponent and add to pool
                frozen = PPOFrozenAgent(
                    obs_shape=obs_shape,
                    action_dim=action_dim,
                    conv_channels=cfg["model"]["conv_channels"],
                    lstm_hidden=int(cfg["model"]["lstm_hidden"]),
                    lstm_layers=int(cfg["model"].get("lstm_layers", 1)),
                    value_tanh_bound=float(cfg["model"].get("value_tanh_bound", 5.0)),
                    device=str(device),
                    model_cfg=cfg["model"],
                    state_dict_path=snap_path,
                )
                pool.add_snapshot(frozen, max_snapshots=snapman.max_keep)
                print(f"[snapshots] added {snap_path}")
            except Exception as e:
                print(f"[snapshots] failed to load snapshot: {e}")

    # Save final
    run_dir = log.run_dir
    os.makedirs(run_dir, exist_ok=True)
    torch.save(policy.state_dict(), os.path.join(run_dir, "policy_final.pt"))
    log.close()


if __name__ == "__main__":
    main()
