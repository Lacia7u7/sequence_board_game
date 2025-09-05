# training/envs/vectorized/subproc_vec_env.py
from __future__ import annotations
import multiprocessing as mp
import os
import pickle
import traceback
from typing import Any, Dict, List, Tuple, Optional

import numpy as np

from collections import OrderedDict

from training.algorithms.ppo_lstm.ppo_lstm_policy import PPORecurrentPolicy
# Worker-side imports (keep top-level for spawn safety)
from training.envs.sequence_env import SequenceEnv
from training.agents.opponent_pool import OpponentPool
from training.agents.baseline.greedy_sequence_agent import GreedySequenceAgent
from training.agents.baseline.center_heuristic_agent import CenterHeuristicAgent
from training.agents.baseline.blocking_agent import BlockingAgent
from training.agents.baseline.random_agent import RandomAgent
from training.agents.advanced.beam_minimax_agent import BeamMinimaxAgent
from training.agents.advanced.ensemble_heuristic_agent import EnsembleHeuristicAgent
from training.agents.advanced.fork_threat_agent import ForkThreatAgent
from training.agents.advanced.pattern_window_agent import PatternWindowAgent
from training.agents.advanced.threat_aware_minimax_agent import ThreatAwareMinimaxAgent
from training.agents.advanced.bonus_proximity_agent import BonusProximityAgent
from training.agents.ppo_frozen_agent import PPOFrozenAgent

# Commands
_CMD_RESET_BATCH   = "reset_batch"
_CMD_STEP_BATCH    = "step_batch"
_CMD_CLOSE         = "close"
_CMD_ADD_SNAPSHOT  = "add_snapshot"
_CMD_SET_HEUR_W    = "set_heuristic_weights"

def _build_heuristics() -> List[Any]:
    return [
        GreedySequenceAgent(),
        CenterHeuristicAgent(),
        BlockingAgent(),
        RandomAgent(),
        BeamMinimaxAgent(),
        EnsembleHeuristicAgent(),
        ForkThreatAgent(),
        PatternWindowAgent(),
        ThreatAwareMinimaxAgent(),
        BonusProximityAgent(),
    ]

class _SnapshotRegistry:
    """Per-worker LRU cache of loaded PPOFrozenAgent models (CPU)."""
    def __init__(self, capacity: int = 2):
        self.capacity = max(1, int(capacity))
        self._cache = OrderedDict()  # path -> PPOFrozenAgent

    def get(self, path: str, policy_kwargs: dict):
        # Return cached agent if available
        agent = self._cache.get(path)
        if agent is not None:
            self._cache.move_to_end(path)
            return agent
        # Load on demand
        agent = PPOFrozenAgent(state_dict_path=path, **policy_kwargs)
        self._cache[path] = agent
        # Evict LRU if over capacity
        while len(self._cache) > self.capacity:
            self._cache.popitem(last=False)
        return agent


class _SnapshotFactory:
    """
    Lightweight proxy that creates/returns a cached PPOFrozenAgent from a registry.
    Provides .make_new_agent(env) so OpponentPool uses it transparently.
    """
    def __init__(self, path: str, registry: _SnapshotRegistry, policy_kwargs: dict):
        self.state_dict_path = path
        self._registry = registry
        self.policy_kwargs = dict(policy_kwargs)
        self.name = f"Snapshot[{os.path.basename(path)}]"

    def make_new_agent(self, env=None):
        # Let FileNotFoundError bubble up; OpponentPool will handle removal/resample.
        return self._registry.get(self.state_dict_path, self.policy_kwargs)


def _worker_loop(
    pipe,
    cfg_bytes: bytes,
    seeds: List[int],
    obs_shape_from_parent: Optional[Tuple[int, ...]],
    pool_probs: Dict[str, float],
    pool_desc: Optional[Dict[str, Any]],
    snapshot_paths: List[str],
) -> None:
    """
    Worker owns multiple envs and a single OpponentPool across them.
    Sends a startup handshake ("ready" | "fatal:<traceback>") and then
    accepts batched RESET/STEP commands.
    """
    try:
        # Keep workers single-threaded to avoid CPU oversubscription
        try:
            import torch
            torch.set_num_threads(1)
        except Exception:
            pass
        os.environ.setdefault("OMP_NUM_THREADS", "1")
        os.environ.setdefault("MKL_NUM_THREADS", "1")

        cfg = pickle.loads(cfg_bytes)

        max_cache = int(cfg.get("training", {}).get(
            "max_snapshot_cache",
            max(1, int(cfg.get("training", {}).get("max_snapshots", 2)))
        ))

        policy_kwargs = dict(
            obs_shape=obs_shape_from_parent,
            action_dim=envs[0].action_dim,
            conv_channels=cfg["model"]["conv_channels"],
            lstm_hidden=int(cfg["model"]["lstm_hidden"]),
            lstm_layers=int(cfg["model"].get("lstm_layers", 1)),
            value_tanh_bound=float(cfg["model"].get("value_tanh_bound", 5.0)),
            device="cpu",
            model_cfg=cfg["model"],
        )

        _registry = _SnapshotRegistry(capacity=max_cache)

        num_local = len(seeds)
        if num_local <= 0:
            pipe.send(("fatal", "Worker started with zero local envs"))
            return

        # 1) Build envs and RESET them immediately (avoids NoneType internals)
        envs: List[SequenceEnv] = [SequenceEnv(cfg) for _ in range(num_local)]
        for li, s in enumerate(seeds):
            envs[li].reset(seed=int(s))

        # 2) Build heuristics and preload snapshots (CPU)
        heuristics = _build_heuristics()
        snapshots = []
        for p in (snapshot_paths or []):
            try:
                snapshots.append(_SnapshotFactory(p, _registry, policy_kwargs))
            except Exception:
                pass

        # 3) Create OpponentPool across these envs
        # Use positional args to match project signature:
        # OpponentPool.create(envs, policy, pool_probabilities, heuristics, snapshots, pool_probabilities_description)
        policy = None  # IMPORTANT: 'current' prob must be 0.0 when policy is None
        aligned_info_list, aligned_obs_list, pool = OpponentPool.create(
            envs,
            policy,
            pool_probs,
            heuristics,
            snapshots,
            pool_desc,
        )

        # 4) Fast-align each env to next learner turn
        def _to_np(obs):
            return np.asarray(obs, dtype=np.float32)

        cur_obs_list: List[np.ndarray] = []
        cur_info_list: List[Dict[str, Any]] = []
        for li in range(num_local):
            # we already reset above; ensure pool tracks the reset
            pool.on_env_reset(li, envs[li])
            next_obs, next_info, _ = pool.skipTo(policy, envs[li], li)
            arr = _to_np(next_obs)
            inf = dict(next_info)
            if inf.get("legal_mask", None) is None:
                inf["legal_mask"] = np.ones((envs[li].action_dim,), np.float32)
            cur_obs_list.append(arr)
            cur_info_list.append(inf)

        # Startup handshake (report shapes, etc.)
        try:
            pipe.send((
                "ready",
                {
                    "obs_shape": tuple(cur_obs_list[0].shape),
                    "action_dim": int(envs[0].action_dim),
                    "num_local_envs": num_local,
                },
            ))
        except Exception:
            return

        # =================== Main loop ===================
        while True:
            cmd, payload = pipe.recv()

            if cmd == _CMD_RESET_BATCH:
                try:
                    local_idxs = payload.get("local_idxs", [])
                    seeds_payload = payload.get("seeds", [None] * len(local_idxs))
                    out_obs, out_infos = [], []
                    for li, s in zip(local_idxs, seeds_payload):
                        seed_val = int(s) if s is not None else None
                        envs[li].reset(seed=seed_val)
                        pool.on_env_reset(li, envs[li])
                        next_obs, next_info, _ = pool.skipTo(policy, envs[li], li)
                        cur_obs_list[li] = _to_np(next_obs)
                        inf = dict(next_info)
                        if inf.get("legal_mask", None) is None:
                            inf["legal_mask"] = np.ones((envs[li].action_dim,), np.float32)
                        cur_info_list[li] = inf
                        out_obs.append(cur_obs_list[li])
                        out_infos.append(cur_info_list[li])
                    pipe.send((out_obs, out_infos))
                except Exception:
                    pipe.send(("error", traceback.format_exc()))
                continue

            if cmd == _CMD_STEP_BATCH:
                try:
                    local_idxs = payload.get("local_idxs", [])
                    actions = payload.get("actions", [])
                    assert len(local_idxs) == len(actions)
                    out_obs, out_infos, out_rewards, out_dones, out_extras = [], [], [], [], []
                    for li, act in zip(local_idxs, actions):
                        # 1) learner action
                        _, rew, term, trunc, _ = envs[li].step(int(act))
                        done_after_own = bool(term or trunc)
                        reward = float(rew)
                        episode_done = False
                        learner_won = False
                        opp_classes: List[str] = []

                        # 2) opponents fast-forward
                        if not done_after_own:
                            next_obs, next_info, rolled_terminal = pool.skipTo(policy, envs[li], li)
                            episode_done = bool(rolled_terminal)
                        else:
                            episode_done = True

                        # 3) if opponent rolled to terminal, add terminal reward
                        if (not done_after_own) and episode_done:
                            try:
                                winners = list(envs[li].game_engine.winner_teams())
                            except Exception:
                                winners = list(getattr(getattr(envs[li], "game_engine", None), "state", {}).get("winners", [])) or []
                            learner_team = 0
                            reward += (envs[li].R_WIN if (winners and learner_team in winners) else envs[li].R_LOSS)

                        # 4) if done, compute outcome and reset+realign
                        if episode_done:
                            try:
                                winners = list(envs[li].game_engine.winner_teams())
                            except Exception:
                                winners = list(getattr(getattr(envs[li], "game_engine", None), "state", {}).get("winners", [])) or []
                            learner_team = 0
                            learner_won = bool(winners and (learner_team in winners))
                            try:
                                opp_classes = pool.get_env_classes(li, filter=[PPORecurrentPolicy.__name__])
                            except Exception:
                                opp_classes = []
                            envs[li].reset()
                            pool.on_env_reset(li, envs[li])
                            next_obs, next_info, _ = pool.skipTo(policy, envs[li], li)

                        cur_obs_list[li] = _to_np(next_obs)
                        inf = dict(next_info)
                        if inf.get("legal_mask", None) is None:
                            inf["legal_mask"] = np.ones((envs[li].action_dim,), np.float32)
                        cur_info_list[li] = inf

                        out_obs.append(cur_obs_list[li])
                        out_infos.append(cur_info_list[li])
                        out_rewards.append(float(reward))
                        out_dones.append(float(episode_done))
                        out_extras.append({
                            "episode_done": bool(episode_done),
                            "learner_won": bool(learner_won),
                            "opp_classes": list(opp_classes),
                        })

                    pipe.send((out_obs, out_infos, out_rewards, out_dones, out_extras))
                except Exception:
                    pipe.send(("error", traceback.format_exc()))
                continue

            if cmd == _CMD_ADD_SNAPSHOT:
                try:
                    path = str(payload["path"])
                    factory = _SnapshotFactory(path, _registry, policy_kwargs)
                    pool.add_snapshot(factory, max_snapshots=int(payload.get("max_keep", 0) or 0))
                    pipe.send(True)
                except Exception:
                    pipe.send(("error", traceback.format_exc()))
                continue

            if cmd == _CMD_SET_HEUR_W:
                try:
                    w = payload["weights"]
                    pool.set_heuristics_weights(w, normalize=True)
                    pipe.send(True)
                except Exception:
                    pipe.send(("error", traceback.format_exc()))
                continue

            if cmd == _CMD_CLOSE:
                try:
                    pipe.send(True)
                except Exception:
                    pass
                break

    except Exception:
        # Fatal during startup: send full traceback
        try:
            pipe.send(("fatal", traceback.format_exc()))
        except Exception:
            pass


class SubprocVecEnv:
    """
    Distributes `num_envs` across `num_workers` processes.
    API (global order):
      - reset() -> (obs_np[N,...], info_list[N])
      - step(actions_np[N]) -> (obs_np, info_list, rewards_np[N], dones_np[N], extras_list[N])
      - add_snapshot(path), set_heuristic_weights(weights), close()
    """
    def __init__(
        self,
        num_envs: int,
        cfg: Dict[str, Any],
        base_seed: int,
        obs_shape: Optional[Tuple[int, ...]],
        pool_probabilities: Dict[str, float],
        pool_probabilities_description: Optional[Dict[str, Any]] = None,
        snapshot_paths: Optional[List[str]] = None,
        num_workers: Optional[int] = None,
    ) -> None:
        assert num_envs >= 1
        self.num_envs = int(num_envs)
        max_cpus = os.cpu_count() or 1
        if num_workers is None:
            num_workers = min(self.num_envs, max_cpus)
        else:
            num_workers = max(1, min(int(num_workers), self.num_envs))

        self.num_workers = int(num_workers)
        self._ctx = mp.get_context("spawn")  # Windows/Colab safe
        self._parents: List[Any] = []
        self._procs: List[mp.Process] = []
        self._obs_shape = tuple(obs_shape) if obs_shape is not None else None  # filled after first reset

        cfg_bytes = pickle.dumps(cfg, protocol=pickle.HIGHEST_PROTOCOL)

        # distribute num_envs into num_workers
        per_worker_base = self.num_envs // self.num_workers
        remainder = self.num_envs % self.num_workers
        self._worker_local_counts: List[int] = []
        for w in range(self.num_workers):
            n_local = per_worker_base + (1 if w < remainder else 0)
            self._worker_local_counts.append(n_local)

        # Build maps
        self._global_to_worker: List[Tuple[int, int]] = []
        self._reverse_map: Dict[Tuple[int, int], int] = {}
        gid = 0
        for w, n_local in enumerate(self._worker_local_counts):
            for li in range(n_local):
                self._global_to_worker.append((w, li))
                self._reverse_map[(w, li)] = gid
                gid += 1
        assert len(self._global_to_worker) == self.num_envs

        # Start workers
        gid = 0
        for w, n_local in enumerate(self._worker_local_counts):
            parent, child = self._ctx.Pipe(duplex=True)
            seed_list = [int(base_seed + 1009 * (gid + i)) for i in range(n_local)]
            gid += n_local
            proc = self._ctx.Process(
                target=_worker_loop,
                args=(
                    child,
                    cfg_bytes,
                    seed_list,
                    self._obs_shape,
                    pool_probabilities,
                    pool_probabilities_description,
                    snapshot_paths or [],
                ),
                daemon=True,
            )
            proc.start()
            child.close()
            self._parents.append(parent)
            self._procs.append(proc)

        # Confirm init
        self._worker_meta: List[Dict[str, Any]] = []
        for w, parent in enumerate(self._parents):
            msg = parent.recv()  # "ready" or "fatal"
            if isinstance(msg, tuple) and msg and msg[0] == "ready":
                self._worker_meta.append(msg[1])
            elif isinstance(msg, tuple) and msg and msg[0] == "fatal":
                raise RuntimeError(f"Worker {w} failed to start:\n{msg[1]}")
            else:
                raise RuntimeError(f"Worker {w} sent unexpected init message: {msg!r}")

        # Check shape consistency (optional)
        if self._worker_meta:
            obs_shape0 = tuple(self._worker_meta[0]["obs_shape"])
            action_dim0 = int(self._worker_meta[0]["action_dim"])
            for m in self._worker_meta[1:]:
                if tuple(m["obs_shape"]) != obs_shape0:
                    raise RuntimeError("Workers report mismatched obs shapes")
                if int(m["action_dim"]) != action_dim0:
                    raise RuntimeError("Workers report mismatched action dims")

        # First reset to fetch obs/infos
        obs_np, info_list = self.reset()
        self._obs_shape = tuple(obs_np.shape[1:])
        lm0 = info_list[0].get("legal_mask", np.ones((1,), np.float32))
        self._action_dim = int(np.asarray(lm0).shape[0])
        self._last_obs = obs_np
        self._last_infos = info_list

    @property
    def obs_shape(self) -> Tuple[int, ...]:
        return self._obs_shape  # type: ignore[return-value]

    @property
    def action_dim(self) -> int:
        return self._action_dim

    def reset(self) -> Tuple[np.ndarray, List[Dict[str, Any]]]:
        per_worker_idxs: List[List[int]] = [[] for _ in range(self.num_workers)]
        per_worker_seeds: List[List[Optional[int]]] = [[] for _ in range(self.num_workers)]
        for g in range(self.num_envs):
            w, li = self._global_to_worker[g]
            per_worker_idxs[w].append(li)
            per_worker_seeds[w].append(None)

        for w in range(self.num_workers):
            if per_worker_idxs[w]:
                self._parents[w].send((_CMD_RESET_BATCH, {
                    "local_idxs": per_worker_idxs[w],
                    "seeds": per_worker_seeds[w],
                }))

        obs_list: List[Optional[np.ndarray]] = [None] * self.num_envs
        info_list: List[Optional[Dict[str, Any]]] = [None] * self.num_envs
        for w in range(self.num_workers):
            if not per_worker_idxs[w]:
                continue
            msg = self._parents[w].recv()
            if isinstance(msg, tuple) and msg and msg[0] in ("error", "fatal"):
                raise RuntimeError(f"Worker {w} reset error:\n{msg[1]}")
            out_obs, out_infos = msg
            for local_idx, o, inf in zip(per_worker_idxs[w], out_obs, out_infos):
                g = self._reverse_map[(w, local_idx)]
                obs_list[g] = np.asarray(o, dtype=np.float32)
                info_list[g] = inf

        obs_np = np.stack(obs_list, axis=0)  # type: ignore[arg-type]
        info_list_typed: List[Dict[str, Any]] = info_list  # type: ignore[assignment]
        self._last_obs = obs_np
        self._last_infos = info_list_typed
        return obs_np, info_list_typed

    def step(self, actions: np.ndarray):
        actions = np.asarray(actions).astype(np.int64).reshape(-1)
        assert actions.shape[0] == self.num_envs

        per_worker_local_idxs: List[List[int]] = [[] for _ in range(self.num_workers)]
        per_worker_actions: List[List[int]] = [[] for _ in range(self.num_workers)]
        for g, act in enumerate(actions):
            w, li = self._global_to_worker[g]
            per_worker_local_idxs[w].append(li)
            per_worker_actions[w].append(int(act))

        for w in range(self.num_workers):
            if per_worker_local_idxs[w]:
                self._parents[w].send((_CMD_STEP_BATCH, {
                    "local_idxs": per_worker_local_idxs[w],
                    "actions": per_worker_actions[w],
                }))

        obs_list: List[Optional[np.ndarray]] = [None] * self.num_envs
        info_list: List[Optional[Dict[str, Any]]] = [None] * self.num_envs
        reward_list: List[float] = [0.0] * self.num_envs
        done_list: List[float] = [0.0] * self.num_envs
        extras_list: List[Optional[Dict[str, Any]]] = [None] * self.num_envs

        for w in range(self.num_workers):
            if not per_worker_local_idxs[w]:
                continue
            msg = self._parents[w].recv()
            if isinstance(msg, tuple) and msg and msg[0] in ("error", "fatal"):
                raise RuntimeError(f"Worker {w} step error:\n{msg[1]}")
            out_obs, out_infos, out_rewards, out_dones, out_extras = msg
            for local_idx, o, inf, r, d, ex in zip(
                per_worker_local_idxs[w], out_obs, out_infos, out_rewards, out_dones, out_extras
            ):
                g = self._reverse_map[(w, local_idx)]
                obs_list[g] = np.asarray(o, dtype=np.float32)
                info_list[g] = inf
                reward_list[g] = float(r)
                done_list[g] = float(d)
                extras_list[g] = ex

        obs_np = np.stack(obs_list, axis=0)  # type: ignore[arg-type]
        rew_np = np.asarray(reward_list, dtype=np.float32)
        done_np = np.asarray(done_list, dtype=np.float32)
        info_list_typed: List[Dict[str, Any]] = info_list  # type: ignore[assignment]
        extras_list_typed: List[Dict[str, Any]] = [ex or {} for ex in extras_list]

        self._last_obs = obs_np
        self._last_infos = info_list_typed
        return obs_np, info_list_typed, rew_np, done_np, extras_list_typed

    def add_snapshot(self, path: str, max_keep: int = 0) -> None:
        for p in self._parents:
            p.send((_CMD_ADD_SNAPSHOT, {"path": path, "max_keep": int(max_keep)}))
        for p in self._parents:
            _ = p.recv()

    def set_heuristic_weights(self, weights: Dict[str, float]) -> None:
        for p in self._parents:
            p.send((_CMD_SET_HEUR_W, {"weights": dict(weights)}))
        for p in self._parents:
            _ = p.recv()

    def close(self):
        for p in self._parents:
            try:
                p.send((_CMD_CLOSE, {}))
            except Exception:
                pass
        for p in self._parents:
            try:
                _ = p.recv()
            except Exception:
                pass
        for proc in self._procs:
            try:
                proc.join(timeout=1.0)
            except Exception:
                pass
