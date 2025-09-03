import random
from typing import Any, Dict, List, Optional

import numpy as np

from .base_agent import BaseAgent
from ..envs.sequence_env import SequenceEnv

# Optional types; we only use duck-typing but these help IDEs.
try:
    from ..algorithms.ppo_lstm.ppo_lstm_policy import PPORecurrentPolicy
except Exception:
    PPORecurrentPolicy = object  # fallback if not available


class OpponentPool:
    """
    Opponent pool that can sample from three buckets: {"current", "snapshots", "heuristics"}.

    Configuration
    -------------
    probabilities: Dict[str, float]
        Top-level bucket mixing weights (must sum to 1 after normalization).
        Example: {"current": 0.0, "snapshots": 0.7, "heuristics": 0.3}

    probabilities_description: Dict[str, Any]
        Optional fine-grained weights that may override defaults:
          - "buckets": {"current": x, "snapshots": y, "heuristics": z}  # overrides top-level split
          - "heuristics": {"AgentClassA": 0.5, "AgentClassB": 0.5, ...}
          - "snapshots": {"SnapshotNameA": 0.7, "SnapshotNameB": 0.3, ...}
        Unspecified classes/names inside a bucket share remaining mass uniformly.

    Notes
    -----
    - A "snapshot" agent is expected to be a frozen policy wrapper (e.g., PPOFrozenAgent);
      we derive its sampling name from `agent.name` if present, else its class name.
    - A "heuristic" agent can be any heuristic implementing `select_action(...)`.
    """

    def __init__(
        self,
        probabilities: Optional[Dict[str, float]] = None,
        probabilities_description: Optional[Dict[str, Dict[str, float]]] = None,
    ):
        self.probabilities = probabilities or {"current": 0.0, "snapshots": 0.7, "heuristics": 0.3}
        # Fine-grained weights per bucket
        self.probabilities_description = probabilities_description or {
            "heuristics": {},
            # "snapshots": {},
            # "buckets": {},
        }

        self.current_policy: Optional[Any] = None
        self.snapshots: List[Any] = []
        self.heuristics: List[Any] = []
        self._env_seats: Dict[int, List[Any]] = {}

    # ---------- Static factory used by train.py ----------

    @staticmethod
    def create(
        envs: List[SequenceEnv],
        policy: "PPORecurrentPolicy",
        pool_probs: Dict[str, float],
        heuristics,
        snapshots,
        pool_probs_desc: Optional[Dict[str, Dict[str, float]]] = None,
    ):
        pool = OpponentPool(probabilities=pool_probs, probabilities_description=pool_probs_desc)
        pool.current_policy = policy
        for h_agent in heuristics:
            pool.add_heuristic(h_agent)
        for s_agent in snapshots:
            pool.add_snapshot(s_agent)
        for i, e in enumerate(envs):
            pool.ensure_env(i, e)

        aligned_obs, aligned_info = [], []
        for i, e in enumerate(envs):
            obs_i, info_i, _ = pool.skipTo(policy, e, i)
            aligned_obs.append(obs_i)
            aligned_info.append(info_i)
        return aligned_info, aligned_obs, pool

    # ---------- Manage pools ----------

    def add_snapshot(self, agent: Any, max_snapshots: int = 15) -> int:
        self.snapshots.append(agent)
        if len(self.snapshots) > max_snapshots:
            # keep the last max_snapshots items (including the newly appended)
            self.snapshots = self.snapshots[-max_snapshots:]
        return len(self.snapshots) - 1

    def add_heuristic(self, agent: Any) -> int:
        self.heuristics.append(agent)
        return len(self.heuristics) - 1

    # ---------- Seat management ----------

    def ensure_env(self, env_idx: int, env) -> None:
        """Ensure seats are assigned for this env index."""
        if env_idx not in self._env_seats:
            self._assign_seats_for_env(env_idx, env)

    def on_env_reset(self, env_idx: int, env) -> None:
        """Re-sample opponents (non-learner seats) for a fresh episode."""
        self._assign_seats_for_env(env_idx, env)

    def _assign_seats_for_env(self, env_idx: int, env) -> None:
        total_players = int(env.gconf.teams) * int(env.gconf.players_per_team)
        assert total_players >= 1
        seats: List[Any] = []
        for seat in range(total_players):
            if seat == 0:
                a = self.current_policy
            else:
                base = self._sample_opponent()
                factory = getattr(base, "make_new_agent", None)
                a = factory(env) if callable(factory) else base  # use as-is if no factory
                if a is None:
                    raise Exception("Unable to create opponent from:", base)
            seats.append(a)

        self._env_seats[env_idx] = seats
        for s, agent in enumerate(seats):
            if agent is self.current_policy:
                continue
            if hasattr(agent, "reset"):
                try:
                    agent.reset(env, seat=s)
                except Exception:
                    try:
                        agent.reset(env)
                    except Exception:
                        print("Failed to reset agent:", agent)

    # ---------- Weighted sampling ----------

    def _agent_name(self, agent: Any) -> str:
        return getattr(agent, "name", type(agent).__name__)

    def _parse_dict_or_pairs(self, raw) -> Dict[str, float]:
        """Supports dict or list-of-dicts / (k, v) pairs."""
        if isinstance(raw, dict):
            return {str(k): float(v) for k, v in raw.items()}
        if isinstance(raw, list):
            acc: Dict[str, float] = {}
            for item in raw:
                if isinstance(item, dict):
                    for k, v in item.items():
                        acc[str(k)] = float(v)
                elif isinstance(item, (list, tuple)) and len(item) == 2:
                    acc[str(item[0])] = float(item[1])
            return acc
        return {}

    # ---- Top-level bucket choice (supports override via probabilities_description["buckets"]) ----
    def _bucket_choice(self) -> str:
        p = dict(self.probabilities or {})
        bdesc = (self.probabilities_description or {}).get("buckets", None)
        if bdesc:
            try:
                bdesc = self._parse_dict_or_pairs(bdesc)
                p.update({k: float(v) for k, v in bdesc.items()})
            except Exception:
                pass
        # Normalize; if degenerate, fall back to defaults
        s = sum(max(0.0, float(x)) for x in p.values())
        if s > 0:
            p = {k: max(0.0, float(v)) / s for k, v in p.items()}
        else:
            p = {"current": 0.0, "snapshots": 0.7, "heuristics": 0.3}
        return random.choices(list(p.keys()), list(p.values()))[0]

    # ---- Inside “heuristics” bucket ----
    def _parse_heuristics_desc(self) -> Dict[str, float]:
        raw = (self.probabilities_description or {}).get("heuristics", {})
        return self._parse_dict_or_pairs(raw)

    def _heuristics_weights(self) -> List[float]:
        """Return weights aligned with self.heuristics, distributing leftover uniformly."""
        if not self.heuristics:
            return []
        names = [self._agent_name(a) for a in self.heuristics]
        specified = self._parse_heuristics_desc()

        sum_spec = sum(max(0.0, float(specified.get(n, 0.0))) for n in names)
        remaining = max(0.0, 1.0 - sum_spec)

        unspecified_indices = [i for i, n in enumerate(names) if specified.get(n) is None]
        per_unspecified = (remaining / len(unspecified_indices)) if unspecified_indices else 0.0

        weights: List[float] = []
        for i, n in enumerate(names):
            w = specified.get(n, None)
            if w is None:
                w = per_unspecified
            weights.append(max(0.0, float(w)))
        return weights

    def _weighted_choice_heuristic(self) -> Optional[Any]:
        if not self.heuristics:
            return None
        weights = self._heuristics_weights()
        if sum(weights) <= 0.0:
            weights = None  # uniform
        return random.choices(self.heuristics, weights=weights, k=1)[0]

    # ---- Inside “snapshots” bucket ----
    def _snapshots_weights(self) -> List[float]:
        """Optional per-snapshot weights via probabilities_description['snapshots']."""
        if not self.snapshots:
            return []
        names = [self._agent_name(a) for a in self.snapshots]
        raw = (self.probabilities_description or {}).get("snapshots", {})
        specified = self._parse_dict_or_pairs(raw)

        sum_spec = sum(max(0.0, float(specified.get(n, 0.0))) for n in names)
        remaining = max(0.0, 1.0 - sum_spec)
        unspecified_idx = [i for i, n in enumerate(names) if specified.get(n) is None]
        per_unspec = (remaining / len(unspecified_idx)) if unspecified_idx else 0.0

        weights: List[float] = []
        for i, n in enumerate(names):
            w = specified.get(n, None)
            if w is None:
                w = per_unspec
            weights.append(max(0.0, float(w)))
        return weights

    def _weighted_choice_snapshot(self) -> Optional[Any]:
        if not self.snapshots:
            return None
        w = self._snapshots_weights()
        if sum(w) <= 0.0:
            w = None  # uniform
        return random.choices(self.snapshots, weights=w, k=1)[0]

    # ---- Final sampler combining all buckets ----
    def _sample_opponent(self) -> BaseAgent:
        bucket_type = self._bucket_choice()

        if bucket_type == "snapshots":
            sampled = self._weighted_choice_snapshot()
            if sampled is not None:
                return sampled
            # fallback if no snapshots/weights
            if self.heuristics:
                return self._weighted_choice_heuristic() or random.choice(self.heuristics)
            return self.current_policy or (random.choice(self.heuristics) if self.heuristics else None)

        if bucket_type == "heuristics":
            sampled = self._weighted_choice_heuristic()
            if sampled is not None:
                return sampled
            # fallback if no heuristics/weights
            if self.snapshots:
                return self._weighted_choice_snapshot() or random.choice(self.snapshots)
            return self.current_policy or (random.choice(self.snapshots) if self.snapshots else None)

        # bucket_type == 'current' (or unknown)
        return self.current_policy or (
            self._weighted_choice_heuristic() or
            (random.choice(self.heuristics) if self.heuristics else None)
        )

    # ---------- Public helpers used by training ----------

    def get_env_classes(self, env_id: int, filter: List[str] = None) -> List[str]:
        """Return class names of seats (optionally excluding names listed in `filter`)."""
        seats = self._env_seats[env_id]
        classes = []
        for seat in seats:
            name__ = str(type(seat).__name__)
            if filter is not None:
                if name__ not in filter:
                    classes.append(name__)
            else:
                classes.append(name__)
        return classes

    def skipTo(self, learner_policy: Any, env, env_idx: int):
        """
        Advance `env` by letting non-learner seats play until it's the learner's turn.
        Returns (obs_np, info_dict, rolled_terminal):
          - obs_np, info_dict: observation and legal mask at the learner's turn
          - rolled_terminal: True if the episode terminated during skipping
        """
        self.ensure_env(env_idx, env)

        seats = self._env_seats[env_idx]
        try:
            learner_seat = seats.index(learner_policy)
        except ValueError:
            learner_seat = 0

        def _is_done() -> bool:
            try:
                return bool(env.game_engine.is_terminal())
            except Exception:
                try:
                    return bool(env.game_engine.winner_teams())
                except Exception:
                    winners = getattr(getattr(env, "game_engine", None), "state", None)
                    winners = getattr(winners, "winners", []) if winners is not None else []
                    return bool(winners)

        rolled_terminal = False

        # Already terminal before skipping: nothing rolled during skip
        if _is_done():
            return None, None, False

        MAX_HOPS = 512  # safety against loops

        for _ in range(MAX_HOPS):
            # Is it the learner's turn now?
            if int(env.current_player) == int(learner_seat):
                legal = env._legal_for(env.current_player)
                obs_np = env.get_obs()  # perspective of current_player
                info = {
                    "current_player": env.current_player,
                    "legal_mask": env._legal_mask(legal),
                }
                return obs_np, info, rolled_terminal

            # Opponent's turn
            seat = int(env.current_player)
            opp_agent = seats[seat]

            obs_np = env.get_obs()
            legal = env._legal_for(seat)
            legal_mask = env._legal_mask(legal)

            action = self._select_action_for_agent(
                agent=opp_agent,
                env=env,
                env_idx=env_idx,
                seat=seat,
                obs_np=obs_np,
                legal_mask=legal_mask,
            )

            # Execute action
            _, _, terminated, truncated, _ = env.step(int(action), fast_run=True)

            if terminated or truncated:
                rolled_terminal = True
                return None, None, True

            # otherwise, continue loop to next current_player

        # If we got here, likely a loop due to illegal moves/bug
        raise RuntimeError("skipTo: exceeded MAX_HOPS; possible loop of illegal actions")

    # ---------- Agent action adapter ----------

    def _select_action_for_agent(
        self,
        agent: Any,
        env,
        env_idx: int,
        seat: int,
        obs_np: np.ndarray,
        legal_mask: np.ndarray,
    ) -> int:
        """
        Call into different agent types:
          - PPORecurrentPolicy-like: not supported here (requires external LSTM state)
          - PPOFrozenAgent-like: select_action(legal_mask, ctx={...})
          - Heuristics: select_action(legal_mask, ctx={'env','seat','obs'})
        """
        if isinstance(agent, PPORecurrentPolicy):
            raise NotImplementedError("OpponentPool cannot manage live PPORecurrentPolicy opponents")

        if hasattr(agent, "select_action"):
            try:
                return int(agent.select_action(legal_mask=legal_mask, ctx={"obs": obs_np, "env": env, "seat": seat}))
            except TypeError:
                # some heuristics may have positional signature
                return int(agent.select_action(legal_mask, {"obs": obs_np, "env": env, "seat": seat}))

        if hasattr(agent, "act"):  # very simple API
            return int(agent.act(env))

        # Fallback: random legal
        legal_idxs = np.flatnonzero(legal_mask > 0.5)
        if legal_idxs.size == 0:
            return 0
        return int(np.random.choice(legal_idxs))

    # ----- Runtime curriculum control -----

    def set_heuristics_weights(self, weights: Dict[str, float], normalize: bool = True) -> None:
        """
        Update fine-grained weights inside the 'heuristics' bucket.
        'weights' keys should be agent class names (or .name if provided).
        Unspecified agents will share the remaining mass uniformly (per our sampler).
        """
        w = {str(k): float(v) for k, v in (weights or {}).items()}
        if normalize:
            s = sum(max(0.0, x) for x in w.values())
            if s > 0:
                for k in list(w.keys()):
                    w[k] = max(0.0, w[k]) / s
        if self.probabilities_description is None:
            self.probabilities_description = {}
        self.probabilities_description["heuristics"] = w

    def current_heuristics_weights(self) -> Dict[str, float]:
        """
        Returns the explicit per-class weights (not including the leftover that will be
        distributed uniformly among unspecified agents).
        """
        hdesc = (self.probabilities_description or {}).get("heuristics", {})
        if isinstance(hdesc, dict):
            return {str(k): float(v) for k, v in hdesc.items()}
        return {}
