import random
from os import name
from typing import Any, Dict, List, Optional, Tuple

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
    Manages opponents for multi-seat turn-based self-play and can 'advance' an env
    by letting non-learner seats act until it's the learner's turn again.

    Key features:
      - Static factory create
        where `policies[0]` is the learner; other seats are sampled from heuristics/snapshots.
      - Per-env seat assignment (re-sampled on episode resets).
      - Per-env/per-seat RNN state for recurrent opponents.
      - `skipTo(learner_policy, env, env_idx)` steps the env forward with opponents'
        actions until it's the learner's turn (or the episode ends/reset).
    """

    def __init__(self, probabilities: Optional[Dict[str, float]] = None):
        self.probabilities = probabilities or {"current": 0.0, "snapshots": 0.7, "heuristics": 0.3}

        # Candidate sources
        self.current_policy: Optional[Any] = None
        self.snapshots: List[Any] = []
        self.heuristics: List[Any] = []

        # Per-env seat assignment: env_idx -> [agent_for_seat0, agent_for_seat1, ...]
        self._env_seats: Dict[int, List[Any]] = {}

    # ---------- Factory & registration ----------

    @staticmethod
    def create(envs: List[SequenceEnv], policy: PPORecurrentPolicy, pool_probs: Dict[str, float], heuristics, snapshots):
        # Create pool and seat all envs
        pool = OpponentPool(probabilities=pool_probs)
        pool.current_policy = policy
        for h_agent in heuristics: pool.add_heuristic(h_agent)
        for s_agent in snapshots:  pool.add_snapshot(s_agent)
        for i, e in enumerate(envs):
            pool.ensure_env(i, e)
        # Align *all envs* to the learner's turn before starting the rollout
        aligned_obs, aligned_info = [], []
        for i, e in enumerate(envs):
            obs_i, info_i, _ = pool.skipTo(policy, e, i)
            aligned_obs.append(obs_i)
            aligned_info.append(info_i)
        return aligned_info, aligned_obs, pool

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
        """Seat 0 = learner; other seats sampled from snapshots/heuristics."""
        total_players = int(env.gconf.teams) * int(env.gconf.players_per_team)
        assert total_players >= 1, "Invalid player count from env config"
        seats: List[Any] = []
        for seat in range(total_players):
            if seat == 0:
                a = self.current_policy
            else:
                a = self._sample_opponent()
                a = a.make_new_agent(env)
                if a is None:
                    raise Exception("Unable to sample opponent by calling make_new_agent on: ", a)
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


    def _sample_opponent(self) -> BaseAgent:
        """Category sample by probabilities, then pick a random agent from that bucket. The agent must be created by .make_new_agent(env) later"""
        p = self.probabilities
        bucket_type = random.choices(list(p.keys()), p.values())[0]
        sampled_agent = None

        if bucket_type == "snapshots":
            sampled_agent =random.choice(self.snapshots)
        elif bucket_type ==  "heuristics":
            sampled_agent = random.choice(self.heuristics)
        else:
            sampled_agent =  self.current_policy

        return sampled_agent

    def get_env_classes(self, env_id: int, filter: List[str] = None) -> List[str]:
        seats = self._env_seats[env_id]
        classes = []
        for seat in seats:
            name__ = str(type(seat).__name__)
            if filter is not None:
                if name__ not in filter :
                    classes.append(name__)
            else:
                classes.append(name__)
        return classes

    # ---------- Public: advance env to learner's turn ----------
    def skipTo(self, learner_policy: Any, env, env_idx: int):
        """
        Advance `env` by letting non-learner seats act until it's the learner's turn.
        Returns (obs_np, info_dict, rolled_terminal), where:
         - obs_np and info_dict are for the learner seat's *current* turn
         - rolled_terminal=True iff an episode ended during the skipping (i.e., opponent won)
        """
        self.ensure_env(env_idx, env)

        seats = self._env_seats[env_idx]
        # Find the learner seat in this env (usually 0 by construction)
        try:
            learner_seat = seats.index(learner_policy)
        except ValueError:
            learner_seat = 0  # fallback

        rolled_terminal = False

        # Loop until learner's turn (or until reset happens)
        while True:
            # If it's learner's turn, return the learner's observation+mask
            if int(env.current_player) == int(learner_seat):
                legal = env._legal_for(env.current_player)
                obs_np = env.get_obs()
                info = {"current_player": env.current_player, "legal_mask": env._legal_mask(legal)}
                return obs_np, info, rolled_terminal

            # Opponent's turn: pick that agent and make a move
            seat = int(env.current_player)
            opp_agent = seats[seat]

            # Build obs + legal mask for that seat
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

            # Step env with opponent action
            _, _, terminated, truncated, _ = env.step(int(action))

            # If episode ended, reset env and re-seat opponents; mark rolled_terminal
            if terminated or truncated:
                rolled_terminal = True
                obs0, _ = env.reset()
                self.on_env_reset(env_idx, env)
                # continue—loop will deliver the learner's first-turn obs after reset

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
          - PPORecurrentPolicy-like: Not supported if not agent and self-managed
          - PPOFrozenAgent-like: has select_action(legal_mask, ctx={...}) and manages its own h/c
          - Heuristics: select_action(legal_mask, ctx={'env', 'seat', 'obs'})
        """
        # Try raw PPORecurrentPolicy (needs external state)
        if isinstance(agent, PPORecurrentPolicy):
            raise NotImplementedError("Opponent pool cannot manage external policies managed by themselves")
        # PPOFrozenAgent or compatible wrappers
        if hasattr(agent, "select_action"):
            try:
                return int(agent.select_action(legal_mask=legal_mask, ctx={"obs": obs_np, "env": env, "seat": seat}))
            except TypeError:
                return int(agent.select_action(legal_mask, {"obs": obs_np, "env": env, "seat": seat}))

        # As a last resort, try a simple API `act(env)` or random legal
        if hasattr(agent, "act"):
            return int(agent.act(env))

        # Fallback: random legal
        legal_idxs = np.flatnonzero(legal_mask > 0.5)
        if legal_idxs.size == 0:
            return 0
        return int(np.random.choice(legal_idxs))
