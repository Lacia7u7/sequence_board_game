# training/scripts/eval.py
from __future__ import annotations
import argparse, json, importlib
from typing import Dict, Any, List, Tuple, Optional

from training.utils.seeding import set_seeds_from_cfg
from ..envs.sequence_env import SequenceEnv
from ..agents.selfplay_manager import SelfPlayManager
from ..agents.base_agent import BaseAgent

def load_ratings_backends():
    try:
        # was: training.agents.ratings  (plural)
        return importlib.import_module("training.agents.rating")
    except Exception:
        return None


def load_agent(path: str, env, kwargs: Dict[str, Any]) -> BaseAgent:
    mod_path = "training." + path.replace(".py", "").replace("/", ".")
    mod = importlib.import_module(mod_path)
    if hasattr(mod, "make_agent"):
        return mod.make_agent(env=env, **kwargs)
    for cls_name in ("Agent","BlockingAgent","GreedySequenceAgent","RandomAgent","PPOLstmAgent","HumanAgent", "CenterHeuristicAgent"):
        if hasattr(mod, cls_name):
            cls = getattr(mod, cls_name)
            try:
                return cls(env=env, **kwargs)
            except TypeError:
                return cls(**kwargs)
    raise RuntimeError(f"Could not create agent from {path}")

def play_match(env: SequenceEnv, a: BaseAgent, b: BaseAgent, seed: Optional[int]) -> Tuple[bool, bool, List[int]]:
    mgr = SelfPlayManager([a, b], env, max_steps=int(env._step_limit))
    out = mgr.play_episode(seed=seed, render=False)
    return out["terminated"], out["truncated"], out.get("winners", [])

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True)
    ap.add_argument("--episodes", type=int, default=200)
    args = ap.parse_args()

    cfg = json.load(open(args.config, "r"))
    eval_cfg = dict(cfg.get("evaluation", {}))

    # These flags are used by ratings backends below
    use_elo = bool(eval_cfg.get("elo", True))
    use_ts  = bool(eval_cfg.get("trueskill", True))

    env = SequenceEnv(cfg)
    seed = set_seeds_from_cfg(cfg, "evaluation")

    obs,info = env.reset(seed=seed)

    bench_paths: List[str] = list(eval_cfg.get("benchmark_agents", []))
    eval_paths:  List[str] = list(eval_cfg.get("evaluated_agents", []))
    agent_kwargs_all: Dict[str, Dict[str, Any]] = dict(eval_cfg.get("agent_kwargs", {}))

    ratings_mod = load_ratings_backends()
    EloCls = getattr(ratings_mod, "Elo", None) if ratings_mod and use_elo else None
    TSCls  = getattr(ratings_mod, "TrueSkill", None) if ratings_mod and use_ts  else None
    elo_sys = EloCls() if EloCls else None
    ts_sys  = TSCls()  if TSCls  else None

    benches = [(p, load_agent(p, env, dict(agent_kwargs_all.get(p, {})))) for p in bench_paths]
    evals   = [(p, load_agent(p, env, dict(agent_kwargs_all.get(p, {})))) for p in eval_paths]

    elo_r: Dict[str, Any] = {}
    ts_r:  Dict[str, Any] = {}

    def ensure(name: str):
        if elo_sys is not None and name not in elo_r:
            elo_r[name] = elo_sys.create() if hasattr(elo_sys, "create") else 1000.0
        if ts_sys is not None and name not in ts_r:
            ts_r[name] = ts_sys.create() if hasattr(ts_sys, "create") else None


    summary: List[Dict[str, Any]] = []

    for ename, eagent in evals:
        ensure(ename)
        for bname, bagent in benches:
            ensure(bname)
            w = l = d = 0
            for ep in range(args.episodes):
                if ep % 2 == 0:
                    terminated, truncated, winners = play_match(env, eagent, bagent, seed=ep*seed)
                    e_won = (0 in winners)
                else:
                    terminated, truncated, winners = play_match(env, bagent, eagent, seed=ep*seed)
                    e_won = (1 in winners)

                if winners:
                    if e_won: w += 1
                    else:     l += 1
                else:
                    d += 1

                # ratings updates
                if elo_sys is not None:
                    a, b = elo_r[ename], elo_r[bname]
                    score = 1.0 if e_won else (0.0 if winners else 0.5)
                    if hasattr(elo_sys, "rate_1vs1"):
                        a, b = elo_sys.rate_1vs1(a, b, score_a=score)
                    else:
                        Ea = 1.0/(1.0+10.0**((b-a)/400.0)); k=32.0
                        a = a + k*(score-Ea); b = b + k*((1.0-score)-(1.0-Ea))
                    elo_r[ename], elo_r[bname] = a, b

                if ts_sys is not None and hasattr(ts_sys, "rate_1vs1"):
                    a, b = ts_r[ename], ts_r[bname]
                    if winners:
                        a, b = ts_sys.rate_1vs1(a, b, draw=False, a_wins=e_won)
                    else:
                        a, b = ts_sys.rate_1vs1(a, b, draw=True)
                    ts_r[ename], ts_r[bname] = a, b

            n = max(1, w+l+d)
            row = {"evaluated": ename, "benchmark": bname, "episodes": n,
                   "win_rate": w/n, "wins": w, "losses": l, "draws": d}
            if ename in elo_r: row["elo_eval"] = elo_r[ename]
            if bname in elo_r: row["elo_bench"] = elo_r[bname]
            if ename in ts_r:  row["ts_eval"]  = ts_r[ename]
            if bname in ts_r:  row["ts_bench"] = ts_r[bname]
            summary.append(row)

    print("\n=== Evaluation Summary ===")
    for r in summary:
        line = (f"{r['evaluated']} vs {r['benchmark']} | n={r['episodes']} | "
                f"WR={r['win_rate']*100:5.1f}% | W/L/D={r['wins']}/{r['losses']}/{r['draws']}")
        if "elo_eval" in r: line += f" | ELO(eval)={r['elo_eval']}"
        if "ts_eval" in r:  line += f" | TS(eval)={r['ts_eval']}"
        print(line)

if __name__ == "__main__":
    main()
