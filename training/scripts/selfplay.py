from __future__ import annotations
import argparse, json

from ..envs.sequence_env import SequenceEnv
from ..agents.selfplay_manager import SelfPlayManager
from ..agents.baseline.greedy_sequence_agent import GreedySequenceAgent
from ..agents.baseline.blocking_agent import BlockingAgent

def main():
    p = argparse.ArgumentParser()
    p.add_argument("--config", required=True)
    p.add_argument("--games", type=int, default=1)
    args = p.parse_args()

    cfg = json.load(open(args.config))
    env = SequenceEnv(cfg)

    agents = [GreedySequenceAgent(env=env), BlockingAgent(env=env)]
    mgr = SelfPlayManager(agents, env, max_steps=int(cfg.get("training", {}).get("episode_cap", 400)))

    for _ in range(args.games):
        _ = mgr.play_episode(render=False)
    print("Self-play completed.")

if __name__ == "__main__":
    main()
