import argparse, json
from ..envs.sequence_env import SequenceEnv
from ..algorithms.baselines.greedy_sequence_policy import GreedySequencePolicy
from ..algorithms.baselines.blocking_policy import BlockingPolicy
from ..agents.selfplay_manager import SelfPlayManager

def main():
    p = argparse.ArgumentParser()
    p.add_argument("--config", required=True)
    p.add_argument("--games", type=int, default=1)
    args = p.parse_args()
    config = json.load(open(args.config))
    env = SequenceEnv(config)
    agent0 = GreedySequencePolicy(env)
    agent1 = BlockingPolicy(env)
    manager = SelfPlayManager([agent0, agent1], env)
    wins = [0,0]
    for _ in range(args.games):
        manager.play_episode(render=False)
        # Simplified: environment reward reported at end; rely on render/logs for details
    print("Self-play completed.")

if __name__ == "__main__":
    main()
