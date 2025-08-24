import argparse, json, torch
from ..envs.sequence_env import SequenceEnv
from ..algorithms.ppo_lstm.policy import PPOPolicy

def main():
    p = argparse.ArgumentParser()
    p.add_argument("--config", required=True)
    p.add_argument("--policy_path", required=True)
    args = p.parse_args()
    config = json.load(open(args.config))
    env = SequenceEnv(config)
    obs_shape = env.observation_space.shape
    action_dim = env.action_space.n
    conv_channels = config["model"].get("conv_channels", [64,64,128,128])
    lstm_hidden = config["model"].get("lstm_hidden", 256)
    lstm_layers = config["model"].get("lstm_layers", 1)
    policy = PPOPolicy(obs_shape, action_dim, conv_channels, lstm_hidden, lstm_layers)
    policy.load_state_dict(torch.load(args.policy_path, map_location="cpu"))
    policy.eval()
    wins = 0; games = 5
    for _ in range(games):
        obs, info = env.reset()
        done = False
        while not done:
            obs_t = torch.from_numpy(obs).unsqueeze(0)
            logits, value, _ = policy(obs_t)
            action = int(torch.argmax(logits, dim=-1).item())
            obs, reward, terminated, truncated, info = env.step(action)
            done = terminated or truncated
        if reward > 0: wins += 1
    print(f"Win-rate over {games} self-play matches: {wins/games:.2%}")

if __name__ == "__main__":
    main()
