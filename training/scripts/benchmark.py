import time, torch
from ..envs.sequence_env import SequenceEnv
from ..algorithms.ppo_lstm.policy import PPOPolicy

def main():
    env = SequenceEnv()
    obs, info = env.reset()
    steps = 200
    t0 = time.time()
    for _ in range(steps):
        a = env.action_space.sample()
        obs, r, d, tr, info = env.step(a)
        if d or tr: env.reset()
    t1 = time.time()
    print(f"Env FPS: {steps/(t1-t0):.1f}")

    obs_shape = env.observation_space.shape
    action_dim = env.action_space.n
    policy = PPOPolicy(obs_shape, action_dim, conv_channels=[32,32], lstm_hidden=64)
    bs = 128
    x = torch.randn((bs,) + obs_shape)
    if torch.cuda.is_available():
        policy = policy.cuda(); x = x.cuda()
    t0 = time.time()
    with torch.no_grad():
        for _ in range(500):
            policy(x)
    t1 = time.time()
    per = (t1 - t0) / 500.0
    print(f"Inference ~{bs/per:.1f} obs/sec")

if __name__ == "__main__":
    main()
