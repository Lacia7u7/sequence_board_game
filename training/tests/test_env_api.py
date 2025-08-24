from training.envs.sequence_env import SequenceEnv
import gymnasium as gym

def test_env_compliance():
    env = SequenceEnv()
    assert isinstance(env, gym.Env)
    obs, info = env.reset()
    assert env.observation_space.contains(obs)
    done = False; steps = 0
    while not done and steps < 10:
        a = env.action_space.sample()
        obs, r, term, trunc, info = env.step(a)
        assert isinstance(r, float)
        done = term or trunc; steps += 1
    env.close()
