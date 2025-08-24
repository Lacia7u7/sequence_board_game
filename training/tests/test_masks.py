from training.envs.sequence_env import SequenceEnv
from training.envs.masks import legal_action_mask

def test_legal_action_mask():
    env = SequenceEnv()
    obs, info = env.reset()
    mask = legal_action_mask(env, env.current_player)
    assert mask.sum() > 0
    assert mask.shape[0] == env.action_space.n
