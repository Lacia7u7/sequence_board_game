import numpy as np
from ..envs.sequence_env import SequenceEnv
def main():
    env = SequenceEnv()
    obs, info = env.reset()
    np.savez("sample_observation.npz", obs=obs)
    print("Saved to sample_observation.npz", obs.shape)
if __name__ == "__main__":
    main()
