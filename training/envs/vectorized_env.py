import multiprocessing as mp
from typing import List, Any
from .sequence_env import SequenceEnv

def _worker(remote, env_fn):
    env = env_fn()
    try:
        while True:
            cmd, data = remote.recv()
            if cmd == "reset":
                ob, info = env.reset(seed=data)
                remote.send((ob, info))
            elif cmd == "step":
                ob, reward, terminated, truncated, info = env.step(data)
                remote.send((ob, reward, terminated, truncated, info))
            elif cmd == "close":
                env.close(); remote.close(); break
            else:
                remote.send(None)
    except EOFError:
        pass

class VectorizedEnv:
    def __init__(self, num_envs: int, env_fn):
        self.num_envs = num_envs
        self.remotes: List[Any] = []
        self.processes: List[mp.Process] = []
        for _ in range(num_envs):
            parent_remote, child_remote = mp.Pipe()
            p = mp.Process(target=_worker, args=(child_remote, env_fn))
            p.daemon = True; p.start()
            child_remote.close()
            self.remotes.append(parent_remote)
            self.processes.append(p)

    def reset(self, seeds: List[int] = None):
        seeds = seeds or [None]*self.num_envs
        for remote, seed in zip(self.remotes, seeds):
            remote.send(("reset", seed))
        results = [remote.recv() for remote in self.remotes]
        obs, infos = zip(*results)
        return list(obs), list(infos)

    def step(self, actions: List[int]):
        for remote, act in zip(self.remotes, actions):
            remote.send(("step", act))
        results = [remote.recv() for remote in self.remotes]
        obs, rewards, dones, truncs, infos = zip(*results)
        return list(obs), list(rewards), list(dones), list(truncs), list(infos)

    def close(self):
        for remote in self.remotes:
            remote.send(("close", None))
        for p in self.processes:
            p.join()
