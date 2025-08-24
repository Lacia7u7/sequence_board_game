import numpy as np
from collections import deque
from typing import Deque

class HistoryEncoder:
    def __init__(self, max_length: int):
        self.max_length = max_length
        self.frames: Deque[np.ndarray] = deque(maxlen=max_length)
    def reset(self):
        self.frames.clear()
    def push(self, obs: np.ndarray):
        if obs is not None:
            self.frames.append(obs)
    def get_stacked(self) -> np.ndarray:
        if not self.frames:
            return np.zeros((0,))
        return np.concatenate(list(self.frames), axis=0)
