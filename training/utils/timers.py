# training/utils/timers.py
from __future__ import annotations
import time

class Timer:
    def __init__(self):
        self._t0 = None

    def start(self) -> None:
        self._t0 = time.time()

    def elapsed(self) -> float:
        return 0.0 if self._t0 is None else (time.time() - self._t0)
