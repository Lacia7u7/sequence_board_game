import time
class Timer:
    def __init__(self): self.t0 = None
    def start(self): self.t0 = time.time()
    def elapsed(self): return (time.time() - self.t0) if self.t0 else 0.0
