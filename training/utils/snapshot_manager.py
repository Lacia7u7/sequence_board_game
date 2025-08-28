from __future__ import annotations
import os
from collections import deque
from typing import Deque, List, Optional
import torch

class SnapshotManager:
    """Disk-backed ring buffer of PPO snapshots.

    Saves CPU state_dicts under <run_dir>/snapshots/ppo_update_XXXXXX.pt
    and keeps only the most recent `max_keep` files.
    """
    def __init__(self, run_dir: str, max_keep: int = 10, subdir: str = "snapshots"):
        self.snap_dir = os.path.join(run_dir, subdir)
        os.makedirs(self.snap_dir, exist_ok=True)
        self.max_keep = int(max_keep)
        self.paths: Deque[str] = deque()

        # If resuming, preload existing paths ordered by mtime
        try:
            entries = [os.path.join(self.snap_dir, f) for f in os.listdir(self.snap_dir) if f.endswith('.pt')]
            entries.sort(key=lambda p: os.path.getmtime(p))
            for p in entries[-self.max_keep:]:
                self.paths.append(p)
        except Exception:
            pass

    def save(self, policy: torch.nn.Module, update_idx: int) -> str:
        path = os.path.join(self.snap_dir, f"ppo_update_{update_idx:06d}.pt")
        state = {k: v.detach().cpu() for k, v in policy.state_dict().items()}
        torch.save(state, path)
        self.paths.append(path)
        while len(self.paths) > self.max_keep:
            old = self.paths.popleft()
            try:
                os.remove(old)
            except OSError:
                pass
        return path

    def latest(self) -> Optional[str]:
        return self.paths[-1] if self.paths else None

    def all(self) -> List[str]:
        return list(self.paths)