# training/utils/seeding.py
from __future__ import annotations
import os
import random
import numpy as np

def set_all_seeds(seed: int) -> None:
    seed = int(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    random.seed(seed)
    np.random.seed(seed)
    try:
        import torch
        torch.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        # For full determinism, uncomment (slower):
        # torch.backends.cudnn.deterministic = True
        # torch.backends.cudnn.benchmark = False
    except Exception:
        pass
