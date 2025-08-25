# training/utils/tb_writer.py
from __future__ import annotations
import os
import time
from typing import Dict, Any, Optional

try:
    from torch.utils.tensorboard import SummaryWriter
except Exception:
    SummaryWriter = None


class TBWriter:
    """
    Safe wrapper around SummaryWriter. If TB isn't installed, it silently no-ops.
    """
    def __init__(self, logdir: str, run_name: Optional[str] = None):
        base = os.path.abspath(logdir or "runs")
        os.makedirs(base, exist_ok=True)
        self.logdir = os.path.join(base, run_name) if run_name else os.path.join(base, time.strftime("run_%Y%m%d-%H%M%S"))
        os.makedirs(self.logdir, exist_ok=True)
        self._writer = SummaryWriter(self.logdir) if SummaryWriter is not None else None

    def add_scalar(self, tag: str, value: float, step: int) -> None:
        if self._writer is not None:
            self._writer.add_scalar(tag, float(value), int(step))

    def add_scalars(self, main_tag: str, scalars: Dict[str, float], step: int) -> None:
        if self._writer is not None:
            self._writer.add_scalars(main_tag, {k: float(v) for k, v in scalars.items()}, int(step))

    def add_hparams_once(self, hparams: Dict[str, Any], metrics: Dict[str, float]) -> None:
        if self._writer is None:
            return
        try:
            self._writer.add_hparams(
                {str(k): (str(v) if isinstance(v, (list, dict)) else v) for k, v in hparams.items()},
                {str(k): float(v) for k, v in metrics.items()},
            )
        except Exception:
            pass

    def flush(self) -> None:
        if self._writer is not None:
            self._writer.flush()

    def close(self) -> None:
        if self._writer is not None:
            self._writer.close()
