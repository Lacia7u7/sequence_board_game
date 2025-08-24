# TensorBoard writer shim (optional dependency)
try:
    from torch.utils.tensorboard import SummaryWriter as _TBW
except Exception:
    _TBW = None

class TBWriter:
    def __init__(self, logdir: str):
        self._writer = _TBW(logdir) if _TBW else None
    def add_scalar(self, tag, scalar_value, global_step=None):
        if self._writer: self._writer.add_scalar(tag, scalar_value, global_step)
    def flush(self):
        if self._writer: self._writer.flush()
    def close(self):
        if self._writer: self._writer.close()
