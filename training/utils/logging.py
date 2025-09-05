# training/utils/logging.py
from __future__ import annotations
import os, csv, json, time
from typing import Dict, Any, Optional
from .tb_writer import TBWriter


class CSVLogger:
    def __init__(self, path: str):
        self.path = path
        os.makedirs(os.path.dirname(self.path), exist_ok=True)
        self._fh = open(self.path, "a", newline="", encoding="utf-8")
        self._w = csv.writer(self._fh)
        if self._fh.tell() == 0:
            self._w.writerow(["step", "key", "value", "timestamp"])

    def log(self, step: int, metrics: Dict[str, float]) -> None:
        ts = int(time.time())
        for k, v in metrics.items():
            self._w.writerow([int(step), k, float(v), ts])
        self._fh.flush()

    def close(self) -> None:
        try:
            self._fh.close()
        except Exception:
            pass


class JSONLLogger:
    def __init__(self, path: str):
        self.path = path
        os.makedirs(os.path.dirname(self.path), exist_ok=True)
        self._fh = open(self.path, "a", encoding="utf-8")

    def log(self, step: int, payload: Dict[str, Any]) -> None:
        out = {"step": int(step), "ts": int(time.time()), **payload}
        self._fh.write(json.dumps(out, ensure_ascii=False) + "\n")
        self._fh.flush()

    def close(self) -> None:
        try:
            self._fh.close()
        except Exception:
            pass


class LoggingMux:
    """
    Multiplex logs to TensorBoard, CSV, and JSONL based on config.
    Creates a per-run directory at '<logdir>/<run_name>/'.
    """
    def __init__(self, cfg: Dict[str, Any]):
        log_cfg = cfg.get("logging", {})
        base_dir = log_cfg.get("logdir", "runs")
        run_name = log_cfg.get("run_name", "run")
        self.run_dir = os.path.join(os.path.abspath(base_dir), run_name)
        os.makedirs(self.run_dir, exist_ok=True)
        print("Run dir: {}".format(self.run_dir))

        self.tb = TBWriter(base_dir, run_name) if log_cfg.get("tensorboard", True) else None
        self.csv = CSVLogger(os.path.join(self.run_dir, "metrics.csv")) if log_cfg.get("csv", True) else None
        self.jsonl = JSONLLogger(os.path.join(self.run_dir, "metrics.jsonl")) if log_cfg.get("jsonl", True) else None
        self.log_private = bool(log_cfg.get("log_private_hands_locally", False))
        self.private_jsonl = JSONLLogger(os.path.join(self.run_dir, "_private.jsonl")) if self.log_private else None

    def hparams(self, hparams: Dict[str, Any], metrics: Dict[str, float]) -> None:
        if self.tb is not None:
            self.tb.add_hparams_once(hparams, metrics)

    def scalars(self, tag: str, scalars: Dict[str, float], step: int) -> None:
        if self.tb is not None:
            self.tb.add_scalars(tag, scalars, step)
        if self.csv is not None:
            # flatten under tag in CSV
            self.csv.log(step, {f"{tag}/{k}": v for k, v in scalars.items()})
        if self.jsonl is not None:
            self.jsonl.log(step, {tag: scalars})

    def scalar(self, tag: str, value: float, step: int) -> None:
        if self.tb is not None:
            self.tb.add_scalar(tag, value, step)
        if self.csv is not None:
            self.csv.log(step, {tag: value})
        if self.jsonl is not None:
            self.jsonl.log(step, {tag: value})

    def private(self, step: int, payload: Dict[str, Any]) -> None:
        if self.private_jsonl is not None:
            self.private_jsonl.log(step, payload)

    def flush(self) -> None:
        if self.tb is not None:
            self.tb.flush()

    def close(self) -> None:
        if self.tb is not None:
            self.tb.close()
        if self.csv is not None:
            self.csv.close()
        if self.jsonl is not None:
            self.jsonl.close()
        if self.private_jsonl is not None:
            self.private_jsonl.close()
    @staticmethod
    def get_run_dir(cfg):
        log_cfg = cfg.get("logging", {})
        base_dir = log_cfg.get("logdir", "runs")
        run_name = log_cfg.get("run_name", "run")
        return os.path.join(os.path.abspath(base_dir), run_name)