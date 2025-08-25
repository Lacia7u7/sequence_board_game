# training/utils/jsonio.py
from __future__ import annotations
import json
import copy
from typing import Any, Dict


def load_json(path: str) -> Dict[str, Any]:
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def save_json(path: str, data: Dict[str, Any]) -> None:
    with open(path, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=2)


def deep_update(base: Dict[str, Any], override: Dict[str, Any]) -> Dict[str, Any]:
    """
    Recursively merge 'override' into 'base' (returns a new dict).
    """
    out = copy.deepcopy(base)
    for k, v in (override or {}).items():
        if isinstance(v, dict) and isinstance(out.get(k), dict):
            out[k] = deep_update(out[k], v)
        else:
            out[k] = v
    return out


def override_config(cfg: dict, overrides: dict) -> dict:
    """
    Dot-path override utility, e.g. {"training.lr":1e-4, "logging.run_name":"exp1"}.
    """
    out = copy.deepcopy(cfg)

    def set_by_path(d: dict, path: str, value: Any) -> None:
        keys = path.split(".")
        for k in keys[:-1]:
            if k not in d or not isinstance(d[k], dict):
                d[k] = {}
            d = d[k]
        d[keys[-1]] = value

    for k, v in (overrides or {}).items():
        set_by_path(out, k, v)
    return out
