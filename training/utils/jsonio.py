import copy
def override_config(cfg: dict, overrides: dict) -> dict:
    out = copy.deepcopy(cfg)
    def set_by_path(d, path, value):
        keys = path.split(".")
        for k in keys[:-1]:
            if k not in d or not isinstance(d[k], dict):
                d[k] = {}
            d = d[k]
        d[keys[-1]] = value
    for k, v in overrides.items():
        set_by_path(out, k, v)
    return out
