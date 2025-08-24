import json, os
def save_replay(path: str, data: dict):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w") as f:
        json.dump(data, f)
def load_replay(path: str) -> dict:
    with open(path, "r") as f:
        return json.load(f)
