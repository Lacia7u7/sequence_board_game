import numpy as np
def legal_action_mask(env, player_index: int) -> np.ndarray:
    mask = np.zeros(env.action_space.n, dtype=np.float32)
    legal = env.game_engine.legal_actions_for(player_index)
    for (r,c) in legal.get("place", []):
        mask[r*10 + c] = 1.0
    for (r,c) in legal.get("remove", []):
        mask[r*10 + c] = 1.0
    for idx in legal.get("discard", []):
        if 100 + idx < env.action_space.n:
            mask[100 + idx] = 1.0
    if env.config.get("action_space", {}).get("include_pass", False):
        if not legal.get("place") and not legal.get("remove") and not legal.get("discard"):
            mask[-1] = 1.0
    return mask
