def flat_to_components(flat_index: int, max_hand: int, include_pass: bool = False):
    pass_index = 100 + max_hand if include_pass else None
    if include_pass and flat_index == pass_index:
        return ("pass", None)
    if 0 <= flat_index < 100:
        r = flat_index // 10; c = flat_index % 10
        return ("cell", (r, c))
    if 100 <= flat_index < 100 + max_hand:
        return ("discard", flat_index - 100)
    return ("unknown", flat_index)

def components_to_flat(action_type: str, detail, max_hand: int, include_pass: bool = False):
    if action_type == "pass" and include_pass:
        return 100 + max_hand
    if action_type == "cell":
        r, c = detail
        return r * 10 + c
    if action_type == "discard":
        return 100 + int(detail)
    raise ValueError(f"Cannot convert action type {action_type}")
