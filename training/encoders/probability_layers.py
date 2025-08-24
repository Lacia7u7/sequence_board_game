import math
from typing import Dict

def probability_cell_playable_next(deck_counts: Dict[str, int], total_remaining: int, cell_card: str) -> float:
    if cell_card == "BONUS":
        return 0.0
    count_match = deck_counts.get(cell_card, 0)
    count_wild = deck_counts.get("JC", 0) + deck_counts.get("JD", 0)
    p_match = count_match / total_remaining if total_remaining > 0 else 0.0
    p_wild = count_wild / total_remaining if total_remaining > 0 else 0.0
    return p_match + p_wild - (p_match * p_wild)

def probability_opponent_target_next(deck_counts: Dict[str, int], total_remaining: int, cell_card: str) -> float:
    return probability_cell_playable_next(deck_counts, total_remaining, cell_card)

def probability_cell_playable_k_steps(p_next: float, k: int) -> float:
    return 1.0 - ((1.0 - p_next) ** k)
