# training/encoders/probability_layers.py
from __future__ import annotations
from typing import Dict


def _count(deck_counts: Dict[str, int], card: str) -> int:
    return max(0, int(deck_counts.get(card, 0)))


def probability_cell_playable_next(deck_counts: Dict[str, int], total_remaining: int, card: str) -> float:
    """
    Probability that after drawing one card (end of this turn),
    the agent will have a card enabling play on this cell on the next turn.
    PUBLIC-only approximation:
      p = p_X + p_J2 - p_X*p_J2, with p_X = c_X/R, p_J2 = (c_JC+c_JD)/R
    """
    if total_remaining <= 0:
        return 0.0
    c_x = _count(deck_counts, card)
    c_j2 = _count(deck_counts, "JC") + _count(deck_counts, "JD")
    p_x = c_x / total_remaining
    p_j2 = c_j2 / total_remaining
    p = p_x + p_j2 - (p_x * p_j2)
    return float(max(0.0, min(1.0, p)))


def probability_cell_playable_k_steps(p1: float, k: int) -> float:
    """
    Mean-field propagation: assume independence and approximate k steps ahead
    as 1 - (1 - p1)^k (chance of at least one enabling draw across k rounds).
    """
    if k <= 0:
        return float(p1)
    q = (1.0 - p1) ** k
    return float(max(0.0, min(1.0, 1.0 - q)))


def probability_opponent_target_next(deck_counts: Dict[str, int], total_remaining: int, card: str) -> float:
    """
    PUBLIC-only estimate of opponent having the enabling card or a two-eyed jack.
    Uses the same formula as the agent but is generic (no private info).
    """
    return probability_cell_playable_next(deck_counts, total_remaining, card)
