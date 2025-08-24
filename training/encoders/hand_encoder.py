from typing import List, Dict
def encode_hand_cards(hand: List[str]) -> Dict[str, int]:
    counts: Dict[str, int] = {}
    for card in hand:
        counts[card] = counts.get(card, 0) + 1
    return counts
