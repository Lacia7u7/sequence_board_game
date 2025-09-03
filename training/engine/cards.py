"""
Card and deck utilities for Sequence.
"""
from typing import List, Optional
import random

RANKS: List[str] = ["A", "2", "3", "4", "5", "6", "7", "8", "9", "10", "J", "Q", "K"]
SUITS: List[str] = ["S", "H", "D", "C"]

def is_two_eyed_jack(card: Optional[str]) -> bool:
    return card in ("JC", "JD")

def is_one_eyed_jack(card: Optional[str]) -> bool:
    return card in ("JH", "JS")

def create_full_deck() -> List[str]:
    return [f"{rank}{suit}" for suit in SUITS for rank in RANKS]

class Deck:
    """Two combined decks (104 cards)."""
    def __init__(self):
        self.cards: List[str] = create_full_deck() + create_full_deck()
        self.discard_pile: List[str] = []
        self.burned_cards: List[str] = []
        self.rng = random.Random()

    def shuffle(self, seed: Optional[int] = None) -> None:
        self.rng.seed(seed)
        self.rng.shuffle(self.cards)

    def draw(self) -> Optional[str]:
        if not self.cards:
            if self.discard_pile:
                self.cards = list(self.discard_pile)
                self.discard_pile.clear()
                self.rng.shuffle(self.cards)
            else:
                return None
        return self.cards.pop()

    def discard(self, card: str, burned: bool = False) -> None:
        self.discard_pile.append(card)
        if burned:
            self.burned_cards.append(card)
