from dataclasses import dataclass, field
from typing import List, Dict, Set, Tuple, Optional

@dataclass
class GameConfig:
    teams: int
    players_per_team: int
    hand_size: int
    allowAdvancedJack: bool
    win_sequences_needed: int
    reset_full_board_no_winner: bool

@dataclass
class GameState:
    hands: List[List[str]]
    board: List[List[Optional[int]]]  # 10x10 int team or None
    sequence_cells: Set[Tuple[int, int]] = field(default_factory=set)
    sequences_count: Dict[int, int] = field(default_factory=lambda: {})
    current_player: int = 0
    round_count: int = 0
    winners: List[int] = field(default_factory=list)
