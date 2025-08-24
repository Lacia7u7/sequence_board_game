# training/engine/state.py
from __future__ import annotations

from dataclasses import dataclass, field, asdict
from typing import Any, Dict, List, Optional


@dataclass
class GameConfig:
    """
    Canonical configuration that the engine/env rely on.
    This class provides a stable structure AND a `from_dict` constructor so
    training scripts can pass a JSON config without manual parsing.
    """
    # Rules
    mode: str = "2v2"                       # "1v1" | "2v2" | "1v1v1"
    teams: int = 2
    players_per_team: int = 2
    hand_size: int = 6
    allowAdvancedJack: bool = False
    win_sequences_needed: int = 2           # 2 for 1v1/2v2; 1 for 1v1v1
    reset_full_board_no_winner: bool = True

    # Engine options
    use_cuda_sequences: bool = True         # fast CUDA path for sequence detection (optional)
    reshuffle_on_empty_deck: bool = True

    # Episode control (env)
    episode_cap: int = 400

    # Reproducibility
    seed: Optional[int] = None

    # --------- factory & helpers ---------
    @classmethod
    def from_dict(cls, cfg: Dict[str, Any]) -> "GameConfig":
        """
        Build a GameConfig from a nested dict like the JSON files under training/configs/.
        Expected top-level keys: "rules", "engine", "training" (all optional).
        """
        rules = dict(cfg.get("rules", {}))
        eng = dict(cfg.get("engine", {}))
        training = dict(cfg.get("training", {}))

        # Determine default win_sequences_needed based on mode if not explicitly provided
        mode = rules.get("mode", "2v2")
        default_needed = 1 if mode == "1v1v1" else 2
        win_needed = int(rules.get("win_sequences_needed", default_needed))

        return cls(
            mode=str(mode),
            teams=int(rules.get("teams", 2)),
            players_per_team=int(rules.get("players_per_team", 2)),
            hand_size=int(rules.get("hand_size", 6)),
            allowAdvancedJack=bool(rules.get("allowAdvancedJack", False)),
            win_sequences_needed=win_needed,
            reset_full_board_no_winner=bool(rules.get("reset_full_board_no_winner", True)),
            use_cuda_sequences=bool(eng.get("use_cuda_sequences", eng.get("use_cuda", True))),
            reshuffle_on_empty_deck=bool(eng.get("reshuffle_on_empty_deck", True)),
            episode_cap=int(training.get("episode_cap", 400)),
            seed=training.get("seed", None),
        )

    def to_dict(self) -> Dict[str, Any]:
        """Round-trip to a plain dict (useful for logging/debug)."""
        return asdict(self)


@dataclass
class GameState:
    """
    Runtime state snapshot used by the engine & env.

    NOTE: This is intentionally minimal and unopinionated; the engine_core.py
    owns the invariants and manipulates these fields.
    """
    board: List[List[Dict[str, Any]]] = field(default_factory=list)
    deck: Dict[str, Any] = field(default_factory=dict)
    hands: Dict[str, List[str]] = field(default_factory=dict)
    turn_index: int = 0

    # Sequence bookkeeping
    sequenceCells: List[Dict[str, int]] = field(default_factory=list)
    sequencesMeta: List[Dict[str, Any]] = field(default_factory=list)
    sequences: Dict[str, int] = field(default_factory=dict)

    # Outcomes & rounds
    winners: List[int] = field(default_factory=list)
    roundCount: int = 0

    # Config bound to this match
    config: Optional[GameConfig] = None
