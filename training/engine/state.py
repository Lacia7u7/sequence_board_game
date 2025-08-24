# training/engine/state.py
from __future__ import annotations

from dataclasses import dataclass, field, asdict
from typing import Any, Dict, List, Optional, Tuple, Set, Iterable


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

    Canonical fields use camelCase to avoid breaking current code.
    Back-compat snake_case properties and helper methods are provided to bridge
    older call sites.
    """

    # 10x10 board of team chips: None (empty) or team index (int)
    board: List[List[Optional[int]]] = field(default_factory=list)

    # Deck bookkeeping (optional; engine usually owns the Deck object)
    deck: Dict[str, Any] = field(default_factory=dict)

    # Hands: one list per player, each with card strings
    hands: List[List[str]] = field(default_factory=list)

    # Whose turn (0..total_players-1)
    turn_index: int = 0

    # --- Sequence bookkeeping ---
    # Set of board coordinates that belong to any confirmed sequence
    sequenceCells: Set[Tuple[int, int]] = field(default_factory=set)

    # Optional richer metadata per sequence (not required by engine_core)
    sequencesMeta: List[Dict[str, Any]] = field(default_factory=list)

    # Canonical per-team sequence counts: team_idx -> count
    sequences: Dict[int, int] = field(default_factory=dict)

    # Outcomes & rounds
    winners: List[int] = field(default_factory=list)
    roundCount: int = 0

    # Config bound to this match (optional)
    config: Optional[GameConfig] = None

    # ----------------- Back-compat aliases (snake_case <-> camelCase) -----------------

    # current_player <-> turn_index
    @property
    def current_player(self) -> int:
        return self.turn_index

    @current_player.setter
    def current_player(self, value: int) -> None:
        self.turn_index = int(value)

    # round_count <-> roundCount
    @property
    def round_count(self) -> int:
        return self.roundCount

    @round_count.setter
    def round_count(self, value: int) -> None:
        self.roundCount = int(value)

    # sequence_cells <-> sequenceCells (as a SET of (r,c) tuples)
    @property
    def sequence_cells(self) -> Set[Tuple[int, int]]:
        return self.sequenceCells  # return the actual set so callers can .add()

    @sequence_cells.setter
    def sequence_cells(self, value: Iterable[Tuple[int, int]]) -> None:
        self.sequenceCells = set((int(r), int(c)) for r, c in value)

    # sequences_meta <-> sequencesMeta
    @property
    def sequences_meta(self) -> List[Dict[str, Any]]:
        return self.sequencesMeta

    @sequences_meta.setter
    def sequences_meta(self, value: List[Dict[str, Any]]) -> None:
        self.sequencesMeta = list(value) if value is not None else []

    # sequences_count <-> sequences (dict[int,int])
    @property
    def sequences_count(self) -> Dict[int, int]:
        return self.sequences  # return the actual dict so callers can mutate

    @sequences_count.setter
    def sequences_count(self, value: Any) -> None:
        self.sequences = self._coerce_sequences_dict(value)

    # ----------------- Helper callable expected by some code -----------------

    def sequences_meta_cells(self) -> List[Tuple[Any, Any, List[Any]]]:
        """
        Return a list of (seq_id, seq_team, cells).

        Priority:
          1) If `sequencesMeta` has structured dicts, build triples from them.
             Expected keys (any of these): id|seq_id, team|team_id|owner, cells|cell_indices|positions
          2) Else, if `sequenceCells` is present, treat it as a single sequence with id=0 and team=None.
          3) Else, return an empty list.
        """
        metas = self.sequencesMeta or []
        if isinstance(metas, list) and metas:
            out: List[Tuple[Any, Any, List[Any]]] = []
            for i, m in enumerate(metas):
                if not isinstance(m, dict):
                    continue
                cells = m.get("cells") or m.get("cell_indices") or m.get("positions") or []
                team = m.get("team") or m.get("team_id") or m.get("owner")
                seq_id = m.get("id") or m.get("seq_id") or i
                out.append((seq_id, team, list(cells)))
            return out

        # Fallback: treat sequenceCells as a single sequence
        if self.sequenceCells:
            return [(0, None, [(r, c) for (r, c) in sorted(self.sequenceCells)])]

        return []

    # ----------------- Helpers -----------------

    @staticmethod
    def _coerce_sequences_dict(value: Any) -> Dict[int, int]:
        """
        Coerce various shapes into Dict[int, int].
        Accepts dict[int|str,int], list[int], tuple[int], or int (stored under -1).
        """
        out: Dict[int, int] = {}
        if isinstance(value, dict):
            for k, v in value.items():
                try:
                    ik = int(k)
                except (TypeError, ValueError):
                    # Very rare: non-numeric key; keep a stable bucket
                    ik = hash(k) % (10**6)
                out[ik] = int(v)
        elif isinstance(value, (list, tuple)):
            out = {i: int(v) for i, v in enumerate(value)}
        elif isinstance(value, int):
            out = {-1: int(value)}  # special "total" bucket
        elif value is None:
            out = {}
        else:
            out = {}
        return out
