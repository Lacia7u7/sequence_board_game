# training/utils/keys.py
from __future__ import annotations
from typing import Dict, Tuple, Sequence
from enum import Enum

# Python <3.11 fallback for StrEnum
try:
    from enum import StrEnum
except Exception:  # pragma: no cover
    class StrEnum(str, Enum):  # type: ignore
        pass


class LegalKey(StrEnum):
    """Canonical field names for legal-actions dictionaries."""
    PLACE = "place"               # list[(r,c)]
    REMOVE = "remove"             # list[(r,c)]
    DISCARD = "discard"           # list[int]
    PASS = "pass"                 # bool
    TARGETS = "targets"           # list[(r,c)] — union(place, remove) or place-only
    DISCARD_SLOTS = "discard_slots"  # alias for DISCARD to satisfy legacy code


# One place to manage every synonym we accept anywhere in the codebase.
LEGAL_ALIASES: Dict[LegalKey, Tuple[str, ...]] = {
    LegalKey.PLACE: (
        "place", "place_targets", "targets", "moves", "cells",
    ),
    LegalKey.REMOVE: (
        "remove", "remove_targets", "captures",
    ),
    LegalKey.DISCARD: (
        "discard", "discard_slots", "burn", "drops",
    ),
    LegalKey.PASS: (
        "pass", "skip", "noop", "do_nothing",
    ),
    # TARGETS/DISCARD_SLOTS are “derived” keys; you usually don't provide them as input.
    LegalKey.TARGETS: ("targets",),
    LegalKey.DISCARD_SLOTS: ("discard_slots",),
}


def get_aliases(key: LegalKey) -> Tuple[str, ...]:
    """Return all accepted input names for a canonical LegalKey."""
    return LEGAL_ALIASES[key]


# Optional: config key for mask encoding
class MaskEncoding(StrEnum):
    ONE_FOR_ALLOWED = "one_for_allowed"  # 1 = legal, 0 = illegal (recommended)
    ONE_FOR_INVALID = "one_for_invalid"  # 1 = illegal, 0 = legal (invert before use)
