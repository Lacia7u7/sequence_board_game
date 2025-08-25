# training/utils/legal_utils.py
from __future__ import annotations
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple, Union
import numpy as np

from .keys import LegalKey, get_aliases, MaskEncoding


CoordLike = Union[Tuple[int, int], Sequence[int], np.ndarray, List[int]]


def _first_present(d: Dict[str, Any], names: Iterable[str], default=None):
    for n in names:
        if n in d:
            return d[n]
    return default


def _to_rc_list(x: Any) -> List[Tuple[int, int]]:
    """Convert input to list[(r,c)] safely, drop malformed, preserve order."""
    if x is None:
        return []
    out: List[Tuple[int, int]] = []
    for item in x:
        if isinstance(item, (tuple, list, np.ndarray)) and len(item) == 2:
            out.append((int(item[0]), int(item[1])))
    # de-dup preserve order
    seen = set()
    uniq = []
    for rc in out:
        if rc not in seen:
            uniq.append(rc)
            seen.add(rc)
    return uniq


def _to_int_list(x: Any) -> List[int]:
    """Convert to list[int] safely, drop malformed, preserve order."""
    if x is None:
        return []
    out: List[int] = []
    for item in x:
        try:
            out.append(int(item))
        except Exception:
            continue
    seen = set()
    uniq = []
    for i in out:
        if i not in seen:
            uniq.append(i)
            seen.add(i)
    return uniq


def normalize_legal(
    legal: Optional[Dict[str, Any]],
    *,
    board_h: int = 10,
    board_w: int = 10,
    max_hand: int = 7,
    include_pass: bool = False,
    union_place_remove_for_targets: bool = True,
    extra_aliases: Optional[Dict[LegalKey, Sequence[str]]] = None,
) -> Dict[str, Any]:
    """
    Normalize any 'legal' dict into the canonical schema:

      {
        "place": [(r,c), ...],
        "remove": [(r,c), ...],
        "discard": [hand_idx, ...],
        "pass": bool,
        "targets": [(r,c), ...],           # place ∪ remove  (or only place)
        "discard_slots": [hand_idx, ...]   # alias of "discard"
      }

    - Drops out-of-bounds board cells and invalid hand indices.
    - De-duplicates while preserving order.
    - 'extra_aliases' lets a caller add new synonyms without touching core code.
    """
    d = dict(legal or {})

    def aliases(key: LegalKey) -> Sequence[str]:
        base = list(get_aliases(key))
        if extra_aliases and key in extra_aliases:
            base += [str(x) for x in extra_aliases[key]]
        return base

    place_raw = _first_present(d, aliases(LegalKey.PLACE), [])
    remove_raw = _first_present(d, aliases(LegalKey.REMOVE), [])
    discard_raw = _first_present(d, aliases(LegalKey.DISCARD), [])
    pass_raw = _first_present(d, aliases(LegalKey.PASS), False)

    place = _to_rc_list(place_raw)
    remove = _to_rc_list(remove_raw)
    discard = _to_int_list(discard_raw)
    pass_flag = bool(pass_raw) and bool(include_pass)

    # bounds
    def in_bounds(rc: Tuple[int, int]) -> bool:
        return 0 <= rc[0] < board_h and 0 <= rc[1] < board_w

    place = [rc for rc in place if in_bounds(rc)]
    remove = [rc for rc in remove if in_bounds(rc)]
    discard = [i for i in discard if 0 <= i < max_hand]

    targets = (place + remove) if union_place_remove_for_targets else list(place)

    return {
        LegalKey.PLACE.value: place,
        LegalKey.REMOVE.value: remove,
        LegalKey.DISCARD.value: discard,
        LegalKey.PASS.value: pass_flag,
        LegalKey.TARGETS.value: targets,
        LegalKey.DISCARD_SLOTS.value: list(discard),
    }


def ensure_mask_convention(mask: np.ndarray, encoding: str | MaskEncoding) -> np.ndarray:
    """
    Ensure mask follows 1=legal, 0=illegal by inverting if the input encoding is ONE_FOR_INVALID.
    """
    enc = MaskEncoding(encoding) if not isinstance(encoding, MaskEncoding) else encoding
    mask = np.asarray(mask, dtype=np.float32)
    if enc == MaskEncoding.ONE_FOR_INVALID:
        mask = 1.0 - mask
    return mask


def build_action_mask(
    action_dim: int,
    *,
    board_h: int,
    board_w: int,
    max_hand: int,
    include_pass: bool,
    canon_legal: Dict[str, Any],
) -> np.ndarray:
    """
    Build a (action_dim,) mask with 1=legal, 0=illegal from a *canonical* legal dict.

    Layout assumed:
      0..(board_h*board_w-1) board cells (place/remove)
      board_h*board_w .. +max_hand-1  discard slots
      +max_hand  optional PASS
    """
    mask = np.zeros((action_dim,), dtype=np.float32)

    # board cells
    for (r, c) in canon_legal.get(LegalKey.TARGETS.value, []):
        if 0 <= r < board_h and 0 <= c < board_w:
            mask[r * board_w + c] = 1.0

    base_disc = canon_legal.get(LegalKey.DISCARD.value, [])
    for hand_idx in base_disc:
        if 0 <= hand_idx < max_hand:
            mask[board_h * board_w + hand_idx] = 1.0

    if include_pass and canon_legal.get(LegalKey.PASS.value, False):
        mask[board_h * board_w + max_hand] = 1.0

    # Safety guard
    if mask.sum() < 1e-6:
        if include_pass:
            mask[board_h * board_w + max_hand] = 1.0
        else:
            # Leave empty (upstream can decide), or raise
            pass

    return mask
