# training/agents/human_agent.py
# A lightweight Pygame UI to replace the console renderer for HumanAgent.
# - Shows 10x10 board with card faces (optional), chips, and sequence overlays
# - Click a hand card to select; click a board cell to play; or click "Burn"/"Pass"
# - Returns an integer action following SequenceEnv's unified action space
#
# Expected assets (optional): PNG files at assets/cards/{rank}{suit}.png, e.g., 7H.png, 10D.png
# If images are missing, the UI will render text placeholders instead.

from __future__ import annotations

from typing import Any, Dict, Optional
import numpy as np

# Import engine/env bits for helpers
try:
    from ..engine.board_layout import BOARD_LAYOUT
    from ..ui.human_agent_ui import BOARD_LAYOUT, HumanAgentUI
    from ..engine.engine_core import is_two_eyed_jack, is_one_eyed_jack
except Exception:  # fallback when running this file stand-alone for tinkering

    pass

# -----------------------------------------------------------------------------
# HumanAgent that uses the Pygame UI (this file lives in training/agents)
# -----------------------------------------------------------------------------

try:
    from .base_agent import BaseAgent  # preferred relative import within package
except Exception:
    from base_agent import BaseAgent  # fallback when package context differs


class HumanAgent(BaseAgent):
    """Drop-in replacement that opens a Pygame window for interactive play.

    Action mapping (SequenceEnv):
      - Click a board cell  -> 0..99
      - Click "Burn"        -> 100 + hand_index (if legal and not a Jack)
      - Click/press "Pass"  -> 100 + max_hand (if include_pass)
    """

    def __init__(self, env, card_img_dir: str = "assets/cards"):
        super().__init__(env)
        self.env = env
        self.ui = HumanAgentUI(card_img_dir=card_img_dir)

    def reset(self, env, seat: int) -> None:
        # nothing required; window persists across steps
        pass

    def select_action(self, legal_mask: Optional[np.ndarray], ctx: Optional[Dict[str, Any]] = None) -> int:
        info = ctx.get("info") if ctx else {}
        return self.ui.choose_action(self.env, legal_mask, info)
