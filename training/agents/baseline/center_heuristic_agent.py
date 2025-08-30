from __future__ import annotations
from typing import Optional
import numpy as np
from overrides import overrides

from ..base_agent import BaseAgent, AgentCtx
from ...algorithms.baselines.center_heuristic_policy import CenterHeuristicPolicy


class CenterHeuristicAgent(BaseAgent):
    """
    Agente heurístico:
      - Coloca cerca del centro (distancia Manhattan).
      - Si puede completar/avanzar secuencia, lo prioriza.
      - Si puede remover (JH/JS), elige la que rompe más "casi-secuencias" del rival.
    """

    def __init__(self, env=None):
        super().__init__(env)
        self.policy = CenterHeuristicPolicy(env=env)

    def reset(self, env, seat: int) -> None:
        # Nada que persistir por episodio; la política consulta el estado del env en cada paso
        self.env = env
        self.policy.env = env

    @overrides
    def select_action(self, legal_mask: Optional[np.ndarray], ctx: Optional[AgentCtx] = None) -> int:
        return self.policy.select_action(legal_mask, ctx=ctx)
