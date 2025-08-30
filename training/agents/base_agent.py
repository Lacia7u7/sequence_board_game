# training/agents/base_agent.py
from __future__ import annotations

from typing import Optional, Dict, Any

import numpy as np

# Optional: a minimal spec you can extend safely
# Agents should tolerate missing keys and raise clear errors only when they truly need a key.
AgentCtx = Dict[str, Any]  # e.g. {"obs": np.ndarray, "info": dict, "seat": int, "env": <SequenceEnv>, "legal_mask": np.ndarray}

class BaseAgent:
    """
    Minimal agent interface that primarily consumes the legal mask.
    The second param is a flexible context dict for dynamic data (obs, info, seat, env, etc.)
    """
    def __init__(self,env):
        self.env = env

    def reset(self, env, seat: int) -> None:
        pass

    def select_action(self, legal_mask: Optional[np.ndarray], ctx: Optional[AgentCtx] = None) -> int:
        raise NotImplementedError

    def make_new_agent(self, env):
        return self.__class__(env)   # or type(self)(env)

class ObsAgent(BaseAgent):
    """
    Same interface; this class is just a semantic tag for agents that *require* observations.
    """
    def select_action(self, legal_mask: Optional[np.ndarray], ctx: Optional[AgentCtx] = None) -> int:
        # Concrete subclasses should *require* ctx and ctx["obs"]
        raise NotImplementedError("ObsAgent subclasses must implement select_action and use ctx['obs'].")
