# training/agents/rating.py
from __future__ import annotations
from dataclasses import dataclass
from typing import Tuple, Optional
import math

# Try to use the real library if present
try:
    import trueskill as _ts
except Exception:
    _ts = None


# -----------------------------
# Simple Elo implementation
# -----------------------------
class Elo:
    def __init__(self, k: float = 32.0, base: float = 10.0, default: float = 1000.0):
        self.k = float(k)
        self.base = float(base)
        self.default = float(default)

    def create(self) -> float:
        """Return a fresh rating."""
        return self.default

    def _expected(self, ra: float, rb: float) -> float:
        return 1.0 / (1.0 + self.base ** ((rb - ra) / 400.0))

    def rate_1vs1(self, a: float, b: float, score_a: float) -> Tuple[float, float]:
        """
        Update two players after a single head-to-head result.
        score_a: 1.0 win, 0.5 draw, 0.0 loss
        """
        ea = self._expected(a, b)
        eb = 1.0 - ea
        a_new = a + self.k * (score_a - ea)
        b_new = b + self.k * ((1.0 - score_a) - eb)
        return a_new, b_new


# -----------------------------
# TrueSkill wrapper or fallback
# -----------------------------
@dataclass
class TSRating:
    mu: float
    sigma: float
    def __str__(self) -> str:
        return f"{self.mu:.2f}±{self.sigma:.2f}"


class TrueSkill:
    """
    If the `trueskill` package is installed, we wrap it.
    Otherwise we use a light-weight fallback with (mu, sigma) updated via a logistic model.
    API:
      - create() -> rating object
      - rate_1vs1(a, b, draw: bool=False, a_wins: Optional[bool]=None) -> (a', b')
    """
    def __init__(
        self,
        mu: float = 25.0,
        sigma: float = 25.0 / 3.0,
        beta: float = 25.0 / 6.0,
        tau: float = 25.0 / 300.0,
        draw_probability: float = 0.10,
    ):
        self._use_lib = _ts is not None
        if self._use_lib:
            # Real TrueSkill environment
            self._env = _ts.TrueSkill(
                mu=mu, sigma=sigma, beta=beta, tau=tau, draw_probability=draw_probability
            )
            # Expose for users if they want to inspect
            self.mu, self.sigma, self.beta, self.tau, self.draw_probability = (
                self._env.mu, self._env.sigma, self._env.beta, self._env.tau, self._env.draw_probability
            )
        else:
            # Fallback parameters
            self.mu = mu
            self.sigma = sigma
            self.beta = beta
            self.tau = tau
            self.draw_probability = draw_probability
            # modest learning-rate scaled to sigma
            self._k = 1.0

    # ----- Common API -----
    def create(self):
        if self._use_lib:
            return self._env.Rating()
        else:
            return TSRating(mu=self.mu, sigma=self.sigma)

    def rate_1vs1(
        self,
        a,
        b,
        draw: bool = False,
        a_wins: Optional[bool] = None,
    ):
        """
        Update two ratings after a 1v1 match.
          - If draw=True: treat as draw.
          - Else: a_wins must be True/False.
        Returns (a_new, b_new).
        """
        if self._use_lib:
            if draw:
                # trueskill supports drawn 1vs1 directly
                return self._env.rate_1vs1(a, b, drawn=True)
            if a_wins is None:
                raise ValueError("TrueSkill.rate_1vs1: a_wins must be provided when draw=False.")
            if a_wins:
                return self._env.rate_1vs1(a, b, drawn=False)
            else:
                # If b wins, flip arguments then swap results
                rb, ra = self._env.rate_1vs1(b, a, drawn=False)
                return ra, rb

        # ---- Fallback update (no external library) ----
        if not isinstance(a, TSRating) or not isinstance(b, TSRating):
            raise TypeError("Fallback TrueSkill expects TSRating instances; use create() to initialize.")

        if draw:
            # Slight uncertainty shrink on draw
            a2 = TSRating(mu=a.mu, sigma=max(1e-3, a.sigma * 0.99))
            b2 = TSRating(mu=b.mu, sigma=max(1e-3, b.sigma * 0.99))
            return a2, b2

        if a_wins is None:
            raise ValueError("Fallback TrueSkill: a_wins must be provided when draw=False.")

        # Logistic expected score based on mu difference and beta
        diff = (a.mu - b.mu) / (math.sqrt(2.0) * max(1e-6, self.beta))
        exp_a = 1.0 / (1.0 + math.exp(-diff))
        score_a = 1.0 if a_wins else 0.0

        # Adjust learning rate by uncertainty (more uncertain -> bigger steps)
        k_a = self._k * (a.sigma / self.sigma)
        k_b = self._k * (b.sigma / self.sigma)

        a_mu = a.mu + k_a * (score_a - exp_a)
        b_mu = b.mu + k_b * ((1.0 - score_a) - (1.0 - exp_a))

        # Small dynamics / decay to avoid sigma collapsing to zero
        a_sig = max(1e-3, math.sqrt(a.sigma * a.sigma * 0.98 + self.tau * self.tau))
        b_sig = max(1e-3, math.sqrt(b.sigma * b.sigma * 0.98 + self.tau * self.tau))

        return TSRating(a_mu, a_sig), TSRating(b_mu, b_sig)
