from __future__ import annotations
from collections import defaultdict, deque
from typing import Deque, Dict, Iterable, List, Tuple


def _wilson_interval(wins: int, games: int, z: float = 1.96) -> Tuple[float, float]:
    if games <= 0:
        return (0.0, 0.0)
    n = float(games)
    p = float(wins) / n
    z2 = z * z
    denom = 1.0 + z2 / (2.0 * n)
    centre = p + z2 / (2.0 * n)
    margin = z * ((p * (1.0 - p) + z2 / (4.0 * n)) / n) ** 0.5
    lo = max(0.0, (centre - margin) / denom)
    hi = min(1.0, (centre + margin) / denom)
    return (lo, hi)


class AgentWinMeter:
    """
    Tracks win-rate overall & by opponent class, with rolling window stats,
    EMA, Wilson CIs, and console summaries.

    Use from train.py:
      meter = AgentWinMeter(window=256, ema_alpha=0.10)
      meter.update(opponent_classes, learner_won)
      meter.log_to(log, step)
      print("[eval]", meter.console_line())
    """

    def __init__(
        self,
        window: int = 256,
        sparkline_len: int = 48,
        min_games_for_ranking: int = 5,
        ema_alpha: float = 0.10,
    ) -> None:
        # Cumulative
        self._global_games = 0
        self._global_wins = 0
        self._per_class: Dict[str, Dict[str, int]] = defaultdict(lambda: {"games": 0, "wins": 0})

        # Rolling (overall)
        self.window = int(max(1, window))
        self._recent: Deque[bool] = deque(maxlen=self.window)
        self._recent_wins = 0

        # Rolling (per-class)
        self._recent_by_class: Dict[str, Deque[bool]] = defaultdict(lambda: deque(maxlen=self.window))

        # EMA (overall + per-class)
        self.ema_alpha = float(max(1e-6, min(1.0, ema_alpha)))
        self._ema_overall = 0.5  # neutral start
        self._ema_by_class: Dict[str, float] = defaultdict(lambda: 0.5)

        # Console
        self.sparkline_len = int(max(8, sparkline_len))
        self.min_games_for_ranking = int(max(1, min_games_for_ranking))

    # -------- Core API --------

    def reset(self) -> None:
        self._global_games = 0
        self._global_wins = 0
        self._per_class.clear()
        self._recent.clear()
        self._recent_wins = 0
        self._recent_by_class.clear()
        self._ema_overall = 0.5
        self._ema_by_class.clear()

    def update(self, opponent_classes: Iterable[str], learner_won: bool) -> None:
        won = bool(learner_won)

        # Cumulative
        self._global_games += 1
        self._global_wins += int(won)
        for cls_name in opponent_classes:
            if "NoneType" in cls_name:
                continue
            d = self._per_class[cls_name]
            d["games"] += 1
            d["wins"] += int(won)

        # Rolling (overall)
        if len(self._recent) == self._recent.maxlen and self._recent and self._recent[0]:
            self._recent_wins -= 1
        self._recent.append(won)
        if won:
            self._recent_wins += 1

        # Rolling (per-class) + EMA
        a = self.ema_alpha
        self._ema_overall = (1 - a) * self._ema_overall + a * (1.0 if won else 0.0)
        for cls_name in opponent_classes:
            if "NoneType" in cls_name:
                continue
            dq = self._recent_by_class[cls_name]
            dq.append(won)
            self._ema_by_class[cls_name] = (1 - a) * self._ema_by_class[cls_name] + a * (1.0 if won else 0.0)

    # -------- Public metrics --------

    def scalars(self) -> Dict[str, float]:
        out: Dict[str, float] = {}

        # Overall cumulative + Wilson CI
        g = self._global_games
        w = self._global_wins
        if g > 0:
            out["eval/winrate_overall"] = w / g
            lo, hi = _wilson_interval(w, g)
            out["eval/winrate_overall_lo"] = lo
            out["eval/winrate_overall_hi"] = hi
        out["eval/games_total"] = float(g)

        # Overall recent + EMA
        if len(self._recent) > 0:
            out[f"eval/winrate_overall_recent{self.window}"] = self._recent_wins / len(self._recent)
        out["eval/winrate_overall_ema"] = float(self._ema_overall)
        out["eval/games_recent"] = float(len(self._recent))

        # Per-class cumulative, recent, EMA
        for cls_name, d in self._per_class.items():
            if d["games"] > 0:
                out[f"eval/winrate_vs/{cls_name}"] = d["wins"] / d["games"]
            dq = self._recent_by_class.get(cls_name)
            if dq and len(dq) > 0:
                out[f"eval/winrate_vs_recent/{cls_name}"] = sum(1 for x in dq if x) / len(dq)
            out[f"eval/winrate_vs_ema/{cls_name}"] = float(self._ema_by_class.get(cls_name, 0.5))
        return out

    def log_to(self, log, step: int) -> None:
        for k, v in self.scalars().items():
            log.scalar(k, float(v), step=step)

    # -------- Data helpers for curriculum --------

    def recent_winrates_by_class(self, min_games: int = 1) -> Dict[str, float]:
        out: Dict[str, float] = {}
        for cls, dq in self._recent_by_class.items():
            if len(dq) >= min_games:
                out[cls] = sum(1 for x in dq if x) / len(dq)
        return out

    def ema_winrates_by_class(self) -> Dict[str, float]:
        return dict(self._ema_by_class)

    # -------- Console --------

    def short_text(self, top: int = 3) -> str:
        return self.console_line(top_hard=top, top_easy=top)

    def console_line(self, top_hard: int = 3, top_easy: int = 1) -> str:
        if self._global_games == 0:
            return "no episodes yet"

        g, w = self._global_games, self._global_wins
        overall = f"overall {w}/{g} ({(w/max(1,g)):.1%})"
        recent_rate = (self._recent_wins / max(1, len(self._recent))) if self._recent else 0.0
        ema_rate = self._ema_overall
        recent = f"recent@{self.window} {recent_rate:.1%} | ema {ema_rate:.1%}"

        spark_samples = list(self._recent)[-self.sparkline_len:] if self._recent else []
        spark = "".join("█" if x else "·" for x in spark_samples) or "(no recent)"

        # Rank opponents by recent winrate (hardest first)
        rank_items: List[Tuple[str, float, int]] = []
        for k, dq in self._recent_by_class.items():
            n = len(dq)
            if n >= self.min_games_for_ranking:
                p = sum(1 for x in dq if x) / n
                rank_items.append((k, p, n))
        rank_items.sort(key=lambda x: x[1])  # hardest first

        hard_txt = ""
        if top_hard > 0 and rank_items:
            chunk = ", ".join(f"{k} {p:.0%}" for k, p, _ in rank_items[:top_hard])
            hard_txt = f" | hard: {chunk}"

        easy_txt = ""
        if top_easy > 0 and rank_items:
            chunk = ", ".join(f"{k} {p:.0%}" for k, p, _ in rank_items[-top_easy:])
            easy_txt = f" | easy: {chunk}"

        return f"{overall} | {recent} {spark}{hard_txt}{easy_txt}"
