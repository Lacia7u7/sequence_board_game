# training/utils/agent_win_meter.py
from __future__ import annotations
from collections import defaultdict
from typing import Dict, Iterable

class AgentWinMeter:
    """
    Acumula win-rate global y por CLASE de oponente (por nombre de clase).
    Úsalo desde train.py: update(opponent_classes, learner_won) y luego log_to(log, step).
    """
    def __init__(self) -> None:
        self._global_games = 0
        self._global_wins = 0
        self._per_class: Dict[str, Dict[str, int]] = defaultdict(lambda: {"games": 0, "wins": 0})

    def reset(self) -> None:
        self._global_games = 0
        self._global_wins = 0
        self._per_class.clear()

    def update(self, opponent_classes: Iterable[str], learner_won: bool) -> None:
        """Registra un episodio completo contra N oponentes (sus clases) y si ganó el learner."""
        self._global_games += 1
        self._global_wins += int(learner_won)
        for cls_name in opponent_classes:
            d = self._per_class[cls_name]
            d["games"] += 1
            d["wins"]  += int(learner_won)

    def scalars(self) -> Dict[str, float]:
        """Devuelve dict con métricas listas para loguear."""
        out: Dict[str, float] = {}
        if self._global_games > 0:
            out["eval/winrate_overall"] = self._global_wins / self._global_games
        for cls_name, d in self._per_class.items():
            if d["games"] > 0:
                out[f"eval/winrate_vs/{cls_name}"] = d["wins"] / d["games"]
        return out

    def log_to(self, log, step: int) -> None:
        """Escribe cada métrica en LoggingMux."""
        for k, v in self.scalars().items():
            log.scalar(k, float(v), step=step)

    # Opcional: texto corto para consola
    def short_text(self, top: int = 3) -> str:
        if self._global_games == 0:
            return "no episodes yet"
        parts = [f"overall {self._global_wins}/{self._global_games} ({self._global_wins/max(1,self._global_games):.1%})"]
        items = sorted(
            ((k, d["wins"], d["games"]) for k, d in self._per_class.items() if d["games"] > 0),
            key=lambda x: (x[1]/max(1,x[2])), reverse=False
        )[:top]
        parts += [f"{k}: {w}/{g} ({w/max(1,g):.1%})" for k, w, g in items]
        return " | ".join(parts)
