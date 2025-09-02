from __future__ import annotations

from typing import Any, Dict, Optional, List, Tuple
import numpy as np

from ..engine.engine_core import GameEngine, EngineError, is_two_eyed_jack, is_one_eyed_jack
from ..engine.state import GameConfig
from ..engine.board_layout import BOARD_LAYOUT
from ..encoders.board_encoder import encode_board_state

from ..utils.legal_utils import normalize_legal, build_action_mask


class SequenceEnv:
    """
    Gymnasium-like single-agent env around engine_core.GameEngine with reward shaping.

    Unified action space:
      0..99                  -> board cells (row-major)
      100..100+H-1           -> discard hand index (H = max_hand)
      100+H                  -> PASS (if enabled)

    Rewards (configurable via config["rewards"]):
      - illegal:   rewards.illegal             (default -0.01)
      - win:       rewards.win                 (default +1.0)
      - loss:      rewards.loss                (default -1.0)
      - seq bonus: rewards.seq_bonus * (#new sequences for acting team)
      - shaping:   potential-based γ·Φ(s') − Φ(s) over normalized features
                   (Φ includes open{2,3,4}, immediate win, forks, run, hot/center/corner,
                    protected, jack_vuln, coverage contamination, mobility, etc.)
      - optional per-step penalty: rewards.step_penalty (default 0.0)

    Performance (optional):
      config["performance"] = {
        "use_torch_features": true/false,   # default False
        "device": "cuda"|"cpu",             # default "cuda"
        "dtype": "float16"|"float32"        # default "float16"
      }
    """

    metadata = {"render.modes": ["human"]}

    def __init__(self, config: Dict[str, Any]):
        self.config: Dict[str, Any] = config
        self.gconf: GameConfig = GameConfig.from_dict(config) if isinstance(config, dict) else config
        self.game_engine: GameEngine = GameEngine()
        self.current_player: int = 0

        # Action space
        self.max_hand: int = int(config.get("action_space", {}).get("max_hand", 7))
        include_pass = bool(config.get("action_space", {}).get("include_pass", False))
        self._include_pass: bfool = include_pass
        self.action_dim: int = 100 + self.max_hand + (1 if include_pass else 0)

        # Episode cap
        self._step_limit: int = int(getattr(self.gconf, "episode_cap", 400))
        self._steps_elapsed: int = 0
        self._global_step: int = 0  # for annealing across episodes

        # Discount (used for potential-based shaping)
        self.discount_gamma: float = float(config.get("training", {}).get("gamma", 0.99))

        # Reward coefficients (base)
        rw = dict(config.get("rewards", {}) or {})
        self.R_ILLEGAL      = float(rw.get("illegal", -0.01))
        self.R_WIN          = float(rw.get("win", 1.0))
        self.R_LOSS         = float(rw.get("loss", -1.0))
        self.R_SEQ_BONUS    = float(rw.get("seq_bonus", 0.3))
        self.R_STEP_PENALTY = float(rw.get("step_penalty", 0.0))

        # Kept for compatibility (weights feed into Φ):
        self.W_OPEN4_SELF   = float(rw.get("open4_self", 0.05))
        self.W_OPEN4_OPP    = float(rw.get("open4_opp", 0.05))
        self.W_OPEN3_SELF   = float(rw.get("open3_self", 0.01))
        self.W_OPEN3_OPP    = float(rw.get("open3_opp", 0.01))

        # Extra rewards (weights feed into Φ and a few action-conditional extras)
        rx = dict(config.get("rewards_extra", {}) or {})
        # Threats & tactics
        self.W_IMM_WIN_SELF       = float(rx.get("imm_win_self",       0.10))
        self.W_IMM_WIN_OPP        = float(rx.get("imm_win_opp",        0.10))
        self.W_FORK_SELF          = float(rx.get("fork_self",          0.10))
        self.W_FORK_OPP           = float(rx.get("fork_opp",           0.10))
        self.W_RUN_SELF           = float(rx.get("run_self",           0.02))
        self.W_RUN_OPP            = float(rx.get("run_opp",            0.02))
        self.W_OPEN2_SELF         = float(rx.get("open2_self",         0.01))
        self.W_OPEN2_OPP          = float(rx.get("open2_opp",          0.01))
        self.W_CLOSED4_BREAKER    = float(rx.get("closed4_breaker",    0.05))  # extra (action-conditional proxy)
        self.W_JACK_REMOVE_IMPACT = float(rx.get("jack_remove_impact", 0.05))  # extra
        self.W_WILD_EFFICIENCY    = float(rx.get("wild_efficiency",    0.05))  # extra
        # Board control
        self.W_HOT_CONTROL_SELF   = float(rx.get("hot_control_self",   0.005))
        self.W_HOT_CONTROL_OPP    = float(rx.get("hot_control_opp",    0.005))
        self.W_CENTER_CONTROL     = float(rx.get("center_control",     0.01))
        self.W_CORNER_SYNERGY     = float(rx.get("corner_synergy",     0.01))
        self.W_COVERAGE_CUT       = float(rx.get("coverage_cut",       0.01))
        self.W_REDUNDANCY_PENALTY = float(rx.get("redundancy_penalty", 0.05))  # extra
        # Safety / robustness
        self.W_PROTECTED_CHIPS    = float(rx.get("protected_chips",    0.03))
        self.W_JACK_VULN_SELF     = float(rx.get("jack_vuln_self",     0.02))
        self.W_JACK_VULN_OPP      = float(rx.get("jack_vuln_opp",      0.02))
        self.W_BLUNDER            = float(rx.get("blunder",            0.10))   # extra
        # Tempo / mano
        self.W_MOBILITY           = float(rx.get("mobility",           0.001))
        self.W_DISCARD_EV         = float(rx.get("discard_ev",         0.01))   # extra
        self.EARLY_FINISH_BONUS   = float(rx.get("early_finish_bonus", 0.50))   # extra (terminal-only)

        # Shaping meta (clips & annealing)
        sh = dict(config.get("shaping", {}) or {})
        self.SHAPING_USE_POTENTIAL   = bool(sh.get("use_potential", True))
        self.SHAPING_PER_TERM_CLIP   = float(sh.get("per_term_clip", 0.25))
        self.SHAPING_GLOBAL_CLIP     = float(sh.get("global_clip", 0.5))
        self.SHAPING_ANNEAL_TOTAL    = int(sh.get("anneal_total_env_steps", 0))   # 0 = no anneal
        self.SHAPING_ANNEAL_MIN      = float(sh.get("anneal_min_scale", 0.0))     # floor of anneal

        # Static caches for windows / square weights
        self._WINDOWS_CACHE = None
        self._SQUARE_WEIGHT = None
        self._CENTER_SET    = {(r, c) for r in range(3, 7) for c in range(3, 7)}  # center 4x4
        self._BONUS_CELLS   = {(0,0),(0,9),(9,0),(9,9)}  # BONUS corners

        # ---- Fast feature backend (Torch/NumPy + card masks for mobility/EV) ----
        perf = dict(config.get("performance", {}) or {})
        self._fast_use_torch = bool(perf.get("use_torch_features", False))
        self._fast_device = str(perf.get("device", "cuda"))
        self._fast_dtype = str(perf.get("dtype", "float16"))
        self._FAST_READY = False
        self._init_fast_backend()

    # ---------------- Gym-like API ----------------

    def reset(self, seed: Optional[int] = None):
        if seed is not None:
            try:
                self.game_engine.seed(int(seed))
            except Exception:
                pass

        self.game_engine.start_new(self.gconf)
        self._steps_elapsed = 0
        self.current_player = self._seat_index()

        legal = self._legal_for(self.current_player)
        obs = encode_board_state(
            self._state(),
            self.current_player,
            self.config,
            legal=legal,
            public=self._public_summary(),
        )
        info = {
            "current_player": self.current_player,
            "legal_mask": self._legal_mask(legal),
        }
        return obs, info

    def get_obs(self):
        return encode_board_state(
            self._state(),
            self.current_player,
            self.config,
            legal=self._legal_for(self.current_player),
            public=self._public_summary(),
        )

    def step(self, action: int, fast_run: bool = False):
        """
        fast_run=True: ejecuta la jugada en el engine y devuelve obs/terminated/truncated/info,
                       pero NO calcula recompensas ni shaping pesados. reward=None.
        fast_run=False (por defecto): comportamiento completo (rewards + breakdown).
        """
        st_before = self._state()
        acting_player = self.current_player
        acting_team = self._player_team(acting_player)

        # --- LEGALIDAD (siempre necesaria) ---
        legal = self._legal_for(self.current_player)
        mask = self._legal_mask(legal)

        # ILEGAL
        if not (0 <= action < self.action_dim) or mask[action] < 0.5:
            obs = encode_board_state(
                st_before, self.current_player, self.config, legal=legal, public=self._public_summary()
            )
            info = {"current_player": self.current_player, "legal_mask": mask, "illegal": True}
            if fast_run:
                # mismo shape del tuple, pero reward=None y sin breakdown pesado
                return obs, None, False, False, info
            else:
                return obs, float(self.R_ILLEGAL), False, False, info

        # --- MAPEO DE ACCIÓN A MOVE ---
        if action < 100:
            r, c = divmod(action, 10)
            move = self._resolve_cell_action(self.current_player, r, c)
        else:
            idx = action - 100
            if self._include_pass and idx == self.max_hand:
                move = {"type": "timeout-skip", "card": None, "target": None, "removed": None}
            else:
                move = self._resolve_discard_action(self.current_player, idx)

        # --- APLICAR EN ENGINE ---
        try:
            st_after, move_record = self.game_engine.step(move)
        except EngineError:
            obs = encode_board_state(
                st_before, self.current_player, self.config, legal=legal, public=self._public_summary()
            )
            info = {"current_player": self.current_player, "legal_mask": mask, "engine_reject": True}
            if fast_run:
                return obs, None, False, False, info
            else:
                return obs, float(self.R_ILLEGAL), False, False, info

        # --- TERMINAL? (siempre lo calculamos; útil incluso en fast_run) ---
        winners = (
            self.game_engine.winner_teams()
            if hasattr(self.game_engine, "winner_teams")
            else (list(getattr(st_after, "winners", [])) if st_after else [])
        )
        terminated = bool(winners)

        # --- SI ES FAST RUN: OMITIR CUALQUIER CÓMPUTO PESADO DE REWARD ---
        if fast_run:
            # Siguiente turno / obs / legal
            self.current_player = self._seat_index()
            next_legal = self._legal_for(self.current_player)
            obs = encode_board_state(
                self._state(), self.current_player, self.config, legal=next_legal, public=self._public_summary()
            )

            # Contadores y truncation por límite de pasos (mantenemos coherencia del episodio)
            self._steps_elapsed += 1
            self._global_step += 1
            truncated = False
            if not terminated and self._step_limit > 0 and self._steps_elapsed >= self._step_limit:
                truncated = True

            info = {
                "current_player": self.current_player,
                "legal_mask": self._legal_mask(next_legal),
                "move": move_record,
                "winners": winners,
                "fast_run": True,  # bandera para que el caller sepa que no hay reward
            }
            # reward=None para indicar explícitamente que no se calculó
            return obs, None, terminated, truncated, info

        # ---------- MODO COMPLETO (recompensas + shaping) ----------
        # BEFORE features (un solo escaneo pesado)
        seq_count_before = int(self._sequences_for_team(st_before, acting_team))
        feats_b = self._features_bundle(st_before, acting_team)
        self_b, opp_b = feats_b["self"], feats_b["opp"]
        coverage_clean_b, mobility_b = feats_b["coverage_clean"], feats_b["mobility"]

        # AFTER features
        seq_count_after = int(self._sequences_for_team(st_after, acting_team))
        feats_a = self._features_bundle(st_after, acting_team)
        self_a, opp_a = feats_a["self"], feats_a["opp"]
        coverage_clean_a, mobility_a = feats_a["coverage_clean"], feats_a["mobility"]

        # Reward assembly
        reward = 0.0

        # Terminal outcome
        if terminated:
            reward += self.R_WIN if acting_team in winners else self.R_LOSS

        # Sequence creation bonus
        seq_delta = max(0, seq_count_after - seq_count_before)
        if seq_delta > 0:
            reward += self.R_SEQ_BONUS * float(seq_delta)

        # Optional per-step penalty
        reward += self.R_STEP_PENALTY

        # Potential-based shaping
        phi_breakdown, phi_sum_clipped, phi_sum_raw, anneal_scale = self._potential_shaping(
            self_b, opp_b, coverage_clean_b, mobility_b,
            self_a, opp_a, coverage_clean_a, mobility_a
        )
        reward += phi_sum_clipped

        # -------- Extras (clipped) --------
        # closed4_breaker
        contrib_closed4 = self.W_CLOSED4_BREAKER * (
                    (self_a["open4"] - self_b["open4"]) + max(0, seq_count_after - seq_count_before))
        contrib_closed4 = float(np.clip(contrib_closed4, -self.SHAPING_PER_TERM_CLIP, self.SHAPING_PER_TERM_CLIP))
        reward += contrib_closed4

        # jack-remove impact
        contrib_jackrem = 0.0
        if isinstance(move_record, dict) and move_record.get("type") == "jack-remove":
            opp_open_b = opp_b["open4"] + opp_b["open3"]
            opp_open_a = opp_a["open4"] + opp_a["open3"]
            contrib_jackrem = self.W_JACK_REMOVE_IMPACT * (opp_open_b - opp_open_a)
            contrib_jackrem = float(np.clip(contrib_jackrem, -self.SHAPING_PER_TERM_CLIP, self.SHAPING_PER_TERM_CLIP))
            reward += contrib_jackrem

        # wild efficiency
        contrib_wild = 0.0
        if isinstance(move_record, dict) and move_record.get("type") == "wild":
            gain = (self_a["open4"] - self_b["open4"]) + 2 * max(0, seq_count_after - seq_count_before)
            if gain >= 2:
                contrib_wild = self.W_WILD_EFFICIENCY * float(gain)
            else:
                contrib_wild = -0.5 * self.W_WILD_EFFICIENCY
            contrib_wild = float(np.clip(contrib_wild, -self.SHAPING_PER_TERM_CLIP, self.SHAPING_PER_TERM_CLIP))
            reward += contrib_wild

        # redundancy penalty
        contrib_redundant = 0.0
        redundant_flag = 0
        if isinstance(move_record, dict) and move_record.get("target") is not None:
            tr = move_record["target"].get("r")
            tc = move_record["target"].get("c")
            if tr is not None and tc is not None:
                improved_threats = (self_a["open4"] + self_a["open3"] + self_a["open2"]) - (
                            self_b["open4"] + self_b["open3"] + self_b["open2"])
                if improved_threats <= 0 and (seq_count_after - seq_count_before) <= 0:
                    self._ensure_windows()
                    board_after = st_after.board  # type: ignore
                    in_complete = False
                    for w in self._WINDOWS_CACHE:
                        if (tr, tc) in w:
                            s = 0
                            for (rr, cc) in w:
                                printed = BOARD_LAYOUT[rr][cc]
                                chip = board_after[rr][cc]
                                if printed == "BONUS" or chip == acting_team:
                                    s += 1
                            if s == 5:
                                in_complete = True
                                break
                    if in_complete:
                        redundant_flag = 1
                        contrib_redundant = - self.W_REDUNDANCY_PENALTY
                        contrib_redundant = float(
                            np.clip(contrib_redundant, -self.SHAPING_PER_TERM_CLIP, self.SHAPING_PER_TERM_CLIP))
                        reward += contrib_redundant

        # blunder
        blunder_delta = (len(opp_a["imm_win_cells"]) - len(opp_b["imm_win_cells"]))
        contrib_blunder = 0.0
        if blunder_delta > 0:
            contrib_blunder = - self.W_BLUNDER * float(blunder_delta)
            contrib_blunder = float(np.clip(contrib_blunder, -self.SHAPING_PER_TERM_CLIP, self.SHAPING_PER_TERM_CLIP))
            reward += contrib_blunder

        # discard EV (solo en burn)
        contrib_disc_ev = 0.0
        if isinstance(move_record, dict) and move_record.get("type") == "burn":
            public_b = self._public_summary()
            deck_b = public_b.get("deck_counts", {})
            tot_b = max(1, int(public_b.get("total_remaining", 1)))
            hand_b = self._hand_for(acting_player)

            base_playable, union_mask, empties_mask, has_twoeyed = self._mobility_base_union(st_before, hand_b)
            ev_after = 0.0
            if not has_twoeyed and tot_b > 0:
                for card, cnt in deck_b.items():
                    if cnt <= 0 or card in hand_b:
                        continue
                    cm = self._CARD_TO_MASK.get(card, None)
                    if cm is None:
                        continue
                    inc = int(np.count_nonzero(cm & empties_mask & (~union_mask)))
                    ev_after += (cnt / tot_b) * inc

            contrib_disc_ev = self.W_DISCARD_EV * float(ev_after)
            contrib_disc_ev = float(np.clip(contrib_disc_ev, -self.SHAPING_PER_TERM_CLIP, self.SHAPING_PER_TERM_CLIP))
            reward += contrib_disc_ev

        # Early finish bonus
        contrib_early = 0.0
        if terminated and (acting_team in winners) and self._step_limit > 0 and self.EARLY_FINISH_BONUS != 0.0:
            frac = 1.0 - min(1.0, float(self._steps_elapsed) / float(self._step_limit))
            contrib_early = self.R_WIN * self.EARLY_FINISH_BONUS * frac
            reward += contrib_early

        # --- Siguiente obs ---
        self.current_player = self._seat_index()
        next_legal = self._legal_for(self.current_player)
        obs = encode_board_state(
            self._state(), self.current_player, self.config, legal=next_legal, public=self._public_summary()
        )

        # --- Truncation por step cap ---
        self._steps_elapsed += 1
        self._global_step += 1
        truncated = False
        if not terminated and self._step_limit > 0 and self._steps_elapsed >= self._step_limit:
            truncated = True

        # --- Breakdown detallado (solo en modo completo) ---
        info = {
            "current_player": self.current_player,
            "legal_mask": self._legal_mask(next_legal),
            "move": move_record,
            "winners": winners,
            "reward_breakdown": {
                "terminal": (
                    self.R_WIN if terminated and acting_team in winners else (self.R_LOSS if terminated else 0.0)),
                "seq_bonus": self.R_SEQ_BONUS * float(seq_delta),
                "step_penalty": float(self.R_STEP_PENALTY),
                "phi_sum_raw": float(phi_sum_raw),
                "phi_sum_clipped": float(phi_sum_clipped),
                "phi_anneal_scale": float(anneal_scale),
                "closed4_breaker": float(contrib_closed4),
                "jack_remove_impact": float(contrib_jackrem),
                "wild_efficiency": float(contrib_wild),
                "redundant_move": int(redundant_flag),
                "redundancy_penalty": float(contrib_redundant),
                "blunder": float(contrib_blunder),
                "discard_ev": float(contrib_disc_ev),
                "early_finish_bonus": float(contrib_early),
            }
        }
        info["reward_breakdown"].update({
            "phi_open4_self": float(phi_breakdown.get("open4_self", 0.0)),
            "phi_open4_opp": float(phi_breakdown.get("open4_opp", 0.0)),
            "phi_open3_self": float(phi_breakdown.get("open3_self", 0.0)),
            "phi_open3_opp": float(phi_breakdown.get("open3_opp", 0.0)),
            "phi_imm_win_self": float(phi_breakdown.get("imm_win_self", 0.0)),
            "phi_imm_win_opp": float(phi_breakdown.get("imm_win_opp", 0.0)),
            "phi_fork_self": float(phi_breakdown.get("fork_self", 0.0)),
            "phi_fork_opp": float(phi_breakdown.get("fork_opp", 0.0)),
            "phi_coverage": float(phi_breakdown.get("coverage_contam", 0.0)),
            "phi_mobility": float(phi_breakdown.get("mobility", 0.0)),
        })

        return obs, float(reward), terminated, truncated, info

    # ---------------- Potential shaping helpers ----------------

    def _anneal_scale(self) -> float:
        if self.SHAPING_ANNEAL_TOTAL <= 0:
            return 1.0
        prog = min(1.0, self._global_step / float(self.SHAPING_ANNEAL_TOTAL))
        return max(self.SHAPING_ANNEAL_MIN, 1.0 - prog)

    def _potential_shaping(
        self,
        self_b: Dict[str, Any], opp_b: Dict[str, Any], coverage_clean_b: int, mobility_b: float,
        self_a: Dict[str, Any], opp_a: Dict[str, Any], coverage_clean_a: int, mobility_a: float
    ):
        """
        Compute per-term contributions of γ·Φ(s') − Φ(s) with normalization + per-term & global clips.
        Returns (per_term_breakdown, clipped_sum, raw_sum, anneal_scale).
        """
        # Normalize both states
        nb = self._normalized_features(self_b, opp_b, coverage_clean_b, mobility_b)
        na = self._normalized_features(self_a, opp_a, coverage_clean_a, mobility_a)

        # Signed weights (opp terms negative by design)
        w = {
            "open4_self":  self.W_OPEN4_SELF,    "open4_opp":  -self.W_OPEN4_OPP,
            "open3_self":  self.W_OPEN3_SELF,    "open3_opp":  -self.W_OPEN3_OPP,
            "open2_self":  self.W_OPEN2_SELF,    "open2_opp":  -self.W_OPEN2_OPP,
            "imm_win_self": self.W_IMM_WIN_SELF, "imm_win_opp": -self.W_IMM_WIN_OPP,
            "fork_self":    self.W_FORK_SELF,    "fork_opp":    -self.W_FORK_OPP,
            "run_self":     self.W_RUN_SELF,     "run_opp":     -self.W_RUN_OPP,
            "hot_self":     self.W_HOT_CONTROL_SELF, "hot_opp": -self.W_HOT_CONTROL_OPP,
            "center":       self.W_CENTER_CONTROL,
            "corner_adj":   self.W_CORNER_SYNERGY,
            "protected":    self.W_PROTECTED_CHIPS,
            "jack_vuln_self": -self.W_JACK_VULN_SELF,
            "jack_vuln_opp":   self.W_JACK_VULN_OPP,
            "coverage_contam": self.W_COVERAGE_CUT,
            "mobility":        self.W_MOBILITY,
        }

        # Compute per-term contributions with potential form
        per_term = {}
        raw_sum = 0.0
        for k, wk in w.items():
            before = nb.get(k, 0.0)
            after  = na.get(k, 0.0)
            contrib = wk * (self.discount_gamma * after - before)
            per_term[k] = float(np.clip(contrib, -self.SHAPING_PER_TERM_CLIP, self.SHAPING_PER_TERM_CLIP))
            raw_sum += contrib

        # Global clip + anneal
        clipped_sum = float(np.clip(sum(per_term.values()), -self.SHAPING_GLOBAL_CLIP, self.SHAPING_GLOBAL_CLIP))
        scale = self._anneal_scale()
        return per_term, clipped_sum * scale, raw_sum, scale

    def _normalized_features(self, self_f: Dict[str, Any], opp_f: Dict[str, Any], coverage_clean: int, mobility: float) -> Dict[str, float]:
        """
        Map raw feature dicts to ~[0,1] ranges.
        """
        self._ensure_windows()
        WINDOWS_TOTAL = len(self._WINDOWS_CACHE)  # 192 on 10x10 with len-5
        EMPTIES_DEN   = 96.0                      # playable non-BONUS cells
        HOT_DEN       = WINDOWS_TOTAL * 5.0       # each window contributes 5 cells => 960

        def div(a, b):  # safe divide
            return float(a) / float(b) if b > 0 else 0.0

        imm_self_cnt = len(self_f.get("imm_win_cells", []))
        imm_opp_cnt  = len(opp_f.get("imm_win_cells", []))
        # Backward/fast-path compatibility: allow providing counts directly
        imm_self_cnt = int(self_f.get("imm_win_count", imm_self_cnt))
        imm_opp_cnt  = int(opp_f.get("imm_win_count",  imm_opp_cnt))

        feat = {
            # open counts (per-window)
            "open4_self":  div(self_f.get("open4", 0), WINDOWS_TOTAL),
            "open4_opp":   div(opp_f.get("open4", 0), WINDOWS_TOTAL),
            "open3_self":  div(self_f.get("open3", 0), WINDOWS_TOTAL),
            "open3_opp":   div(opp_f.get("open3", 0), WINDOWS_TOTAL),
            "open2_self":  div(self_f.get("open2", 0), WINDOWS_TOTAL),
            "open2_opp":   div(opp_f.get("open2", 0), WINDOWS_TOTAL),
            # immediate wins / forks
            "imm_win_self": div(imm_self_cnt, EMPTIES_DEN),
            "imm_win_opp":  div(imm_opp_cnt,  EMPTIES_DEN),
            "fork_self":    div(self_f.get("forks", 0), EMPTIES_DEN),
            "fork_opp":     div(opp_f.get("forks", 0), EMPTIES_DEN),
            # runs
            "run_self":     div(self_f.get("max_run", 0), 5.0),
            "run_opp":      div(opp_f.get("max_run", 0), 5.0),
            # spatial
            "hot_self":     div(self_f.get("hot_control", 0), HOT_DEN),
            "hot_opp":      div(opp_f.get("hot_control", 0), HOT_DEN),
            "center":       div(self_f.get("center_count", 0), 16.0),
            "corner_adj":   div(self_f.get("corner_adj", 0), 20.0),
            # safety
            "protected":      div(self_f.get("protected_chips", 0), EMPTIES_DEN),
            "jack_vuln_self": div(self_f.get("jack_vuln", 0), WINDOWS_TOTAL),
            "jack_vuln_opp":  div(opp_f.get("jack_vuln", 0), WINDOWS_TOTAL),
            # opponent windows contaminated by our chips
            "coverage_contam": 1.0 - div(coverage_clean, WINDOWS_TOTAL),
            # tempo
            "mobility": div(mobility, EMPTIES_DEN),
        }
        # Clamp to [0,1] defensively
        for k in list(feat.keys()):
            feat[k] = float(np.clip(feat[k], 0.0, 1.0))
        return feat

    # ---------------- Helpers: legality & mapping ----------------

    def _legal_for(self, seat: int) -> Dict[str, Any]:
        try:
            legal = self.game_engine.legal_actions_for(seat)
            if isinstance(legal, dict):
                return legal
        except Exception:
            pass
        return {}

    def _legal_mask(self, legal: Dict[str, Any]) -> np.ndarray:
        canon = normalize_legal(
            legal,
            board_h=10, board_w=10,
            max_hand=self.max_hand,
            include_pass=self._include_pass,
            union_place_remove_for_targets=True,
        )
        return build_action_mask(
            self.action_dim,
            board_h=10, board_w=10,
            max_hand=self.max_hand,
            include_pass=self._include_pass,
            canon_legal=canon,
        )

    def _resolve_cell_action(self, seat: int, r: int, c: int) -> Dict[str, Any]:
        # Prefer engine-native resolver if present
        if hasattr(self.game_engine, "resolve_cell_action"):
            try:
                return self.game_engine.resolve_cell_action(seat, r, c)  # type: ignore
            except Exception:
                pass

        board_cell = self._cell(r, c)
        printed = BOARD_LAYOUT[r][c]
        my_team = self._player_team(seat)
        hand = self._hand_for(seat)

        # Opponent chip present?
        if board_cell is not None and isinstance(board_cell, int) and board_cell != my_team:
            for card in hand:
                if is_one_eyed_jack(card):
                    return {"type": "jack-remove", "card": card, "target": None, "removed": {"r": r, "c": c}}

        # Empty placement
        is_empty = (board_cell is None) and (printed != "BONUS")
        if is_empty:
            # exact printed card
            for card in hand:
                if card == printed:
                    return {"type": "play", "card": card, "target": {"r": r, "c": c}, "removed": None}
            # two-eyed jack wild
            for card in hand:
                if is_two_eyed_jack(card):
                    return {"type": "wild", "card": card, "target": {"r": r, "c": c}, "removed": None}

        # fallback to burn first discard slot if any
        disc = self._first_discard_slot(seat)
        if disc is not None:
            return {"type": "burn", "card": hand[disc], "target": None, "removed": None}

        return {"type": "timeout-skip", "card": None, "target": None, "removed": None}

    def _resolve_discard_action(self, seat: int, hand_index: int) -> Dict[str, Any]:
        if hasattr(self.game_engine, "resolve_discard_action"):
            try:
                return self.game_engine.resolve_discard_action(seat, hand_index)  # type: ignore
            except Exception:
                pass
        hand = self._hand_for(seat)
        if 0 <= hand_index < len(hand):
            return {"type": "burn", "card": hand[hand_index], "target": None, "removed": None}
        return {"type": "timeout-skip", "card": None, "target": None, "removed": None}

    # ---------------- Feature extraction (fast paths + fallbacks) ----------------

    def _init_fast_backend(self):
        """Prepare window indices, static masks, and per-card masks for vectorized feature computation."""
        # Precompute windows and weights (CPU)
        self._ensure_windows()

        # Flattened indices for each 5-cell window: (W,5)
        idx = []
        for w in self._WINDOWS_CACHE:
            idx.append([r*10 + c for (r, c) in w])
        self._WIN_IDX = np.asarray(idx, dtype=np.int64)  # (W,5)

        # Static flattened masks (100,)
        bonus = np.zeros((100,), dtype=np.bool_)
        center = np.zeros((100,), dtype=np.bool_)
        adj_bonus = np.zeros((100,), dtype=np.bool_)
        sqw = np.zeros((100,), dtype=np.float32)
        printed_flat = []

        for r in range(10):
            for c in range(10):
                f = r*10 + c
                printed = BOARD_LAYOUT[r][c]
                printed_flat.append(printed)
                if printed == "BONUS":
                    bonus[f] = True
                if (r, c) in self._CENTER_SET:
                    center[f] = True
                # adjacency to any BONUS
                for (rr, cc) in self._neighbors8(r, c):
                    if BOARD_LAYOUT[rr][cc] == "BONUS":
                        adj_bonus[f] = True
                        break
                sqw[f] = float(self._SQUARE_WEIGHT[(r, c)])

        self._PRINTED_FLAT = np.array(printed_flat, dtype=object)

        # Build per-card boolean masks (non-BONUS cells only)
        self._CARD_TO_MASK: Dict[str, np.ndarray] = {}
        for s in "SHDC":
            for r in ["A","2","3","4","5","6","7","8","9","10","J","Q","K"]:
                card = f"{r}{s}"
                self._CARD_TO_MASK[card] = (self._PRINTED_FLAT == card)

        # Save NumPy versions
        self._NP_WIN_IDX = self._WIN_IDX
        self._NP_BONUS = bonus
        self._NP_CENTER = center
        self._NP_ADJ_BONUS = adj_bonus
        self._NP_SQW = sqw

        # Try Torch tensors on chosen device
        self._FAST_TORCH = None
        if self._fast_use_torch:
            try:
                import torch
                # choose device
                if "cuda" in self._fast_device and torch.cuda.is_available():
                    device = torch.device("cuda")
                else:
                    device = torch.device("cpu")
                dtypf = torch.float16 if self._fast_dtype == "float16" else torch.float32
                self._T_device = device
                self._T_float = dtypf
                self._T_WIN_IDX   = torch.as_tensor(self._WIN_IDX, device=device, dtype=torch.long)   # (W,5)
                self._T_BONUS     = torch.as_tensor(self._NP_BONUS, device=device, dtype=torch.bool)  # (100,)
                self._T_CENTER    = torch.as_tensor(self._NP_CENTER, device=device, dtype=torch.bool) # (100,)
                self._T_ADJ_BONUS = torch.as_tensor(self._NP_ADJ_BONUS, device=device, dtype=torch.bool)
                self._T_SQW       = torch.as_tensor(self._NP_SQW, device=device, dtype=dtypf)

                self._FAST_TORCH = True
            except Exception:
                self._FAST_TORCH = False

        self._FAST_READY = True

    def _board_flat_numpy(self, st) -> np.ndarray:
        """Flatten board to array (100,) with -1 for empty else team id."""
        arr = np.empty((100,), dtype=np.int16)
        i = 0
        b = st.board  # 10x10
        for r in range(10):
            row = b[r]
            for c in range(10):
                x = row[c]
                arr[i] = -1 if x is None else int(x)
                i += 1
        return arr

    def _team_features_fast(self, st, team: int):
        """CUDA/NumPy vectorized replacement for _team_features(). Returns same-structure dict."""
        if not self._FAST_READY:
            return self._team_features(st, team)

        flat = self._board_flat_numpy(st)  # (100,)

        # Torch path
        if self._FAST_TORCH:
            import torch
            flat_t = torch.as_tensor(flat, device=self._T_device)

            team_plane  = (flat_t == team)       # (100,)
            empty_plane = (flat_t == -1)
            opp_plane   = (~empty_plane) & (~team_plane)

            sat_plane = team_plane | self._T_BONUS            # satisfied wrt team
            sat_w = sat_plane[self._T_WIN_IDX]                # (W,5)
            emp_w = empty_plane[self._T_WIN_IDX]              # (W,5)
            opp_w = opp_plane[self._T_WIN_IDX]                # (W,5)

            sat_cnt = sat_w.sum(dim=1)                        # (W,)
            emp_cnt = emp_w.sum(dim=1)
            opp_cnt = opp_w.sum(dim=1)

            open4_mask = (opp_cnt == 0) & (sat_cnt == 4) & (emp_cnt == 1)
            open3_mask = (opp_cnt == 0) & (sat_cnt == 3) & (emp_cnt == 2)
            open2_mask = (opp_cnt == 0) & (sat_cnt == 2) & (emp_cnt == 3)
            open1_mask = (opp_cnt == 0) & (sat_cnt == 1) & (emp_cnt == 4)

            open4 = int(open4_mask.sum().item())
            open3 = int(open3_mask.sum().item())
            open2 = int(open2_mask.sum().item())
            open1 = int(open1_mask.sum().item())

            # immediate win cells (set of (r,c)) and forks count
            if open4 > 0:
                emp_rows  = emp_w[open4_mask]                # (N,5) bool
                empty_pos = torch.argmax(emp_rows.int(), dim=1)     # (N,)
                win_rows  = self._T_WIN_IDX[open4_mask]      # (N,5)
                win_flat  = torch.gather(win_rows, 1, empty_pos.view(-1,1)).squeeze(1)  # (N,)
                uniq = torch.unique(win_flat)
                imm_cnt = int(uniq.numel())
                imm_set = {(int(f.item())//10, int(f.item())%10) for f in uniq}
                counts = torch.bincount(win_flat, minlength=100)
                forks = int((counts >= 2).sum().item())
            else:
                imm_cnt = 0
                imm_set = set()
                forks = 0

            # max_run (contiguous satisfied within window)
            max_run = 0
            if sat_w.numel() > 0:
                v = sat_w.float().unsqueeze(1)               # (W,1,5)
                for k in (5,4,3,2,1):
                    ker = torch.ones((1,1,k), device=self._T_device)
                    conv = torch.nn.functional.conv1d(v, ker)
                    if (conv == k).any():
                        max_run = k
                        break

            # protected chips: own chip in fully satisfied window
            prot_positions = None
            prot = 0
            mask5 = (sat_cnt == 5)
            if mask5.any():
                win5 = self._T_WIN_IDX[mask5]                # (M,5)
                own_in5 = (flat_t[win5] == team)             # bool (M,5)
                if own_in5.any():
                    prot_flat = win5[own_in5]
                    prot_positions = torch.unique(prot_flat)
                    prot = int(prot_positions.numel())

            # jack vulnerability: open4 with ≥1 own non-protected chip
            jack_vuln = 0
            if open4 > 0:
                win_open4 = self._T_WIN_IDX[open4_mask]      # (N,5)
                own_in_win = (flat_t[win_open4] == team)     # (N,5)
                if prot_positions is not None and prot > 0:
                    prot_mask = torch.zeros((100,), dtype=torch.bool, device=self._T_device)
                    prot_mask[prot_positions] = True
                    any_unprot = (own_in_win & (~prot_mask[win_open4])).any(dim=1)
                else:
                    any_unprot = own_in_win.any(dim=1)
                jack_vuln = int(any_unprot.sum().item())

            # spatial
            hot = float((team_plane.float() * self._T_SQW).sum().item())
            center_cnt = int((team_plane & self._T_CENTER).sum().item())
            corner_adj = int((team_plane & self._T_ADJ_BONUS).sum().item())

            return {
                "open1": open1, "open2": open2, "open3": open3, "open4": open4,
                "imm_win_cells": imm_set, "imm_win_count": imm_cnt,
                "forks": forks, "max_run": max_run,
                "protected_chips": prot, "jack_vuln": jack_vuln,
                "hot_control": hot, "center_count": center_cnt, "corner_adj": corner_adj,
            }

        # -------- NumPy fallback (still vectorized) --------
        team_plane  = (flat == team)
        empty_plane = (flat == -1)
        opp_plane   = (~empty_plane) & (~team_plane)

        sat_plane = team_plane | self._NP_BONUS
        sat_w = sat_plane[self._NP_WIN_IDX]               # (W,5)
        emp_w = empty_plane[self._NP_WIN_IDX]
        opp_w = opp_plane[self._NP_WIN_IDX]

        sat_cnt = sat_w.sum(axis=1)
        emp_cnt = emp_w.sum(axis=1)
        opp_cnt = opp_w.sum(axis=1)

        open4_mask = (opp_cnt == 0) & (sat_cnt == 4) & (emp_cnt == 1)
        open3_mask = (opp_cnt == 0) & (sat_cnt == 3) & (emp_cnt == 2)
        open2_mask = (opp_cnt == 0) & (sat_cnt == 2) & (emp_cnt == 3)
        open1_mask = (opp_cnt == 0) & (sat_cnt == 1) & (emp_cnt == 4)

        open4 = int(open4_mask.sum()); open3 = int(open3_mask.sum())
        open2 = int(open2_mask.sum()); open1 = int(open1_mask.sum())

        if open4:
            emp_rows = emp_w[open4_mask]                  # (N,5)
            empty_idx = emp_rows.argmax(axis=1)           # (N,)
            win_rows  = self._NP_WIN_IDX[open4_mask]      # (N,5)
            empties   = win_rows[np.arange(len(win_rows)), empty_idx]   # (N,)
            uniq = np.unique(empties)
            imm_cnt = int(uniq.size)
            imm_set = {(int(f)//10, int(f)%10) for f in uniq.tolist()}
            bc = np.bincount(empties, minlength=100)
            forks = int((bc >= 2).sum())
        else:
            imm_cnt = 0
            imm_set = set()
            forks = 0

        # max_run via rolling sums
        max_run = 0
        if sat_w.size:
            # compute rolling sums for k and check if equals k anywhere
            for k in (5,4,3,2,1):
                c = np.cumsum(sat_w, axis=1)
                roll = c[:, k-1:] - np.pad(c[:, :-k], ((0,0),(1,0)), mode='constant')
                if (roll == k).any():
                    max_run = k
                    break

        # protected chips
        prot = 0
        prot_positions = None
        mask5 = (sat_cnt == 5)
        if mask5.any():
            win5 = self._NP_WIN_IDX[mask5]
            own = (flat[win5] == team)
            if own.any():
                prot_positions = np.unique(win5[own])
                prot = int(prot_positions.size)

        # jack vulnerability
        jack_vuln = 0
        if open4:
            win_open4 = self._NP_WIN_IDX[open4_mask]
            own_in_win = (flat[win_open4] == team)
            if prot_positions is not None and prot > 0:
                prot_mask = np.zeros((100,), dtype=np.bool_)
                prot_mask[prot_positions] = True
                any_unprot = (own_in_win & (~prot_mask[win_open4])).any(axis=1)
            else:
                any_unprot = own_in_win.any(axis=1)
            jack_vuln = int(any_unprot.sum())

        hot = float(np.dot(team_plane.astype(np.float32), self._NP_SQW))
        center_cnt = int((team_plane & self._NP_CENTER).sum())
        corner_adj = int((team_plane & self._NP_ADJ_BONUS).sum())

        return {
            "open1": open1, "open2": open2, "open3": open3, "open4": open4,
            "imm_win_cells": imm_set, "imm_win_count": imm_cnt,
            "forks": forks, "max_run": max_run,
            "protected_chips": prot, "jack_vuln": jack_vuln,
            "hot_control": hot, "center_count": center_cnt, "corner_adj": corner_adj,
        }

    def _coverage_clean_windows_for_opp_fast(self, st, acting_team: int):
        """Vectorized count of windows with NO acting_team chip."""
        if not self._FAST_READY:
            return self._coverage_clean_windows_for_opp(st, acting_team)

        flat = self._board_flat_numpy(st)     # (100,)
        has_self_w = (flat[self._NP_WIN_IDX] == acting_team).any(axis=1)
        return int((~has_self_w).sum())

    def _features_bundle(self, st, acting_team: int):
        """Aggregate features for self and opponents + coverage and mobility estimators (fast path)."""
        teams = max(1, int(self.gconf.teams))

        self_f = self._team_features_fast(st, acting_team)

        opp = {"open1":0,"open2":0,"open3":0,"open4":0,"forks":0,"max_run":0,
               "imm_win_cells": set(), "imm_win_count": 0,
               "protected_chips":0, "jack_vuln":0,
               "hot_control":0, "center_count":0, "corner_adj":0}
        for t in range(teams):
            if t == acting_team:
                continue
            f = self._team_features_fast(st, t)
            for k in ("open1","open2","open3","open4","forks","max_run",
                      "protected_chips","jack_vuln","hot_control","center_count","corner_adj"):
                opp[k] += f[k]
            # union of immediate-win cells; sum of counts too
            opp["imm_win_cells"] |= f["imm_win_cells"]
            opp["imm_win_count"] += int(f.get("imm_win_count", len(f["imm_win_cells"])))

        coverage_clean = self._coverage_clean_windows_for_opp_fast(st, acting_team)
        mobility = self._expected_mobility(st, acting_team)
        return {"self": self_f, "opp": opp, "coverage_clean": coverage_clean, "mobility": mobility}

    # ---------------- Mobility / EV helpers (vectorized) ----------------

    def _empty_mask(self, st) -> np.ndarray:
        """Boolean mask (100,) of empty, non-BONUS cells."""
        flat = self._board_flat_numpy(st)
        # NOTE: BONUS cells are excluded by not being in any card mask
        return (flat == -1)

    def _mobility_base_union(self, st, hand: List[str]):
        """Return (base_count, union_mask, empties_mask, has_twoeyed)."""
        empties_mask = self._empty_mask(st)
        has_twoeyed = any(is_two_eyed_jack(c) for c in hand)
        if has_twoeyed:
            union_mask = empties_mask.copy()
            base = int(np.count_nonzero(empties_mask))
            return base, union_mask, empties_mask, True

        union_mask = np.zeros((100,), dtype=np.bool_)
        for card in set(hand):
            cm = self._CARD_TO_MASK.get(card, None)
            if cm is not None:
                union_mask |= cm
        base = int(np.count_nonzero(union_mask & empties_mask))
        return base, union_mask, empties_mask, False

    def _expected_mobility(self, st, seat: int):
        """
        Estimate playable placements next turn: empties matching hand + EV of drawing 1 (vectorized).
        """
        public = self._public_summary()
        deck_counts = public.get("deck_counts", {})
        total_rem = max(1, int(public.get("total_remaining", 1)))

        hand = self._hand_for(seat)

        base, union_mask, empties_mask, has_twoeyed = self._mobility_base_union(st, hand)

        add_ev = 0.0
        if (not has_twoeyed) and total_rem > 0:
            for card, cnt in deck_counts.items():
                if cnt <= 0 or card in hand:
                    continue
                cm = self._CARD_TO_MASK.get(card, None)
                if cm is None:
                    continue
                # New cells unlocked by this card only (not already in union)
                inc = int(np.count_nonzero(cm & empties_mask & (~union_mask)))
                add_ev += (cnt / total_rem) * inc

        return float(base) + float(add_ev)

    # ---------------- Legacy slow fallbacks (kept for parity & safety) ----------------
    def _sequences_for_team(self, st, team: int) -> int:
        seqs = getattr(st, "sequences", None)

        if isinstance(seqs, dict) and team in seqs:
            return int(seqs[team])  # alias through GameState helper if present
        if hasattr(st, "sequences_count"):
            try:
                return int(st.sequences_count.get(team, 0)) # type: ignore
            except Exception:
                pass
        return 0
    def _team_features(self, st, team: int):
        """
        Slow Python fallback (kept for safety/parity).
        Returns dict with:
          open1, open2, open3, open4,
          imm_win_cells (set),
          forks (count),
          max_run,
          protected_chips,
          jack_vuln,
          hot_control, center_count, corner_adj
        """
        self._ensure_windows()
        board = st.board  # type: ignore
        windows = self._WINDOWS_CACHE
        sqw = self._SQUARE_WEIGHT

        open_counts = {1:0,2:0,3:0,4:0}
        imm_cells = []
        max_run = 0
        protected_cells = set()

        for w in windows:
            s = e = o = 0
            empties = []
            team_cells = []
            for (r,c) in w:
                printed = BOARD_LAYOUT[r][c]
                chip = board[r][c]
                if printed == "BONUS":
                    s += 1
                elif chip is None:
                    e += 1
                    empties.append((r,c))
                elif chip == team:
                    s += 1
                    team_cells.append((r,c))
                else:
                    o += 1

            # runs
            run = 0
            for (r,c) in w:
                chip = board[r][c]
                printed = BOARD_LAYOUT[r][c]
                if printed == "BONUS" or chip == team:
                    run += 1
                    max_run = max(max_run, run)
                else:
                    run = 0

            if o == 0:
                if s == 4 and e == 1:
                    open_counts[4] += 1
                    imm_cells.extend(empties)
                elif s == 3 and e == 2:
                    open_counts[3] += 1
                elif s == 2 and e == 3:
                    open_counts[2] += 1
                elif s == 1 and e == 4:
                    open_counts[1] += 1

            # protected: any own chip in a fully satisfied 5
            if s == 5:
                for (r,c) in w:
                    chip = board[r][c]
                    if chip == team:
                        protected_cells.add((r,c))

        # forks: count cells that appear ≥2 times as immediate-win empties
        from collections import Counter
        forks = 0
        if imm_cells:
            cnt = Counter(imm_cells)
            forks = sum(1 for _, v in cnt.items() if v >= 2)

        # jack vulnerability: open4 with at least one own (non-protected) chip
        jack_vuln = 0
        for w in self._WINDOWS_CACHE:
            s = e = o = 0
            team_nonbonus = []
            for (r,c) in w:
                printed = BOARD_LAYOUT[r][c]
                chip = board[r][c]
                if printed == "BONUS":
                    s += 1
                elif chip is None:
                    e += 1
                elif chip == team:
                    s += 1
                    team_nonbonus.append((r,c))
                else:
                    o += 1
            if o == 0 and s == 4 and e == 1:
                if any((rc not in protected_cells) for rc in team_nonbonus):
                    jack_vuln += 1

        hot = 0
        center_cnt = 0
        corner_adj = 0
        for r in range(10):
            for c in range(10):
                chip = board[r][c]
                if chip == team:
                    hot += sqw[(r,c)]
                    if (r,c) in self._CENTER_SET:
                        center_cnt += 1
                    for (rr,cc) in self._neighbors8(r,c):
                        if BOARD_LAYOUT[rr][cc] == "BONUS":
                            corner_adj += 1
                            break

        return {
            "open1": open_counts[1], "open2": open_counts[2], "open3": open_counts[3], "open4": open_counts[4],
            "imm_win_cells": set(imm_cells), "imm_win_count": len(set(imm_cells)),
            "forks": forks, "max_run": max_run,
            "protected_chips": len(protected_cells), "jack_vuln": jack_vuln,
            "hot_control": hot, "center_count": center_cnt, "corner_adj": corner_adj,
        }

    def _coverage_clean_windows_for_opp(self, st, acting_team: int):
        """Slow fallback: count windows that contain no acting_team chip."""
        self._ensure_windows()
        board = st.board  # type: ignore
        clean = 0
        for w in self._WINDOWS_CACHE:
            has_self = False
            for (r,c) in w:
                chip = board[r][c]
                if chip == acting_team:
                    has_self = True
                    break
            if not has_self:
                clean += 1
        return clean

    # ---------------- Engine accessors & summaries ----------------

    def _state(self):
        return self.game_engine.state

    def _seat_index(self) -> int:
        st = self._state()
        return int(getattr(st, "current_player", getattr(st, "turn_index", 0)))

    def _player_team(self, seat: int) -> int:
        teams = max(1, int(self.gconf.teams))
        return int(seat) % teams

    def _cell(self, r: int, c: int):
        st = self._state()
        return st.board[r][c]  # type: ignore

    def _hand_for(self, seat: int) -> List[str]:
        st = self._state()
        hands = getattr(st, "hands", None)
        if isinstance(hands, list) and 0 <= seat < len(hands):
            return list(hands[seat])
        return []

    def _first_discard_slot(self, seat: int) -> Optional[int]:
        legal = self._legal_for(seat)
        disc = legal.get("discard") or legal.get("discard_slots")
        if isinstance(disc, list) and len(disc) > 0:
            try:
                return int(disc[0])
            except Exception:
                return None
        hand = self._hand_for(seat)
        return 0 if len(hand) > 0 else None

    def _public_summary(self) -> Dict[str, Any]:
        # 2 copies per card (double deck), minus discard pile, minus occupied printed cells (approx.)
        base: Dict[str, int] = {f"{r}{s}": 2 for s in "SHDC" for r in ["A", "2", "3", "4", "5", "6", "7", "8", "9", "10", "J", "Q", "K"]}
        deck = getattr(self.game_engine, "deck", None)
        discard_pile = []
        if deck is not None:
            dp = getattr(deck, "discard_pile", getattr(deck, "discardPile", None))
            if isinstance(dp, list):
                discard_pile = list(dp)
        for d in discard_pile:
            if d in base:
                base[d] = max(0, base[d] - 1)
        st = self._state()
        for rr in range(10):
            for cc in range(10):
                printed = BOARD_LAYOUT[rr][cc]
                if printed == "BONUS":
                    continue
                chip = st.board[rr][cc]
                if chip is not None and printed in base:
                    base[printed] = max(0, base[printed] - 1)
        total_remaining = int(sum(base.values()))
        return {"deck_counts": base, "total_remaining": total_remaining, "discard_pile": discard_pile}

    # ---------------- Geometry & windows ----------------

    def _ensure_windows(self):
        """Precompute all 5-cell windows and per-square participation weights."""
        if self._WINDOWS_CACHE is not None and self._SQUARE_WEIGHT is not None:
            return
        windows = []
        # Horizontal
        for r in range(10):
            for c in range(10 - 5 + 1):
                windows.append([(r, c+i) for i in range(5)])
        # Vertical
        for r in range(10 - 5 + 1):
            for c in range(10):
                windows.append([(r+i, c) for i in range(5)])
        # Diagonal ↘︎
        for r in range(10 - 5 + 1):
            for c in range(10 - 5 + 1):
                windows.append([(r+i, c+i) for i in range(5)])
        # Diagonal ↙︎
        for r in range(10 - 5 + 1):
            for c in range(4, 10):
                windows.append([(r+i, c-i) for i in range(5)])

        sqw = {(r, c): 0 for r in range(10) for c in range(10)}
        for w in windows:
            for (rr, cc) in w:
                sqw[(rr, cc)] += 1

        self._WINDOWS_CACHE = windows
        self._SQUARE_WEIGHT = sqw

    def _neighbors8(self, r, c):
        for dr in (-1,0,1):
            for dc in (-1,0,1):
                if dr == 0 and dc == 0:
                    continue
                rr, cc = r+dr, c+dc
                if 0 <= rr < 10 and 0 <= cc < 10:
                    yield (rr, cc)

    # ---------------- Optional ASCII render ----------------

    def render(self, mode: str = "human"):
        st = self._state()
        if st is None or getattr(st, "board", None) is None:
            print("<no board>")
            return
        out = []
        for r in range(10):
            row = []
            for c in range(10):
                cell = st.board[r][c]
                row.append("." if cell is None else str(int(cell)))
            out.append(" ".join(row))
        print("\n".join(out))
        return out
