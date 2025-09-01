# training/ui/human_agent_ui.py
# UI Pygame para HumanAgent con:
# - Hints de secuencia más claros (banda de color por eje y equipo + leyenda)
# - Scroll del tablero (rueda: vertical | Shift+rueda: horizontal)
# - Scroll horizontal de la mano (rueda sobre la mano o arrastre)
# - Elementos más grandes vía UI_SCALE
#
# Carga tablero desde training/assets/boards/standard_10x10.json (BOARD_LAYOUT)
# Renderiza 10x10 con imágenes opcionales en assets/cards/{rank}{suit}.png
# Interacción:
#   - Click en carta para seleccionarla
#   - Click en celda para jugar
#   - Botón "Burn"/"Pass" o atajos [B]/[P]
# Devuelve la acción entera según el action space unificado de SequenceEnv
#
# Notas:
# - La UI asume 10x10 por el mapeo r*10+c
# - UI_SCALE ajusta tamaños; si el tablero excede la ventana, usa el scroll
# - Las “jugadas ganadoras inmediatas” parpadean (anillo verde)

from __future__ import annotations

import os
import time
from typing import Optional, Dict, List, Tuple, Any, Set

import pygame
import numpy as np

from ..engine.board_layout import BOARD_LAYOUT
assert BOARD_LAYOUT[0][1] == "2S" and BOARD_LAYOUT[0][8] == "9S"

# --------------------------------------------------------------------------------------
# Constantes de UI (con escala global)
# --------------------------------------------------------------------------------------

UI_SCALE = 1.25  # <---- ajusta aquí el tamaño general de la UI

def _S(x: float) -> int:
    return int(round(x * UI_SCALE))

PANEL_W = _S(260)
CELL_W, CELL_H = _S(56), _S(68)
GRID_W, GRID_H = 10 * CELL_W, 10 * CELL_H
HAND_PANEL_H = _S(140)
MARG = _S(16)
GAP = _S(8)

# Tamaño de ventana “target”; si excede, el contenido usa scroll
WIN_W_TARGET = _S(1280)
WIN_H_TARGET = _S(880)

FPS = 60

TEAM_COLORS = {
    0: (220, 38, 38),   # rojo
    1: (16, 185, 129),  # esmeralda
    2: (59, 130, 246),  # azul
    3: (234, 179, 8),   # ámbar
}
MINE_RING = (255, 255, 255)
BG = (248, 250, 252)     # slate-50
BORDER = (203, 213, 225) # slate-300
TEXT = (30, 41, 59)      # slate-800
SUBTLE = (100, 116, 139) # slate-500
INDIGO = (99, 102, 241)
ROSE = (244, 63, 94)
GREEN = (16, 185, 129)
SCROLL_TRACK = (241, 245, 249)
SCROLL_THUMB = (148, 163, 184)

CARD_IMG_CACHE: Dict[str, pygame.Surface] = {}

# --------------------------------------------------------------------------------------
# Seguridad: tablero 10x10 (SequenceEnv mapea r*10+c)
# --------------------------------------------------------------------------------------
assert len(BOARD_LAYOUT) == 10 and all(len(r) == 10 for r in BOARD_LAYOUT), "Board must be 10x10."

# --------------------------------------------------------------------------------------
# Jacks helpers
# --------------------------------------------------------------------------------------
try:
    from ..engine.engine_core import is_two_eyed_jack, is_one_eyed_jack  # type: ignore
except Exception:
    def is_two_eyed_jack(card: Optional[str]) -> bool:
        return card in {"JC", "JD"}

    def is_one_eyed_jack(card: Optional[str]) -> bool:
        return card in {"JH", "JS"}

# --------------------------------------------------------------------------------------
# Tipografías
# --------------------------------------------------------------------------------------

def _safe_font(size: int, bold: bool = False) -> pygame.font.Font:
    try:
        f = pygame.font.SysFont("Inter, Segoe UI, Arial, Helvetica, sans-serif", size, bold=bold)
    except Exception:
        f = pygame.font.SysFont("Arial", size, bold=bold)
    return f


class HumanAgentUI:
    """
    Pygame UI para elegir acciones en SequenceEnv.

    Uso:
        ui = HumanAgentUI(card_img_dir="assets/cards")
        action = ui.choose_action(env, legal_mask, ctx_info)
    """

    def __init__(self, card_img_dir: str = "assets/cards"):
        self.card_img_dir = card_img_dir
        self._screen: Optional[pygame.Surface] = None
        self._clock: Optional[pygame.time.Clock] = None

        self._selected_hand_idx: Optional[int] = None
        self._status_text: str = ""
        self._status_until: float = 0.0

        # Geometría base
        self._board_rect = pygame.Rect(MARG, MARG, GRID_W, GRID_H)  # actual viewport se ajusta en _layout()
        self._hand_rects: List[Tuple[pygame.Rect, int]] = []  # (rect, hand_idx)
        self._burn_rect: Optional[pygame.Rect] = None
        self._pass_rect: Optional[pygame.Rect] = None
        self._legend_rect: Optional[pygame.Rect] = None

        # Scroll estado
        self._board_view: pygame.Rect = pygame.Rect(0, 0, GRID_W, GRID_H)  # viewport visible dentro del grid
        self._board_scroll_x = 0
        self._board_scroll_y = 0

        self._hand_scroll_x = 0
        self._hand_dragging = False
        self._hand_drag_x0 = 0
        self._hand_drag_start = 0

        # Reutilizable
        self._boot()

    # ---------- API público ----------

    def choose_action(self, env, legal_mask: Optional[np.ndarray], info: Optional[Dict[str, Any]] = None) -> int:
        if legal_mask is None:
            legal_mask = np.ones((env.action_dim,), dtype=np.float32)
        legal_set = {i for i, v in enumerate(legal_mask.tolist()) if v >= 0.5}

        current_player = int((info or {}).get("current_player", getattr(env, "current_player", 0)))
        my_team = int(getattr(env, "_player_team")(current_player)) if hasattr(env, "_player_team") else current_player

        chosen: Optional[int] = None
        while chosen is None:
            chosen = self._frame(env, legal_set, my_team)
        return chosen

    # ---------- Pygame setup & frame ----------

    def _boot(self):
        if self._screen is not None:
            return
        pygame.init()
        pygame.display.set_caption("Sequence – HumanAgentUI")
        # Ajusta ventana objetivo y calcula viewport con scroll si excede
        win_w = min(GRID_W + PANEL_W + 3*MARG, WIN_W_TARGET)
        win_h = min(GRID_H + HAND_PANEL_H + 3*MARG, WIN_H_TARGET)
        self._screen = pygame.display.set_mode((win_w, win_h), pygame.RESIZABLE)
        self._clock = pygame.time.Clock()

    def _layout(self):
        """Calcula rects visibles (board viewport, panel, mano) considerando tamaño ventana."""
        assert self._screen is not None
        sw, sh = self._screen.get_size()

        # Área principal para el tablero
        board_w_avail = max(_S(600), sw - PANEL_W - 3*MARG)
        board_h_avail = max(_S(480), sh - HAND_PANEL_H - 3*MARG)
        self._board_rect = pygame.Rect(MARG, MARG, board_w_avail, board_h_avail)

        # Panel lateral
        self._panel_rect = pygame.Rect(self._board_rect.right + MARG, MARG, PANEL_W, board_h_avail)

        # Mano
        self._hand_area = pygame.Rect(MARG, self._board_rect.bottom + MARG, sw - 2*MARG, HAND_PANEL_H)

        # Limites de scroll del tablero
        max_scroll_x = max(0, GRID_W - self._board_rect.width)
        max_scroll_y = max(0, GRID_H - self._board_rect.height)
        self._board_scroll_x = max(0, min(self._board_scroll_x, max_scroll_x))
        self._board_scroll_y = max(0, min(self._board_scroll_y, max_scroll_y))
        self._board_view = pygame.Rect(self._board_scroll_x, self._board_scroll_y,
                                       self._board_rect.width, self._board_rect.height)

    def _handle_scroll_events(self, event: pygame.event.Event):
        # Mouse wheel para tablero y mano
        keys = pygame.key.get_mods()
        shift = keys & pygame.KMOD_SHIFT
        mx, my = pygame.mouse.get_pos()

        if event.type == pygame.MOUSEWHEEL:
            # Sobre tablero
            if self._board_rect.collidepoint(mx, my):
                if shift:
                    self._board_scroll_x -= event.y * _S(40)
                else:
                    self._board_scroll_y -= event.y * _S(40)
                self._clamp_board_scroll()
            # Sobre mano
            elif self._hand_area.collidepoint(mx, my):
                self._hand_scroll_x -= event.y * _S(60)
                self._clamp_hand_scroll()

        # Arrastre horizontal de la mano (drag)
        if event.type == pygame.MOUSEBUTTONDOWN and event.button == 1:
            if self._hand_area.collidepoint(mx, my):
                self._hand_dragging = True
                self._hand_drag_x0 = mx
                self._hand_drag_start = self._hand_scroll_x
        if event.type == pygame.MOUSEBUTTONUP and event.button == 1:
            self._hand_dragging = False
        if event.type == pygame.MOUSEMOTION and self._hand_dragging:
            dx = mx - self._hand_drag_x0
            self._hand_scroll_x = self._hand_drag_start - dx
            self._clamp_hand_scroll()

    def _clamp_board_scroll(self):
        max_x = max(0, GRID_W - self._board_rect.width)
        max_y = max(0, GRID_H - self._board_rect.height)
        self._board_scroll_x = max(0, min(self._board_scroll_x, max_x))
        self._board_scroll_y = max(0, min(self._board_scroll_y, max_y))

    def _clamp_hand_scroll(self):
        # Límite se calcula en _draw_hand al conocer el ancho total de cartas
        if hasattr(self, "_hand_total_w"):
            max_x = max(0, self._hand_total_w - (self._hand_area.width - 2*MARG))
            self._hand_scroll_x = max(0, min(self._hand_scroll_x, max_x))

    def _frame(self, env, legal_set: Set[int], my_team: int) -> Optional[int]:
        assert self._screen is not None and self._clock is not None

        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                pygame.quit()
                raise KeyboardInterrupt("UI closed")
            if event.type == pygame.VIDEORESIZE:
                # Recalcular layout tras resize
                self._layout()
            if event.type == pygame.KEYDOWN:
                if event.key == pygame.K_ESCAPE:
                    self._selected_hand_idx = None
                if event.key == pygame.K_b:  # burn
                    return self._maybe_burn(env, legal_set)
                if event.key == pygame.K_p:  # pass
                    return self._maybe_pass(env, legal_set)

            # Scroll / drag
            self._handle_scroll_events(event)

            # Clicks
            if event.type == pygame.MOUSEBUTTONDOWN and event.button == 1:
                mx, my = event.pos
                # Mano
                for rect, hidx in self._hand_rects:
                    if rect.collidepoint(mx, my):
                        self._selected_hand_idx = hidx
                        self._flash_status(self._jack_badge_text(env, hidx))
                        return None
                # Botones
                if self._burn_rect and self._burn_rect.collidepoint(mx, my):
                    return self._maybe_burn(env, legal_set)
                if self._pass_rect and self._pass_rect.collidepoint(mx, my):
                    return self._maybe_pass(env, legal_set)
                # Tablero
                if self._board_rect.collidepoint(mx, my):
                    # Coordenadas dentro del grid con scroll aplicado
                    gx = mx - self._board_rect.left + self._board_scroll_x
                    gy = my - self._board_rect.top + self._board_scroll_y
                    c = int(gx // CELL_W)
                    r = int(gy // CELL_H)
                    if 0 <= r < 10 and 0 <= c < 10:
                        action = int(r * 10 + c)
                        if action in legal_set:
                            return action
                        else:
                            self._flash_status("No es legal en esa casilla.")

        # Dibujo
        self._layout()
        self._screen.fill(BG)
        self._draw_board(env, legal_set, my_team)
        self._draw_panel(env, legal_set, my_team)
        self._draw_hand(env)
        self._draw_status()
        pygame.display.flip()
        self._clock.tick(FPS)
        return None

    # ---------- Dibujo ----------

    def _draw_board(self, env, legal_set: Set[int], my_team: int):
        s = self._screen; assert s is not None

        # Fondo del viewport
        pygame.draw.rect(s, (255, 255, 255), self._board_rect, border_radius=_S(10))
        pygame.draw.rect(s, BORDER, self._board_rect, width=2, border_radius=_S(10))

        # Pre-calcula secuencias completas por equipo y eje + celdas de win inmediato
        seq_overlays = self._detect_sequences_with_teams(env)  # {(r,c): [(axis,team), ...]}
        imm_cells = self._imm_win_cells(env, my_team)

        st = env.game_engine.state

        # Dibuja solo celdas visibles en viewport (con scroll)
        x0 = self._board_rect.left - self._board_scroll_x
        y0 = self._board_rect.top - self._board_scroll_y

        now = time.time()
        pulse = (1 + 0.25 * np.sin(now * 5.0))  # factor 1..1.25

        for r in range(10):
            for c in range(10):
                cell_rect = pygame.Rect(x0 + c * CELL_W, y0 + r * CELL_H, CELL_W, CELL_H)
                if not cell_rect.colliderect(self._board_rect):
                    continue

                printed = BOARD_LAYOUT[r][c]
                chip = st.board[r][c]

                # Fondo por celda
                if printed == "BONUS":
                    pygame.draw.rect(s, (254, 240, 138), cell_rect)  # yellow-200
                else:
                    pygame.draw.rect(s, (255, 255, 255), cell_rect)

                # Cara de la carta (imagen o texto)
                if printed and printed != "BONUS":
                    img = self._card_image(printed)
                    if img is not None:
                        img_rect = img.get_rect()
                        img_rect.center = cell_rect.center
                        scale = min((CELL_W - _S(6)) / img_rect.w, (CELL_H - _S(6)) / img_rect.h)
                        if scale < 1.0:
                            img = pygame.transform.smoothscale(img, (int(img_rect.w * scale), int(img_rect.h * scale)))
                            img_rect = img.get_rect(center=cell_rect.center)
                        s.blit(img, img_rect)
                    else:
                        self._blit_centered_text(s, printed, _S(16), cell_rect, SUBTLE)
                elif printed == "BONUS":
                    self._blit_centered_text(s, "★", _S(20), cell_rect, (234, 179, 8), bold=True)

                # Borde de celda si es legal
                a = r * 10 + c
                if a in legal_set:
                    pygame.draw.rect(s, (199, 210, 254), cell_rect, width=_S(3), border_radius=_S(6))  # indigo-200

                # Chip
                if chip is not None:
                    color = TEAM_COLORS.get(int(chip), (107, 114, 128))
                    pygame.draw.circle(s, color, (cell_rect.right - _S(12), cell_rect.top + _S(12)), _S(10))
                    pygame.draw.circle(s, MINE_RING, (cell_rect.right - _S(12), cell_rect.top + _S(12)), _S(10), _S(2))

                # Overlays de secuencia (más gruesos y coloreados por equipo)
                axes = seq_overlays.get((r, c), [])
                for axis, team in axes:
                    col = TEAM_COLORS.get(team, INDIGO)
                    self._draw_sequence_band(s, cell_rect, axis, col, alpha=110, thickness=_S(8))

                # Win inmediato para mi equipo (anillo pulsante + texto)
                if (r, c) in imm_cells:
                    rad = int(_S(9) * pulse)
                    pygame.draw.circle(s, GREEN, (cell_rect.centerx, cell_rect.centery), rad, _S(3))
                    self._blit_centered_text(s, "WIN", _S(10), cell_rect.move(0, _S(16)), GREEN, bold=True)

                # Bordes finos de celda
                pygame.draw.rect(s, BORDER, cell_rect, width=1)

        # Scrollbars del tablero (si procede)
        self._draw_board_scrollbars(s)

    def _draw_sequence_band(self, s: pygame.Surface, rect: pygame.Rect,
                            axis: str, color: Tuple[int, int, int], alpha: int = 110, thickness: int = 6):
        # Dibujar una banda semitransparente gruesa por eje
        band = pygame.Surface((rect.width, rect.height), pygame.SRCALPHA)
        col = (*color, alpha)
        cx, cy = rect.centerx, rect.centery
        if axis == 'H':
            pygame.draw.rect(band, col, (GAP//2, rect.height//2 - thickness//2, rect.width - GAP, thickness), border_radius=thickness//2)
        elif axis == 'V':
            pygame.draw.rect(band, col, (rect.width//2 - thickness//2, GAP//2, thickness, rect.height - GAP), border_radius=thickness//2)
        elif axis == 'D1':  # \
            pygame.draw.line(band, col, (0, rect.height), (rect.width, 0), thickness)
        elif axis == 'D2':  # /
            pygame.draw.line(band, col, (0, 0), (rect.width, rect.height), thickness)
        s.blit(band, rect.topleft)

    def _draw_board_scrollbars(self, s: pygame.Surface):
        # Muestra barras si el grid excede el viewport
        need_h = GRID_W > self._board_rect.width
        need_v = GRID_H > self._board_rect.height
        if not (need_h or need_v):
            return

        # Track
        if need_h:
            track = pygame.Rect(self._board_rect.left, self._board_rect.bottom - _S(10),
                                self._board_rect.width, _S(8))
            pygame.draw.rect(s, SCROLL_TRACK, track, border_radius=_S(4))
            # Thumb
            thumb_w = max(_S(40), int(self._board_rect.width * self._board_rect.width / GRID_W))
            max_scroll = GRID_W - self._board_rect.width
            x = 0 if max_scroll == 0 else int(self._board_scroll_x * (self._board_rect.width - thumb_w) / max_scroll)
            thumb = pygame.Rect(track.left + x, track.top, thumb_w, track.height)
            pygame.draw.rect(s, SCROLL_THUMB, thumb, border_radius=_S(4))

        if need_v:
            track = pygame.Rect(self._board_rect.right - _S(10), self._board_rect.top,
                                _S(8), self._board_rect.height)
            pygame.draw.rect(s, SCROLL_TRACK, track, border_radius=_S(4))
            thumb_h = max(_S(40), int(self._board_rect.height * self._board_rect.height / GRID_H))
            max_scroll = GRID_H - self._board_rect.height
            y = 0 if max_scroll == 0 else int(self._board_scroll_y * (self._board_rect.height - thumb_h) / max_scroll)
            thumb = pygame.Rect(track.left, track.top + y, track.width, thumb_h)
            pygame.draw.rect(s, SCROLL_THUMB, thumb, border_radius=_S(4))

    def _draw_panel(self, env, legal_set: Set[int], my_team: int):
        s = self._screen; assert s is not None
        panel = self._panel_rect

        pygame.draw.rect(s, (255, 255, 255), panel, border_radius=_S(10))
        pygame.draw.rect(s, BORDER, panel, 2, border_radius=_S(10))

        # Header
        self._blit_text(s, "Jugadores", _S(22), (panel.left + _S(12), panel.top + _S(12)), bold=True)

        # Lista de asientos
        try:
            current_player = int(env.current_player)
        except Exception:
            current_player = 0
        teams = max(1, int(getattr(env.gconf, "teams", 2)))
        seats = max(2, int(getattr(env.gconf, "players", 2)))

        y = panel.top + _S(48)
        row_h = _S(32)
        for seat in range(seats):
            team_idx = seat % teams
            row = pygame.Rect(panel.left + _S(8), y, panel.width - _S(16), row_h)
            if seat == current_player:
                pygame.draw.rect(s, (254, 243, 199), row, border_radius=_S(6))
            name = f"Asiento {seat}"
            team_badge = f"Equipo {team_idx + 1}"
            self._blit_text(s, name, _S(15), (row.left + _S(8), row.centery - _S(10)))
            self._blit_text(s, team_badge, _S(12), (row.right - _S(100), row.centery - _S(9)), color=SUBTLE)
            y += row_h + _S(4)

        # Leyenda (hints)
        y += _S(8)
        legend = pygame.Rect(panel.left + _S(10), y, panel.width - _S(20), _S(110))
        pygame.draw.rect(s, (249, 250, 251), legend, border_radius=_S(8))
        pygame.draw.rect(s, BORDER, legend, 1, border_radius=_S(8))
        self._blit_text(s, "Leyenda", _S(16), (legend.left + _S(8), legend.top + _S(8)), bold=True)
        # Muestras
        lx = legend.left + _S(12)
        ly = legend.top + _S(34)
        # Legal
        legal_box = pygame.Rect(lx, ly, _S(28), _S(18))
        pygame.draw.rect(s, (255, 255, 255), legal_box)
        pygame.draw.rect(s, (199, 210, 254), legal_box, _S(3), border_radius=_S(4))
        self._blit_text(s, "Jugada legal", _S(13), (legal_box.right + _S(8), ly))
        ly += _S(24)
        # Secuencia (banda)
        seq_box = pygame.Rect(lx, ly, _S(28), _S(18))
        pygame.draw.rect(s, (255, 255, 255), seq_box)
        self._draw_sequence_band(s, seq_box, 'H', TEAM_COLORS.get(0, INDIGO), alpha=130, thickness=_S(8))
        pygame.draw.rect(s, BORDER, seq_box, 1)
        self._blit_text(s, "Secuencia completa", _S(13), (seq_box.right + _S(8), ly))
        ly += _S(24)
        # Win inmediato
        win_box = pygame.Rect(lx, ly, _S(28), _S(18))
        pygame.draw.rect(s, (255, 255, 255), win_box)
        pygame.draw.circle(s, GREEN, win_box.center, _S(8), _S(3))
        self._blit_text(s, "Ganas si juegas aquí", _S(13), (win_box.right + _S(8), ly))

        # Ganadores
        winners: List[int] = []
        try:
            winners = list(env.game_engine.winner_teams())  # type: ignore
        except Exception:
            try:
                winners = list(getattr(env.game_engine.state, "winners", []))  # type: ignore
            except Exception:
                winners = []
        if winners:
            self._blit_text(s, f"Ganador: Equipo {int(winners[0]) + 1}", _S(20),
                            (panel.left + _S(12), panel.bottom - _S(40)),
                            color=(22, 163, 74), bold=True)

        # Botones
        btn_y = panel.bottom - _S(100)
        self._burn_rect = self._button(s, "Burn Card [B]", (panel.left + _S(12), btn_y),
                                       enabled=self._can_burn(env, legal_set))
        btn_y += _S(48)
        include_pass = bool(getattr(env, "_include_pass", False))
        self._pass_rect = self._button(s, "Pass [P]", (panel.left + _S(12), btn_y),
                                       enabled=self._can_pass(env, legal_set) if include_pass else False)

    def _draw_hand(self, env):
        s = self._screen; assert s is not None
        area = self._hand_area
        pygame.draw.rect(s, (255, 255, 255), area, border_radius=_S(10))
        pygame.draw.rect(s, BORDER, area, 2, border_radius=_S(10))
        self._blit_text(s, "Tu Mano", _S(22), (area.left + _S(12), area.top + _S(6)), bold=True)

        hand = self._hand(env)
        gap = GAP
        available_w = area.width - 2*MARG
        max_cards = max(1, len(hand))

        # Calcula tamaño de carta base (más grande) y ancho total
        card_w = min(int(CELL_W * 0.95), max(_S(50), int((available_w - (max_cards - 1) * gap) / max_cards)))
        card_h = min(int(HAND_PANEL_H - _S(56)), int(CELL_H * 0.95))
        total_w = max_cards * card_w + (max_cards - 1) * gap
        self._hand_total_w = total_w
        self._clamp_hand_scroll()

        # Ventana visible dentro de area
        x = area.left + MARG - self._hand_scroll_x
        y = area.top + _S(40)
        self._hand_rects.clear()

        for i, card in enumerate(hand):
            rect = pygame.Rect(x, y, card_w, card_h)
            if rect.right >= area.left + _S(16) and rect.left <= area.right - _S(16):
                # Dibuja carta (imagen/texto)
                if card and card != "BONUS":
                    img = self._card_image(card)
                    if img is not None:
                        img_scaled = pygame.transform.smoothscale(img, (rect.w, rect.h))
                        s.blit(img_scaled, rect)
                    else:
                        pygame.draw.rect(s, (249, 250, 251), rect, border_radius=_S(8))
                        pygame.draw.rect(s, BORDER, rect, 1, border_radius=_S(8))
                        self._blit_centered_text(s, card, _S(16), rect, SUBTLE)
                else:
                    pygame.draw.rect(s, (249, 250, 251), rect, border_radius=_S(8))
                    pygame.draw.rect(s, BORDER, rect, 1, border_radius=_S(8))
                    self._blit_centered_text(s, card or "?", _S(16), rect, SUBTLE)

                # Selección
                if self._selected_hand_idx == i:
                    pygame.draw.rect(s, INDIGO, rect.inflate(_S(10), _S(10)), _S(3), border_radius=_S(10))

                # Badges de Jack
                badge = None
                if is_two_eyed_jack(card):
                    badge = ("Wild", INDIGO)
                elif is_one_eyed_jack(card):
                    badge = ("Cut", ROSE)
                if badge:
                    btxt, col = badge
                    brect = pygame.Rect(rect.left, rect.top - _S(20), _S(46), _S(18))
                    pygame.draw.rect(s, col, brect, border_radius=_S(6))
                    self._blit_centered_text(s, btxt, _S(11), brect, (255, 255, 255), bold=True)

                self._hand_rects.append((rect, i))
            x += rect.w + gap

        # Scrollbar de la mano (si overflow)
        if total_w > (area.width - 2*MARG):
            track = pygame.Rect(area.left + _S(10), area.bottom - _S(12), area.width - _S(20), _S(6))
            pygame.draw.rect(s, SCROLL_TRACK, track, border_radius=_S(3))
            thumb_w = max(_S(40), int(track.width * (track.width / total_w)))
            max_scroll = total_w - track.width
            x = 0 if max_scroll <= 0 else int(self._hand_scroll_x * (track.width - thumb_w) / max_scroll)
            thumb = pygame.Rect(track.left + x, track.top, thumb_w, track.height)
            pygame.draw.rect(s, SCROLL_THUMB, thumb, border_radius=_S(3))

    def _draw_status(self):
        if not self._status_text or time.time() > self._status_until:
            return
        s = self._screen; assert s is not None
        banner = pygame.Rect(MARG, _S(8), s.get_width() - 2*MARG, _S(32))
        pygame.draw.rect(s, (15, 23, 42), banner, border_radius=_S(8))
        self._blit_centered_text(s, self._status_text, _S(16), banner, (255, 255, 255), bold=True)

    # ---------- Acciones & validación ----------

    def _maybe_burn(self, env, legal_set: Set[int]) -> Optional[int]:
        if self._selected_hand_idx is None:
            self._flash_status("Selecciona una carta primero.")
            return None
        card = self._hand(env)[self._selected_hand_idx]
        if is_two_eyed_jack(card) or is_one_eyed_jack(card):
            self._flash_status("No puedes quemar Jacks.")
            return None
        action = 100 + int(self._selected_hand_idx)
        if action in legal_set:
            self._selected_hand_idx = None
            return action
        self._flash_status("Burn no es legal ahora.")
        return None

    def _maybe_pass(self, env, legal_set: Set[int]) -> Optional[int]:
        include_pass = bool(getattr(env, "_include_pass", False))
        if not include_pass:
            self._flash_status("Pass deshabilitado en esta configuración.")
            return None
        h = int(getattr(env, "max_hand", 7))
        action = 100 + h
        if action in legal_set:
            return action
        self._flash_status("Pass no es legal ahora.")
        return None

    def _can_burn(self, env, legal_set: Set[int]) -> bool:
        if self._selected_hand_idx is None:
            return False
        card = self._hand(env)[self._selected_hand_idx]
        if is_two_eyed_jack(card) or is_one_eyed_jack(card):
            return False
        return (100 + int(self._selected_hand_idx)) in legal_set

    def _can_pass(self, env, legal_set: Set[int]) -> bool:
        include_pass = bool(getattr(env, "_include_pass", False))
        if not include_pass:
            return False
        h = int(getattr(env, "max_hand", 7))
        return (100 + h) in legal_set

    def _jack_badge_text(self, env, hand_idx: int) -> str:
        try:
            card = self._hand(env)[hand_idx]
        except Exception:
            return ""
        if is_two_eyed_jack(card):
            return "Wild Jack seleccionado"
        if is_one_eyed_jack(card):
            return "Cut Jack seleccionado"
        return card or ""

    # ---------- Datos ----------

    def _hand(self, env) -> List[str]:
        try:
            return list(env._hand_for(env.current_player))
        except Exception:
            return []

    def _imm_win_cells(self, env, my_team: int) -> Set[Tuple[int, int]]:
        try:
            f = env._team_features_fast(env.game_engine.state, my_team)
            return set(f.get("imm_win_cells", set()))
        except Exception:
            return set()

    def _detect_sequences_with_teams(self, env) -> Dict[Tuple[int, int], List[Tuple[str, int]]]:
        """Devuelve celdas pertenecientes a ventanas completas por eje y equipo.
        Mapa: (r,c) -> [(axis, team), ...] con axis en {'H','V','D1','D2'}
        """
        axes: Dict[Tuple[int, int], List[Tuple[str, int]]] = {}
        try:
            env._ensure_windows()
            st = env.game_engine.state
            teams = max(1, int(getattr(env.gconf, "teams", 2)))
            for w in env._WINDOWS_CACHE:  # type: ignore[attr-defined]
                r0, c0 = w[0]
                r4, c4 = w[-1]
                if r0 == r4:
                    axis = 'H'
                elif c0 == c4:
                    axis = 'V'
                elif (r4 - r0) == (c4 - c0):
                    axis = 'D1'
                else:
                    axis = 'D2'
                for team in range(teams):
                    ok = True
                    for (r, c) in w:
                        printed = BOARD_LAYOUT[r][c]
                        chip = st.board[r][c]
                        if printed == "BONUS":
                            continue
                        if chip != team:
                            ok = False
                            break
                    if ok:
                        for rc in w:
                            lst = axes.get(rc, [])
                            if (axis, team) not in lst:
                                lst.append((axis, team))
                            axes[rc] = lst
        except Exception:
            pass
        return axes

    # ---------- Assets & widgets ----------

    def _card_image(self, code: str) -> Optional[pygame.Surface]:
        if not code or code == "BONUS":
            return None
        if code in CARD_IMG_CACHE:
            return CARD_IMG_CACHE[code]
        path = os.path.join(self.card_img_dir, f"{code}.png")
        if os.path.exists(path):
            try:
                img = pygame.image.load(path).convert_alpha()
                CARD_IMG_CACHE[code] = img
                return img
            except Exception:
                return None
        return None

    def _button(self, s: pygame.Surface, label: str, pos: Tuple[int, int], enabled: bool = True) -> pygame.Rect:
        rect = pygame.Rect(pos[0], pos[1], PANEL_W - _S(24), _S(40))
        pygame.draw.rect(s, (229, 231, 235) if not enabled else (51, 65, 85), rect, border_radius=_S(10))
        self._blit_centered_text(s, label, _S(16), rect, (71, 85, 105) if not enabled else (255, 255, 255), bold=True)
        pygame.draw.rect(s, BORDER, rect, 1, border_radius=_S(10))
        return rect

    def _blit_text(self, s: pygame.Surface, txt: str, size: int, topleft: Tuple[int, int],
                   color: Tuple[int, int, int] = TEXT, bold: bool = False):
        font = _safe_font(size, bold=bold)
        img = font.render(txt, True, color)
        s.blit(img, topleft)

    def _blit_centered_text(self, s: pygame.Surface, txt: str, size: int, rect: pygame.Rect,
                            color: Tuple[int, int, int] = TEXT, bold: bool = False):
        font = _safe_font(size, bold=bold)
        img = font.render(txt, True, color)
        img_rect = img.get_rect(center=rect.center)
        s.blit(img, img_rect)

    def _flash_status(self, text: str, dur: float = 1.4):
        if not text:
            return
        self._status_text = text
        self._status_until = time.time() + dur
