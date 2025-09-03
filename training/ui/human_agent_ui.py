# training/ui/human_agent_ui.py
# UI Pygame para HumanAgent con:
# - Hints de secuencia más claros (banda de color por eje y equipo + leyenda)
# - Scroll del tablero (rueda: vertical | Shift+rueda: horizontal)
# - Scroll horizontal de la mano (rueda sobre la mano o arrastre)
# - Elementos más grandes vía UI_SCALE
# - Zoom del tablero con barra deslizante + atajos:
#       * Ctrl + rueda del mouse (equivale al gesto de zoom de Windows/trackpad)
#       * Botones +/- en el panel derecho
#       * Botón Reset
# - Pan del tablero con botón central del mouse (MMB)
#
# Carga tablero desde training/assets/boards/standard_10x10.json (BOARD_LAYOUT)
# Renderiza 10x10 con imágenes opcionales en assets/cards/{rank}{suit}.png
# Interacción:
#   - Click en carta para seleccionarla
#   - Click en celda para jugar
#   - Botón "Burn"/"Pass" o atajos [B]/[P]
#   - Zoom: Ctrl + rueda, botones +/- o deslizador
#   - Pan: arrastre con botón central del mouse
# Devuelve la acción entera según el action space unificado de SequenceEnv
#
# Notas:
# - La UI asume 10x10 por el mapeo r*10+c
# - UI_SCALE ajusta tamaños base; el zoom modifica el tamaño del grid (celdas) dinámicamente
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

UI_SCALE = 0.8  # <---- ajusta aquí el tamaño general de la UI

def _S(x: float) -> int:
    return int(round(x * UI_SCALE))

PANEL_W = _S(260)
CELL_W, CELL_H = _S(56), _S(68)         # tamaños base de celda (se escalan con ZOOM)
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

# Add these under the existing color constants
PLAYABLE_BORDER = (250, 204, 21)       # amber-400, bright yellow
PLAYABLE_BORDER_DARK = (202, 138, 4)   # amber-700, contrast stroke


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

# ------ Zoom (dinámico sobre el grid) ------
ZOOM_MIN = 0.6
ZOOM_MAX = 1.6
ZOOM_STEP = 0.08

# Guardamos tamaños base del grid/celda (antes de aplicar zoom)
BASE_CELL_W = CELL_W
BASE_CELL_H = CELL_H

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
        self._board_rect = pygame.Rect(MARG, MARG, GRID_W, GRID_H)
        self._hand_rects: List[Tuple[pygame.Rect, int]] = []  # (rect, hand_idx)
        self._burn_rect: Optional[pygame.Rect] = None
        self._pass_rect: Optional[pygame.Rect] = None

        # Controles Zoom (se crean en _draw_panel)
        self.zoom: float = 1.0
        self._zoom_track_rect: Optional[pygame.Rect] = None
        self._zoom_thumb_rect: Optional[pygame.Rect] = None
        self._zoom_minus_rect: Optional[pygame.Rect] = None
        self._zoom_plus_rect: Optional[pygame.Rect] = None
        self._zoom_reset_rect: Optional[pygame.Rect] = None
        self._zoom_thumb_w: int = _S(24)
        self._zoom_dragging: bool = False

        # Scroll estado
        self._board_view: pygame.Rect = pygame.Rect(0, 0, GRID_W, GRID_H)  # viewport visible dentro del grid
        self._board_scroll_x = 0
        self._board_scroll_y = 0

        self._hand_scroll_x = 0
        self._hand_dragging = False
        self._hand_drag_x0 = 0
        self._hand_drag_start = 0

        # Pan del tablero (MMB)
        self._board_dragging = False
        self._board_drag_x0 = 0
        self._board_drag_y0 = 0
        self._board_scroll_x0 = 0
        self._board_scroll_y0 = 0

        # Scrollbars interactivos del tablero
        self._hscroll_track_rect: Optional[pygame.Rect] = None
        self._vscroll_track_rect: Optional[pygame.Rect] = None
        self._hscroll_thumb_rect: Optional[pygame.Rect] = None
        self._vscroll_thumb_rect: Optional[pygame.Rect] = None
        self._hscroll_dragging = False
        self._vscroll_dragging = False
        self._hscroll_drag_dx = 0
        self._vscroll_drag_dy = 0

        self._boot()

    # ---------- Helpers dinámicos de tamaño (dependen de zoom) ----------

    @property
    def cell_w(self) -> int:
        return max(12, int(BASE_CELL_W * self.zoom))

    @property
    def cell_h(self) -> int:
        return max(12, int(BASE_CELL_H * self.zoom))

    @property
    def grid_w(self) -> int:
        return 10 * self.cell_w

    @property
    def grid_h(self) -> int:
        return 10 * self.cell_h

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
        # Ventana objetivo inicial basada en tamaños base
        win_w = min(GRID_W + PANEL_W + 3*MARG, WIN_W_TARGET)
        win_h = min(GRID_H + HAND_PANEL_H + 3*MARG, WIN_H_TARGET)
        self._screen = pygame.display.set_mode((win_w, win_h), pygame.RESIZABLE)
        self._clock = pygame.time.Clock()

    def _layout(self):
        """Calcula rects visibles (board viewport, panel, mano) considerando tamaño ventana."""
        assert self._screen is not None
        sw, sh = self._screen.get_size()

        # Tablero a la izquierda
        board_w_avail = max(_S(600), sw - PANEL_W - 3*MARG)
        board_h_avail = max(_S(480), sh - HAND_PANEL_H - 3*MARG)
        self._board_rect = pygame.Rect(MARG, MARG, board_w_avail, board_h_avail)

        # Panel a la DERECHA
        self._panel_rect = pygame.Rect(self._board_rect.right + MARG, MARG, PANEL_W, board_h_avail)

        # Mano
        self._hand_area = pygame.Rect(MARG, self._board_rect.bottom + MARG, sw - 2*MARG, HAND_PANEL_H)

        # Limites de scroll del tablero (con dimensiones dinámicas por zoom)
        max_scroll_x = max(0, self.grid_w - self._board_rect.width)
        max_scroll_y = max(0, self.grid_h - self._board_rect.height)
        self._board_scroll_x = max(0, min(self._board_scroll_x, max_scroll_x))
        self._board_scroll_y = max(0, min(self._board_scroll_y, max_scroll_y))
        self._board_view = pygame.Rect(self._board_scroll_x, self._board_scroll_y,
                                       self._board_rect.width, self._board_rect.height)

    def _handle_scroll_events(self, event: pygame.event.Event):
        # Mouse wheel: tablero y mano / Ctrl+wheel = ZOOM (gesto Windows/trackpad)
        keys = pygame.key.get_mods()
        shift = keys & pygame.KMOD_SHIFT
        ctrl = keys & pygame.KMOD_CTRL
        mx, my = pygame.mouse.get_pos()

        if event.type == pygame.MOUSEWHEEL:
            # Ctrl + rueda -> Zoom (si el puntero está sobre el tablero, anclamos al cursor)
            if ctrl:
                anchor = (mx, my) if self._board_rect.collidepoint(mx, my) else None
                factor = (1.0 + ZOOM_STEP) if event.y > 0 else (1.0 / (1.0 + ZOOM_STEP))
                self._set_zoom(self.zoom * factor, anchor_px=anchor)
                return

            # Scroll sobre tablero
            if self._board_rect.collidepoint(mx, my):
                if shift:
                    self._board_scroll_x -= event.y * _S(40)
                else:
                    self._board_scroll_y -= event.y * _S(40)
                self._clamp_board_scroll()
            # Scroll sobre mano
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

        # --- Scrollbars del tablero (clic y arrastre estilo Windows) ---
        if event.type == pygame.MOUSEBUTTONDOWN and event.button == 1:
            # Horizontal
            if self._hscroll_thumb_rect and self._hscroll_thumb_rect.collidepoint(mx, my):
                self._hscroll_dragging = True
                self._hscroll_drag_dx = mx - self._hscroll_thumb_rect.left
            elif self._hscroll_track_rect and self._hscroll_track_rect.collidepoint(mx, my):
                thumb_w = self._hscroll_thumb_rect.width if self._hscroll_thumb_rect else _S(40)
                rel = (mx - self._hscroll_track_rect.left - thumb_w/2) / max(1, (self._hscroll_track_rect.width - thumb_w))
                rel = max(0.0, min(1.0, rel))
                max_scroll = max(0, self.grid_w - self._board_rect.width)
                self._board_scroll_x = int(rel * max_scroll)
                self._clamp_board_scroll()
                # Empieza drag inmediatamente (sensación nativa)
                self._hscroll_dragging = True
                self._hscroll_drag_dx = thumb_w // 2

            # Vertical
            if self._vscroll_thumb_rect and self._vscroll_thumb_rect.collidepoint(mx, my):
                self._vscroll_dragging = True
                self._vscroll_drag_dy = my - self._vscroll_thumb_rect.top
            elif self._vscroll_track_rect and self._vscroll_track_rect.collidepoint(mx, my):
                thumb_h = self._vscroll_thumb_rect.height if self._vscroll_thumb_rect else _S(40)
                rel = (my - self._vscroll_track_rect.top - thumb_h/2) / max(1, (self._vscroll_track_rect.height - thumb_h))
                rel = max(0.0, min(1.0, rel))
                max_scroll = max(0, self.grid_h - self._board_rect.height)
                self._board_scroll_y = int(rel * max_scroll)
                self._clamp_board_scroll()
                self._vscroll_dragging = True
                self._vscroll_drag_dy = thumb_h // 2

        if event.type == pygame.MOUSEMOTION:
            if self._hscroll_dragging and self._hscroll_track_rect and self._hscroll_thumb_rect:
                thumb_w = self._hscroll_thumb_rect.width
                x = mx - self._hscroll_drag_dx
                x = max(self._hscroll_track_rect.left, min(x, self._hscroll_track_rect.right - thumb_w))
                rel = (x - self._hscroll_track_rect.left) / max(1, (self._hscroll_track_rect.width - thumb_w))
                max_scroll = max(0, self.grid_w - self._board_rect.width)
                self._board_scroll_x = int(rel * max_scroll)
                self._clamp_board_scroll()
            if self._vscroll_dragging and self._vscroll_track_rect and self._vscroll_thumb_rect:
                thumb_h = self._vscroll_thumb_rect.height
                y = my - self._vscroll_drag_dy
                y = max(self._vscroll_track_rect.top, min(y, self._vscroll_track_rect.bottom - thumb_h))
                rel = (y - self._vscroll_track_rect.top) / max(1, (self._vscroll_track_rect.height - thumb_h))
                max_scroll = max(0, self.grid_h - self._board_rect.height)
                self._board_scroll_y = int(rel * max_scroll)
                self._clamp_board_scroll()

        if event.type == pygame.MOUSEBUTTONUP and event.button == 1:
            self._hscroll_dragging = False
            self._vscroll_dragging = False

        # Pan del tablero con botón central (MMB)
        if event.type == pygame.MOUSEBUTTONDOWN and event.button == 2:
            if self._board_rect.collidepoint(mx, my):
                self._board_dragging = True
                self._board_drag_x0 = mx
                self._board_drag_y0 = my
                self._board_scroll_x0 = self._board_scroll_x
                self._board_scroll_y0 = self._board_scroll_y
        if event.type == pygame.MOUSEBUTTONUP and event.button == 2:
            self._board_dragging = False
        if event.type == pygame.MOUSEMOTION and self._board_dragging:
            dx = mx - self._board_drag_x0
            dy = my - self._board_drag_y0
            self._board_scroll_x = self._board_scroll_x0 - dx
            self._board_scroll_y = self._board_scroll_y0 - dy
            self._clamp_board_scroll()

    def _clamp_board_scroll(self):
        max_x = max(0, self.grid_w - self._board_rect.width)
        max_y = max(0, self.grid_h - self._board_rect.height)
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
                if event.key in (pygame.K_PLUS, pygame.K_EQUALS):  # '+'
                    self._set_zoom(self.zoom + ZOOM_STEP)
                if event.key in (pygame.K_MINUS,):
                    self._set_zoom(self.zoom - ZOOM_STEP)
                if event.key == pygame.K_0 and (pygame.key.get_mods() & pygame.KMOD_CTRL):
                    self._set_zoom(1.0)

            # Scroll / drag / pan / zoom
            self._handle_scroll_events(event)

            # Clicks + controles de zoom + tablero
            if event.type == pygame.MOUSEBUTTONDOWN and event.button == 1:
                mx, my = event.pos

                # Controles de zoom
                if self._zoom_minus_rect and self._zoom_minus_rect.collidepoint(mx, my):
                    self._set_zoom(self.zoom - ZOOM_STEP, anchor_px=(mx, my))
                    return None
                if self._zoom_plus_rect and self._zoom_plus_rect.collidepoint(mx, my):
                    self._set_zoom(self.zoom + ZOOM_STEP, anchor_px=(mx, my))
                    return None
                if self._zoom_reset_rect and self._zoom_reset_rect.collidepoint(mx, my):
                    self._set_zoom(1.0)
                    return None
                # Click en la pista del deslizador: saltar y empezar drag
                if self._zoom_track_rect and self._zoom_track_rect.collidepoint(mx, my):
                    self._update_zoom_from_slider(mx)
                    self._zoom_dragging = True
                    return None
                if self._zoom_thumb_rect and self._zoom_thumb_rect.collidepoint(mx, my):
                    self._zoom_dragging = True
                    return None

                # Mano
                for rect, hidx in self._hand_rects:
                    if rect.collidepoint(mx, my):
                        self._selected_hand_idx = hidx
                        self._flash_status(self._jack_badge_text(env, hidx))
                        return None
                # Botones principales
                if self._burn_rect and self._burn_rect.collidepoint(mx, my):
                    return self._maybe_burn(env, legal_set)
                if self._pass_rect and self._pass_rect.collidepoint(mx, my):
                    return self._maybe_pass(env, legal_set)
                # Tablero
                if self._board_rect.collidepoint(mx, my):
                    # Coordenadas dentro del grid con scroll y zoom aplicado
                    gx = mx - self._board_rect.left + self._board_scroll_x
                    gy = my - self._board_rect.top + self._board_scroll_y
                    c = int(gx // self.cell_w)
                    r = int(gy // self.cell_h)
                    if 0 <= r < 10 and 0 <= c < 10:
                        action = int(r * 10 + c)
                        if action in legal_set:
                            return action
                        else:
                            self._flash_status("No es legal en esa casilla.")

            if event.type == pygame.MOUSEBUTTONUP and event.button == 1:
                self._zoom_dragging = False

            if event.type == pygame.MOUSEMOTION and self._zoom_dragging:
                mx, _ = event.pos
                self._update_zoom_from_slider(mx)

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

        # Offset para pintar celdas actuales considerando scroll (con dimensiones dinámicas)
        x0 = self._board_rect.left - self._board_scroll_x
        y0 = self._board_rect.top - self._board_scroll_y

        now = time.time()
        pulse = (1 + 0.25 * np.sin(now * 5.0))  # factor 1..1.25

        cw, ch = self.cell_w, self.cell_h

        for r in range(10):
            for c in range(10):
                cell_rect = pygame.Rect(x0 + c * cw, y0 + r * ch, cw, ch)
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
                        scale = min((cw - _S(6)) / img_rect.w, (ch - _S(6)) / img_rect.h)
                        if scale < 1.0 or scale > 1.0:
                            img = pygame.transform.smoothscale(img, (int(img_rect.w * scale), int(img_rect.h * scale)))
                            img_rect = img.get_rect(center=cell_rect.center)
                        s.blit(img, img_rect)
                    else:
                        self._blit_centered_text(s, printed, _S(16), cell_rect, SUBTLE)
                elif printed == "BONUS":
                    self._blit_centered_text(s, "★", _S(20), cell_rect, (234, 179, 8), bold=True)

                # Borde de celda si es legal
                # Borde de celda si es legal (amarillo y grueso)
                a = r * 10 + c
                if a in legal_set:
                    border_w = max(3, cw // 8)  # más grueso que antes
                    pygame.draw.rect(s, PLAYABLE_BORDER, cell_rect, width=border_w, border_radius=_S(8))
                    # Trazo interior más oscuro para contraste
                    inner_rect = cell_rect.inflate(-max(2, border_w // 2), -max(2, border_w // 2))
                    pygame.draw.rect(s, PLAYABLE_BORDER_DARK, inner_rect, width=max(2, border_w // 4),
                                     border_radius=_S(6))

                # Chip (centrado y grande)
                if chip is not None:
                    color = TEAM_COLORS.get(int(chip), (107, 114, 128))
                    center = cell_rect.center
                    rad = max(_S(10), int(min(cw, ch) * 0.32))
                    # cuerpo
                    pygame.draw.circle(s, color, center, rad)
                    # anillo blanco nítido
                    pygame.draw.circle(s, MINE_RING, center, rad, max(2, rad // 4))

                # Overlays de secuencia (más gruesos y coloreados por equipo)
                axes = seq_overlays.get((r, c), [])
                for axis, team in axes:
                    col = TEAM_COLORS.get(team, INDIGO)
                    self._draw_sequence_band(s, cell_rect, axis, col, alpha=110, thickness=max(4, cw // 7))

                # Win inmediato para mi equipo (anillo pulsante + texto)
                if (r, c) in imm_cells:
                    rad = int(max(6, cw // 6) * pulse)
                    pygame.draw.circle(s, GREEN, (cell_rect.centerx, cell_rect.centery), rad, max(2, cw // 30))
                    self._blit_centered_text(s, "WIN", max(10, cw // 8), cell_rect.move(0, _S(16)), GREEN, bold=True)

                # Bordes finos de celda
                pygame.draw.rect(s, BORDER, cell_rect, width=1)

        # Scrollbars del tablero (si procede)
        self._draw_board_scrollbars(s)

    def _draw_sequence_band(self, s: pygame.Surface, rect: pygame.Rect,
                            axis: str, color: Tuple[int, int, int], alpha: int = 110, thickness: int = 6):
        # Dibujar una banda semitransparente gruesa por eje
        band = pygame.Surface((rect.width, rect.height), pygame.SRCALPHA)
        col = (*color, alpha)
        if axis == 'H':
            pygame.draw.rect(band, col, (GAP//2, rect.height//2 - thickness//2, rect.width - GAP, thickness), border_radius=max(2, thickness//2))
        elif axis == 'V':
            pygame.draw.rect(band, col, (rect.width//2 - thickness//2, GAP//2, thickness, rect.height - GAP), border_radius=max(2, thickness//2))
        elif axis == 'D1':  # \
            pygame.draw.line(band, col, (0, rect.height), (rect.width, 0), thickness)
        elif axis == 'D2':  # /
            pygame.draw.line(band, col, (0, 0), (rect.width, rect.height), thickness)
        s.blit(band, rect.topleft)

    def _draw_board_scrollbars(self, s: pygame.Surface):
        # Muestra barras si el grid excede el viewport
        need_h = self.grid_w > self._board_rect.width
        need_v = self.grid_h > self._board_rect.height
        self._hscroll_track_rect = None
        self._vscroll_track_rect = None
        self._hscroll_thumb_rect = None
        self._vscroll_thumb_rect = None
        if not (need_h or need_v):
            return

        # Track horizontal
        if need_h:
            track = pygame.Rect(self._board_rect.left, self._board_rect.bottom - _S(10),
                                self._board_rect.width, _S(8))
            pygame.draw.rect(s, SCROLL_TRACK, track, border_radius=_S(4))
            # Thumb
            thumb_w = max(_S(40), int(self._board_rect.width * self._board_rect.width / self.grid_w))
            max_scroll = self.grid_w - self._board_rect.width
            x = 0 if max_scroll == 0 else int(self._board_scroll_x * (self._board_rect.width - thumb_w) / max_scroll)
            thumb = pygame.Rect(track.left + x, track.top, thumb_w, track.height)
            pygame.draw.rect(s, SCROLL_THUMB, thumb, border_radius=_S(4))
            self._hscroll_track_rect = track
            self._hscroll_thumb_rect = thumb

        # Track vertical
        if need_v:
            track = pygame.Rect(self._board_rect.right - _S(10), self._board_rect.top,
                                _S(8), self._board_rect.height)
            pygame.draw.rect(s, SCROLL_TRACK, track, border_radius=_S(4))
            thumb_h = max(_S(40), int(self._board_rect.height * self._board_rect.height / self.grid_h))
            max_scroll = self.grid_h - self._board_rect.height
            y = 0 if max_scroll == 0 else int(self._board_scroll_y * (self._board_rect.height - thumb_h) / max_scroll)
            thumb = pygame.Rect(track.left, track.top + y, track.width, thumb_h)
            pygame.draw.rect(s, SCROLL_THUMB, thumb, border_radius=_S(4))
            self._vscroll_track_rect = track
            self._vscroll_thumb_rect = thumb

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
        # Legal (coincide con el borde amarillo grueso del tablero)
        legal_box = pygame.Rect(lx, ly, _S(28), _S(18))
        pygame.draw.rect(s, (255, 255, 255), legal_box, border_radius=_S(4))
        pygame.draw.rect(s, PLAYABLE_BORDER, legal_box, _S(4), border_radius=_S(4))
        inner_demo = legal_box.inflate(-_S(6), -_S(6))
        pygame.draw.rect(s, PLAYABLE_BORDER_DARK, inner_demo, _S(2), border_radius=_S(3))
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

        # Ganadores (si aplica)
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
                            (panel.left + _S(12), panel.bottom - _S(140)),
                            color=(22, 163, 74), bold=True)

        # Controles de ZOOM
        zoom_top = panel.bottom - _S(140)
        self._draw_zoom_controls(s, pygame.Rect(panel.left + _S(12), zoom_top, panel.width - _S(24), _S(56)))

        # Botones
        btn_y = panel.bottom - _S(88)
        self._burn_rect = self._button(s, "Burn Card [B]", (panel.left + _S(12), btn_y),
                                       enabled=self._can_burn(env, legal_set))
        btn_y += _S(48)
        include_pass = bool(getattr(env, "_include_pass", False))
        self._pass_rect = self._button(s, "Pass [P]", (panel.left + _S(12), btn_y),
                                       enabled=self._can_pass(env, legal_set) if include_pass else False)

    def _draw_zoom_controls(self, s: pygame.Surface, area: pygame.Rect):
        """Barra de zoom con botones +/- y deslizador."""
        # Etiqueta + valor
        self._blit_text(s, "Zoom", _S(16), (area.left, area.top), bold=True)
        pct = int(round(self.zoom * 100))
        self._blit_text(s, f"{pct}%", _S(14), (area.right - _S(56), area.top + _S(2)), color=SUBTLE)

        # Botones +/- (cuadrados)
        btn_size = _S(28)
        minus = pygame.Rect(area.left, area.bottom - btn_size, btn_size, btn_size)
        plus = pygame.Rect(area.left + btn_size + _S(8), area.bottom - btn_size, btn_size, btn_size)
        pygame.draw.rect(s, (51, 65, 85), minus, border_radius=_S(6))
        pygame.draw.rect(s, (51, 65, 85), plus, border_radius=_S(6))
        self._blit_centered_text(s, "–", _S(18), minus, (255, 255, 255), bold=True)
        self._blit_centered_text(s, "+", _S(18), plus, (255, 255, 255), bold=True)
        self._zoom_minus_rect, self._zoom_plus_rect = minus, plus

        # Botón Reset
        reset_w = _S(64)
        reset = pygame.Rect(area.right - reset_w, area.bottom - btn_size, reset_w, btn_size)
        pygame.draw.rect(s, (71, 85, 105), reset, border_radius=_S(6))
        self._blit_centered_text(s, "Reset", _S(14), reset, (255, 255, 255), bold=True)
        self._zoom_reset_rect = reset

        # Track del deslizador
        track_left = plus.right + _S(10)
        track_right = reset.left - _S(10)
        track = pygame.Rect(track_left,
                            area.bottom - btn_size//2 - _S(4),
                            max(_S(60), track_right - track_left),
                            _S(8))
        pygame.draw.rect(s, SCROLL_TRACK, track, border_radius=_S(4))

        # Thumb según valor
        t = (self.zoom - ZOOM_MIN) / (ZOOM_MAX - ZOOM_MIN)
        t = max(0.0, min(1.0, t))
        thumb_w = self._zoom_thumb_w
        thumb_x = track.left + int(t * (track.width - thumb_w))
        thumb = pygame.Rect(thumb_x, track.top - _S(4), thumb_w, track.height + _S(8))
        pygame.draw.rect(s, SCROLL_THUMB, thumb, border_radius=_S(6))
        self._zoom_track_rect = track
        self._zoom_thumb_rect = thumb

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

        # Tamaño de carta base relacionado a la celda y al panel disponible
        cw, ch = self.cell_w, self.cell_h
        card_w = min(int(cw * 0.95), max(_S(50), int((available_w - (max_cards - 1) * gap) / max_cards)))
        card_h = min(int(HAND_PANEL_H - _S(56)), int(ch * 0.95))
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

    # ---------- Zoom helpers ----------

    def _set_zoom(self, new_zoom: float, anchor_px: Optional[Tuple[int, int]] = None):
        """Ajusta el zoom del grid con anclaje opcional en coordenada pantalla (sobre el tablero)."""
        old_zoom = self.zoom
        z = max(ZOOM_MIN, min(ZOOM_MAX, float(new_zoom)))
        if abs(z - old_zoom) < 1e-4:
            return
        old_cw, old_ch = self.cell_w, self.cell_h  # con old zoom
        self.zoom = z
        # Mantener el punto bajo el cursor estable al cambiar zoom (si el cursor está sobre el tablero)
        if anchor_px and self._board_rect.collidepoint(*anchor_px):
            mx, my = anchor_px
            # Coordenadas del punto dentro del grid (antes)
            gx_old = self._board_scroll_x + (mx - self._board_rect.left)
            gy_old = self._board_scroll_y + (my - self._board_rect.top)
            cell_x = gx_old / max(1, old_cw)
            cell_y = gy_old / max(1, old_ch)
            # Nuevas coordenadas objetivo en pixeles
            new_cw, new_ch = self.cell_w, self.cell_h
            gx_new = cell_x * new_cw
            gy_new = cell_y * new_ch
            self._board_scroll_x = int(gx_new - (mx - self._board_rect.left))
            self._board_scroll_y = int(gy_new - (my - self._board_rect.top))
        # Re-clamp para respetar nuevos límites
        self._clamp_board_scroll()

    def _update_zoom_from_slider(self, mouse_x: int):
        if not self._zoom_track_rect:
            return
        track = self._zoom_track_rect
        thumb_w = self._zoom_thumb_w
        t = (mouse_x - (track.left + thumb_w / 2)) / max(1, (track.width - thumb_w))
        t = max(0.0, min(1.0, t))
        self._set_zoom(ZOOM_MIN + t * (ZOOM_MAX - ZOOM_MIN))

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
