"""Tkinter user interface for the Sequence game.

This module defines a GUI for playing Sequence with configurable teams and
opponents.  It supports human vs human, human vs AI, or AI vs AI games,
with up to three teams.  The UI displays the board with card faces,
allows players to select cards from their hand and click board positions
to play or remove chips, and manages the game loop.  A quick training
button trains a reinforcement learning agent using self‑play and saves
weights for future use.

The interface consists of two main screens: a configuration screen for
selecting the number of teams, players per team, and whether each team
is controlled by a human or the AI; and the gameplay screen showing the
board, player hands, and status messages.
"""

from __future__ import annotations

import os
import threading
import tkinter as tk
from tkinter import messagebox, filedialog
from typing import Dict, Optional, Tuple

from PIL import ImageTk

from .game import Game
from .cards import generate_card_image, generate_all_cards, CARD_WIDTH, CARD_HEIGHT
from .rl.agent import RLAgent, HeuristicAgent
from .rl.train import train as rl_train

# Default path for saving/loading agent weights
DEFAULT_WEIGHTS_PATH = os.path.join(
    os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
    "assets",
    "weights",
    "agent_default.npz",
)


class GameUI:
    def __init__(self) -> None:
        # Pre-generate all card images
        generate_all_cards()
        self.root = tk.Tk()
        self.root.title("Sequence RL Game")
        self.root.protocol("WM_DELETE_WINDOW", self.root.quit)
        # Variables for configuration
        self.num_teams_var = tk.IntVar(value=2)
        self.players_per_team_var = tk.IntVar(value=1)
        self.team_human_vars: Dict[str, tk.BooleanVar] = {
            'B': tk.BooleanVar(value=True),
            'R': tk.BooleanVar(value=True),
            'Y': tk.BooleanVar(value=True),
        }
        # Storage for RL agents per team
        self.agents: Dict[str, RLAgent] = {}
        # Current game
        self.game: Optional[Game] = None
        # Selected card
        self.selected_card: Optional[str] = None
        # Mapping from board item id to position
        self.board_item_positions: Dict[int, Tuple[int, int]] = {}
        # Photo cache to prevent garbage collection
        self.photo_cache: Dict[str, ImageTk.PhotoImage] = {}
        # Setup screens
        self._create_start_screen()

    def _create_start_screen(self) -> None:
        self.start_frame = tk.Frame(self.root)
        self.start_frame.pack(padx=10, pady=10)
        # Title
        tk.Label(self.start_frame, text="Sequence Game Setup", font=("Arial", 16)).pack(pady=5)
        # Number of teams
        tk.Label(self.start_frame, text="Number of teams:").pack(anchor="w")
        tk.OptionMenu(self.start_frame, self.num_teams_var, 2, 3).pack(fill="x")
        # Players per team
        tk.Label(self.start_frame, text="Players per team (1 or even number):").pack(anchor="w")
        tk.OptionMenu(self.start_frame, self.players_per_team_var, 1, 2, 4).pack(fill="x")
        # Team controllers
        tk.Label(self.start_frame, text="Team controllers:").pack(anchor="w")
        for team, var in self.team_human_vars.items():
            frame = tk.Frame(self.start_frame)
            frame.pack(fill="x")
            tk.Label(frame, text=f"Team {team}").pack(side="left")
            tk.Checkbutton(frame, text="Human", variable=var).pack(side="left")
        # Buttons
        btn_frame = tk.Frame(self.start_frame)
        btn_frame.pack(pady=10)
        tk.Button(btn_frame, text="Start Game", command=self.start_game).pack(side="left", padx=5)
        tk.Button(btn_frame, text="Quick Train RL", command=self.quick_train_rl).pack(side="left", padx=5)
        tk.Button(btn_frame, text="Load Weights", command=self.load_weights).pack(side="left", padx=5)

    def start_game(self) -> None:
        """Initialize and start a new game based on configuration."""
        num_teams = self.num_teams_var.get()
        players_per_team = self.players_per_team_var.get()
        # Validate players_per_team
        if players_per_team != 1 and players_per_team % 2 != 0:
            messagebox.showerror("Invalid Input", "Players per team must be 1 or an even number.")
            return
        # Build team_humans dict
        active_teams = ['B', 'R', 'Y'][:num_teams]
        team_humans = {t: self.team_human_vars[t].get() for t in active_teams}
        # Create game instance
        self.game = Game(num_teams=num_teams, players_per_team=players_per_team,
                         team_humans=team_humans)
        # Create agents for AI teams
        self.agents.clear()
        for t in active_teams:
            if not team_humans[t]:
                agent = RLAgent(team=t)
                # Load existing weights if available
                if os.path.exists(DEFAULT_WEIGHTS_PATH):
                    agent.load(DEFAULT_WEIGHTS_PATH)
                self.agents[t] = agent
        # Hide start screen and show game screen
        self.start_frame.pack_forget()
        self._create_game_screen()
        # Begin game
        self.update_status()
        # If first player is AI, let it move
        self.root.after(100, self._maybe_ai_move)

    def _create_game_screen(self) -> None:
        self.game_frame = tk.Frame(self.root)
        self.game_frame.pack()
        # Board canvas
        self.board_canvas = tk.Canvas(self.game_frame, width=600, height=600)
        self.board_canvas.grid(row=0, column=0, padx=10, pady=10)
        self.board_canvas.bind("<Button-1>", self.on_board_click)
        # Hand frame
        self.hand_frame = tk.Frame(self.game_frame)
        self.hand_frame.grid(row=1, column=0, pady=5)
        # Status bar
        self.status_var = tk.StringVar()
        tk.Label(self.game_frame, textvariable=self.status_var, anchor="w").grid(row=2, column=0, sticky="w")
        # Draw board
        self.draw_board()
        # Bind keypresses
        self.root.bind("n", lambda event: self._maybe_ai_move())
        self.root.bind("t", lambda event: self.quick_train_rl())

    def draw_board(self) -> None:
        """Render the board and hand."""
        if not self.game:
            return
        # Determine cell size based on canvas dimensions
        canvas_w = int(self.board_canvas['width'])
        canvas_h = int(self.board_canvas['height'])
        cell_w = canvas_w // 10
        cell_h = canvas_h // 10
        self.board_canvas.delete("all")
        self.board_item_positions.clear()
        # Draw card images and tokens
        for y, row in enumerate(self.game.board):
            for x, cell in enumerate(row):
                card_code = cell.card
                # Fetch or generate photo
                pil_img = generate_card_image(card_code)
                # Resize to fit cell
                img = pil_img.resize((cell_w - 4, int((cell_w - 4) * CARD_HEIGHT / CARD_WIDTH)), ImageTk.Image.BILINEAR)
                photo = ImageTk.PhotoImage(img)
                self.photo_cache[f"{card_code}_{y}_{x}"] = photo
                cx = x * cell_w + 2
                cy = y * cell_h + 2
                item_id = self.board_canvas.create_image(cx, cy, anchor='nw', image=photo)
                self.board_item_positions[item_id] = (y, x)
                # Draw token overlay if present
                if cell.token and cell.token != 'W':
                    # Draw a coloured oval centered in the cell
                    centre_x = x * cell_w + cell_w / 2
                    centre_y = y * cell_h + cell_h / 2
                    radius = min(cell_w, cell_h) * 0.25
                    colour = {'B': 'blue', 'R': 'red', 'Y': 'yellow'}.get(cell.token, 'grey')
                    self.board_canvas.create_oval(
                        centre_x - radius, centre_y - radius,
                        centre_x + radius, centre_y + radius,
                        fill=colour, outline='black'
                    )
        # Draw player hand
        self.draw_hand()

    def draw_hand(self) -> None:
        """Render the current player's hand as buttons."""
        # Clear existing widgets
        for widget in self.hand_frame.winfo_children():
            widget.destroy()
        if not self.game:
            return
        player = self.game.current_player()
        # Display label
        tk.Label(self.hand_frame, text=f"Team {player.team} hand: ").pack(side="left")
        for idx, card in enumerate(player.hand):
            pil_img = generate_card_image(card)
            # scale down hand cards to 60px width
            width = 50
            height = int(width * CARD_HEIGHT / CARD_WIDTH)
            img = pil_img.resize((width, height), ImageTk.Image.BILINEAR)
            photo = ImageTk.PhotoImage(img)
            self.photo_cache[f"hand_{idx}_{card}"] = photo
            btn = tk.Button(self.hand_frame, image=photo, command=lambda c=card: self.select_card(c))
            # Bind right-click for discard
            btn.bind("<Button-3>", lambda event, c=card: self.discard_card(c))
            btn.pack(side="left", padx=2)

    def select_card(self, card: str) -> None:
        """Select a card from the hand for play."""
        if not self.game:
            return
        # If same card clicked twice, deselect
        if self.selected_card == card:
            self.selected_card = None
        else:
            self.selected_card = card
        self.update_status()

    def on_board_click(self, event) -> None:
        """Handle clicks on the board canvas to place or remove chips."""
        if not self.game or self.selected_card is None:
            return
        # Determine which cell was clicked
        canvas_w = int(self.board_canvas['width'])
        canvas_h = int(self.board_canvas['height'])
        cell_w = canvas_w // 10
        cell_h = canvas_h // 10
        x = event.x // cell_w
        y = event.y // cell_h
        # Validate coordinates
        if x < 0 or x >= 10 or y < 0 or y >= 10:
            return
        # Attempt to play the selected card on that cell
        success = self.game.play_turn(self.selected_card, (y, x))
        if success:
            self.selected_card = None
            self.draw_board()
            # Check for game end
            if self.game.winner:
                messagebox.showinfo("Game Over", f"Team {self.game.winner} wins!")
                self.return_to_start()
                return
            else:
                # If next player is AI, schedule AI move
                self.update_status()
                self.root.after(100, self._maybe_ai_move)
        else:
            messagebox.showwarning("Invalid Move", "That move is not allowed.")

    def discard_card(self, card: str) -> None:
        """Discard a dead card from the hand."""
        if not self.game:
            return
        if self.game.discard_card(card):
            self.selected_card = None
            self.draw_board()
            self.update_status()
            # After discarding, advance to next player
            self.root.after(100, self._maybe_ai_move)

    def update_status(self) -> None:
        """Update the status bar with current information."""
        if not self.game:
            self.status_var.set("")
            return
        player = self.game.current_player()
        # Build status message
        msg = f"Team {player.team}'s turn. "
        if self.selected_card:
            msg += f"Selected card: {self.selected_card}. "
        # Sequence counts
        counts = {t: 0 for t in ['B', 'R', 'Y']}
        for p in self.game.players:
            counts[p.team] = p.sequences
        msg += "Sequences - " + ", ".join(f"{t}:{c}" for t, c in counts.items() if c or t in counts)
        self.status_var.set(msg)

    def _maybe_ai_move(self) -> None:
        """If the current player is controlled by an AI, make its move."""
        if not self.game or self.game.winner:
            return
        player = self.game.current_player()
        if player.team in self.agents:
            agent = self.agents[player.team]
            card, target = agent.select_move(self.game, greedy=True)
            if card and target:
                success = self.game.play_turn(card, target)
                if success:
                    # Clear selected card to avoid interference
                    self.selected_card = None
                    self.draw_board()
                    if self.game.winner:
                        messagebox.showinfo("Game Over", f"Team {self.game.winner} wins!")
                        self.return_to_start()
                        return
                    else:
                        self.update_status()
                        # If next player is also AI, continue
                        self.root.after(100, self._maybe_ai_move)
                else:
                    # Fallback: if move failed, just skip
                    self.game.next_player()
                    self.root.after(100, self._maybe_ai_move)
        else:
            # Human player's turn
            self.draw_board()
            self.update_status()

    def quick_train_rl(self) -> None:
        """Run a quick training session in a separate thread."""
        def train_thread():
            messagebox.showinfo("Training", "Starting quick training... this may take a moment.")
            # Run training for a small number of episodes
            rl_train(num_episodes=100, seed=42, save_path=DEFAULT_WEIGHTS_PATH, verbose=False)
            messagebox.showinfo("Training", "Training complete and weights saved.")
        # Spawn training in another thread to avoid blocking UI
        threading.Thread(target=train_thread, daemon=True).start()

    def load_weights(self) -> None:
        """Load agent weights from a .npz file and update any existing agents."""
        path = filedialog.askopenfilename(filetypes=[("NumPy weights", "*.npz"), ("All files", "*.*")])
        if not path:
            return
        # Load into each agent
        for agent in self.agents.values():
            agent.load(path)
        messagebox.showinfo("Load Weights", f"Loaded weights from {path}")

    def return_to_start(self) -> None:
        """Return to the start screen after a game ends."""
        # Destroy game screen and show start screen again
        self.game_frame.destroy()
        self.start_frame.pack(padx=10, pady=10)
        self.game = None
        self.agents.clear()
        self.selected_card = None
        self.update_status()

    def run(self) -> None:
        self.root.mainloop()


if __name__ == "__main__":
    ui = GameUI()
    ui.run()