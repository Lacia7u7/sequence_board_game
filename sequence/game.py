"""Core game logic for the Sequence board game.

This module models the board, deck, hands and rules for Sequence.  It provides
stateful classes to represent a game, validate moves, apply the effects of
playing cards (including Jacks) and detect completed sequences.  The goal is
to separate the pure rules from the user interface and the reinforcement
learning agent.
"""

from __future__ import annotations

import random
from dataclasses import dataclass, field
from typing import List, Tuple, Dict, Optional, Iterable

from .board_layouts import load_layout, card_positions

# Represent a token colour; empty string means no token, 'W' is reserved for
# corners (wild for everyone).
Token = str  # '', 'B', 'R', 'Y', or 'W'

@dataclass
class Cell:
    card: str  # e.g. '6S' or 'W'
    token: Token = ''  # which team has placed a chip; 'W' for corner
    protected: bool = False  # part of a completed sequence


@dataclass
class Player:
    team: str  # team identifier: 'B', 'R', 'Y'
    is_human: bool
    hand: List[str] = field(default_factory=list)
    sequences: int = 0


class Game:
    """Manage the state and rules of a Sequence game."""

    def __init__(self, num_teams: int = 2, players_per_team: int = 1,
                 team_humans: Dict[str, bool] | None = None, seed: Optional[int] = None):
        if num_teams not in (2, 3):
            raise ValueError("num_teams must be 2 or 3")
        if players_per_team != 1 and players_per_team % 2 != 0:
            raise ValueError("players_per_team must be 1 or an even number")
        self.random = random.Random(seed)
        self.layout = load_layout()
        self.positions = card_positions(self.layout)
        # Build board of Cell objects
        self.board: List[List[Cell]] = [[Cell(card=val, token=('W' if val == 'W' else '')) for val in row]
                                        for row in self.layout]
        # Build players and teams
        team_order = ['B', 'R', 'Y'][:num_teams]
        self.players: List[Player] = []
        team_humans = team_humans or {}
        for team in team_order:
            for _ in range(players_per_team):
                self.players.append(Player(team=team, is_human=team_humans.get(team, True)))
        # Determine sequences needed to win
        self.sequences_needed = 2 if num_teams == 2 else 1
        # Build deck (two decks minus jokers)
        self.deck: List[str] = self._build_deck()
        self.random.shuffle(self.deck)
        # Determine hand size based on number of players
        num_players = len(self.players)
        if num_players == 2:
            self.hand_size = 7
        elif num_players in (3, 4):
            self.hand_size = 6
        elif num_players == 6:
            self.hand_size = 5
        elif num_players in (8, 9):
            self.hand_size = 4
        else:
            self.hand_size = 3
        # Deal hands
        self.current_player_index = 0
        self._deal_hands()
        # Precompute all 5‑space lines (row, col, diag)
        self._lines = self._compute_lines()
        # Track whether the game is over
        self.winner: Optional[str] = None

    def _build_deck(self) -> List[str]:
        # Build two standard decks without jokers; represent one‑eyed jacks as J1
        # (J♠ and J♥), two‑eyed jacks as J2 (J♣ and J♦)
        ranks = ["A", "2", "3", "4", "5", "6", "7", "8", "9", "T", "Q", "K"]
        suits = ["S", "H", "D", "C"]
        deck = []
        for _ in range(2):
            for r in ranks:
                for s in suits:
                    if r == 'J':
                        continue
                    deck.append(r + s)
            # Add one‑eyed and two‑eyed jacks
            deck.extend(['J1'] * 2)  # J♠ and J♥
            deck.extend(['J2'] * 2)  # J♣ and J♦
        return deck

    def _deal_hands(self) -> None:
        for _ in range(self.hand_size):
            for p in self.players:
                if not self.deck:
                    self._reshuffle_discard_piles()
                p.hand.append(self.deck.pop())

    def _reshuffle_discard_piles(self) -> None:
        # Placeholder: for simplicity we reshuffle discards when deck runs out
        # In this simplified implementation we do nothing; the deck is large enough
        if not self.deck:
            raise RuntimeError("out of cards")

    def _compute_lines(self) -> List[List[Tuple[int, int]]]:
        lines: List[List[Tuple[int, int]]] = []
        # Rows
        for y in range(10):
            for x in range(6):
                lines.append([(y, x + i) for i in range(5)])
        # Columns
        for x in range(10):
            for y in range(6):
                lines.append([(y + i, x) for i in range(5)])
        # Diagonal down (\)
        for y in range(6):
            for x in range(6):
                lines.append([(y + i, x + i) for i in range(5)])
        # Diagonal up (/)
        for y in range(4, 10):
            for x in range(6):
                lines.append([(y - i, x + i) for i in range(5)])
        return lines

    def current_player(self) -> Player:
        return self.players[self.current_player_index]

    def next_player(self) -> None:
        self.current_player_index = (self.current_player_index + 1) % len(self.players)

    def play_turn(self, card: str, target: Optional[Tuple[int, int]] = None) -> bool:
        """Execute a player's turn by playing a card.

        :param card: the card string from the player's hand (e.g. '7H', 'J1', 'J2')
        :param target: the board position to place or remove a chip; required
            for normal cards and J1 removals; ignored for J2 (wild)
        :returns: True if the move was successful, False otherwise
        """
        player = self.current_player()
        if card not in player.hand:
            return False
        # Determine card type
        if card == 'J2':
            # Two‑eyed jack: place on any empty non‑corner
            if target is None:
                return False
            y, x = target
            cell = self.board[y][x]
            if cell.card == 'W' or cell.token:
                return False
            cell.token = player.team
        elif card == 'J1':
            # One‑eyed jack: remove an opponent chip that is not protected
            if target is None:
                return False
            y, x = target
            cell = self.board[y][x]
            if cell.token in ('', player.team, 'W'):
                return False
            if cell.protected:
                return False
            cell.token = ''
        else:
            # Normal card
            positions = self.positions.get(card, [])
            if not positions:
                return False
            # Validate chosen position is one of the two
            if target not in positions:
                return False
            y, x = target
            cell = self.board[y][x]
            if cell.token:
                # If one instance is occupied, maybe the other is empty
                return False
            cell.token = player.team
        # Remove the card from hand
        player.hand.remove(card)
        # Draw a replacement card if available
        if self.deck:
            player.hand.append(self.deck.pop())
        # Update sequences after the move
        self._update_sequences()
        # Check winning condition
        if player.sequences >= self.sequences_needed:
            self.winner = player.team
        # Advance to next player
        self.next_player()
        return True

    def discard_card(self, card: str) -> bool:
        """Allow a player to discard a dead card and draw a replacement."""
        player = self.current_player()
        if card not in player.hand:
            return False
        # Dead only if both positions of a normal card are occupied
        if card in ('J1', 'J2'):
            return False
        positions = self.positions.get(card, [])
        if not positions:
            return False
        if all(self.board[y][x].token for y, x in positions):
            player.hand.remove(card)
            if self.deck:
                player.hand.append(self.deck.pop())
            return True
        return False

    def _update_sequences(self) -> None:
        """Recalculate sequences for all teams and mark protected chips."""
        # Reset protections and counts
        for row in self.board:
            for cell in row:
                cell.protected = False
        for p in self.players:
            p.sequences = 0
        # Evaluate each line
        for line in self._lines:
            # Determine if a team owns all cells in the line (consider corners)
            owner: Optional[str] = None
            tokens_in_line = set()
            for y, x in line:
                cell = self.board[y][x]
                if cell.card == 'W':
                    continue
                if cell.token == '':
                    owner = None
                    break
                tokens_in_line.add(cell.token)
            else:
                # All positions have tokens or are corners
                if len(tokens_in_line) == 1:
                    owner = tokens_in_line.pop()
            if owner:
                # Mark all chips in line as protected and increment sequence count
                for y, x in line:
                    self.board[y][x].protected = True
                for p in self.players:
                    if p.team == owner:
                        p.sequences += 1

    def get_valid_targets(self, card: str) -> List[Tuple[int, int]]:
        """Return a list of valid board positions for the given card.

        For J2 cards this returns all empty, non‑corner positions.  For J1 cards
        returns all opponent tokens that are not protected.  For normal cards
        returns the unoccupied matching positions.  Useful for UI and agents.
        """
        if card == 'J2':
            targets = []
            for y, row in enumerate(self.board):
                for x, cell in enumerate(row):
                    if cell.card != 'W' and not cell.token:
                        targets.append((y, x))
            return targets
        if card == 'J1':
            team = self.current_player().team
            targets = []
            for y, row in enumerate(self.board):
                for x, cell in enumerate(row):
                    if cell.token not in ('', team, 'W') and not cell.protected:
                        targets.append((y, x))
            return targets
        # Normal card
        positions = self.positions.get(card, [])
        return [(y, x) for y, x in positions if not self.board[y][x].token]


    # Convenience method to convert board into a tensor for agents
    def state_tensor(self, team: str) -> List[List[List[int]]]:
        """Return a 3D tensor representation of the board for a given team.

        Channels are: my chips, opponent chips, empty squares, corners, protected.
        Each channel is a 10×10 binary matrix.
        """
        import numpy as np
        my_chip = np.zeros((10, 10), dtype=np.float32)
        opp_chip = np.zeros((10, 10), dtype=np.float32)
        empty = np.zeros((10, 10), dtype=np.float32)
        corners = np.zeros((10, 10), dtype=np.float32)
        protected = np.zeros((10, 10), dtype=np.float32)
        for y in range(10):
            for x in range(10):
                cell = self.board[y][x]
                if cell.card == 'W':
                    corners[y, x] = 1.0
                elif cell.token == '':
                    empty[y, x] = 1.0
                elif cell.token == team:
                    my_chip[y, x] = 1.0
                elif cell.token != 'W':
                    opp_chip[y, x] = 1.0
                if cell.protected:
                    protected[y, x] = 1.0
        # Stack channels
        return np.stack([my_chip, opp_chip, empty, corners, protected], axis=0)
