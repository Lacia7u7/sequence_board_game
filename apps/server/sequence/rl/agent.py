"""Agents for playing Sequence.

This module defines two agents: a learning agent backed by a small CNN
(`RLAgent`) and a simple heuristic agent for baseline play (`HeuristicAgent`).
The RL agent uses a policy gradient approach with a fixed architecture
implemented in ``model_numpy.py``.  It can be trained via self-play or
against a heuristic opponent.  Legal move masking ensures only valid
actions are considered.

Both agents expose a ``select_move`` method that, given a ``Game`` instance,
returns a tuple ``(card, target)`` representing the chosen card from the
player's hand and the board coordinates where it should be used.  If no
legal move exists the method returns ``(None, None)``.
"""

from __future__ import annotations

import os
import random
from typing import List, Optional, Tuple

import numpy as np

from ..game import Game
from .model_numpy import SequenceCNN


class RLAgent:
    """Reinforcement learning agent using a simple CNN policy network."""

    def __init__(self, team: str, model: Optional[SequenceCNN] = None,
                 epsilon: float = 0.1, seed: int | None = None) -> None:
        self.team = team
        self.model = model or SequenceCNN()
        self.epsilon = epsilon  # exploration rate
        self.rng = np.random.default_rng(seed)
        # Memory for training (states, action indices)
        self.states: List[np.ndarray] = []
        self.actions: List[int] = []

    def select_move(self, game: Game, greedy: bool = False) -> Tuple[Optional[str], Optional[Tuple[int, int]]]:
        """Select a move (card and target) for the current game state.

        :param game: current game instance
        :param greedy: if True, always pick the highest scoring action; if
          False, sample from the masked softmax distribution
        :returns: (card code, (row, col)) if a move exists; otherwise (None, None)
        """
        player = game.current_player()
        assert player.team == self.team, "select_move called for wrong team"
        # Compute state tensor for the current player
        state = game.state_tensor(self.team)
        logits, _ = self.model.forward(state)
        # Mask invalid actions: 100 entries (row-major order)
        mask = np.full((100,), -np.inf, dtype=np.float32)
        # Build a mapping from target cell to possible cards in hand
        valid_moves: List[Tuple[int, int, str]] = []  # (action_idx, cell_index, card)
        for idx in range(100):
            y, x = divmod(idx, 10)
            cell = game.board[y][x]
            # Skip corners and my own token
            if cell.card == 'W':
                continue
            # Determine if any card in hand can target this cell
            chosen_card = None
            # 1. Removal: if cell has opponent token and not protected
            if cell.token and cell.token not in ('', self.team, 'W') and not cell.protected:
                if 'J1' in player.hand:
                    chosen_card = 'J1'
            else:
                # 2. Placement: if empty and normal card matches cell or J2
                if not cell.token:
                    # Check if any normal card matches the board card
                    for card in player.hand:
                        if card in ('J1', 'J2'):
                            continue
                        # Card codes may be e.g. 'T' vs '10' on board; unify tens
                        board_code = cell.card
                        if board_code == card:
                            chosen_card = card
                            break
                    # If none found, check J2
                    if chosen_card is None and 'J2' in player.hand:
                        chosen_card = 'J2'
            if chosen_card is not None:
                mask[idx] = 0.0  # valid action
                valid_moves.append((idx, idx, chosen_card))
        if not valid_moves:
            return None, None
        # Compute probabilities using masked softmax
        masked_logits = logits + mask
        # For numerical stability subtract max
        maxlog = np.max(masked_logits)
        exp_logits = np.exp(masked_logits - maxlog)
        probs = exp_logits / exp_logits.sum()
        # Exploration/exploitation: epsilon-greedy
        if greedy or self.rng.random() > self.epsilon:
            action_idx = int(np.argmax(probs))
        else:
            # Sample from valid actions only using their probabilities
            valid_indices = [idx for idx, _, _ in valid_moves]
            valid_probs = probs[valid_indices]
            valid_probs /= valid_probs.sum()
            action_idx = int(self.rng.choice(valid_indices, p=valid_probs))
        # Determine card for chosen action
        y, x = divmod(action_idx, 10)
        chosen_card = None
        cell = game.board[y][x]
        if cell.token and cell.token not in ('', self.team, 'W') and not cell.protected:
            chosen_card = 'J1'
        else:
            # find matching normal card
            for card in player.hand:
                if card not in ('J1', 'J2'):
                    if card == cell.card:
                        chosen_card = card
                        break
            if chosen_card is None and 'J2' in player.hand:
                chosen_card = 'J2'
        # Record state and action for training
        self.states.append(state)
        self.actions.append(action_idx)
        return chosen_card, (y, x)

    def record_outcome(self, reward: float) -> None:
        """Update the policy network using REINFORCE.

        After an episode ends, call this method with +1 for a win or -1 for
        a loss.  The stored states and actions from the episode are used
        to compute policy gradients.  A constant baseline of zero is
        assumed.  After updating, the memory is cleared.
        """
        if not self.states:
            return
        for state, action_idx in zip(self.states, self.actions):
            logits, cache = self.model.forward(state)
            # Softmax over all 100 actions (no mask during training)
            maxlog = np.max(logits)
            exp_logits = np.exp(logits - maxlog)
            probs = exp_logits / exp_logits.sum()
            # Gradient of log prob for chosen action
            d_logits = probs.copy()
            d_logits[action_idx] -= 1.0
            # Multiply by negative reward for gradient descent
            d_logits *= -reward
            self.model.backward(cache, d_logits[np.newaxis, :])
        # Clear episode memory
        self.states.clear()
        self.actions.clear()

    def save(self, path: str) -> None:
        self.model.save(path)

    def load(self, path: str) -> None:
        if os.path.exists(path):
            self.model.load(path)


class HeuristicAgent:
    """Simple baseline agent that prioritises centre and blocks sequences."""

    def __init__(self, team: str, seed: int | None = None) -> None:
        self.team = team
        self.rng = random.Random(seed)
        # Precompute centrality weights: Manhattan distance from centre reversed
        self.weights = np.zeros((10, 10), dtype=np.float32)
        for y in range(10):
            for x in range(10):
                self.weights[y, x] = 1.0 / (1 + abs(y - 4.5) + abs(x - 4.5))

    def select_move(self, game: Game, greedy: bool = False) -> Tuple[Optional[str], Optional[Tuple[int, int]]]:
        player = game.current_player()
        assert player.team == self.team
        best_score = -1e9
        best_move: Optional[Tuple[str, Tuple[int, int]]] = None
        # For each card in hand check possible targets and evaluate
        for card in player.hand:
            # Determine type
            if card == 'J2':
                # Place anywhere empty (non-corner)
                for y in range(10):
                    for x in range(10):
                        cell = game.board[y][x]
                        if cell.card != 'W' and not cell.token:
                            score = self.weights[y, x]
                            if score > best_score:
                                best_score = score
                                best_move = (card, (y, x))
            elif card == 'J1':
                # Remove opponent chip not protected
                for y in range(10):
                    for x in range(10):
                        cell = game.board[y][x]
                        if cell.token not in ('', player.team, 'W') and not cell.protected:
                            # Removing an opponent chip slightly reduces their potential
                            score = 0.5 + self.weights[y, x]
                            if score > best_score:
                                best_score = score
                                best_move = (card, (y, x))
            else:
                # Normal card: get positions on board
                positions = game.positions.get(card, [])
                for (y, x) in positions:
                    cell = game.board[y][x]
                    if not cell.token:
                        score = self.weights[y, x]
                        if score > best_score:
                            best_score = score
                            best_move = (card, (y, x))
        # Fallback: if no move found (should not happen) choose random
        if best_move is None:
            # Choose any card and any valid position for that card
            for card in player.hand:
                if card == 'J2':
                    # random empty cell
                    empties = [(y, x) for y in range(10) for x in range(10)
                               if game.board[y][x].card != 'W' and not game.board[y][x].token]
                    if empties:
                        return card, self.rng.choice(empties)
                elif card == 'J1':
                    ops = [(y, x) for y in range(10) for x in range(10)
                           if game.board[y][x].token not in ('', player.team, 'W') and not game.board[y][x].protected]
                    if ops:
                        return card, self.rng.choice(ops)
                else:
                    positions = game.positions.get(card, [])
                    empties = [(y, x) for (y, x) in positions if not game.board[y][x].token]
                    if empties:
                        return card, self.rng.choice(empties)
            return None, None
        return best_move
