"""Training script for the Sequence RL agent.

This module provides functions to train a reinforcement learning agent via
self-play against a simple heuristic opponent.  It uses a policy gradient
approach with a CNN defined in ``model_numpy.py``.  The training loop
supports running a specified number of episodes and saving intermediate
checkpoints.
"""

from __future__ import annotations

import os
import time
from typing import Optional

from ..game import Game
from .agent import RLAgent, HeuristicAgent


def train(num_episodes: int = 200, seed: Optional[int] = None,
          save_path: Optional[str] = None, verbose: bool = True) -> RLAgent:
    """Train an RL agent against a heuristic opponent.

    :param num_episodes: number of games to play during training
    :param seed: optional random seed for reproducibility
    :param save_path: if provided, save the final model weights to this path
    :param verbose: whether to print progress information
    :returns: the trained ``RLAgent`` instance
    """
    # Create a game with two teams (Blue vs Red), one player each
    # RL agent always plays Blue, heuristic plays Red
    agent = RLAgent(team='B', epsilon=0.1, seed=seed)
    opponent = HeuristicAgent(team='R', seed=seed)
    wins = 0
    for episode in range(1, num_episodes + 1):
        game = Game(num_teams=2, players_per_team=1,
                    team_humans={'B': False, 'R': False}, seed=seed)
        # Assign teams in order: Blue (agent) goes first, then Red (heuristic)
        # The Game class orders players by team order ['B','R'] so this is fine
        while game.winner is None:
            current = game.current_player()
            if current.team == 'B':
                card, target = agent.select_move(game)
            else:
                card, target = opponent.select_move(game)
            # If no valid move, skip (should rarely happen)
            if card is None or target is None:
                # Skip by discarding a random card (if possible)
                h = current.hand
                # Attempt to discard a dead card
                discarded = False
                for c in h:
                    if game.discard_card(c):
                        discarded = True
                        break
                if not discarded:
                    # Pick and play first available card-target pair
                    for c in h:
                        targets = game.get_valid_targets(c)
                        if targets:
                            game.play_turn(c, targets[0])
                            break
                continue
            success = game.play_turn(card, target)
            if not success:
                # If for some reason the selected move is invalid, skip
                continue
        # Episode finished; assign reward
        reward = 1.0 if game.winner == 'B' else -1.0
        if game.winner == 'B':
            wins += 1
        agent.record_outcome(reward)
        if verbose and episode % max(1, num_episodes // 10) == 0:
            print(f"Episode {episode}/{num_episodes}: Wins so far {wins}")
    if verbose:
        print(f"Training complete. Win rate: {wins}/{num_episodes} ({wins / num_episodes:.2%})")
    # Save model if path given
    if save_path:
        # Ensure directory exists
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        agent.save(save_path)
        if verbose:
            print(f"Model saved to {save_path}")
    return agent


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="Train Sequence RL agent")
    parser.add_argument('--episodes', type=int, default=200)
    parser.add_argument('--seed', type=int, default=None)
    parser.add_argument('--save', type=str, default=None)
    args = parser.parse_args()
    train(args.episodes, args.seed, args.save)