import unittest

from sequence.board_layouts import load_layout, card_positions
from sequence.game import Game


class TestSequenceGame(unittest.TestCase):
    def test_layout_integrity(self):
        layout = load_layout()
        # Check dimensions
        self.assertEqual(len(layout), 10)
        for row in layout:
            self.assertEqual(len(row), 10)
        # Count corners and card occurrences
        positions = card_positions(layout)
        count = {}
        for card, coords in positions.items():
            count[card] = len(coords)
        # Each card appears twice
        for card, c in count.items():
            self.assertEqual(c, 2, f"Card {card} appears {c} times")
        # Total non-corner cells
        self.assertEqual(sum(count.values()), 96)

    def test_jack_two_place_anywhere(self):
        game = Game()
        player = game.current_player()
        # Force player to have J2 in hand
        player.hand = ['J2'] + player.hand[:-1]
        # Try placing on an empty non-corner
        for y in range(10):
            for x in range(10):
                cell = game.board[y][x]
                if cell.card != 'W' and not cell.token:
                    self.assertTrue(game.play_turn('J2', (y, x)))
                    return
        self.fail("Could not find empty non-corner for J2")

    def test_jack_one_removes_opponent(self):
        game = Game()
        # Give current player J1
        player = game.current_player()
        player.hand = ['J1'] + player.hand[:-1]
        # Place opponent chip
        game.board[1][1].token = 'R'
        # Attempt removal
        self.assertTrue(game.play_turn('J1', (1, 1)))
        self.assertEqual(game.board[1][1].token, '')

    def test_sequence_detection_and_protection(self):
        game = Game()
        # Build a horizontal sequence for team 'B' on row 0 from col1 to col5
        # Remove any chips there and play matching cards
        positions = game.positions
        # set first five positions (0,1) to (0,5)
        cards = [game.board[0][i].card for i in range(1, 6)]
        # Force player's hand to contain these cards in order
        player = game.current_player()
        player.hand = cards + player.hand[len(cards):]
        for i, card in enumerate(cards, start=1):
            # Find the correct position on row 0 for this card
            played = False
            for pos in positions[card]:
                if pos == (0, i):
                    success = game.play_turn(card, pos)
                    self.assertTrue(success)
                    # Advance back to first player (play_turn already advanced)
                    game.next_player()
                    played = True
                    break
            self.assertTrue(played, f"Card {card} could not be played on row 0 col {i}")
        # After playing 5 cards, sequence should be detected
        player_idx = 0  # first player
        # Next player's turn after playing 5 cards
        # sequences updated automatically
        self.assertGreaterEqual(game.players[player_idx].sequences, 1)
        # Chips should be protected
        for i in range(1, 6):
            self.assertTrue(game.board[0][i].protected)
        # Opponent cannot remove protected chip
        opp = game.players[1]
        opp.hand = ['J1'] + opp.hand[:-1]
        # Opponent tries to remove first chip
        self.assertFalse(game.play_turn('J1', (0, 1)))


if __name__ == '__main__':
    unittest.main()