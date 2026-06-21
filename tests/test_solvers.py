import unittest

from catchup.board import BOARD
from catchup.game import GameState
from catchup.solvers import RandomPlayer, random_playout


class RandomPlayerTest(unittest.TestCase):
    def test_choose_action_returns_a_legal_action(self) -> None:
        state = GameState.new().apply_action(0)
        player = RandomPlayer.with_seed(7)

        action = player.choose_action(state)

        self.assertIn(action, state.legal_actions())

    def test_choose_action_rejects_terminal_state(self) -> None:
        terminal = random_playout(player=RandomPlayer.with_seed(1))

        with self.assertRaises(ValueError):
            RandomPlayer.with_seed(2).choose_action(terminal)

    def test_random_playout_reaches_terminal_state(self) -> None:
        terminal = random_playout(player=RandomPlayer.with_seed(11))

        self.assertTrue(terminal.is_terminal())
        self.assertEqual(len(terminal.selected), 0)
        self.assertEqual(
            sum(terminal.group_sizes(0)) + sum(terminal.group_sizes(1)),
            BOARD.cell_count - terminal.tracker.empty_count(),
        )
        if terminal.tracker.empty_count() > 0:
            self.assertIsNotNone(terminal.proven_winner())
        self.assertIn(terminal.winner(), (0, 1, None))

    def test_random_playout_is_reproducible_with_seed(self) -> None:
        first = random_playout(player=RandomPlayer.with_seed(23))
        second = random_playout(player=RandomPlayer.with_seed(23))

        self.assertEqual(first.tracker.cell_owners, second.tracker.cell_owners)
        self.assertEqual(first.group_sizes(0), second.group_sizes(0))
        self.assertEqual(first.group_sizes(1), second.group_sizes(1))

    def test_random_playout_does_not_mutate_start_state(self) -> None:
        state = GameState.new().apply_action(30)
        before_cell_owners = state.tracker.cell_owners.copy()
        before_selected = state.selected

        terminal = random_playout(state, RandomPlayer.with_seed(31))

        self.assertTrue(terminal.is_terminal())
        self.assertEqual(state.tracker.cell_owners, before_cell_owners)
        self.assertEqual(state.selected, before_selected)
        self.assertFalse(state.is_terminal())

    def test_random_playout_max_actions_guard(self) -> None:
        with self.assertRaises(RuntimeError):
            random_playout(player=RandomPlayer.with_seed(1), max_actions=1)


if __name__ == "__main__":
    unittest.main()
