import unittest

from catchup.board import BOARD
from catchup.game import GameState
from catchup.solvers import MCTSPlayer, RandomPlayer, random_playout


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


class MCTSPlayerTest(unittest.TestCase):
    def test_choose_action_returns_a_legal_action(self) -> None:
        state = GameState.new().apply_action(0)
        player = MCTSPlayer.with_seed(5, simulations=8)

        action = player.choose_action(state)

        self.assertIn(action, state.legal_actions())

    def test_choose_action_rejects_terminal_state(self) -> None:
        terminal = random_playout(player=RandomPlayer.with_seed(1))

        with self.assertRaises(ValueError):
            MCTSPlayer.with_seed(2, simulations=4).choose_action(terminal)

    def test_search_tracks_root_visits_and_children(self) -> None:
        state = GameState.new()
        player = MCTSPlayer.with_seed(7, simulations=12)

        root = player.search(state)

        self.assertEqual(root.visits, 12)
        self.assertLessEqual(len(root.children), 12)
        self.assertTrue(root.children)
        for action, child in root.children.items():
            self.assertIn(action, state.legal_actions())
            self.assertEqual(child.action_from_parent, action)
            self.assertIs(child.parent, root)
            self.assertGreater(child.visits, 0)

    def test_mcts_is_reproducible_with_seed(self) -> None:
        state = GameState.new().apply_action(20)

        first = MCTSPlayer.with_seed(13, simulations=15).choose_action(state)
        second = MCTSPlayer.with_seed(13, simulations=15).choose_action(state)

        self.assertEqual(first, second)

    def test_search_does_not_mutate_start_state(self) -> None:
        state = GameState.new().apply_action(30)
        before_cell_owners = state.tracker.cell_owners.copy()
        before_selected = state.selected

        MCTSPlayer.with_seed(17, simulations=10).search(state)

        self.assertEqual(state.tracker.cell_owners, before_cell_owners)
        self.assertEqual(state.selected, before_selected)

    def test_constructor_rejects_invalid_settings(self) -> None:
        with self.assertRaises(ValueError):
            MCTSPlayer(simulations=0)
        with self.assertRaises(ValueError):
            MCTSPlayer(exploration_weight=-1.0)


if __name__ == "__main__":
    unittest.main()
