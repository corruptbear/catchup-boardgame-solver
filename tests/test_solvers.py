import random
import unittest

from catchup.board import BOARD
from catchup.game import FINISH, GameState
from catchup.solvers import (
    MCTSPlayer,
    RandomPlayer,
    fast_random_playout,
    random_playout,
    undo_random_playout_result,
)


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
        self.assertIn(terminal.winner(), (0, 1))

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

    def test_fast_random_playout_matches_random_playout_with_seed(self) -> None:
        state = GameState.new().apply_action(20)

        slow = random_playout(state, RandomPlayer.with_seed(41))
        fast = fast_random_playout(state, random.Random(41))

        self.assertTrue(fast.is_terminal())
        self.assertEqual(fast.tracker.cell_owners, slow.tracker.cell_owners)
        self.assertEqual(fast.group_sizes(0), slow.group_sizes(0))
        self.assertEqual(fast.group_sizes(1), slow.group_sizes(1))
        self.assertEqual(fast.winner(), slow.winner())

    def test_fast_random_playout_does_not_mutate_start_state(self) -> None:
        state = GameState.new().apply_action(30)
        before_cell_owners = state.tracker.cell_owners.copy()
        before_selected = state.selected

        terminal = fast_random_playout(state, random.Random(31))

        self.assertTrue(terminal.is_terminal())
        self.assertEqual(state.tracker.cell_owners, before_cell_owners)
        self.assertEqual(state.selected, before_selected)
        self.assertFalse(state.is_terminal())

    def test_fast_random_playout_max_actions_guard(self) -> None:
        with self.assertRaises(RuntimeError):
            fast_random_playout(rng=random.Random(1), max_actions=1)

    def test_undo_random_playout_matches_fast_playout_with_seed(self) -> None:
        state = GameState.new().apply_action(20)

        fast = fast_random_playout(state, random.Random(41))
        undo = undo_random_playout_result(state, random.Random(41))

        self.assertEqual(undo.result_for(0), fast.result_for(0))
        self.assertEqual(undo.result_for(1), fast.result_for(1))

    def test_undo_random_playout_restores_start_state(self) -> None:
        state = GameState.new().apply_action(30)
        before_cell_owners = state.tracker.cell_owners.copy()
        before_empty_components = state.tracker.empty_components()
        before_blue_sizes = state.group_sizes(0)
        before_white_sizes = state.group_sizes(1)

        result = undo_random_playout_result(state, random.Random(31))

        self.assertIn(result.result_for(0), (-1, 1))
        self.assertEqual(state.tracker.cell_owners, before_cell_owners)
        self.assertEqual(state.tracker.empty_components(), before_empty_components)
        self.assertEqual(state.group_sizes(0), before_blue_sizes)
        self.assertEqual(state.group_sizes(1), before_white_sizes)
        self.assertIsNone(state.tracker._undo_log)

    def test_undo_random_playout_restores_after_max_actions_guard(self) -> None:
        state = GameState.new().apply_action(30)
        before_cell_owners = state.tracker.cell_owners.copy()
        before_empty_count = state.tracker.empty_count()

        with self.assertRaises(RuntimeError):
            undo_random_playout_result(state, random.Random(1), max_actions=1)

        self.assertEqual(state.tracker.cell_owners, before_cell_owners)
        self.assertEqual(state.tracker.empty_count(), before_empty_count)
        self.assertIsNone(state.tracker._undo_log)


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
        self.assertGreaterEqual(player.last_transposition_table_size, len(root.children) + 1)

    def test_transposition_table_reuses_equivalent_state_nodes(self) -> None:
        state = GameState.new().apply_action(20)
        player = MCTSPlayer.with_seed(3, simulations=1)
        table = {}

        first = player._node_for_state(state.copy(), table)
        second = player._node_for_state(state.copy(), table)

        self.assertIs(first, second)
        self.assertEqual(len(table), 1)

    def test_transposition_key_uses_state_values_not_object_identity(self) -> None:
        first_state = (
            GameState.new()
            .apply_action(0)
            .apply_action(1)
            .apply_action(FINISH)
            .apply_action(2)
            .apply_action(FINISH)
        )
        second_state = (
            GameState.new()
            .apply_action(2)
            .apply_action(1)
            .apply_action(FINISH)
            .apply_action(0)
            .apply_action(FINISH)
        )
        player = MCTSPlayer.with_seed(3, simulations=1)
        table = {}

        self.assertIsNot(first_state, second_state)
        self.assertEqual(
            first_state.tracker.cell_owners,
            second_state.tracker.cell_owners,
        )
        self.assertEqual(player._state_key(first_state), player._state_key(second_state))
        self.assertIs(
            player._node_for_state(first_state, table),
            player._node_for_state(second_state, table),
        )
        self.assertEqual(len(table), 1)

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

    def test_undo_rollout_search_does_not_mutate_start_state(self) -> None:
        state = GameState.new().apply_action(30)
        before_cell_owners = state.tracker.cell_owners.copy()
        before_empty_components = state.tracker.empty_components()
        before_selected = state.selected

        MCTSPlayer(
            simulations=10,
            rng=random.Random(17),
            use_undo_rollout=True,
        ).search(state)

        self.assertEqual(state.tracker.cell_owners, before_cell_owners)
        self.assertEqual(state.tracker.empty_components(), before_empty_components)
        self.assertEqual(state.selected, before_selected)
        self.assertIsNone(state.tracker._undo_log)

    def test_constructor_rejects_invalid_settings(self) -> None:
        with self.assertRaises(ValueError):
            MCTSPlayer(simulations=0)
        with self.assertRaises(ValueError):
            MCTSPlayer(exploration_weight=-1.0)


if __name__ == "__main__":
    unittest.main()
