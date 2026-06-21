import unittest

from catchup.board import BOARD
from catchup.components import EMPTY, PLAYER_ONE, PLAYER_TWO, ComponentTracker
from catchup.game import EARLY_WIN_CHECK_MIN_FILLED_CELLS, FINISH, GameState, compare_size_vectors


class GameStateTest(unittest.TestCase):
    def test_initial_move_claims_one_cell_and_does_not_grant_bonus(self) -> None:
        state = GameState.new()

        self.assertEqual(state.current_player, PLAYER_ONE)
        self.assertEqual(state.max_claims, 1)
        self.assertNotIn(FINISH, state.legal_actions())
        self.assertEqual(len(state.legal_actions()), 61)

        state = state.apply_action(0)

        self.assertEqual(state.current_player, PLAYER_TWO)
        self.assertEqual(state.completed_turns, 1)
        self.assertEqual(state.max_claims, 2)
        self.assertEqual(state.turn_start_largest, 1)
        self.assertEqual(state.selected, ())

    def test_claim_two_that_increases_global_largest_grants_next_player_three(self) -> None:
        state = GameState.new()
        state = state.apply_action(0)

        state = state.apply_action(1)
        self.assertIn(FINISH, state.legal_actions())
        state = state.apply_action(2)

        self.assertEqual(state.current_player, PLAYER_ONE)
        self.assertEqual(state.max_claims, 3)
        self.assertEqual(state.turn_start_largest, 2)
        self.assertEqual(state.group_sizes(PLAYER_TWO), (2,))

    def test_finish_after_one_claim_is_legal_after_opening(self) -> None:
        state = GameState.new().apply_action(0)
        state = state.apply_action(10)

        self.assertIn(FINISH, state.legal_actions())
        state = state.apply_action(FINISH)

        self.assertEqual(state.current_player, PLAYER_ONE)
        self.assertEqual(state.max_claims, 2)
        self.assertEqual(state.group_sizes(PLAYER_TWO), (1,))

    def test_finish_before_claim_is_illegal(self) -> None:
        state = GameState.new()

        with self.assertRaises(ValueError):
            state.apply_action(FINISH)

    def test_multi_cell_turns_use_canonical_increasing_order(self) -> None:
        state = GameState.new().apply_action(0)
        state = state.apply_action(10)

        legal = state.legal_actions()
        self.assertIn(FINISH, legal)
        self.assertNotIn(9, legal)
        self.assertIn(11, legal)

        with self.assertRaises(ValueError):
            state.apply_action(9)

    def test_apply_turn_rejects_empty_turn(self) -> None:
        with self.assertRaises(ValueError):
            GameState.new().apply_turn(())

    def test_winner_uses_terminal_component_vectors(self) -> None:
        state = GameState.new()
        for cell in range(BOARD.cell_count - 1):
            state.tracker.claim(PLAYER_ONE, cell)
        state.tracker.claim(PLAYER_TWO, BOARD.cell_count - 1)

        self.assertTrue(state.is_terminal())
        self.assertEqual(state.winner(), PLAYER_ONE)
        self.assertEqual(state.result_for(PLAYER_ONE), 1)
        self.assertEqual(state.result_for(PLAYER_TWO), -1)

    def test_reachable_bounds_can_prove_early_winner(self) -> None:
        cell_owners = [PLAYER_ONE] * BOARD.cell_count
        cell_owners[0] = PLAYER_TWO
        cell_owners[1] = EMPTY
        state = GameState(
            tracker=ComponentTracker(cell_owners=cell_owners),
        )

        self.assertGreater(state.tracker.empty_count(), 0)
        self.assertGreaterEqual(
            BOARD.cell_count - state.tracker.empty_count(),
            EARLY_WIN_CHECK_MIN_FILLED_CELLS,
        )
        self.assertEqual(state.proven_winner(), PLAYER_ONE)
        self.assertTrue(state.is_terminal())
        self.assertEqual(state.legal_actions(), ())
        self.assertEqual(state.winner(), PLAYER_ONE)

    def test_reachable_bounds_are_skipped_before_minimum_filled_cells(self) -> None:
        cell_owners = [EMPTY] * BOARD.cell_count
        filled_cells = EARLY_WIN_CHECK_MIN_FILLED_CELLS - 1
        for cell in range(filled_cells):
            cell_owners[cell] = PLAYER_ONE
        state = GameState(tracker=ComponentTracker(cell_owners=cell_owners))

        self.assertEqual(BOARD.cell_count - state.tracker.empty_count(), filled_cells)
        self.assertIsNone(state.proven_winner())
        self.assertFalse(state.is_terminal())

    def test_reachable_bounds_do_not_stop_during_partial_turn(self) -> None:
        state = GameState.new().apply_action(0).apply_action(1)

        self.assertEqual(state.selected, (1,))
        self.assertIsNone(state.proven_winner())
        self.assertFalse(state.is_terminal())
        self.assertIn(FINISH, state.legal_actions())

    def test_compare_size_vectors_uses_later_groups_as_tie_breakers(self) -> None:
        self.assertGreater(compare_size_vectors((5, 3), (5, 2, 2)), 0)
        self.assertLess(compare_size_vectors((4, 4), (5,)), 0)
        self.assertEqual(compare_size_vectors((3, 2), (3, 2)), 0)


if __name__ == "__main__":
    unittest.main()
