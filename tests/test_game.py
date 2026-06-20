import unittest

from catchup.board import BOARD
from catchup.components import PLAYER_ONE, PLAYER_TWO
from catchup.game import FINISH, GameState, compare_size_vectors


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

    def test_compare_size_vectors_uses_later_groups_as_tie_breakers(self) -> None:
        self.assertGreater(compare_size_vectors((5, 3), (5, 2, 2)), 0)
        self.assertLess(compare_size_vectors((4, 4), (5,)), 0)
        self.assertEqual(compare_size_vectors((3, 2), (3, 2)), 0)


if __name__ == "__main__":
    unittest.main()
