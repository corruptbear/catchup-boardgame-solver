import random
import unittest

from catchup.game import GameState
from catchup.neural_puct import (
    ACTION_COUNT,
    NeuralPuctPlayer,
    PuctEvaluation,
    root_choices,
)


class FixedEvaluator:
    def __init__(self, preferred_action: int = 0, value: float = 0.25) -> None:
        self.preferred_action = preferred_action
        self.value = value
        self.calls = 0

    def evaluate(self, state: GameState) -> PuctEvaluation:
        self.calls += 1
        priors = [0.0] * ACTION_COUNT
        for action in state.legal_actions():
            priors[action] = 0.1
        if self.preferred_action in state.legal_actions():
            priors[self.preferred_action] = 10.0
        return PuctEvaluation(tuple(priors), self.value)


class NeuralPuctPlayerTest(unittest.TestCase):
    def test_first_simulation_expands_root_edges(self) -> None:
        evaluator = FixedEvaluator(preferred_action=3)
        player = NeuralPuctPlayer(evaluator, simulations=1, rng=random.Random(1))

        root = player.search(GameState.new())

        self.assertEqual(root.visits, 1)
        self.assertEqual(evaluator.calls, 1)
        self.assertTrue(root.edges)
        self.assertFalse(root.children)
        self.assertEqual(root_choices(root)[0]["action"], 3)

    def test_choose_action_returns_legal_action(self) -> None:
        state = GameState.new()
        player = NeuralPuctPlayer(
            FixedEvaluator(preferred_action=5),
            simulations=4,
            rng=random.Random(2),
        )

        action = player.choose_action(state)

        self.assertIn(action, state.legal_actions())

    def test_search_does_not_mutate_start_state(self) -> None:
        state = GameState.new().apply_action(20)
        before_cell_owners = state.tracker.cell_owners.copy()
        before_selected = state.selected
        player = NeuralPuctPlayer(
            FixedEvaluator(preferred_action=21),
            simulations=5,
            rng=random.Random(3),
        )

        player.search(state)

        self.assertEqual(state.tracker.cell_owners, before_cell_owners)
        self.assertEqual(state.selected, before_selected)

    def test_backpropagates_value_from_leaf_player_perspective(self) -> None:
        player = NeuralPuctPlayer(
            FixedEvaluator(preferred_action=0, value=1.0),
            simulations=2,
            rng=random.Random(4),
        )

        root = player.search(GameState.new())

        self.assertEqual(root.visits, 2)
        self.assertAlmostEqual(root.total_value, 0.0)
        self.assertEqual(root_choices(root)[0]["action"], 0)
        self.assertEqual(root_choices(root)[0]["visits"], 1)

    def test_rejects_terminal_state(self) -> None:
        state = GameState.new()
        while not state.is_terminal():
            state = state.apply_action(state.legal_actions()[0])

        with self.assertRaises(ValueError):
            NeuralPuctPlayer(FixedEvaluator()).search(state)


if __name__ == "__main__":
    unittest.main()
