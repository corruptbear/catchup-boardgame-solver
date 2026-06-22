import unittest

from catchup.cpp_solver import cpp_solver_args, find_cpp_solver, suggest_with_cpp_mcts
from catchup.game import GameState


class CppSolverTest(unittest.TestCase):
    def test_cpp_solver_args_include_current_state(self) -> None:
        state = GameState.new().apply_action(20)

        args = cpp_solver_args(state, simulations=7, seed=3)
        owners = args[args.index("--owners") + 1].split(",")

        self.assertIn("--owners", args)
        self.assertIn("--selected", args)
        self.assertEqual(owners[20], "0")
        self.assertIn("--current-player", args)
        self.assertIn(str(state.current_player), args)
        self.assertIn("--simulations", args)
        self.assertIn("7", args)
        self.assertIn("--seed", args)
        self.assertIn("3", args)

    def test_cpp_solver_returns_legal_action_when_built(self) -> None:
        if find_cpp_solver() is None:
            self.skipTest("C++ solver binary is not built")

        state = GameState.new()

        result = suggest_with_cpp_mcts(state, simulations=4, seed=1)

        self.assertIsNotNone(result)
        assert result is not None
        self.assertIn(result["action"], state.legal_actions())
        self.assertEqual(result["player"], state.current_player)
        self.assertEqual(result["simulations"], 4)
        self.assertEqual(
            sum(choice["visits"] for choice in result["choices"]),
            4,
        )

    def test_cpp_solver_handles_partial_turn_when_built(self) -> None:
        if find_cpp_solver() is None:
            self.skipTest("C++ solver binary is not built")

        state = GameState.new().apply_action(20).apply_action(21)

        result = suggest_with_cpp_mcts(state, simulations=4, seed=2)

        self.assertIsNotNone(result)
        assert result is not None
        self.assertIn(result["action"], state.legal_actions())


if __name__ == "__main__":
    unittest.main()
