import unittest

from catchup.components import EMPTY, PLAYER_ONE, ComponentTracker
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
        self.assertNotIn("--state-mode", args)
        self.assertNotIn("--engine", args)

    def test_cpp_solver_args_include_non_default_engine(self) -> None:
        state = GameState.new()

        args = cpp_solver_args(
            state,
            simulations=7,
            seed=3,
            engine="puct",
            puct_prior="flat",
            puct_rollout="biased",
        )

        self.assertIn("--engine", args)
        self.assertEqual(args[args.index("--engine") + 1], "puct")
        self.assertEqual(args[args.index("--puct-prior") + 1], "flat")
        self.assertEqual(args[args.index("--puct-rollout") + 1], "biased")

    def test_cpp_solver_args_include_neural_options(self) -> None:
        state = GameState.new()

        args = cpp_solver_args(
            state,
            simulations=7,
            seed=3,
            engine="neural-puct",
            neural_model="data/models/model.safetensors",
            neural_backend="mlx",
        )

        self.assertEqual(args[args.index("--engine") + 1], "neural-puct")
        self.assertEqual(args[args.index("--model") + 1], "data/models/model.safetensors")
        self.assertEqual(args[args.index("--neural-backend") + 1], "mlx")
        self.assertNotIn("--neural-batch-size", args)

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
        self.assertEqual(result["engine"], "random")
        self.assertEqual(
            sum(choice["visits"] for choice in result["choices"]),
            4,
        )
        self.assertEqual(result["state_mode"], "tracked")

    def test_cpp_solver_handles_partial_turn_when_built(self) -> None:
        if find_cpp_solver() is None:
            self.skipTest("C++ solver binary is not built")

        state = GameState.new().apply_action(20).apply_action(21)

        result = suggest_with_cpp_mcts(state, simulations=4, seed=2)

        self.assertIsNotNone(result)
        assert result is not None
        self.assertIn(result["action"], state.legal_actions())

    def test_cpp_solver_can_use_puct_when_built(self) -> None:
        if find_cpp_solver() is None:
            self.skipTest("C++ solver binary is not built")

        state = GameState.new()

        result = suggest_with_cpp_mcts(
            state,
            simulations=4,
            seed=3,
            engine="puct",
            puct_prior="heuristic",
            puct_rollout="biased",
        )

        self.assertIsNotNone(result)
        assert result is not None
        self.assertEqual(result["engine"], "puct")
        self.assertEqual(result["puct_prior"], "heuristic")
        self.assertEqual(result["puct_rollout"], "biased")
        self.assertIn(result["action"], state.legal_actions())
        self.assertEqual(
            sum(choice["visits"] for choice in result["choices"]),
            4,
        )

    def test_puct_prior_prefers_connecting_own_groups_when_built(self) -> None:
        if find_cpp_solver() is None:
            self.skipTest("C++ solver binary is not built")

        owners = [EMPTY] * 61
        owners[20] = PLAYER_ONE
        owners[22] = PLAYER_ONE
        state = GameState(
            tracker=ComponentTracker(cell_owners=owners),
            current_player=PLAYER_ONE,
            max_claims=1,
            turn_start_largest=4,
            opening_turn=False,
            completed_turns=2,
        )

        result = suggest_with_cpp_mcts(
            state,
            simulations=1,
            seed=1,
            engine="puct",
            puct_prior="heuristic",
            puct_rollout="biased",
        )

        self.assertIsNotNone(result)
        assert result is not None
        self.assertEqual(result["action"], 21)


if __name__ == "__main__":
    unittest.main()
