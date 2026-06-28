import unittest
from tempfile import TemporaryDirectory

import catchup.ui_server as ui_server
from catchup.components import PLAYER_ONE, PLAYER_TWO
from catchup.game import FINISH, GameState
from catchup.ui_server import PLAYER_NAMES, GameSession, action_description, state_payload


class UiServerTest(unittest.TestCase):
    def test_state_payload_includes_component_sizes_for_both_players(self) -> None:
        state = GameState.new()
        state = state.apply_action(0)
        state = state.apply_action(1)
        state = state.apply_action(2)

        payload = state_payload(state)
        players = {player["id"]: player for player in payload["players"]}

        self.assertEqual(players[PLAYER_ONE]["name"], PLAYER_NAMES[PLAYER_ONE])
        self.assertEqual(players[PLAYER_ONE]["group_sizes"], (1,))
        self.assertEqual(players[PLAYER_TWO]["name"], PLAYER_NAMES[PLAYER_TWO])
        self.assertEqual(players[PLAYER_TWO]["group_sizes"], (2,))
        self.assertEqual(players[PLAYER_TWO]["largest_group"], 2)

    def test_state_payload_includes_empty_component_boundaries(self) -> None:
        state = GameState.new().apply_action(30)

        payload = state_payload(state)

        self.assertEqual(len(payload["empty_components"]), 1)
        region = payload["empty_components"][0]
        self.assertEqual(region["size"], 60)
        self.assertNotIn(30, region["cells"])
        self.assertEqual(region["blue"], [{"root": 30, "size": 1}])
        self.assertEqual(region["white"], [])

    def test_neural_model_options_lists_actual_mlx_checkpoints(self) -> None:
        original_model_dir = ui_server.NEURAL_MODEL_DIR
        original_repo_dir = ui_server.REPO_DIR

        with TemporaryDirectory() as tmpdir:
            model_dir = ui_server.Path(tmpdir) / "data" / "models"
            model_dir.mkdir(parents=True)
            (model_dir / "a_mlx.safetensors").write_bytes(b"")
            (model_dir / "b.pt2").write_bytes(b"")

            ui_server.REPO_DIR = ui_server.Path(tmpdir)
            ui_server.NEURAL_MODEL_DIR = model_dir
            try:
                models = ui_server.neural_model_options()
            finally:
                ui_server.REPO_DIR = original_repo_dir
                ui_server.NEURAL_MODEL_DIR = original_model_dir

        self.assertEqual(
            models,
            [{"label": "a_mlx.safetensors", "path": "data/models/a_mlx.safetensors"}],
        )

    def test_session_action_reset_and_undo(self) -> None:
        session = GameSession()

        after_claim = session.apply_action(0)
        self.assertEqual(after_claim["players"][0]["group_sizes"], (1,))
        self.assertEqual(after_claim["current_player"], PLAYER_TWO)

        undone = session.undo()
        self.assertEqual(undone["players"][0]["group_sizes"], ())
        self.assertEqual(undone["current_player"], PLAYER_ONE)

        reset = session.reset()
        self.assertEqual(reset["finish_action"], FINISH)
        self.assertEqual(reset["empty_count"], 61)
        self.assertTrue(session._state.early_win_enabled)

    def test_session_suggest_action_uses_early_win_state(self) -> None:
        original = ui_server.suggest_with_cpp_mcts
        seen_early_win_flags = []

        def fake_cpp_suggest(
            state,
            simulations,
            seed=1,
            engine="random",
            puct_prior=None,
            puct_rollout=None,
            neural_model=None,
            neural_backend=None,
        ):
            seen_early_win_flags.append(state.early_win_enabled)
            return {
                "action": 0,
                "choices": [{"action": 0, "visits": simulations, "value": 0.0}],
                "engine": engine,
                "state_mode": "tracked",
            }

        ui_server.suggest_with_cpp_mcts = fake_cpp_suggest
        try:
            GameSession().suggest_action(simulations=4)
        finally:
            ui_server.suggest_with_cpp_mcts = original

        self.assertEqual(seen_early_win_flags, [True])

    def test_session_allows_human_out_of_order_multi_cell_turn(self) -> None:
        session = GameSession()
        session.apply_action(22)
        after_first_white_cell = session.apply_action(21)

        self.assertIn(0, after_first_white_cell["legal_actions"])

        after_second_white_cell = session.apply_action(0)
        players = {player["id"]: player for player in after_second_white_cell["players"]}

        self.assertEqual(after_second_white_cell["current_player"], PLAYER_ONE)
        self.assertEqual(after_second_white_cell["completed_turns"], 2)
        self.assertEqual(players[PLAYER_TWO]["group_sizes"], (1, 1))

    def test_session_suggest_action_does_not_apply_action(self) -> None:
        session = GameSession()

        payload = session.suggest_action(simulations=4)
        suggestion = payload["suggestion"]
        choices = payload["choices"]
        state = payload["state"]

        self.assertIn(suggestion["action"], GameState.new().legal_actions())
        self.assertEqual(suggestion["player"], PLAYER_ONE)
        self.assertEqual(suggestion["player_name"], PLAYER_NAMES[PLAYER_ONE])
        self.assertEqual(suggestion["simulations"], 4)
        self.assertIsInstance(suggestion["seed"], int)
        self.assertIn(suggestion["engine"], ("cpp/mcts", "python"))
        self.assertIn("Claim #", suggestion["label"])
        self.assertEqual(suggestion["action"], choices[0]["action"])
        self.assertEqual(sum(choice["visits"] for choice in choices), 4)
        self.assertEqual(
            [choice["visits"] for choice in choices],
            sorted((choice["visits"] for choice in choices), reverse=True),
        )
        self.assertEqual(state["current_player"], PLAYER_ONE)
        self.assertEqual(state["empty_count"], 61)

    def test_session_suggest_action_falls_back_to_python_mcts(self) -> None:
        original = ui_server.suggest_with_cpp_mcts
        ui_server.suggest_with_cpp_mcts = (
            lambda state,
            simulations,
            seed=1,
            engine="random",
            puct_prior=None,
            puct_rollout=None,
            neural_model=None,
            neural_backend=None: None
        )
        try:
            payload = GameSession().suggest_action(simulations=4)
        finally:
            ui_server.suggest_with_cpp_mcts = original

        self.assertEqual(payload["suggestion"]["engine"], "python")
        self.assertEqual(sum(choice["visits"] for choice in payload["choices"]), 4)

    def test_session_suggest_action_uses_fresh_seed_for_cpp_mcts(self) -> None:
        original = ui_server.suggest_with_cpp_mcts
        seeds = []

        def fake_cpp_suggest(
            state,
            simulations,
            seed=1,
            engine="random",
            puct_prior=None,
            puct_rollout=None,
            neural_model=None,
            neural_backend=None,
        ):
            seeds.append(seed)
            return {
                "action": 0,
                "choices": [{"action": 0, "visits": simulations, "value": 0.0}],
                "engine": engine,
                "state_mode": "tracked",
            }

        ui_server.suggest_with_cpp_mcts = fake_cpp_suggest
        try:
            first = GameSession().suggest_action(simulations=4)
            second = GameSession().suggest_action(simulations=4)
        finally:
            ui_server.suggest_with_cpp_mcts = original

        self.assertEqual(len(seeds), 2)
        self.assertNotEqual(seeds[0], seeds[1])
        self.assertEqual(first["suggestion"]["seed"], seeds[0])
        self.assertEqual(second["suggestion"]["seed"], seeds[1])

    def test_session_suggest_action_passes_puct_options(self) -> None:
        original = ui_server.suggest_with_cpp_mcts
        calls = []

        def fake_cpp_suggest(
            state,
            simulations,
            seed=1,
            engine="random",
            puct_prior=None,
            puct_rollout=None,
            neural_model=None,
            neural_backend=None,
        ):
            calls.append((engine, puct_prior, puct_rollout))
            return {
                "action": 0,
                "choices": [{"action": 0, "visits": simulations, "value": 0.0}],
                "engine": engine,
                "puct_prior": puct_prior,
                "puct_rollout": puct_rollout,
                "state_mode": "tracked",
            }

        ui_server.suggest_with_cpp_mcts = fake_cpp_suggest
        try:
            payload = GameSession().suggest_action(
                simulations=4,
                solver="puct",
                puct_prior="flat",
                puct_rollout="biased",
            )
        finally:
            ui_server.suggest_with_cpp_mcts = original

        self.assertEqual(calls, [("puct", "flat", "biased")])
        self.assertEqual(payload["suggestion"]["solver"], "puct")
        self.assertEqual(payload["suggestion"]["puct_prior"], "flat")
        self.assertEqual(payload["suggestion"]["puct_rollout"], "biased")
        self.assertEqual(payload["suggestion"]["engine"], "cpp/puct/prior=flat/rollout=biased")

    def test_session_suggest_action_passes_neural_options(self) -> None:
        original = ui_server.suggest_with_cpp_mcts
        calls = []

        def fake_cpp_suggest(
            state,
            simulations,
            seed=1,
            engine="random",
            puct_prior=None,
            puct_rollout=None,
            neural_model=None,
            neural_backend=None,
        ):
            calls.append((engine, neural_model, neural_backend))
            return {
                "action": 0,
                "choices": [{"action": 0, "visits": simulations, "value": 0.0}],
                "engine": engine,
                "state_mode": "tracked",
            }

        ui_server.suggest_with_cpp_mcts = fake_cpp_suggest
        try:
            with TemporaryDirectory() as tmpdir:
                model_path = f"{tmpdir}/model.safetensors"
                open(model_path, "wb").close()

                payload = GameSession().suggest_action(
                    simulations=4,
                    solver="neural-puct",
                    neural_model=model_path,
                    neural_backend="mlx",
                )
        finally:
            ui_server.suggest_with_cpp_mcts = original

        self.assertEqual(calls, [("neural-puct", model_path, "mlx")])
        self.assertEqual(payload["suggestion"]["solver"], "neural-puct")
        self.assertEqual(payload["suggestion"]["neural_model"], model_path)
        self.assertEqual(payload["suggestion"]["neural_backend"], "mlx")
        self.assertNotIn("neural_batch_size", payload["suggestion"])
        self.assertEqual(payload["suggestion"]["engine"], "cpp/neural-puct/backend=mlx")

    def test_session_suggest_action_rejects_puct_without_cpp(self) -> None:
        original = ui_server.suggest_with_cpp_mcts
        ui_server.suggest_with_cpp_mcts = (
            lambda state,
            simulations,
            seed=1,
            engine="random",
            puct_prior=None,
            puct_rollout=None,
            neural_model=None,
            neural_backend=None: None
        )
        try:
            with self.assertRaises(RuntimeError):
                GameSession().suggest_action(simulations=4, solver="puct")
        finally:
            ui_server.suggest_with_cpp_mcts = original

    def test_action_description_formats_finish_and_claim(self) -> None:
        state = GameState.new()

        claim = action_description(state, 30)
        finish = action_description(state, FINISH)

        self.assertEqual(claim["kind"], "claim")
        self.assertEqual(claim["label"], "Claim #30 (0,0)")
        self.assertEqual(finish["kind"], "finish")
        self.assertEqual(finish["label"], "Finish turn")


if __name__ == "__main__":
    unittest.main()
