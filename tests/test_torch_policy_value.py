import importlib
import unittest
from pathlib import Path


def import_trainer():
    try:
        return importlib.import_module("catchup.training.torch_policy_value")
    except ModuleNotFoundError as exc:
        if exc.name == "torch":
            raise unittest.SkipTest("PyTorch is not installed for this Python") from exc
        raise


class TorchPolicyValueTest(unittest.TestCase):
    def test_gnn_forward_returns_policy_and_value_shapes(self) -> None:
        trainer = import_trainer()
        torch = importlib.import_module("torch")

        model = trainer.GraphPolicyValueNet(hidden_size=16, gnn_layers=2)
        features = torch.zeros((3, trainer.FEATURE_COUNT), dtype=torch.float32)

        policy_logits, value = model(features)

        self.assertEqual(tuple(policy_logits.shape), (3, trainer.ACTION_COUNT))
        self.assertEqual(tuple(value.shape), (3,))

    def test_directional_cnn_forward_returns_policy_and_value_shapes(self) -> None:
        trainer = import_trainer()
        torch = importlib.import_module("torch")

        model = trainer.HexDirectionalCnnPolicyValueNet(hidden_size=16, cnn_layers=2)
        features = torch.zeros((3, trainer.FEATURE_COUNT), dtype=torch.float32)

        policy_logits, value = model(features)

        self.assertEqual(tuple(policy_logits.shape), (3, trainer.ACTION_COUNT))
        self.assertEqual(tuple(value.shape), (3,))

    def test_directional_cnn_claim_gather_matches_grid_to_cell_matrix(self) -> None:
        trainer = import_trainer()
        torch = importlib.import_module("torch")

        model = trainer.HexDirectionalCnnPolicyValueNet(hidden_size=16, cnn_layers=1)
        claim_grid = torch.randn((2, trainer.GRID_CELL_COUNT), dtype=torch.float32)

        gathered = claim_grid.index_select(1, model.cell_grid_indices)
        matrix_readout = torch.matmul(claim_grid, model.grid_to_cell)

        self.assertTrue(torch.equal(gathered, matrix_readout))

    def test_sample_features_do_not_include_absolute_current_player_scalar(self) -> None:
        trainer = import_trainer()

        sample = {
            "state": {
                "owners": [-1] * trainer.CELL_COUNT,
                "current_player": 1,
                "selected_this_turn": [3, 8],
                "claimed_this_turn": 2,
                "max_claims": 3,
                "turn_start_largest": 7,
                "opening_turn": False,
                "legal_mask": [False] * trainer.ACTION_COUNT,
            },
            "policy_target": [0.0] * trainer.ACTION_COUNT,
            "value_target": -1.0,
        }

        features, _, _ = trainer.sample_to_arrays(sample)

        self.assertEqual(trainer.GLOBAL_FEATURE_COUNT, 4)
        self.assertEqual(trainer.FEATURE_COUNT, 310)
        expected = [2.0 / 3.0, 1.0, 7.0 / trainer.CELL_COUNT, 0.0]
        self.assertEqual(len(features[trainer.SCALAR_OFFSET:]), len(expected))
        for actual, expected_value in zip(features[trainer.SCALAR_OFFSET:], expected):
            self.assertAlmostEqual(float(actual), expected_value)

    def test_build_model_from_metadata_defaults_old_checkpoints_to_mlp(self) -> None:
        trainer = import_trainer()

        model = trainer.build_model_from_metadata({"hidden_size": 16})

        self.assertIsInstance(model, trainer.PolicyValueNet)

    def test_build_model_from_metadata_accepts_directional_cnn_architecture(self) -> None:
        trainer = import_trainer()

        directional = trainer.build_model_from_metadata({
            "architecture": trainer.DIRECTIONAL_CNN_ARCHITECTURE,
            "hidden_size": 16,
            "cnn_layers": 2,
        })

        self.assertIsInstance(directional, trainer.HexDirectionalCnnPolicyValueNet)

    def test_normalize_state_dict_accepts_old_gnn_policy_piece_names(self) -> None:
        trainer = import_trainer()

        model = trainer.GraphPolicyValueNet(hidden_size=16, gnn_layers=1)
        legacy_state = {}
        for key, value in model.state_dict().items():
            legacy_key = key
            legacy_key = legacy_key.replace("claim_policy_scorer.", "claim_policy_head.")
            legacy_key = legacy_key.replace("finish_policy_scorer.", "finish_policy_head.")
            legacy_state[legacy_key] = value

        normalized = trainer.normalize_model_state_dict(legacy_state)

        self.assertIn("claim_policy_scorer.weight", normalized)
        self.assertIn("finish_policy_scorer.0.weight", normalized)
        self.assertNotIn("claim_policy_head.weight", normalized)
        self.assertNotIn("finish_policy_head.0.weight", normalized)
        model.load_state_dict(normalized)

    def test_replay_age_weights_are_oldest_to_newest(self) -> None:
        trainer = import_trainer()

        weights = trainer.replay_age_weights(3, 0.8)

        self.assertEqual(len(weights), 3)
        self.assertAlmostEqual(weights[0], 0.64)
        self.assertAlmostEqual(weights[1], 0.8)
        self.assertAlmostEqual(weights[2], 1.0)

    def test_replay_train_batches_ramps_while_buffer_fills(self) -> None:
        trainer = import_trainer()
        samples = tuple({} for _ in range(100))
        config = trainer.TrainConfig(
            replay_window_generations=5,
            target_lifetime_coverage=2.0,
            batch_size=32,
        )

        one_generation = [
            trainer.ReplayGeneration(path=Path("iter_0001.jsonl"), samples=samples),
        ]
        two_generations = [
            trainer.ReplayGeneration(path=Path("iter_0001.jsonl"), samples=samples),
            trainer.ReplayGeneration(path=Path("iter_0002.jsonl"), samples=samples),
        ]

        self.assertEqual(trainer.replay_train_batches(config, one_generation), 2)
        self.assertEqual(trainer.replay_train_batches(config, two_generations), 3)

    def test_parse_args_accepts_replay_settings(self) -> None:
        trainer = import_trainer()

        config = trainer.parse_args([
            "--replay-data-glob",
            "data/neural_self_play/iter_*.jsonl",
            "--init-checkpoint",
            "data/models/start.pt",
            "--replay-window-generations",
            "5",
            "--replay-gamma",
            "0.8",
            "--target-lifetime-coverage",
            "2.0",
            "--train-batches",
            "26",
        ])

        self.assertEqual(config.replay_data_glob, "data/neural_self_play/iter_*.jsonl")
        self.assertEqual(config.init_checkpoint, Path("data/models/start.pt"))
        self.assertEqual(config.replay_window_generations, 5)
        self.assertEqual(config.replay_gamma, 0.8)
        self.assertEqual(config.target_lifetime_coverage, 2.0)
        self.assertEqual(config.train_batches, 26)


if __name__ == "__main__":
    unittest.main()
