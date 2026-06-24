import importlib
import unittest


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


if __name__ == "__main__":
    unittest.main()
