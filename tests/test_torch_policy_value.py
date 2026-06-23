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

    def test_build_model_from_metadata_defaults_old_checkpoints_to_mlp(self) -> None:
        trainer = import_trainer()

        model = trainer.build_model_from_metadata({"hidden_size": 16})

        self.assertIsInstance(model, trainer.PolicyValueNet)

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
