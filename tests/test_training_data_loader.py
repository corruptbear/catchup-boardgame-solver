import json
import tempfile
import unittest
from pathlib import Path

from catchup.game import FINISH
from catchup.symmetry import SYMMETRIES
from catchup.training.data_loader import (
    all_symmetry_variants,
    can_augment_with_symmetry,
    iter_jsonl,
    iter_training_samples,
    transform_sample,
)


class FixedRng:
    def __init__(self, choice):
        self._choice = choice

    def choice(self, choices):
        self.assert_choice_present(choices)
        return self._choice

    def assert_choice_present(self, choices):
        if self._choice not in choices:
            raise AssertionError("fixed choice was not available")


def sample_with_marker(cell: int = 5) -> dict:
    owners = [-1] * FINISH
    owners[cell] = 0
    legal_mask = [False] * (FINISH + 1)
    legal_mask[cell] = True
    legal_mask[FINISH] = True
    policy_target = [0.0] * (FINISH + 1)
    policy_target[cell] = 0.75
    policy_target[FINISH] = 0.25

    return {
        "state": {
            "owners": owners,
            "current_player": 0,
            "selected_this_turn": [],
            "claimed_this_turn": 0,
            "max_claims": 2,
            "turn_start_largest": 7,
            "opening_turn": False,
            "legal_mask": legal_mask,
        },
        "policy_target": policy_target,
        "value_target": 1,
        "terminal": {
            "winner": 0,
            "blue_group_sizes": [12],
            "white_group_sizes": [10, 1],
            "filled_cells": 23,
            "completed_turns": 11,
        },
        "meta": {"game_id": 4},
    }


class TrainingDataLoaderTest(unittest.TestCase):
    def test_transform_sample_moves_owner_legal_mask_and_policy(self) -> None:
        symmetry = SYMMETRIES[1]
        sample = sample_with_marker()
        old_cell = 5
        new_cell = symmetry.map_cell(old_cell)

        transformed = transform_sample(sample, symmetry)

        self.assertEqual(transformed["state"]["owners"][old_cell], -1)
        self.assertEqual(transformed["state"]["owners"][new_cell], 0)
        self.assertFalse(transformed["state"]["legal_mask"][old_cell])
        self.assertTrue(transformed["state"]["legal_mask"][new_cell])
        self.assertTrue(transformed["state"]["legal_mask"][FINISH])
        self.assertEqual(transformed["policy_target"][old_cell], 0.0)
        self.assertEqual(transformed["policy_target"][new_cell], 0.75)
        self.assertEqual(transformed["policy_target"][FINISH], 0.25)
        self.assertEqual(transformed["value_target"], 1)
        self.assertEqual(transformed["terminal"], sample["terminal"])

    def test_all_symmetry_variants_returns_twelve_samples(self) -> None:
        variants = list(all_symmetry_variants(sample_with_marker()))

        self.assertEqual(len(variants), 12)
        self.assertEqual(
            {tuple(variant["state"]["owners"]) for variant in variants},
            {
                tuple(transform_sample(sample_with_marker(), symmetry)["state"]["owners"])
                for symmetry in SYMMETRIES
            },
        )

    def test_mid_turn_samples_are_not_augmented_by_loader(self) -> None:
        sample = sample_with_marker()
        sample["state"]["selected_this_turn"] = [5]
        sample["state"]["claimed_this_turn"] = 1

        self.assertFalse(can_augment_with_symmetry(sample))
        with tempfile.TemporaryDirectory() as tmpdir:
            path = Path(tmpdir) / "samples.jsonl"
            path.write_text(json.dumps(sample) + "\n", encoding="utf-8")

            loaded = list(iter_training_samples(
                path,
                augment_symmetry=True,
                rng=FixedRng(SYMMETRIES[1]),
            ))

        self.assertEqual(loaded, [sample])

    def test_transform_sample_rejects_mid_turn_by_default(self) -> None:
        sample = sample_with_marker()
        sample["state"]["selected_this_turn"] = [5]

        with self.assertRaises(ValueError):
            transform_sample(sample, SYMMETRIES[1])

    def test_iter_training_samples_augments_turn_boundary_rows(self) -> None:
        sample = sample_with_marker()
        symmetry = SYMMETRIES[1]
        with tempfile.TemporaryDirectory() as tmpdir:
            path = Path(tmpdir) / "samples.jsonl"
            path.write_text(json.dumps(sample) + "\n", encoding="utf-8")

            loaded = list(iter_training_samples(
                path,
                augment_symmetry=True,
                rng=FixedRng(symmetry),
            ))

        self.assertEqual(loaded, [sample, transform_sample(sample, symmetry)])

    def test_iter_training_samples_keeps_raw_sample_and_adds_augmented_views(self) -> None:
        sample = sample_with_marker()
        symmetry = SYMMETRIES[1]
        with tempfile.TemporaryDirectory() as tmpdir:
            path = Path(tmpdir) / "samples.jsonl"
            path.write_text(json.dumps(sample) + "\n", encoding="utf-8")

            loaded = list(iter_training_samples(
                path,
                augment_symmetry=True,
                symmetry_copies=3,
                rng=FixedRng(symmetry),
            ))

        self.assertEqual(loaded, [sample] + [transform_sample(sample, symmetry)] * 3)

    def test_iter_jsonl_accepts_one_path_or_many_paths(self) -> None:
        first = sample_with_marker(5)
        second = sample_with_marker(6)
        with tempfile.TemporaryDirectory() as tmpdir:
            first_path = Path(tmpdir) / "first.jsonl"
            second_path = Path(tmpdir) / "second.jsonl"
            first_path.write_text(json.dumps(first) + "\n", encoding="utf-8")
            second_path.write_text(json.dumps(second) + "\n", encoding="utf-8")

            self.assertEqual(list(iter_jsonl(first_path)), [first])
            self.assertEqual(list(iter_jsonl([first_path, second_path])), [first, second])


if __name__ == "__main__":
    unittest.main()
