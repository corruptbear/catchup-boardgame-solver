import json
import subprocess
import tempfile
import unittest
from pathlib import Path


CPP_SELF_PLAY = Path("catchup/cpp/build/catchup_self_play")


class CppSelfPlayTest(unittest.TestCase):
    def test_cpp_self_play_jsonl_smoke_when_built(self) -> None:
        if not CPP_SELF_PLAY.is_file():
            self.skipTest("C++ self-play binary is not built")

        with tempfile.TemporaryDirectory() as tmpdir:
            output_path = Path(tmpdir) / "samples.jsonl"
            completed = subprocess.run(
                [
                    str(CPP_SELF_PLAY),
                    "--games",
                    "1",
                    "--simulations",
                    "1",
                    "--threads",
                    "1",
                    "--out",
                    str(output_path),
                ],
                check=False,
                capture_output=True,
                text=True,
            )

            self.assertEqual(completed.returncode, 0, completed.stderr)
            lines = output_path.read_text(encoding="utf-8").splitlines()

        self.assertTrue(lines)
        sample = json.loads(lines[0])
        self.assertEqual(set(sample), {"state", "policy_target", "value_target", "terminal", "meta"})
        self.assertEqual(len(sample["state"]["owners"]), 61)
        self.assertEqual(len(sample["state"]["legal_mask"]), 62)
        self.assertEqual(len(sample["policy_target"]), 62)
        self.assertIn(sample["value_target"], (-1, 1))
        self.assertIn(sample["terminal"]["winner"], (0, 1))
        self.assertIsInstance(sample["terminal"]["blue_group_sizes"], list)
        self.assertIsInstance(sample["terminal"]["white_group_sizes"], list)
        self.assertGreaterEqual(sample["terminal"]["filled_cells"], 0)
        self.assertLessEqual(sample["terminal"]["filled_cells"], 61)
        self.assertGreaterEqual(sample["terminal"]["completed_turns"], 1)
        self.assertEqual(
            sample["meta"]["teacher"],
            "puct:1:prior=heuristic:rollout=biased:"
            "visit_temperature=max(0.050000,empty_count/61)",
        )

    def test_cpp_self_play_can_disable_early_win_when_built(self) -> None:
        if not CPP_SELF_PLAY.is_file():
            self.skipTest("C++ self-play binary is not built")

        with tempfile.TemporaryDirectory() as directory:
            output_path = Path(directory) / "samples.jsonl"
            completed = subprocess.run(
                [
                    str(CPP_SELF_PLAY),
                    "--games",
                    "1",
                    "--simulations",
                    "1",
                    "--early-win",
                    "false",
                    "--out",
                    str(output_path),
                ],
                capture_output=True,
                text=True,
            )

            self.assertEqual(completed.returncode, 0, completed.stderr)
            sample = json.loads(output_path.read_text(encoding="utf-8").splitlines()[0])

        self.assertFalse(sample["meta"]["early_win"])
        self.assertEqual(sample["terminal"]["filled_cells"], 61)


if __name__ == "__main__":
    unittest.main()
