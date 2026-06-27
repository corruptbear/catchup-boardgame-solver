import json
import subprocess
import unittest
from pathlib import Path


CPP_ARENA = Path("catchup/cpp/build/catchup_arena")


class CppArenaTest(unittest.TestCase):
    def test_cpp_arena_json_smoke_when_built(self) -> None:
        if not CPP_ARENA.is_file():
            self.skipTest("C++ arena binary is not built")

        completed = subprocess.run(
            [
                str(CPP_ARENA),
                "--agent-a",
                "puct:1:prior=heuristic:rollout=biased",
                "--agent-b",
                "mcts:1",
                "--pairs",
                "1",
                "--seed",
                "5",
                "--json",
            ],
            capture_output=True,
            check=True,
            text=True,
        )

        payload = json.loads(completed.stdout)

        self.assertEqual(payload["agent_a"], "puct:1:prior=heuristic:rollout=biased")
        self.assertEqual(payload["agent_b"], "mcts:1")
        self.assertEqual(payload["pairs"], 1)
        self.assertEqual(payload["threads"], 1)
        self.assertEqual(payload["summary"]["games"], 2)
        self.assertEqual(len(payload["games"]), 2)
        self.assertEqual(payload["games"][0]["blue_side"], "A")
        self.assertEqual(payload["games"][1]["blue_side"], "B")

    def test_cpp_arena_parallel_records_match_single_thread_when_built(self) -> None:
        if not CPP_ARENA.is_file():
            self.skipTest("C++ arena binary is not built")

        base_args = [
            str(CPP_ARENA),
            "--agent-a",
            "puct:1:prior=heuristic:rollout=biased",
            "--agent-b",
            "mcts:1",
            "--pairs",
            "3",
            "--seed",
            "5",
            "--json",
        ]
        single = subprocess.run(
            [*base_args, "--threads", "1"],
            capture_output=True,
            check=True,
            text=True,
        )
        parallel = subprocess.run(
            [*base_args, "--threads", "2"],
            capture_output=True,
            check=True,
            text=True,
        )

        single_payload = json.loads(single.stdout)
        parallel_payload = json.loads(parallel.stdout)

        self.assertEqual(single_payload["summary"], parallel_payload["summary"])
        self.assertEqual(single_payload["games"], parallel_payload["games"])
        self.assertEqual(single_payload["threads"], 1)
        self.assertEqual(parallel_payload["threads"], 2)

    def test_cpp_arena_game_seeds_do_not_overlap_for_adjacent_base_seeds(self) -> None:
        if not CPP_ARENA.is_file():
            self.skipTest("C++ arena binary is not built")

        def game_seeds(seed: int) -> set[int]:
            completed = subprocess.run(
                [
                    str(CPP_ARENA),
                    "--agent-a",
                    "random",
                    "--agent-b",
                    "random",
                    "--pairs",
                    "64",
                    "--seed",
                    str(seed),
                    "--json",
                ],
                capture_output=True,
                check=True,
                text=True,
            )
            payload = json.loads(completed.stdout)
            return {int(game["game_seed"]) for game in payload["games"]}

        first = game_seeds(1)
        second = game_seeds(2)

        self.assertEqual(len(first), 128)
        self.assertEqual(len(second), 128)
        self.assertTrue(first.isdisjoint(second))


if __name__ == "__main__":
    unittest.main()
