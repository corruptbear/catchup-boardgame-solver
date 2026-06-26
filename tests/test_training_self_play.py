import random
import tempfile
import unittest
from pathlib import Path

from catchup.game import FINISH, GameState
import catchup.training.self_play as self_play
from catchup.training.self_play import (
    ACTION_COUNT,
    build_parser,
    generate_game_samples,
    generate_samples,
    policy_target_from_choices,
    sample_action_from_choices,
    state_payload,
    suggest_with_teacher_puct,
    terminal_payload,
    write_jsonl,
)


def first_legal_teacher(
    state: GameState,
    simulations: int,
    seed: int,
    puct_prior: str,
    puct_rollout: str,
) -> dict[str, object]:
    del seed, puct_prior, puct_rollout
    action = state.legal_actions()[0]
    return {
        "action": action,
        "choices": [{"action": action, "visits": simulations, "value": 0.0}],
    }


class TrainingSelfPlayTest(unittest.TestCase):
    def test_state_payload_has_raw_state_and_legal_mask(self) -> None:
        state = GameState.new()

        payload = state_payload(state)

        self.assertEqual(len(payload["owners"]), 61)
        self.assertEqual(payload["current_player"], state.current_player)
        self.assertEqual(payload["selected_this_turn"], [])
        self.assertEqual(payload["claimed_this_turn"], 0)
        self.assertEqual(len(payload["legal_mask"]), ACTION_COUNT)
        self.assertTrue(payload["legal_mask"][0])
        self.assertFalse(payload["legal_mask"][FINISH])

    def test_policy_target_normalizes_root_visits(self) -> None:
        target = policy_target_from_choices(
            [
                {"action": 1, "visits": 3},
                {"action": FINISH, "visits": 1},
            ]
        )

        self.assertEqual(len(target), ACTION_COUNT)
        self.assertEqual(target[1], 0.75)
        self.assertEqual(target[FINISH], 0.25)
        self.assertEqual(sum(target), 1.0)

    def test_sample_action_from_choices_uses_visit_weights(self) -> None:
        action = sample_action_from_choices(
            [
                {"action": 1, "visits": 0},
                {"action": 2, "visits": 5},
            ],
            random.Random(1),
        )

        self.assertEqual(action, 2)

    def test_terminal_payload_has_final_game_summary(self) -> None:
        terminal = GameState.new()
        while not terminal.is_terminal():
            terminal = terminal.apply_action(terminal.legal_actions()[0])

        payload = terminal_payload(terminal)

        self.assertIn(payload["winner"], (0, 1))
        self.assertGreater(payload["filled_cells"], 0)
        self.assertGreater(payload["completed_turns"], 0)
        self.assertEqual(
            sum(payload["blue_group_sizes"]) + sum(payload["white_group_sizes"]),
            payload["filled_cells"],
        )

    def test_generate_game_samples_fills_value_targets(self) -> None:
        samples = generate_game_samples(
            game_id=3,
            simulations=2,
            seed=7,
            max_actions=200,
            teacher=first_legal_teacher,
        )

        self.assertTrue(samples)
        for sample in samples:
            self.assertIn(sample["value_target"], (-1, 1))
            self.assertEqual(len(sample["policy_target"]), ACTION_COUNT)
            self.assertIn(sample["terminal"]["winner"], (0, 1))
            self.assertGreaterEqual(sample["terminal"]["filled_cells"], 0)
            self.assertLessEqual(sample["terminal"]["filled_cells"], 61)
            self.assertGreaterEqual(sample["terminal"]["completed_turns"], 1)
            self.assertEqual(sample["meta"]["teacher"], "puct:2:prior=heuristic:rollout=biased")
            self.assertEqual(sample["meta"]["game_id"], 3)

    def test_generate_samples_rejects_custom_teacher_with_workers(self) -> None:
        with self.assertRaises(ValueError):
            generate_samples(
                games=2,
                simulations=1,
                seed=1,
                workers=2,
                teacher=first_legal_teacher,
            )

    def test_teacher_wrapper_calls_cpp_solver_in_puct_mode(self) -> None:
        calls: list[tuple[str, str | None, str | None]] = []
        original = self_play.suggest_with_cpp_mcts

        def fake_cpp_solver(
            state: GameState,
            simulations: int,
            seed: int = 1,
            engine: str = "random",
            puct_prior: str | None = None,
            puct_rollout: str | None = None,
        ) -> dict[str, object]:
            del state, simulations, seed
            calls.append((engine, puct_prior, puct_rollout))
            return {"action": 0, "choices": [{"action": 0, "visits": 1, "value": 0.0}]}

        self_play.suggest_with_cpp_mcts = fake_cpp_solver
        try:
            suggest_with_teacher_puct(GameState.new(), 5, 7, "heuristic", "biased")
        finally:
            self_play.suggest_with_cpp_mcts = original

        self.assertEqual(calls, [("puct", "heuristic", "biased")])

    def test_parser_accepts_workers(self) -> None:
        args = build_parser().parse_args(
            [
                "--games",
                "2",
                "--simulations",
                "100",
                "--out",
                "samples.jsonl",
                "--workers",
                "12",
            ]
        )

        self.assertEqual(args.workers, 12)

    def test_write_jsonl_writes_one_line_per_sample(self) -> None:
        samples = [
            {
                "state": state_payload(GameState.new()),
                "policy_target": [0.0] * ACTION_COUNT,
                "value_target": 0,
                "terminal": {
                    "winner": 0,
                    "blue_group_sizes": [1],
                    "white_group_sizes": [],
                    "filled_cells": 1,
                    "completed_turns": 1,
                },
                "meta": {"game_id": 0},
            }
        ]

        with tempfile.TemporaryDirectory() as tmpdir:
            output = Path(tmpdir) / "samples.jsonl"
            write_jsonl(samples, output)

            lines = output.read_text(encoding="utf-8").splitlines()

        self.assertEqual(len(lines), 1)
        self.assertIn('"value_target":0', lines[0])


if __name__ == "__main__":
    unittest.main()
