import unittest

from catchup.arena import (
    GameRecord,
    arena_game_seed,
    parse_agent_spec,
    run_arena,
    summarize_records,
)
from catchup.components import PLAYER_ONE, PLAYER_TWO
from catchup.cpp_solver import find_cpp_solver


class ArenaTest(unittest.TestCase):
    def test_parse_agent_spec_accepts_cpp_engines(self) -> None:
        mcts = parse_agent_spec("mcts:100")
        puct = parse_agent_spec("puct:200:prior=flat:rollout=biased")

        self.assertEqual(mcts.label, "mcts:100")
        self.assertEqual(puct.label, "puct:200:prior=flat:rollout=biased")

    def test_parse_agent_spec_accepts_random_and_neural_agents(self) -> None:
        random_spec = parse_agent_spec("random")
        greedy = parse_agent_spec("neural-greedy:data/models/model.pt:device=mps")
        puct = parse_agent_spec("neural-puct:25:data/models/model.pt:device=cpu")

        self.assertEqual(random_spec.label, "random")
        self.assertEqual(greedy.kind, "neural-greedy")
        self.assertEqual(greedy.checkpoint, "data/models/model.pt")
        self.assertEqual(greedy.device, "mps")
        self.assertEqual(puct.kind, "neural-puct")
        self.assertEqual(puct.simulations, 25)
        self.assertEqual(puct.checkpoint, "data/models/model.pt")
        self.assertEqual(puct.device, "cpu")

    def test_parse_agent_spec_rejects_unknown_engines(self) -> None:
        with self.assertRaises(ValueError):
            parse_agent_spec("python-mcts:10")
        with self.assertRaises(ValueError):
            parse_agent_spec("puct:10")
        with self.assertRaises(ValueError):
            parse_agent_spec("neural-greedy")

    def test_run_arena_can_compare_random_players_without_cpp(self) -> None:
        report = run_arena(
            parse_agent_spec("random"),
            parse_agent_spec("random"),
            pairs=1,
            seed=5,
        )

        self.assertEqual(len(report.records), 2)
        self.assertEqual(report.records[0].blue_side, "A")
        self.assertEqual(report.records[1].blue_side, "B")
        self.assertEqual(report.summary["games"], 2)

    def test_arena_game_seed_does_not_overlap_for_adjacent_base_seeds(self) -> None:
        first = {
            arena_game_seed(base_seed=1, pair_index=pair_index, game_in_pair=game_in_pair)
            for pair_index in range(64)
            for game_in_pair in range(2)
        }
        second = {
            arena_game_seed(base_seed=2, pair_index=pair_index, game_in_pair=game_in_pair)
            for pair_index in range(64)
            for game_in_pair in range(2)
        }

        self.assertEqual(len(first), 128)
        self.assertEqual(len(second), 128)
        self.assertTrue(first.isdisjoint(second))

    def test_summarize_records_counts_agent_sides_not_labels(self) -> None:
        records = (
            GameRecord(
                pair_index=0,
                game_index=0,
                agent_a="mcts:1",
                agent_b="mcts:1",
                blue_agent="mcts:1",
                white_agent="mcts:1",
                blue_side="A",
                white_side="B",
                winner_side="A",
                winner_agent="mcts:1",
                winner_player=PLAYER_ONE,
                completed_turns=10,
                internal_actions=20,
                filled_cells=30,
            ),
            GameRecord(
                pair_index=0,
                game_index=1,
                agent_a="mcts:1",
                agent_b="mcts:1",
                blue_agent="mcts:1",
                white_agent="mcts:1",
                blue_side="B",
                white_side="A",
                winner_side="B",
                winner_agent="mcts:1",
                winner_player=PLAYER_ONE,
                completed_turns=12,
                internal_actions=24,
                filled_cells=32,
            ),
        )

        summary = summarize_records(records)

        self.assertEqual(summary["agent_a_wins"], 1)
        self.assertEqual(summary["agent_b_wins"], 1)

    def test_run_arena_uses_paired_colors_when_cpp_is_built(self) -> None:
        if find_cpp_solver() is None:
            self.skipTest("C++ solver binary is not built")

        report = run_arena(
            parse_agent_spec("mcts:1"),
            parse_agent_spec("puct:1:prior=heuristic:rollout=biased"),
            pairs=1,
            seed=5,
        )

        self.assertEqual(len(report.records), 2)
        self.assertEqual(report.records[0].blue_side, "A")
        self.assertEqual(report.records[0].white_side, "B")
        self.assertEqual(report.records[1].blue_side, "B")
        self.assertEqual(report.records[1].white_side, "A")
        self.assertEqual(report.summary["games"], 2)
        self.assertIn(report.records[0].winner_player, (PLAYER_ONE, PLAYER_TWO))


if __name__ == "__main__":
    unittest.main()
