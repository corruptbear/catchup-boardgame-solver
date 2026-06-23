"""Headless strength arena for Catchup agents."""

from __future__ import annotations

import argparse
import json
import math
import random
from dataclasses import dataclass
from typing import Protocol

from .components import PLAYER_ONE, PLAYER_TWO
from .cpp_solver import suggest_with_cpp_mcts
from .game import GameState


class ArenaAgent(Protocol):
    """Agent used by the arena to choose one legal factored action."""

    def choose_action(self, state: GameState) -> int:
        """Return one action from ``state.legal_actions()``."""


@dataclass(frozen=True, slots=True)
class AgentSpec:
    """Parsed arena agent configuration."""

    kind: str
    simulations: int | None = None
    puct_prior: str | None = None
    puct_rollout: str | None = None

    @property
    def label(self) -> str:
        if self.kind == "puct":
            return (
                f"puct:{self.simulations}:"
                f"prior={self.puct_prior}:rollout={self.puct_rollout}"
            )
        return f"{self.kind}:{self.simulations}"


@dataclass(frozen=True, slots=True)
class GameRecord:
    """One completed arena game."""

    pair_index: int
    game_index: int
    agent_a: str
    agent_b: str
    blue_agent: str
    white_agent: str
    blue_side: str
    white_side: str
    winner_side: str | None
    winner_agent: str | None
    winner_player: int | None
    completed_turns: int
    internal_actions: int
    filled_cells: int


@dataclass(frozen=True, slots=True)
class ArenaReport:
    """All game records plus the aggregate summary."""

    agent_a: str
    agent_b: str
    pairs: int
    seed: int
    records: tuple[GameRecord, ...]
    summary: dict[str, object]


@dataclass(slots=True)
class CppMctsAgent:
    """Arena wrapper around the C++ MCTS binary."""

    simulations: int
    engine: str
    rng: random.Random
    puct_prior: str | None = None
    puct_rollout: str | None = None

    def choose_action(self, state: GameState) -> int:
        payload = suggest_with_cpp_mcts(
            state,
            simulations=self.simulations,
            seed=self.rng.randrange(1 << 63),
            engine=self.engine,
            puct_prior=self.puct_prior,
            puct_rollout=self.puct_rollout,
        )
        if payload is None:
            raise RuntimeError(
                "C++ MCTS binary is not built; run `make -C catchup/cpp` first"
            )
        action = int(payload["action"])
        if action not in state.legal_actions():
            raise RuntimeError(f"C++ MCTS returned illegal action: {action}")
        return action


def parse_agent_spec(text: str) -> AgentSpec:
    """Parse a C++ arena agent spec."""

    normalized = text.strip().lower()
    parts = normalized.split(":")
    if len(parts) < 2:
        raise ValueError(
            "agent must be mcts:N or puct:N:prior=flat|heuristic:rollout=flat|biased"
        )

    kind = parts[0]
    simulations = int(parts[1])
    if simulations <= 0:
        raise ValueError("agent simulations must be positive")
    if kind == "mcts":
        if len(parts) != 2:
            raise ValueError("mcts agent must be mcts:N")
        return AgentSpec(kind, simulations)
    if kind != "puct":
        raise ValueError(
            "agent must be mcts:N or puct:N:prior=flat|heuristic:rollout=flat|biased"
        )

    options: dict[str, str] = {}
    for option in parts[2:]:
        name, separator, value = option.partition("=")
        if not separator:
            raise ValueError("puct options must use name=value")
        options[name] = value
    if set(options) != {"prior", "rollout"}:
        raise ValueError(
            "puct agent must include prior=flat|heuristic and rollout=flat|biased"
        )
    if options["prior"] not in {"flat", "heuristic"}:
        raise ValueError("puct prior must be flat or heuristic")
    if options["rollout"] not in {"flat", "biased"}:
        raise ValueError("puct rollout must be flat or biased")
    return AgentSpec(kind, simulations, options["prior"], options["rollout"])


def make_agent(spec: AgentSpec, seed: int) -> ArenaAgent:
    """Create a fresh stateful agent for one arena game."""

    rng = random.Random(seed)
    if spec.kind == "mcts":
        return CppMctsAgent(spec.simulations, "random", rng)
    if spec.kind == "puct":
        return CppMctsAgent(
            spec.simulations,
            "puct",
            rng,
            spec.puct_prior,
            spec.puct_rollout,
        )
    raise ValueError(f"unknown agent kind: {spec.kind}")


def play_game(
    blue_spec: AgentSpec,
    white_spec: AgentSpec,
    *,
    agent_a: str,
    agent_b: str,
    blue_side: str,
    white_side: str,
    seed: int,
    pair_index: int,
    game_index: int,
    max_actions: int = 512,
) -> GameRecord:
    """Play one game with fixed Blue and White agents."""

    state = GameState.new()
    blue_agent = make_agent(blue_spec, seed * 2 + 1)
    white_agent = make_agent(white_spec, seed * 2 + 2)
    internal_actions = 0

    while not state.is_terminal():
        if internal_actions >= max_actions:
            raise RuntimeError("arena game exceeded max_actions")
        player = blue_agent if state.current_player == PLAYER_ONE else white_agent
        action = player.choose_action(state)
        state = state.apply_action(action)
        internal_actions += 1

    winner_player = state.winner()
    if winner_player == PLAYER_ONE:
        winner_side = blue_side
        winner_agent = blue_spec.label
    elif winner_player == PLAYER_TWO:
        winner_side = white_side
        winner_agent = white_spec.label
    else:
        winner_side = None
        winner_agent = None

    return GameRecord(
        pair_index=pair_index,
        game_index=game_index,
        agent_a=agent_a,
        agent_b=agent_b,
        blue_agent=blue_spec.label,
        white_agent=white_spec.label,
        blue_side=blue_side,
        white_side=white_side,
        winner_side=winner_side,
        winner_agent=winner_agent,
        winner_player=winner_player,
        completed_turns=state.completed_turns,
        internal_actions=internal_actions,
        filled_cells=state.board.cell_count - state.tracker.empty_count(),
    )


def run_arena(
    agent_a: AgentSpec,
    agent_b: AgentSpec,
    *,
    pairs: int,
    seed: int = 1,
    max_actions: int = 512,
) -> ArenaReport:
    """Run paired-color games and summarize agent A's result against agent B."""

    if pairs <= 0:
        raise ValueError("pairs must be positive")

    records: list[GameRecord] = []
    for pair_index in range(pairs):
        first_seed = seed + pair_index * 2
        records.append(
            play_game(
                agent_a,
                agent_b,
                agent_a=agent_a.label,
                agent_b=agent_b.label,
                blue_side="A",
                white_side="B",
                seed=first_seed,
                pair_index=pair_index,
                game_index=len(records),
                max_actions=max_actions,
            )
        )
        records.append(
            play_game(
                agent_b,
                agent_a,
                agent_a=agent_a.label,
                agent_b=agent_b.label,
                blue_side="B",
                white_side="A",
                seed=first_seed + 1,
                pair_index=pair_index,
                game_index=len(records),
                max_actions=max_actions,
            )
        )

    records_tuple = tuple(records)
    return ArenaReport(
        agent_a=agent_a.label,
        agent_b=agent_b.label,
        pairs=pairs,
        seed=seed,
        records=records_tuple,
        summary=summarize_records(records_tuple),
    )


def summarize_records(records: tuple[GameRecord, ...]) -> dict[str, object]:
    """Compute aggregate arena statistics."""

    games = len(records)
    a_wins = sum(record.winner_side == "A" for record in records)
    b_wins = sum(record.winner_side == "B" for record in records)
    ties = games - a_wins - b_wins
    a_score = a_wins + 0.5 * ties
    a_score_rate = a_score / games
    ci_radius = 1.96 * math.sqrt(a_score_rate * (1.0 - a_score_rate) / games)

    a_blue = [record for record in records if record.blue_side == "A"]
    a_white = [record for record in records if record.white_side == "A"]

    return {
        "games": games,
        "agent_a_wins": a_wins,
        "agent_b_wins": b_wins,
        "ties": ties,
        "agent_a_score_rate": a_score_rate,
        "agent_a_score_ci95": (
            max(0.0, a_score_rate - ci_radius),
            min(1.0, a_score_rate + ci_radius),
        ),
        "agent_a_as_blue": _color_summary(a_blue),
        "agent_a_as_white": _color_summary(a_white),
        "average_completed_turns": sum(record.completed_turns for record in records) / games,
        "average_internal_actions": sum(record.internal_actions for record in records) / games,
        "average_filled_cells": sum(record.filled_cells for record in records) / games,
    }


def report_to_dict(report: ArenaReport) -> dict[str, object]:
    """Return a JSON-serializable report."""

    return {
        "agent_a": report.agent_a,
        "agent_b": report.agent_b,
        "pairs": report.pairs,
        "seed": report.seed,
        "summary": report.summary,
        "games": [
            {
                "pair_index": record.pair_index,
                "game_index": record.game_index,
                "blue_agent": record.blue_agent,
                "white_agent": record.white_agent,
                "blue_side": record.blue_side,
                "white_side": record.white_side,
                "winner_side": record.winner_side,
                "winner_agent": record.winner_agent,
                "winner_player": record.winner_player,
                "completed_turns": record.completed_turns,
                "internal_actions": record.internal_actions,
                "filled_cells": record.filled_cells,
            }
            for record in report.records
        ],
    }


def format_report(report: ArenaReport) -> str:
    """Format a compact human-readable report."""

    summary = report.summary
    ci_low, ci_high = summary["agent_a_score_ci95"]
    return "\n".join(
        [
            f"Arena: A={report.agent_a} vs B={report.agent_b}",
            f"Pairs: {report.pairs}  Games: {summary['games']}  Seed: {report.seed}",
            (
                "Result: "
                f"A wins {summary['agent_a_wins']}, "
                f"B wins {summary['agent_b_wins']}, "
                f"ties {summary['ties']}"
            ),
            (
                "A score rate: "
                f"{summary['agent_a_score_rate']:.1%} "
                f"(95% CI {ci_low:.1%}..{ci_high:.1%})"
            ),
            f"A as Blue: {_format_color_summary(summary['agent_a_as_blue'])}",
            f"A as White: {_format_color_summary(summary['agent_a_as_white'])}",
            (
                "Averages: "
                f"{summary['average_completed_turns']:.1f} turns, "
                f"{summary['average_internal_actions']:.1f} internal actions, "
                f"{summary['average_filled_cells']:.1f} filled cells"
            ),
        ]
    )


def main(argv: list[str] | None = None) -> int:
    """CLI entry point for ``python3 -m catchup.arena``."""

    parser = argparse.ArgumentParser(description="Run paired Catchup arena games.")
    parser.add_argument("--agent-a", default="puct:1000:prior=heuristic:rollout=biased")
    parser.add_argument("--agent-b", default="mcts:1000")
    parser.add_argument("--pairs", type=int, default=10)
    parser.add_argument("--seed", type=int, default=1)
    parser.add_argument("--max-actions", type=int, default=512)
    parser.add_argument("--json", action="store_true")
    args = parser.parse_args(argv)

    report = run_arena(
        parse_agent_spec(args.agent_a),
        parse_agent_spec(args.agent_b),
        pairs=args.pairs,
        seed=args.seed,
        max_actions=args.max_actions,
    )
    if args.json:
        print(json.dumps(report_to_dict(report), indent=2, sort_keys=True))
    else:
        print(format_report(report))
    return 0


def _color_summary(records: list[GameRecord]) -> dict[str, int]:
    return {
        "games": len(records),
        "wins": sum(record.winner_side == "A" for record in records),
        "losses": sum(record.winner_side == "B" for record in records),
        "ties": sum(record.winner_side is None for record in records),
    }


def _format_color_summary(summary: dict[str, int]) -> str:
    return (
        f"{summary['wins']}-{summary['losses']}-{summary['ties']} "
        f"in {summary['games']} games"
    )


if __name__ == "__main__":
    raise SystemExit(main())
