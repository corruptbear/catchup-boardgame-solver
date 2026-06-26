"""Generate policy/value bootstrap data from C++ PUCT self-play."""

from __future__ import annotations

import argparse
import json
import multiprocessing
import random
import traceback
from collections.abc import Callable, Sequence
from pathlib import Path
from typing import Any

from ..components import PLAYER_ONE, PLAYER_TWO
from ..cpp_solver import suggest_with_cpp_mcts
from ..game import FINISH, GameState

ACTION_COUNT = FINISH + 1
Teacher = Callable[[GameState, int, int, str, str], dict[str, Any] | None]


def suggest_with_teacher_puct(
    state: GameState,
    simulations: int,
    seed: int,
    puct_prior: str,
    puct_rollout: str,
) -> dict[str, Any] | None:
    """Call the C++ solver in PUCT mode for one teacher search."""

    return suggest_with_cpp_mcts(
        state,
        simulations=simulations,
        seed=seed,
        engine="puct",
        puct_prior=puct_prior,
        puct_rollout=puct_rollout,
    )


def state_payload(state: GameState) -> dict[str, Any]:
    """Return the raw state fields stored with each training sample."""

    legal_mask = [False] * ACTION_COUNT
    for action in state.legal_actions():
        legal_mask[action] = True

    return {
        "owners": list(state.tracker.cell_owners),
        "current_player": state.current_player,
        "selected_this_turn": list(state.selected),
        "claimed_this_turn": len(state.selected),
        "max_claims": state.max_claims,
        "turn_start_largest": state.turn_start_largest,
        "opening_turn": state.opening_turn,
        "legal_mask": legal_mask,
    }


def terminal_payload(state: GameState) -> dict[str, Any]:
    """Return final-game metadata shared by every sample from one game."""

    return {
        "winner": state.winner(),
        "blue_group_sizes": list(state.group_sizes(PLAYER_ONE)),
        "white_group_sizes": list(state.group_sizes(PLAYER_TWO)),
        "filled_cells": state.board.cell_count - state.tracker.empty_count(),
        "completed_turns": state.completed_turns,
    }


def policy_target_from_choices(choices: Sequence[dict[str, Any]]) -> list[float]:
    """Convert C++ root visit counts into a 62-action policy target."""

    target = [0.0] * ACTION_COUNT
    total_visits = sum(int(choice["visits"]) for choice in choices)
    if total_visits == 0:
        raise ValueError("teacher returned no visited actions")

    for choice in choices:
        action = int(choice["action"])
        target[action] = int(choice["visits"]) / total_visits
    return target


def sample_action_from_choices(
    choices: Sequence[dict[str, Any]],
    rng: random.Random,
) -> int:
    """Sample the played move from root visit counts."""

    actions = [int(choice["action"]) for choice in choices]
    weights = [int(choice["visits"]) for choice in choices]
    if sum(weights) == 0:
        raise ValueError("teacher returned no visited actions")
    return rng.choices(actions, weights=weights, k=1)[0]


def terminal_values(
    samples: list[dict[str, Any]],
    terminal_state: GameState,
) -> list[dict[str, Any]]:
    """Fill value targets from the terminal result."""

    completed: list[dict[str, Any]] = []
    terminal = terminal_payload(terminal_state)
    for sample in samples:
        player = int(sample["state"]["current_player"])
        completed_sample = dict(sample)
        completed_sample["value_target"] = terminal_state.result_for(player)
        completed_sample["terminal"] = terminal
        completed.append(completed_sample)
    return completed


def generate_game_samples(
    *,
    game_id: int,
    simulations: int,
    seed: int,
    puct_prior: str = "heuristic",
    puct_rollout: str = "biased",
    max_actions: int = 512,
    early_win_enabled: bool = True,
    teacher: Teacher = suggest_with_teacher_puct,
) -> list[dict[str, Any]]:
    """Play one teacher self-play game and return completed training samples."""

    rng = random.Random(seed)
    state = GameState.new(early_win_enabled=early_win_enabled)
    samples: list[dict[str, Any]] = []

    for ply in range(max_actions):
        if state.is_terminal():
            return terminal_values(samples, state)

        search_seed = rng.randrange(1 << 63)
        payload = teacher(state, simulations, search_seed, puct_prior, puct_rollout)
        if payload is None:
            raise RuntimeError("C++ MCTS binary is not built; run `make -C catchup/cpp` first")

        choices = payload["choices"]
        samples.append(
            {
                "state": state_payload(state),
                "policy_target": policy_target_from_choices(choices),
                "value_target": None,
                "terminal": None,
                "meta": {
                    "action_count": ACTION_COUNT,
                    "finish_action": FINISH,
                    "teacher": f"puct:{simulations}:prior={puct_prior}:rollout={puct_rollout}",
                    "early_win": early_win_enabled,
                    "game_id": game_id,
                    "ply": ply,
                    "seed": search_seed,
                },
            }
        )
        state = state.apply_action(sample_action_from_choices(choices, rng))

    raise RuntimeError(f"self-play game exceeded max_actions={max_actions}")


def generate_samples(
    *,
    games: int,
    simulations: int,
    seed: int | None = None,
    puct_prior: str = "heuristic",
    puct_rollout: str = "biased",
    max_actions: int = 512,
    early_win_enabled: bool = True,
    workers: int = 1,
    teacher: Teacher = suggest_with_teacher_puct,
) -> list[dict[str, Any]]:
    """Generate samples for several independent self-play games."""

    if workers > 1 and teacher is not suggest_with_teacher_puct:
        raise ValueError("custom teacher injection is only supported with workers=1")

    base_seed = seed if seed is not None else random.SystemRandom().randrange(1 << 63)
    all_samples: list[dict[str, Any]] = []
    if workers > 1:
        return generate_samples_parallel(
            games=games,
            simulations=simulations,
            seed=base_seed,
            puct_prior=puct_prior,
            puct_rollout=puct_rollout,
            max_actions=max_actions,
            early_win_enabled=early_win_enabled,
            workers=workers,
        )

    for game_id in range(games):
        all_samples.extend(
            generate_game_samples(
                game_id=game_id,
                simulations=simulations,
                seed=base_seed + game_id,
                puct_prior=puct_prior,
                puct_rollout=puct_rollout,
                max_actions=max_actions,
                early_win_enabled=early_win_enabled,
                teacher=teacher,
            )
        )
    return all_samples


def generate_samples_parallel(
    *,
    games: int,
    simulations: int,
    seed: int,
    puct_prior: str,
    puct_rollout: str,
    max_actions: int,
    early_win_enabled: bool,
    workers: int,
) -> list[dict[str, Any]]:
    """Generate samples in several worker processes."""

    worker_count = min(workers, games)
    game_ids_by_worker = [
        list(range(worker_id, games, worker_count))
        for worker_id in range(worker_count)
    ]
    context = multiprocessing_context()
    parent_connections = []
    processes = []

    for game_ids in game_ids_by_worker:
        parent_conn, child_conn = context.Pipe(duplex=False)
        process = context.Process(
            target=generate_samples_worker,
            args=(
                child_conn,
                game_ids,
                simulations,
                seed,
                puct_prior,
                puct_rollout,
                max_actions,
                early_win_enabled,
            ),
        )
        process.start()
        child_conn.close()
        parent_connections.append(parent_conn)
        processes.append(process)

    game_batches: list[tuple[int, list[dict[str, Any]]]] = []
    errors: list[str] = []
    for connection in parent_connections:
        message = connection.recv()
        connection.close()
        if message["ok"]:
            game_batches.extend(message["games"])
        else:
            errors.append(message["traceback"])

    for process in processes:
        process.join()
        if process.exitcode and process.exitcode != 0:
            errors.append(f"worker process exited with code {process.exitcode}")

    if errors:
        raise RuntimeError("\n".join(errors))

    all_samples: list[dict[str, Any]] = []
    for _, samples in sorted(game_batches, key=lambda item: item[0]):
        all_samples.extend(samples)
    return all_samples


def multiprocessing_context() -> multiprocessing.context.BaseContext:
    """Return a process context that avoids semaphore preflight where possible."""

    methods = multiprocessing.get_all_start_methods()
    if "fork" in methods:
        return multiprocessing.get_context("fork")
    return multiprocessing.get_context()


def generate_samples_worker(
    connection: multiprocessing.connection.Connection,
    game_ids: Sequence[int],
    simulations: int,
    seed: int,
    puct_prior: str,
    puct_rollout: str,
    max_actions: int,
    early_win_enabled: bool,
) -> None:
    """Generate assigned games and send them back to the parent process."""

    try:
        game_batches = []
        for game_id in game_ids:
            game_batches.append(
                (
                    game_id,
                    generate_game_samples(
                        game_id=game_id,
                        simulations=simulations,
                        seed=seed + game_id,
                        puct_prior=puct_prior,
                        puct_rollout=puct_rollout,
                        max_actions=max_actions,
                        early_win_enabled=early_win_enabled,
                    ),
                )
            )
        connection.send({"ok": True, "games": game_batches})
    except BaseException:
        connection.send({"ok": False, "traceback": traceback.format_exc()})
    finally:
        connection.close()


def write_jsonl(samples: Sequence[dict[str, Any]], output_path: Path) -> None:
    """Write training samples as JSON Lines."""

    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", encoding="utf-8") as handle:
        for sample in samples:
            handle.write(json.dumps(sample, separators=(",", ":")))
            handle.write("\n")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--games", type=int, required=True)
    parser.add_argument("--simulations", type=int, required=True)
    parser.add_argument("--seed", type=int)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--puct-prior", choices=("flat", "heuristic"), default="heuristic")
    parser.add_argument("--puct-rollout", choices=("flat", "biased"), default="biased")
    parser.add_argument("--max-actions", type=int, default=512)
    parser.add_argument("--early-win", choices=("true", "false"), default="true")
    parser.add_argument("--workers", type=int, default=1)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    if args.games <= 0:
        parser.error("--games must be positive")
    if args.simulations <= 0:
        parser.error("--simulations must be positive")
    if args.workers <= 0:
        parser.error("--workers must be positive")

    samples = generate_samples(
        games=args.games,
        simulations=args.simulations,
        seed=args.seed,
        puct_prior=args.puct_prior,
        puct_rollout=args.puct_rollout,
        max_actions=args.max_actions,
        early_win_enabled=args.early_win == "true",
        workers=args.workers,
    )
    write_jsonl(samples, args.out)
    print(f"wrote {len(samples)} samples to {args.out} using {min(args.workers, args.games)} workers")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
