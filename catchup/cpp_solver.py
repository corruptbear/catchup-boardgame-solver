"""Bridge from the Python UI/server to the optional C++ MCTS binary."""

from __future__ import annotations

import json
import os
import subprocess
from pathlib import Path
from typing import Any

from .game import GameState


DEFAULT_SOLVER_PATH = Path(__file__).resolve().parent.parent / "cpp" / "build" / "catchup_mcts"


def find_cpp_solver() -> Path | None:
    """Return the configured C++ solver binary, if it exists."""

    configured = os.environ.get("CATCHUP_MCTS_BINARY")
    path = Path(configured) if configured else DEFAULT_SOLVER_PATH
    return path if path.is_file() else None


def cpp_solver_args(
    state: GameState,
    simulations: int,
    seed: int = 1,
) -> list[str]:
    """Build argv for the C++ solver from a Python game state."""

    return [
        "--owners",
        ",".join(str(owner) for owner in state.tracker.cell_owners),
        "--selected",
        ",".join(str(cell) for cell in state.selected),
        "--current-player",
        str(state.current_player),
        "--max-claims",
        str(state.max_claims),
        "--turn-start-largest",
        str(state.turn_start_largest),
        "--opening-turn",
        "1" if state.opening_turn else "0",
        "--completed-turns",
        str(state.completed_turns),
        "--simulations",
        str(simulations),
        "--seed",
        str(seed),
    ]


def suggest_with_cpp_mcts(
    state: GameState,
    simulations: int,
    seed: int = 1,
) -> dict[str, Any] | None:
    """Return C++ MCTS JSON output, or None when the binary is not built."""

    binary = find_cpp_solver()
    if binary is None:
        return None

    command = [str(binary), *cpp_solver_args(state, simulations, seed)]
    completed = subprocess.run(
        command,
        capture_output=True,
        check=False,
        text=True,
        timeout=max(5.0, simulations / 100.0),
    )
    if completed.returncode != 0:
        stderr = completed.stderr.strip()
        raise RuntimeError(stderr or "C++ MCTS solver failed")

    payload = json.loads(completed.stdout)
    if "action" not in payload or "choices" not in payload:
        raise RuntimeError("C++ MCTS solver returned an invalid payload")
    return payload
