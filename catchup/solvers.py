"""Baseline players and playout helpers for Catchup."""

from __future__ import annotations

import random
from dataclasses import dataclass, field
from typing import Protocol

from .game import GameState


class ActionPlayer(Protocol):
    """Chooses one legal factored action from a game state."""

    def choose_action(self, state: GameState) -> int:
        """Return one action from ``state.legal_actions()``."""


@dataclass
class RandomPlayer:
    """Uniform random baseline over the engine's legal factored actions."""

    rng: random.Random = field(default_factory=random.Random)

    @classmethod
    def with_seed(cls, seed: int) -> "RandomPlayer":
        return cls(random.Random(seed))

    def choose_action(self, state: GameState) -> int:
        actions = state.legal_actions()
        if not actions:
            raise ValueError("cannot choose an action from a terminal state")
        return self.rng.choice(actions)


def random_playout(
    state: GameState | None = None,
    player: ActionPlayer | None = None,
    max_actions: int | None = None,
) -> GameState:
    """Play random legal actions until a terminal state is reached."""

    current = state.copy() if state is not None else GameState.new()
    action_player = player if player is not None else RandomPlayer()
    actions_played = 0

    while not current.is_terminal():
        if max_actions is not None and actions_played >= max_actions:
            raise RuntimeError("random playout exceeded max_actions")
        current = current.apply_action(action_player.choose_action(current))
        actions_played += 1

    return current
