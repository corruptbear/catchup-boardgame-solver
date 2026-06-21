"""Baseline players and playout helpers for Catchup."""

from __future__ import annotations

import math
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


@dataclass(eq=False)
class MCTSNode:
    """One node in a plain UCT search tree."""

    state: GameState
    parent: "MCTSNode | None" = None
    action_from_parent: int | None = None
    children: dict[int, "MCTSNode"] = field(default_factory=dict)
    visits: int = 0
    total_value: float = 0.0
    untried_actions: list[int] = field(init=False)

    def __post_init__(self) -> None:
        self.untried_actions = list(self.state.legal_actions())

    def mean_value(self) -> float:
        if self.visits == 0:
            return 0.0
        return self.total_value / self.visits


@dataclass
class MCTSPlayer:
    """Basic non-neural UCT player with random rollouts."""

    simulations: int = 100
    exploration_weight: float = 1.4
    rng: random.Random = field(default_factory=random.Random)
    rollout_player: ActionPlayer | None = None
    max_rollout_actions: int | None = None

    def __post_init__(self) -> None:
        if self.simulations <= 0:
            raise ValueError("simulations must be positive")
        if self.exploration_weight < 0:
            raise ValueError("exploration_weight must be non-negative")

    @classmethod
    def with_seed(
        cls,
        seed: int,
        simulations: int = 100,
        exploration_weight: float = 1.4,
    ) -> "MCTSPlayer":
        return cls(
            simulations=simulations,
            exploration_weight=exploration_weight,
            rng=random.Random(seed),
        )

    def choose_action(self, state: GameState) -> int:
        if state.is_terminal():
            raise ValueError("cannot choose an action from a terminal state")

        root = self.search(state)
        return self._best_root_action(root)

    def search(self, state: GameState) -> MCTSNode:
        if state.is_terminal():
            raise ValueError("cannot search from a terminal state")

        root = MCTSNode(state.copy())
        for _ in range(self.simulations):
            node = self._select_and_expand(root)
            terminal_state = self._rollout(node.state)
            self._backpropagate(node, terminal_state)
        return root

    def _select_and_expand(self, node: MCTSNode) -> MCTSNode:
        while not node.state.is_terminal() and not node.untried_actions and node.children:
            node = self._select_child(node)

        if node.state.is_terminal() or not node.untried_actions:
            return node

        action = self._pop_random_untried_action(node)
        child = MCTSNode(
            state=node.state.apply_action(action),
            parent=node,
            action_from_parent=action,
        )
        node.children[action] = child
        return child

    def _select_child(self, node: MCTSNode) -> MCTSNode:
        parent_visits = max(node.visits, 1)
        best_score = float("-inf")
        best_children: list[MCTSNode] = []

        for child in node.children.values():
            if child.visits == 0:
                score = float("inf")
            else:
                exploitation = self._mean_value_for_player(child, node.state.current_player)
                exploration = self.exploration_weight * math.sqrt(
                    math.log(parent_visits) / child.visits
                )
                score = exploitation + exploration

            if score > best_score:
                best_score = score
                best_children = [child]
            elif score == best_score:
                best_children.append(child)

        return self.rng.choice(best_children)

    def _rollout(self, state: GameState) -> GameState:
        rollout_player = self.rollout_player
        if rollout_player is None:
            rollout_player = RandomPlayer(self.rng)
        return random_playout(
            state,
            player=rollout_player,
            max_actions=self.max_rollout_actions,
        )

    @staticmethod
    def _backpropagate(node: MCTSNode, terminal_state: GameState) -> None:
        while node is not None:
            node.visits += 1
            node.total_value += terminal_state.result_for(node.state.current_player)
            node = node.parent

    def _best_root_action(self, root: MCTSNode) -> int:
        best_score = None
        best_actions: list[int] = []
        for action, child in root.children.items():
            score = (
                child.visits,
                self._mean_value_for_player(child, root.state.current_player),
            )
            if best_score is None or score > best_score:
                best_score = score
                best_actions = [action]
            elif score == best_score:
                best_actions.append(action)
        return self.rng.choice(best_actions)

    def _pop_random_untried_action(self, node: MCTSNode) -> int:
        index = self.rng.randrange(len(node.untried_actions))
        return node.untried_actions.pop(index)

    @staticmethod
    def _mean_value_for_player(node: MCTSNode, player: int) -> float:
        value = node.mean_value()
        if node.state.current_player == player:
            return value
        return -value


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
