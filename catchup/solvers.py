"""Baseline players and playout helpers for Catchup."""

from __future__ import annotations

import math
import random
from dataclasses import dataclass, field
from typing import Protocol, TypeAlias

from .board import BOARD, Board
from .components import PLAYER_ONE, PLAYER_TWO, PLAYERS, ComponentTracker
from .game import (
    EARLY_WIN_CHECK_MIN_FILLED_CELLS,
    FINISH,
    GameState,
    compare_size_vectors,
    other_player,
)


StateKey: TypeAlias = tuple[tuple[int, ...], int, tuple[int, ...], int, int, bool]


@dataclass(frozen=True, slots=True)
class RolloutResult:
    """Terminal rollout value without keeping a terminal board object."""

    winner: int

    def result_for(self, player: int) -> int:
        if player not in PLAYERS:
            raise ValueError(f"invalid player: {player}")
        return 1 if self.winner == player else -1


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


@dataclass(slots=True)
class FastRolloutState:
    """Mutable rollout-only state that avoids copying after every action."""

    board: Board
    tracker: ComponentTracker
    current_player: int
    selected: list[int]
    max_claims: int
    turn_start_largest: int
    opening_turn: bool
    early_win_enabled: bool
    completed_turns: int

    @classmethod
    def from_game_state(cls, state: GameState) -> "FastRolloutState":
        return cls(
            board=state.board,
            tracker=state.tracker.copy(),
            current_player=state.current_player,
            selected=list(state.selected),
            max_claims=state.max_claims,
            turn_start_largest=state.turn_start_largest,
            opening_turn=state.opening_turn,
            early_win_enabled=state.early_win_enabled,
            completed_turns=state.completed_turns,
        )

    @classmethod
    def new(cls, board: Board = BOARD, *, early_win_enabled: bool = True) -> "FastRolloutState":
        return cls.from_game_state(GameState.new(board, early_win_enabled=early_win_enabled))

    def legal_actions(self) -> tuple[int, ...]:
        if self.is_terminal():
            return ()

        actions: list[int] = []
        if self.selected:
            actions.append(FINISH)

        if len(self.selected) < self.max_claims:
            min_cell = self.selected[-1] + 1 if self.selected else 0
            actions.extend(self.tracker.empty_cell_indices(min_cell))

        return tuple(actions)

    def choose_random_action(self, rng: random.Random) -> int:
        actions = self.legal_actions()
        if not actions:
            raise ValueError("cannot choose an action from a terminal state")
        return rng.choice(actions)

    def apply_action(self, action: int) -> None:
        if action == FINISH:
            self._finish_turn()
            return

        self.tracker.claim(self.current_player, action)
        self.selected.append(action)
        if len(self.selected) == self.max_claims or self.tracker.empty_count() == 0:
            self._finish_turn()

    def is_terminal(self) -> bool:
        if self.selected:
            return False
        empty_count = self.tracker.empty_count()
        if empty_count == 0:
            return True
        if not self.early_win_enabled:
            return False
        if self.board.cell_count - empty_count < EARLY_WIN_CHECK_MIN_FILLED_CELLS:
            return False
        return self.proven_winner() is not None

    def group_sizes(self, player: int) -> tuple[int, ...]:
        return self.tracker.group_sizes(player)

    def proven_winner(self) -> int | None:
        if (
            self.selected
            or self.board.cell_count - self.tracker.empty_count()
            < EARLY_WIN_CHECK_MIN_FILLED_CELLS
        ):
            return None

        blue_sizes = self.tracker.group_sizes(PLAYER_ONE)
        white_sizes = self.tracker.group_sizes(PLAYER_TWO)
        blue_bound = self.tracker.reachable_group_bounds(PLAYER_ONE)
        white_bound = self.tracker.reachable_group_bounds(PLAYER_TWO)

        if compare_size_vectors(blue_sizes, white_bound) > 0:
            return PLAYER_ONE
        if compare_size_vectors(white_sizes, blue_bound) > 0:
            return PLAYER_TWO
        return None

    def winner(self) -> int:
        if not self.is_terminal():
            raise ValueError("winner is only defined for terminal states")

        comparison = compare_size_vectors(
            self.tracker.group_sizes(PLAYER_ONE),
            self.tracker.group_sizes(PLAYER_TWO),
        )
        if comparison > 0:
            return PLAYER_ONE
        if comparison < 0:
            return PLAYER_TWO
        raise RuntimeError("terminal Catchup state has equal component vectors")

    def result_for(self, player: int) -> int:
        if player not in PLAYERS:
            raise ValueError(f"invalid player: {player}")
        return 1 if self.winner() == player else -1

    def _finish_turn(self) -> None:
        if not self.selected:
            raise ValueError("cannot finish a turn before claiming a cell")

        new_largest = self.tracker.largest_group_size()
        increased_global_largest = new_largest > self.turn_start_largest
        next_max_claims = 2 if self.opening_turn or not increased_global_largest else 3

        self.selected.clear()
        self.current_player = other_player(self.current_player)
        self.max_claims = next_max_claims
        self.turn_start_largest = new_largest
        self.opening_turn = False
        self.completed_turns += 1


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
    use_transposition_table: bool = True
    use_undo_rollout: bool = False
    last_transposition_table_size: int = field(default=0, init=False)

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

        table: dict[StateKey, MCTSNode] | None = None
        root_state = state.copy()
        if self.use_transposition_table:
            table = {}
            root = self._node_for_state(root_state, table)
        else:
            root = MCTSNode(root_state)

        for _ in range(self.simulations):
            path = self._select_and_expand(root, table)
            terminal_state = self._rollout(path[-1].state)
            self._backpropagate(path, terminal_state)

        self.last_transposition_table_size = len(table) if table is not None else 0
        return root

    def _select_and_expand(
        self,
        node: MCTSNode,
        table: dict[StateKey, MCTSNode] | None,
    ) -> list[MCTSNode]:
        path = [node]
        while not node.state.is_terminal() and not node.untried_actions and node.children:
            node = self._select_child(node)
            path.append(node)

        if node.state.is_terminal() or not node.untried_actions:
            return path

        action = self._pop_random_untried_action(node)
        child = self._child_for_action(node, action, table)
        node.children[action] = child
        path.append(child)
        return path

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

    def _rollout(self, state: GameState) -> GameState | FastRolloutState | RolloutResult:
        rollout_player = self.rollout_player
        if rollout_player is None:
            if self.use_undo_rollout:
                return undo_random_playout_result(
                    state,
                    rng=self.rng,
                    max_actions=self.max_rollout_actions,
                )
            return fast_random_playout(
                state,
                rng=self.rng,
                max_actions=self.max_rollout_actions,
            )
        return random_playout(
            state,
            player=rollout_player,
            max_actions=self.max_rollout_actions,
        )

    @staticmethod
    def _backpropagate(
        path: list[MCTSNode],
        terminal_state: GameState | FastRolloutState | RolloutResult,
    ) -> None:
        for node in reversed(path):
            node.visits += 1
            node.total_value += terminal_state.result_for(node.state.current_player)

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

    def _child_for_action(
        self,
        node: MCTSNode,
        action: int,
        table: dict[StateKey, MCTSNode] | None,
    ) -> MCTSNode:
        child_state = node.state.apply_action(action)
        if table is None:
            return MCTSNode(
                state=child_state,
                parent=node,
                action_from_parent=action,
            )

        child = self._node_for_state(child_state, table)
        if child.parent is None:
            child.parent = node
            child.action_from_parent = action
        return child

    @staticmethod
    def _node_for_state(
        state: GameState,
        table: dict[StateKey, MCTSNode],
    ) -> MCTSNode:
        key = MCTSPlayer._state_key(state)
        node = table.get(key)
        if node is None:
            node = MCTSNode(state)
            table[key] = node
        return node

    @staticmethod
    def _state_key(state: GameState) -> StateKey:
        return (
            tuple(state.tracker.cell_owners),
            state.current_player,
            state.selected,
            state.max_claims,
            state.turn_start_largest,
            state.opening_turn,
        )

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


def fast_random_playout(
    state: GameState | None = None,
    rng: random.Random | None = None,
    max_actions: int | None = None,
) -> FastRolloutState:
    """Play a default random rollout in place after one initial state copy."""

    current = (
        FastRolloutState.from_game_state(state)
        if state is not None
        else FastRolloutState.new()
    )
    action_rng = rng if rng is not None else random.Random()
    actions_played = 0

    while not current.is_terminal():
        if max_actions is not None and actions_played >= max_actions:
            raise RuntimeError("random playout exceeded max_actions")
        current.apply_action(current.choose_random_action(action_rng))
        actions_played += 1

    return current


def undo_random_playout_result(
    state: GameState,
    rng: random.Random | None = None,
    max_actions: int | None = None,
) -> RolloutResult:
    """Play a random rollout in place and roll back with tracker deltas."""

    checkpoint = state.tracker.undo_checkpoint()
    current = FastRolloutState(
        board=state.board,
        tracker=state.tracker,
        current_player=state.current_player,
        selected=list(state.selected),
        max_claims=state.max_claims,
        turn_start_largest=state.turn_start_largest,
        opening_turn=state.opening_turn,
        early_win_enabled=state.early_win_enabled,
        completed_turns=state.completed_turns,
    )
    action_rng = rng if rng is not None else random.Random()
    actions_played = 0

    try:
        while not current.is_terminal():
            if max_actions is not None and actions_played >= max_actions:
                raise RuntimeError("random playout exceeded max_actions")
            current.apply_action(current.choose_random_action(action_rng))
            actions_played += 1
        return RolloutResult(current.winner())
    finally:
        state.tracker.rollback_to(checkpoint)
