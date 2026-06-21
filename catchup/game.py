"""Rules engine for the Catchup hex claiming game."""

from __future__ import annotations

from dataclasses import dataclass, field

from .board import BOARD, Board
from .components import EMPTY, PLAYER_ONE, PLAYER_TWO, PLAYERS, ComponentTracker


FINISH = BOARD.cell_count
EARLY_WIN_CHECK_MIN_FILLED_CELLS = 30


@dataclass
class GameState:
    """A factored-action Catchup state.

    A real turn is represented as up to three internal actions: select a cell,
    select another cell or finish, then optionally select a third cell or finish.
    """

    board: Board = BOARD
    tracker: ComponentTracker = field(default_factory=ComponentTracker)
    current_player: int = PLAYER_ONE
    selected: tuple[int, ...] = ()
    max_claims: int = 1
    turn_start_largest: int = 0
    opening_turn: bool = True
    completed_turns: int = 0

    @classmethod
    def new(cls, board: Board = BOARD) -> "GameState":
        return cls(board=board, tracker=ComponentTracker(board=board))

    def copy(self) -> "GameState":
        return GameState(
            board=self.board,
            tracker=self.tracker.copy(),
            current_player=self.current_player,
            selected=self.selected,
            max_claims=self.max_claims,
            turn_start_largest=self.turn_start_largest,
            opening_turn=self.opening_turn,
            completed_turns=self.completed_turns,
        )

    def legal_actions(self) -> tuple[int, ...]:
        """Return legal internal actions. Cell actions are canonicalized."""

        if self.is_terminal():
            return ()

        actions: list[int] = []
        if self.selected:
            actions.append(FINISH)

        if len(self.selected) < self.max_claims:
            min_cell = self.selected[-1] + 1 if self.selected else 0
            actions.extend(self.tracker.empty_cell_indices(min_cell))

        return tuple(actions)

    def apply_action(self, action: int) -> "GameState":
        """Return the state after one legal internal action."""

        if action not in self.legal_actions():
            raise ValueError(f"illegal action: {action}")

        next_state = self.copy()
        if action == FINISH:
            return next_state._finish_turn()

        next_state.tracker.claim(next_state.current_player, action)
        next_state.selected = (*next_state.selected, action)
        if len(next_state.selected) == next_state.max_claims or next_state.tracker.empty_count() == 0:
            return next_state._finish_turn()
        return next_state

    def apply_turn(self, cells: tuple[int, ...] | list[int]) -> "GameState":
        """Apply one complete real turn, useful for tests and scripts."""

        if not cells:
            raise ValueError("a turn must claim at least one cell")

        state = self
        for cell in cells:
            state = state.apply_action(cell)
        if state.selected:
            state = state.apply_action(FINISH)
        return state

    def is_terminal(self) -> bool:
        if self.selected:
            return False
        empty_count = self.tracker.empty_count()
        if empty_count == 0:
            return True
        if self.board.cell_count - empty_count < EARLY_WIN_CHECK_MIN_FILLED_CELLS:
            return False
        return self.proven_winner() is not None

    def group_sizes(self, player: int) -> tuple[int, ...]:
        return self.tracker.group_sizes(player)

    def proven_winner(self) -> int | None:
        """Return a winner only when reachable-region bounds prove the result."""

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

    def winner(self) -> int | None:
        """Return the winning player for a terminal state, or None for a tie."""

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
        return None

    def result_for(self, player: int) -> int:
        """Return 1 for win, -1 for loss, 0 for tie from player's perspective."""

        if player not in PLAYERS:
            raise ValueError(f"invalid player: {player}")
        winner = self.winner()
        if winner is None:
            return 0
        return 1 if winner == player else -1

    def _finish_turn(self) -> "GameState":
        if not self.selected:
            raise ValueError("cannot finish a turn before claiming a cell")

        new_largest = self.tracker.largest_group_size()
        increased_global_largest = new_largest > self.turn_start_largest
        next_max_claims = 2 if self.opening_turn or not increased_global_largest else 3

        self.selected = ()
        self.current_player = other_player(self.current_player)
        self.max_claims = next_max_claims
        self.turn_start_largest = new_largest
        self.opening_turn = False
        self.completed_turns += 1
        return self


def other_player(player: int) -> int:
    if player == PLAYER_ONE:
        return PLAYER_TWO
    if player == PLAYER_TWO:
        return PLAYER_ONE
    raise ValueError(f"invalid player: {player}")


def compare_size_vectors(first: tuple[int, ...], second: tuple[int, ...]) -> int:
    """Lexicographically compare sorted component-size vectors."""

    max_len = max(len(first), len(second))
    for index in range(max_len):
        first_size = first[index] if index < len(first) else 0
        second_size = second[index] if index < len(second) else 0
        if first_size != second_size:
            return 1 if first_size > second_size else -1
    return 0
