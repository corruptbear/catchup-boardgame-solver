"""Incremental connected-component tracking for Catchup positions."""

from __future__ import annotations

from dataclasses import dataclass

from .board import BOARD, Board


EMPTY = -1
PLAYER_ONE = 0
PLAYER_TWO = 1
PLAYERS = (PLAYER_ONE, PLAYER_TWO)


@dataclass(frozen=True)
class Component:
    """Read-only description of one connected group."""

    player: int
    root: int
    size: int
    cells: tuple[int, ...]


class ComponentTracker:
    """Tracks both players' groups with incremental union-find updates.

    Cells are only ever added in this game, so same-color groups merge but never
    split. That makes union-find a good fit for search: applying a move touches
    only the claimed cell and its six neighbors.
    """

    def __init__(
        self,
        board: Board = BOARD,
        owners: list[int] | None = None,
        parents: list[list[int]] | None = None,
        sizes: list[list[int]] | None = None,
        roots: list[set[int]] | None = None,
        component_cells: list[dict[int, set[int]]] | None = None,
        empty_count: int | None = None,
    ) -> None:
        self.board = board
        self.owners = owners if owners is not None else [EMPTY] * board.cell_count
        self.parents = parents if parents is not None else [[-1] * board.cell_count for _ in PLAYERS]
        self.sizes = sizes if sizes is not None else [[0] * board.cell_count for _ in PLAYERS]
        self.roots = roots if roots is not None else [set(), set()]
        self.component_cells = (
            component_cells if component_cells is not None else [dict(), dict()]
        )
        self._empty_count = (
            empty_count
            if empty_count is not None
            else board.cell_count if owners is None else self.owners.count(EMPTY)
        )

    def copy(self) -> "ComponentTracker":
        """Return a cheap independent copy suitable for search branches."""

        return ComponentTracker(
            board=self.board,
            owners=self.owners.copy(),
            parents=[parent.copy() for parent in self.parents],
            sizes=[size.copy() for size in self.sizes],
            roots=[root_set.copy() for root_set in self.roots],
            component_cells=[
                {root: cells.copy() for root, cells in player_cells.items()}
                for player_cells in self.component_cells
            ],
            empty_count=self._empty_count,
        )

    def owner(self, cell: int) -> int:
        self.board.require_cell(cell)
        return self.owners[cell]

    def is_empty(self, cell: int) -> bool:
        return self.owner(cell) == EMPTY

    def empty_count(self) -> int:
        return self._empty_count

    def claim(self, player: int, cell: int) -> None:
        """Claim one empty cell and merge adjacent same-color groups."""

        self._require_player(player)
        self.board.require_cell(cell)
        if self.owners[cell] != EMPTY:
            raise ValueError(f"cell {cell} is already claimed")

        self.owners[cell] = player
        self._empty_count -= 1
        self.parents[player][cell] = cell
        self.sizes[player][cell] = 1
        self.roots[player].add(cell)
        self.component_cells[player][cell] = {cell}

        for neighbor in self.board.neighbors[cell]:
            if self.owners[neighbor] == player:
                self._union(player, cell, neighbor)

    def group_sizes(self, player: int) -> tuple[int, ...]:
        """Return this player's component sizes in descending order."""

        self._require_player(player)
        return tuple(sorted((self.sizes[player][root] for root in self.roots[player]), reverse=True))

    def largest_group_size(self, player: int | None = None) -> int:
        """Return one player's largest group, or the global largest group."""

        if player is not None:
            self._require_player(player)
            return max((self.sizes[player][root] for root in self.roots[player]), default=0)
        return max((self.largest_group_size(player) for player in PLAYERS), default=0)

    def components(self, player: int) -> tuple[Component, ...]:
        """Return current connected groups for inspection and tests."""

        self._require_player(player)
        return tuple(
            sorted(
                (
                    Component(
                        player=player,
                        root=root,
                        size=self.sizes[player][root],
                        cells=tuple(sorted(self.component_cells[player][root])),
                    )
                    for root in self.roots[player]
                ),
                key=lambda component: (-component.size, component.root),
            )
        )

    def _union(self, player: int, first: int, second: int) -> int:
        first_root = self._find(player, first)
        second_root = self._find(player, second)
        if first_root == second_root:
            return first_root

        if self.sizes[player][first_root] < self.sizes[player][second_root]:
            first_root, second_root = second_root, first_root

        self.parents[player][second_root] = first_root
        self.sizes[player][first_root] += self.sizes[player][second_root]
        self.sizes[player][second_root] = 0
        self.component_cells[player][first_root].update(self.component_cells[player].pop(second_root))
        self.roots[player].discard(second_root)
        return first_root

    def _find(self, player: int, cell: int) -> int:
        parent = self.parents[player][cell]
        if parent == -1:
            raise ValueError(f"cell {cell} is not claimed by player {player}")
        if parent != cell:
            self.parents[player][cell] = self._find(player, parent)
        return self.parents[player][cell]

    @staticmethod
    def _require_player(player: int) -> None:
        if player not in PLAYERS:
            raise ValueError(f"invalid player: {player}")
