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


@dataclass(frozen=True)
class ComponentMembership:
    """Immutable linked membership tree for one claimed component."""

    cell: int | None = None
    left: "ComponentMembership | None" = None
    right: "ComponentMembership | None" = None

    @classmethod
    def leaf(cls, cell: int) -> "ComponentMembership":
        return cls(cell=cell)

    @classmethod
    def merge(
        cls,
        left: "ComponentMembership",
        right: "ComponentMembership",
    ) -> "ComponentMembership":
        return cls(left=left, right=right)

    def iter_cells(self):
        if self.cell is not None:
            yield self.cell
            return
        if self.left is not None:
            yield from self.left.iter_cells()
        if self.right is not None:
            yield from self.right.iter_cells()


@dataclass(frozen=True)
class EmptyComponent:
    """Read-only description of one empty connected region."""

    root: int
    size: int
    cells: tuple[int, ...]
    blue_roots: tuple[int, ...]
    white_roots: tuple[int, ...]


class ComponentTracker:
    """Tracks both players' groups with incremental union-find updates.

    Cells are only ever added in this game, so same-color groups merge but never
    split. That makes union-find a good fit for search: applying a move touches
    only the claimed cell and its six neighbors.
    """

    def __init__(
        self,
        board: Board = BOARD,
        cell_owners: list[int] | None = None,
    ) -> None:
        self.board = board
        if cell_owners is not None and len(cell_owners) != board.cell_count:
            raise ValueError("cell_owners must have one entry per board cell")

        self.cell_owners = (
            cell_owners.copy()
            if cell_owners is not None
            else [EMPTY] * board.cell_count
        )
        self.parents = [[-1] * board.cell_count for _ in PLAYERS]
        self.sizes = [[0] * board.cell_count for _ in PLAYERS]
        self.roots = [set(), set()]
        self.component_members = [dict(), dict()]
        self.empty_cells = {
            cell for cell, owner in enumerate(self.cell_owners) if owner == EMPTY
        }
        self.empty_component_of = [-1] * board.cell_count
        self.empty_component_cells: dict[int, set[int]] = {}
        self.empty_adjacency: dict[int, list[set[int]]] = {}
        self.claimed_adjacent_empty: list[dict[int, set[int]]] = [dict(), dict()]

        if cell_owners is not None:
            self._rebuild_claimed_components_from_cell_owners()
        self._rebuild_empty_components_from_cells(self.empty_cells.copy())

    def copy(self) -> "ComponentTracker":
        """Return a cheap independent copy suitable for search branches."""

        clone = type(self).__new__(type(self))
        clone.board = self.board
        clone.cell_owners = self.cell_owners.copy()
        clone.parents = [parent.copy() for parent in self.parents]
        clone.sizes = [size.copy() for size in self.sizes]
        clone.roots = [root_set.copy() for root_set in self.roots]
        clone.component_members = [
            player_members.copy()
            for player_members in self.component_members
        ]
        clone.empty_cells = self.empty_cells.copy()
        clone.empty_component_of = self.empty_component_of.copy()
        clone.empty_component_cells = {
            root: cells.copy()
            for root, cells in self.empty_component_cells.items()
        }
        clone.empty_adjacency = {
            root: [player_roots.copy() for player_roots in adjacency]
            for root, adjacency in self.empty_adjacency.items()
        }
        clone.claimed_adjacent_empty = [
            {
                root: empty_roots.copy()
                for root, empty_roots in player_roots.items()
            }
            for player_roots in self.claimed_adjacent_empty
        ]
        return clone

    def owner(self, cell: int) -> int:
        self.board.require_cell(cell)
        return self.cell_owners[cell]

    def is_empty(self, cell: int) -> bool:
        self.board.require_cell(cell)
        return cell in self.empty_cells

    def empty_count(self) -> int:
        return len(self.empty_cells)

    def claim(self, player: int, cell: int) -> None:
        """Claim one empty cell and merge adjacent same-color groups."""

        self._require_player(player)
        self.board.require_cell(cell)
        if self.cell_owners[cell] != EMPTY:
            raise ValueError(f"cell {cell} is already claimed")

        old_empty_root = self.empty_component_of[cell]
        self.cell_owners[cell] = player
        self.empty_cells.remove(cell)
        self.parents[player][cell] = cell
        self.sizes[player][cell] = 1
        self.roots[player].add(cell)
        self.component_members[player][cell] = ComponentMembership.leaf(cell)

        for neighbor in self.board.neighbors[cell]:
            if self.cell_owners[neighbor] == player:
                self._union(player, cell, neighbor)

        self._split_empty_region_after_claim(old_empty_root, cell)

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
                        cells=tuple(sorted(self.component_members[player][root].iter_cells())),
                    )
                    for root in self.roots[player]
                ),
                key=lambda component: (-component.size, component.root),
            )
        )

    def empty_components(self) -> tuple[EmptyComponent, ...]:
        """Return current empty connected regions and their claimed boundaries."""

        return tuple(
            sorted(
                (
                    EmptyComponent(
                        root=root,
                        size=len(cells),
                        cells=tuple(sorted(cells)),
                        blue_roots=tuple(sorted(self.empty_adjacency[root][PLAYER_ONE])),
                        white_roots=tuple(sorted(self.empty_adjacency[root][PLAYER_TWO])),
                    )
                    for root, cells in self.empty_component_cells.items()
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
        self.component_members[player][first_root] = ComponentMembership.merge(
            self.component_members[player][first_root],
            self.component_members[player].pop(second_root),
        )
        self.roots[player].discard(second_root)
        self._replace_adjacent_claimed_root(player, second_root, first_root)
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

    def _rebuild_claimed_components_from_cell_owners(self) -> None:
        self.parents = [[-1] * self.board.cell_count for _ in PLAYERS]
        self.sizes = [[0] * self.board.cell_count for _ in PLAYERS]
        self.roots = [set(), set()]
        self.component_members = [dict(), dict()]

        for cell, owner in enumerate(self.cell_owners):
            if owner in PLAYERS:
                self.parents[owner][cell] = cell
                self.sizes[owner][cell] = 1
                self.roots[owner].add(cell)
                self.component_members[owner][cell] = ComponentMembership.leaf(cell)

        for cell, owner in enumerate(self.cell_owners):
            if owner in PLAYERS:
                for neighbor in self.board.neighbors[cell]:
                    if neighbor > cell and self.cell_owners[neighbor] == owner:
                        self._union(owner, cell, neighbor)

    def _split_empty_region_after_claim(self, old_root: int, claimed_cell: int) -> None:
        old_cells = self._unregister_empty_component(old_root)
        old_cells.discard(claimed_cell)
        self._rebuild_empty_components_from_cells(old_cells)

    def _rebuild_empty_components_from_cells(self, cells: set[int]) -> None:
        remaining = set(cells)
        while remaining:
            start = next(iter(remaining))
            component = self._flood_empty_component(start, remaining)
            self._register_empty_component(min(component), component)

    def _flood_empty_component(self, start: int, remaining: set[int]) -> set[int]:
        remaining.remove(start)
        cells = {start}
        stack = [start]

        while stack:
            cell = stack.pop()
            for neighbor in self.board.neighbors[cell]:
                if neighbor in remaining:
                    remaining.remove(neighbor)
                    cells.add(neighbor)
                    stack.append(neighbor)
        return cells

    def _register_empty_component(self, root: int, cells: set[int]) -> None:
        self.empty_component_cells[root] = cells
        for cell in cells:
            self.empty_component_of[cell] = root

        adjacency = self._compute_empty_adjacency(cells)
        self.empty_adjacency[root] = adjacency
        for player in PLAYERS:
            for claimed_root in adjacency[player]:
                self.claimed_adjacent_empty[player].setdefault(claimed_root, set()).add(root)

    def _unregister_empty_component(self, root: int) -> set[int]:
        if root == -1:
            return set()

        cells = self.empty_component_cells.pop(root)
        for cell in cells:
            self.empty_component_of[cell] = -1

        adjacency = self.empty_adjacency.pop(root)
        for player in PLAYERS:
            for claimed_root in adjacency[player]:
                empty_roots = self.claimed_adjacent_empty[player].get(claimed_root)
                if empty_roots is not None:
                    empty_roots.discard(root)
                    if not empty_roots:
                        del self.claimed_adjacent_empty[player][claimed_root]
        return cells

    def _compute_empty_adjacency(self, cells: set[int]) -> list[set[int]]:
        adjacency: list[set[int]] = [set(), set()]
        for cell in cells:
            for neighbor in self.board.neighbors[cell]:
                owner = self.cell_owners[neighbor]
                if owner in PLAYERS:
                    adjacency[owner].add(self._find(owner, neighbor))
        return adjacency

    def _replace_adjacent_claimed_root(self, player: int, old_root: int, new_root: int) -> None:
        empty_roots = self.claimed_adjacent_empty[player].pop(old_root, set())
        if not empty_roots:
            return

        new_empty_roots = self.claimed_adjacent_empty[player].setdefault(new_root, set())
        for empty_root in empty_roots:
            adjacency = self.empty_adjacency.get(empty_root)
            if adjacency is None:
                continue
            adjacency[player].discard(old_root)
            adjacency[player].add(new_root)
            new_empty_roots.add(empty_root)
