"""Incremental connected-component tracking for Catchup positions.

The tracker maintains two related views of the board:

* claimed components, separately for each player, with union-find;
* empty connected regions, with boundary links to nearby claimed components.

Claimed components only merge as the game unfolds, but empty regions can split
when a cell is claimed. The empty-region code therefore rebuilds only the old
empty region that contained the claimed cell, not the whole board.
"""

from __future__ import annotations

from dataclasses import dataclass
from itertools import chain

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
class EmptyComponent:
    """Read-only description of one empty connected region.

    ``blue_roots`` and ``white_roots`` are claimed-component roots touching this
    empty region. They are roots in the claimed union-find structure, not cells
    chosen specially for display.
    """

    root: int
    size: int
    cells: tuple[int, ...]
    blue_roots: tuple[int, ...]
    white_roots: tuple[int, ...]


class ComponentTracker:
    """Tracks claimed groups and empty regions for one board position.

    Same-color claimed groups only merge, so union-find is a natural fit. Empty
    regions are different: claiming a cell removes one empty cell and may split
    its old empty region into several new regions. The tracker updates that one
    affected region incrementally.
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

        # Claimed-component state. A cell has parent -1 while empty; once
        # claimed, cell_owners[cell] tells which player owns its component.
        # Active component roots are also kept in roots[player] for component
        # iteration and empty-boundary references. size_histogram builds Catchup
        # score vectors without sorting roots on every query.
        self.parents = [-1] * board.cell_count
        self.sizes = [0] * board.cell_count
        self.roots = [set(), set()]
        self.size_histogram = [
            [0] * (board.cell_count + 1)
            for _ in PLAYERS
        ]
        self.max_group_size = [0, 0]

        # Empty-region state. cell_owners is the single source of truth for
        # whether a cell is empty; _empty_count caches the count for O(1) access.
        # empty_component_of maps each empty cell to the root of its current
        # empty region. Claimed cells have empty_component_of -1.
        self._empty_count = self.cell_owners.count(EMPTY)
        self.empty_component_of = [-1] * board.cell_count
        self.empty_component_cells: dict[int, set[int]] = {}

        # Boundary maps connect empty regions to claimed components and back.
        # The forward map is useful for reachable-region bounds and UI display.
        # The reverse map makes claimed-component merges update only the empty
        # regions that actually referenced the merged root.
        self.empty_adjacency: dict[int, list[set[int]]] = {}
        self.claimed_adjacent_empty: list[dict[int, set[int]]] = [dict(), dict()]

        if cell_owners is not None:
            self._rebuild_claimed_components_from_cell_owners()
        self._rebuild_empty_components_from_cells({
            cell
            for cell, owner in enumerate(self.cell_owners)
            if owner == EMPTY
        })

    def copy(self) -> "ComponentTracker":
        """Return a cheap independent copy suitable for search branches."""

        clone = type(self).__new__(type(self))
        clone.board = self.board
        clone.cell_owners = self.cell_owners.copy()
        clone.parents = self.parents.copy()
        clone.sizes = self.sizes.copy()
        clone.roots = [root_set.copy() for root_set in self.roots]
        clone.size_histogram = [
            histogram.copy()
            for histogram in self.size_histogram
        ]
        clone.max_group_size = self.max_group_size.copy()
        clone._empty_count = self._empty_count
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
        """Return the raw owner for UI/debug display; not used by MCTS."""

        self.board.require_cell(cell)
        return self.cell_owners[cell]

    def is_empty(self, cell: int) -> bool:
        """Return whether one cell is empty for UI/tests; not used by MCTS."""

        self.board.require_cell(cell)
        return self.cell_owners[cell] == EMPTY

    def empty_count(self) -> int:
        return self._empty_count

    def empty_cell_indices(self, min_cell: int = 0) -> tuple[int, ...]:
        """Return empty cells in increasing index order."""

        if min_cell >= self.board.cell_count:
            return ()

        start = max(0, min_cell)
        return tuple(
            cell
            for cell in range(start, self.board.cell_count)
            if self.cell_owners[cell] == EMPTY
        )

    def claim(self, player: int, cell: int) -> None:
        """Claim one empty cell and merge adjacent same-color groups."""

        self.board.require_cell(cell)
        if self.cell_owners[cell] != EMPTY:
            raise ValueError(f"cell {cell} is already claimed")

        # Capture the old empty region before changing ownership. After the
        # claim, only that old region can have split.
        old_empty_root = self.empty_component_of[cell]
        self.cell_owners[cell] = player
        self._empty_count -= 1
        self.parents[cell] = cell
        self.sizes[cell] = 1
        self.roots[player].add(cell)
        self._add_group_size(player, 1)

        for neighbor in self.board.neighbors[cell]:
            if self.cell_owners[neighbor] == player:
                self._union(player, cell, neighbor)

        self._split_empty_region_after_claim(old_empty_root, cell)

    def group_sizes(self, player: int) -> tuple[int, ...]:
        """Return this player's component sizes in descending order."""

        sizes: list[int] = []
        for size in range(self.max_group_size[player], 0, -1):
            count = self.size_histogram[player][size]
            if count:
                sizes.extend([size] * count)
        return tuple(sizes)

    def largest_group_size(self, player: int | None = None) -> int:
        """Return one player's largest group, or the global largest group."""

        if player is not None:
            return self.max_group_size[player]
        return max(self.max_group_size)

    def components(self, player: int) -> tuple[Component, ...]:
        """Return current connected groups for inspection/tests; not used by MCTS."""

        cells_by_root = {root: [] for root in self.roots[player]}
        for cell, owner in enumerate(self.cell_owners):
            if owner == player:
                cells_by_root[self._find(cell)].append(cell)

        return tuple(
            sorted(
                (
                    Component(
                        player=player,
                        root=root,
                        size=self.sizes[root],
                        cells=tuple(cells),
                    )
                    for root, cells in cells_by_root.items()
                ),
                key=lambda component: (-component.size, component.root),
            )
        )

    def empty_components(self) -> tuple[EmptyComponent, ...]:
        """Return empty regions for UI/inspection/tests; not used by MCTS."""

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

    def reachable_group_bounds(self, player: int) -> tuple[int, ...]:
        """Return optimistic component-size bounds through own cells plus empty cells."""

        # Traverse a bipartite graph:
        #   claimed component root <-> adjacent empty region root
        # Opponent claimed components are absent, so they act as walls. Each
        # connected part of this graph is one optimistic region the player could
        # eventually connect through empty cells.
        #
        # Claimed nodes are encoded as their root cell id. Empty-region nodes are
        # encoded as empty_node_offset + root. This avoids allocating
        # ("claimed", root) / ("empty", root) tuples and temporary neighbor tuples
        # in this hot early-terminal path.
        empty_node_offset = self.board.cell_count
        all_nodes = chain(
            self.roots[player],
            (empty_node_offset + root for root in self.empty_component_cells),
        )

        visited: set[int] = set()
        bounds: list[int] = []
        for node in all_nodes:
            if node in visited:
                continue

            # DFS over the claimed/empty component graph for one reachable region.
            reachable_size = 0
            stack = [node]
            visited.add(node)
            while stack:
                current = stack.pop()
                if current < empty_node_offset:
                    root = current
                    reachable_size += self.sizes[root]
                    for empty_root in self.claimed_adjacent_empty[player].get(root, ()):
                        linked_node = empty_node_offset + empty_root
                        if linked_node not in visited:
                            visited.add(linked_node)
                            stack.append(linked_node)
                else:
                    root = current - empty_node_offset
                    reachable_size += len(self.empty_component_cells[root])
                    for touching_claimed_root in self.empty_adjacency[root][player]:
                        if touching_claimed_root not in visited:
                            visited.add(touching_claimed_root)
                            stack.append(touching_claimed_root)

            bounds.append(reachable_size)

        return tuple(sorted(bounds, reverse=True))

    def _union(self, player: int, first: int, second: int) -> int:
        first_root = self._find(first)
        second_root = self._find(second)
        if first_root == second_root:
            return first_root

        if self.sizes[first_root] < self.sizes[second_root]:
            first_root, second_root = second_root, first_root

        first_size = self.sizes[first_root]
        second_size = self.sizes[second_root]
        merged_size = first_size + second_size

        # Attach the smaller claimed component to the larger root. The size is
        # stored only on active roots; non-root sizes are cleared to catch
        # accidental use in debugging.
        self.parents[second_root] = first_root
        self.sizes[first_root] = merged_size
        self.sizes[second_root] = 0
        self._remove_group_size(player, first_size)
        self._remove_group_size(player, second_size)
        self._add_group_size(player, merged_size)

        self.roots[player].discard(second_root)
        self._replace_adjacent_claimed_root(player, second_root, first_root)
        return first_root

    def _find(self, cell: int) -> int:
        parent = self.parents[cell]
        if parent == -1:
            raise ValueError(f"cell {cell} is not claimed")
        if parent != cell:
            self.parents[cell] = self._find(parent)
        return self.parents[cell]

    def _add_group_size(self, player: int, size: int) -> None:
        self.size_histogram[player][size] += 1
        if size > self.max_group_size[player]:
            self.max_group_size[player] = size

    def _remove_group_size(self, player: int, size: int) -> None:
        # max_group_size never decreases in Catchup: claimed components only grow
        # by adding cells or merging with other same-color components.
        self.size_histogram[player][size] -= 1

    def _rebuild_claimed_components_from_cell_owners(self) -> None:
        """Build claimed union-find state from an existing owner array."""

        self.parents = [-1] * self.board.cell_count
        self.sizes = [0] * self.board.cell_count
        self.roots = [set(), set()]
        self.size_histogram = [
            [0] * (self.board.cell_count + 1)
            for _ in PLAYERS
        ]
        self.max_group_size = [0, 0]

        for cell, owner in enumerate(self.cell_owners):
            if owner in PLAYERS:
                self.parents[cell] = cell
                self.sizes[cell] = 1
                self.roots[owner].add(cell)
                self._add_group_size(owner, 1)

        for cell, owner in enumerate(self.cell_owners):
            if owner in PLAYERS:
                for neighbor in self.board.neighbors[cell]:
                    # Union each same-color edge once.
                    if neighbor > cell and self.cell_owners[neighbor] == owner:
                        self._union(owner, cell, neighbor)

    def _split_empty_region_after_claim(self, old_root: int, claimed_cell: int) -> None:
        """Replace one old empty region after a cell is claimed from it."""

        if old_root == -1:
            return

        old_cells = self.empty_component_cells[old_root]
        remaining_cells = set(old_cells)
        remaining_cells.discard(claimed_cell)
        if not remaining_cells:
            self._unregister_empty_component(old_root)
            return

        # DFS flood-fill first, before touching metadata. If all remaining cells
        # are reached and the old root is still empty, the region did not split
        # and can keep its root.
        unvisited = set(remaining_cells)
        component, adjacency = self._flood_empty_component(
            next(iter(unvisited)),
            unvisited,
        )
        if not unvisited and old_root != claimed_cell:
            old_cells.remove(claimed_cell)
            self.empty_component_of[claimed_cell] = -1
            self._refresh_empty_component_adjacency(old_root, adjacency)
            return

        self._unregister_empty_component(old_root)
        self._register_empty_component(min(component), component, adjacency)
        self._rebuild_empty_components_from_cells(unvisited)

    def _rebuild_empty_components_from_cells(self, cells: set[int]) -> None:
        """Flood-fill a set of empty cells into registered empty regions."""

        remaining = set(cells)
        while remaining:
            start = next(iter(remaining))
            # DFS flood-fill; this mutates remaining by removing every reached cell.
            component, adjacency = self._flood_empty_component(start, remaining)
            self._register_empty_component(min(component), component, adjacency)

    def _flood_empty_component(
        self,
        start: int,
        remaining: set[int],
    ) -> tuple[set[int], list[set[int]]]:
        """Remove and return one empty component plus its claimed boundaries."""

        remaining.remove(start)
        cells = {start}
        adjacency: list[set[int]] = [set(), set()]
        # DFS over adjacent empty cells inside the supplied remaining set.
        stack = [start]

        while stack:
            cell = stack.pop()
            for neighbor in self.board.neighbors[cell]:
                if neighbor in remaining:
                    remaining.remove(neighbor)
                    cells.add(neighbor)
                    stack.append(neighbor)
                else:
                    owner = self.cell_owners[neighbor]
                    if owner in PLAYERS:
                        adjacency[owner].add(self._find(neighbor))
        return cells, adjacency

    def _register_empty_component(
        self,
        root: int,
        cells: set[int],
        adjacency: list[set[int]],
    ) -> None:
        """Add one empty region and its claimed-component boundary links."""

        self.empty_component_cells[root] = cells
        for cell in cells:
            self.empty_component_of[cell] = root

        self.empty_adjacency[root] = adjacency
        self._add_empty_adjacency_reverse_links(root, adjacency)

    def _unregister_empty_component(self, root: int) -> set[int]:
        """Remove one empty region and clean its reverse boundary links."""

        if root == -1:
            return set()

        cells = self.empty_component_cells.pop(root)
        for cell in cells:
            self.empty_component_of[cell] = -1

        adjacency = self.empty_adjacency.pop(root)
        self._remove_empty_adjacency_reverse_links(root, adjacency)
        return cells

    def _refresh_empty_component_adjacency(
        self,
        root: int,
        new_adjacency: list[set[int]],
    ) -> None:
        """Recompute boundary links for one empty region without changing its cells."""

        old_adjacency = self.empty_adjacency[root]
        self._remove_empty_adjacency_reverse_links(root, old_adjacency)
        self.empty_adjacency[root] = new_adjacency
        self._add_empty_adjacency_reverse_links(root, new_adjacency)

    def _add_empty_adjacency_reverse_links(
        self,
        empty_root: int,
        adjacency: list[set[int]],
    ) -> None:
        for player in PLAYERS:
            for touching_claimed_root in adjacency[player]:
                self.claimed_adjacent_empty[player].setdefault(
                    touching_claimed_root,
                    set(),
                ).add(empty_root)

    def _remove_empty_adjacency_reverse_links(
        self,
        empty_root: int,
        adjacency: list[set[int]],
    ) -> None:
        for player in PLAYERS:
            for touching_claimed_root in adjacency[player]:
                empty_roots = self.claimed_adjacent_empty[player].get(
                    touching_claimed_root,
                )
                if empty_roots is not None:
                    empty_roots.discard(empty_root)
                    if not empty_roots:
                        del self.claimed_adjacent_empty[player][touching_claimed_root]

    def _replace_adjacent_claimed_root(self, player: int, old_root: int, new_root: int) -> None:
        """Update empty-region boundary links after two claimed roots merge."""

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
