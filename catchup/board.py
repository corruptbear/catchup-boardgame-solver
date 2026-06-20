"""Board topology for the 61-cell Catchup hex board."""

from __future__ import annotations

from dataclasses import dataclass


Coordinate = tuple[int, int]


@dataclass(frozen=True)
class Board:
    """A radius-4 hexagon stored as row-major axial coordinates."""

    row_lengths: tuple[int, ...] = (5, 6, 7, 8, 9, 8, 7, 6, 5)

    def __post_init__(self) -> None:
        radius = (len(self.row_lengths) - 1) // 2
        coords: list[Coordinate] = []
        rows: list[tuple[int, ...]] = []

        for row, r in enumerate(range(-radius, radius + 1)):
            q_min = max(-radius, -r - radius)
            q_max = min(radius, -r + radius)
            row_indices: list[int] = []
            for q in range(q_min, q_max + 1):
                row_indices.append(len(coords))
                coords.append((q, r))
            if len(row_indices) != self.row_lengths[row]:
                raise ValueError("row_lengths do not describe a regular hexagon")
            rows.append(tuple(row_indices))

        coord_to_index = {coord: index for index, coord in enumerate(coords)}
        neighbor_directions = ((1, 0), (-1, 0), (0, 1), (0, -1), (1, -1), (-1, 1))
        neighbors: list[tuple[int, ...]] = []
        for q, r in coords:
            cell_neighbors = []
            for dq, dr in neighbor_directions:
                neighbor = coord_to_index.get((q + dq, r + dr))
                if neighbor is not None:
                    cell_neighbors.append(neighbor)
            neighbors.append(tuple(cell_neighbors))

        object.__setattr__(self, "radius", radius)
        object.__setattr__(self, "coords", tuple(coords))
        object.__setattr__(self, "coord_to_index", coord_to_index)
        object.__setattr__(self, "rows", tuple(rows))
        object.__setattr__(self, "neighbors", tuple(neighbors))
        object.__setattr__(self, "cell_count", len(coords))

    def index(self, q: int, r: int) -> int:
        """Return the cell index for an axial coordinate."""

        return self.coord_to_index[(q, r)]

    def coordinate(self, index: int) -> Coordinate:
        """Return the axial coordinate for a cell index."""

        self.require_cell(index)
        return self.coords[index]

    def require_cell(self, index: int) -> None:
        """Raise if index is not a valid board cell."""

        if index < 0 or index >= self.cell_count:
            raise ValueError(f"invalid cell index: {index}")


BOARD = Board()
