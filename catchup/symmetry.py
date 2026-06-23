"""Board symmetry permutations for the radius-4 Catchup hex board."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Sequence, TypeVar

from .board import BOARD, Board, Coordinate
from .game import FINISH


T = TypeVar("T")


@dataclass(frozen=True)
class Symmetry:
    """A board symmetry represented as old cell index -> new cell index."""

    name: str
    cell_map: tuple[int, ...]

    def map_cell(self, cell: int) -> int:
        return self.cell_map[cell]

    def map_action(self, action: int) -> int:
        return FINISH if action == FINISH else self.map_cell(action)


def rotate_coordinate(coord: Coordinate, turns: int) -> Coordinate:
    """Rotate an axial coordinate by `turns * 60` degrees around the center."""

    q, r = coord
    for _ in range(turns % 6):
        q, r = -r, q + r
    return q, r


def reflect_coordinate(coord: Coordinate) -> Coordinate:
    """Reflect an axial coordinate across one hex mirror axis."""

    q, r = coord
    return r, q


def build_symmetries(board: Board = BOARD) -> tuple[Symmetry, ...]:
    """Return the 6 rotations and 6 reflections of the board."""

    symmetries: list[Symmetry] = []
    transforms = [
        (
            f"rot{turns * 60}",
            lambda coord, turns=turns: rotate_coordinate(coord, turns),
        )
        for turns in range(6)
    ]
    transforms.extend(
        (
            f"ref{turns * 60}",
            lambda coord, turns=turns: rotate_coordinate(reflect_coordinate(coord), turns),
        )
        for turns in range(6)
    )

    for name, transform in transforms:
        cell_map = tuple(
            board.index(*transform(board.coordinate(cell)))
            for cell in range(board.cell_count)
        )
        symmetries.append(Symmetry(name=name, cell_map=cell_map))
    return tuple(symmetries)


SYMMETRIES = build_symmetries()


def transform_cell_values(values: Sequence[T], symmetry: Symmetry) -> list[T]:
    """Move one value per cell through a symmetry."""

    transformed = list(values)
    for old_cell, new_cell in enumerate(symmetry.cell_map):
        transformed[new_cell] = values[old_cell]
    return transformed


def transform_action_values(values: Sequence[T], symmetry: Symmetry) -> list[T]:
    """Move one value per action through a symmetry; FINISH stays fixed."""

    transformed = list(values)
    for old_cell, new_cell in enumerate(symmetry.cell_map):
        transformed[new_cell] = values[old_cell]
    transformed[FINISH] = values[FINISH]
    return transformed


def transform_cells(cells: Sequence[int], symmetry: Symmetry) -> list[int]:
    return [symmetry.map_cell(cell) for cell in cells]
