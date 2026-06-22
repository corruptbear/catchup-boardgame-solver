"""Exact maximum independent set checks for the Catchup board graph."""

from __future__ import annotations

from functools import cache
from pathlib import Path
import sys


sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from catchup.board import BOARD


def maximum_independent_set() -> int:
    """Return one maximum independent set as a bitmask of board cells."""

    neighbor_masks = _neighbor_masks()
    closed_neighbor_masks = [
        (1 << cell) | neighbor_masks[cell] for cell in range(BOARD.cell_count)
    ]

    @cache
    def search(remaining: int) -> int:
        if remaining == 0:
            return 0

        isolated = _isolated_vertices(remaining, neighbor_masks)
        if isolated:
            return isolated | search(remaining & ~isolated)

        cell = _highest_degree_vertex(remaining, neighbor_masks)
        cell_bit = 1 << cell

        include = cell_bit | search(remaining & ~closed_neighbor_masks[cell])
        exclude = search(remaining & ~cell_bit)

        if include.bit_count() >= exclude.bit_count():
            return include
        return exclude

    return search((1 << BOARD.cell_count) - 1)


def row_mask_maximum_independent_set() -> int:
    """Return one maximum independent set using exhaustive row-mask search."""

    rows = BOARD.rows
    valid_masks = [_valid_row_masks(len(row)) for row in rows]
    scores_by_row: list[dict[int, int]] = []
    parents_by_row: list[dict[int, int | None]] = []

    for row_index, masks in enumerate(valid_masks):
        scores: dict[int, int] = {}
        parents: dict[int, int | None] = {}
        for current_mask in masks:
            if row_index == 0:
                scores[current_mask] = current_mask.bit_count()
                parents[current_mask] = None
                continue

            best_score = -1
            best_previous = None
            for previous_mask, previous_score in scores_by_row[-1].items():
                if _compatible_row_masks(row_index, previous_mask, current_mask):
                    score = previous_score + current_mask.bit_count()
                    if score > best_score:
                        best_score = score
                        best_previous = previous_mask

            scores[current_mask] = best_score
            parents[current_mask] = best_previous

        scores_by_row.append(scores)
        parents_by_row.append(parents)

    row_masks: list[int] = []
    current_mask = max(scores_by_row[-1], key=scores_by_row[-1].__getitem__)
    for row_index in range(len(rows) - 1, -1, -1):
        row_masks.append(current_mask)
        previous = parents_by_row[row_index][current_mask]
        current_mask = 0 if previous is None else previous

    row_masks.reverse()
    return _board_mask_from_row_masks(row_masks)


def _valid_row_masks(row_length: int) -> list[int]:
    return [
        mask
        for mask in range(1 << row_length)
        if mask & (mask << 1) == 0
    ]


def _compatible_row_masks(
    row_index: int,
    previous_mask: int,
    current_mask: int,
) -> bool:
    previous_cells = {
        BOARD.rows[row_index - 1][index] for index in _mask_bit_indices(previous_mask)
    }
    for index in _mask_bit_indices(current_mask):
        cell = BOARD.rows[row_index][index]
        if previous_cells & set(BOARD.neighbors[cell]):
            return False
    return True


def _board_mask_from_row_masks(row_masks: list[int]) -> int:
    board_mask = 0
    for row_index, row_mask in enumerate(row_masks):
        for index in _mask_bit_indices(row_mask):
            board_mask |= 1 << BOARD.rows[row_index][index]
    return board_mask


def _mask_bit_indices(mask: int) -> list[int]:
    indices = []
    scan = mask
    while scan:
        bit = scan & -scan
        indices.append(bit.bit_length() - 1)
        scan ^= bit
    return indices


def _neighbor_masks() -> list[int]:
    masks = []
    for neighbors in BOARD.neighbors:
        mask = 0
        for neighbor in neighbors:
            mask |= 1 << neighbor
        masks.append(mask)
    return masks


def _isolated_vertices(remaining: int, neighbor_masks: list[int]) -> int:
    isolated = 0
    scan = remaining
    while scan:
        vertex_bit = scan & -scan
        vertex = vertex_bit.bit_length() - 1
        if neighbor_masks[vertex] & remaining == 0:
            isolated |= vertex_bit
        scan ^= vertex_bit
    return isolated


def _highest_degree_vertex(remaining: int, neighbor_masks: list[int]) -> int:
    best_vertex = -1
    best_degree = -1
    scan = remaining
    while scan:
        vertex_bit = scan & -scan
        vertex = vertex_bit.bit_length() - 1
        degree = (neighbor_masks[vertex] & remaining).bit_count()
        if degree > best_degree:
            best_vertex = vertex
            best_degree = degree
        scan ^= vertex_bit
    return best_vertex


def _cells(mask: int) -> list[int]:
    cells = []
    scan = mask
    while scan:
        vertex_bit = scan & -scan
        cells.append(vertex_bit.bit_length() - 1)
        scan ^= vertex_bit
    return cells


def _assert_independent(mask: int) -> None:
    for cell in _cells(mask):
        for neighbor in BOARD.neighbors[cell]:
            if mask & (1 << neighbor):
                raise AssertionError(f"adjacent cells in result: {cell}, {neighbor}")


if __name__ == "__main__":
    branch_result = maximum_independent_set()
    row_mask_result = row_mask_maximum_independent_set()
    _assert_independent(branch_result)
    _assert_independent(row_mask_result)
    if branch_result.bit_count() != row_mask_result.bit_count():
        raise AssertionError("maximum independent set implementations disagree")

    print(f"branch-and-reduce size: {branch_result.bit_count()}")
    print("branch-and-reduce cells:", " ".join(str(cell) for cell in _cells(branch_result)))
    print(f"row-mask size: {row_mask_result.bit_count()}")
    print("row-mask cells:", " ".join(str(cell) for cell in _cells(row_mask_result)))
