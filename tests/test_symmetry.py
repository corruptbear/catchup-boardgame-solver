import unittest

from catchup.board import BOARD
from catchup.game import FINISH
from catchup.symmetry import (
    SYMMETRIES,
    transform_action_values,
    transform_cell_values,
    transform_cells,
)


class SymmetryTest(unittest.TestCase):
    def test_has_twelve_unique_symmetries(self) -> None:
        self.assertEqual(len(SYMMETRIES), 12)
        self.assertEqual(len({symmetry.cell_map for symmetry in SYMMETRIES}), 12)

    def test_each_symmetry_is_a_cell_permutation(self) -> None:
        expected = tuple(range(BOARD.cell_count))
        for symmetry in SYMMETRIES:
            self.assertEqual(tuple(sorted(symmetry.cell_map)), expected)

    def test_each_symmetry_preserves_neighbors(self) -> None:
        neighbor_sets = [set(neighbors) for neighbors in BOARD.neighbors]
        for symmetry in SYMMETRIES:
            for cell, neighbors in enumerate(BOARD.neighbors):
                mapped_cell = symmetry.map_cell(cell)
                mapped_neighbors = {
                    symmetry.map_cell(neighbor)
                    for neighbor in neighbors
                }
                self.assertEqual(mapped_neighbors, neighbor_sets[mapped_cell])

    def test_transform_cell_values_moves_values_to_mapped_cells(self) -> None:
        symmetry = SYMMETRIES[1]
        values = list(range(BOARD.cell_count))

        transformed = transform_cell_values(values, symmetry)

        for old_cell, new_cell in enumerate(symmetry.cell_map):
            self.assertEqual(transformed[new_cell], values[old_cell])

    def test_transform_action_values_moves_cells_and_keeps_finish(self) -> None:
        symmetry = SYMMETRIES[1]
        values = list(range(FINISH + 1))
        values[FINISH] = 999

        transformed = transform_action_values(values, symmetry)

        for old_cell, new_cell in enumerate(symmetry.cell_map):
            self.assertEqual(transformed[new_cell], values[old_cell])
        self.assertEqual(transformed[FINISH], 999)

    def test_transform_cells_preserves_input_order(self) -> None:
        symmetry = SYMMETRIES[1]
        cells = [0, 12, 60]

        self.assertEqual(
            transform_cells(cells, symmetry),
            [symmetry.map_cell(cell) for cell in cells],
        )


if __name__ == "__main__":
    unittest.main()
