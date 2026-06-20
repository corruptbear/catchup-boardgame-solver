import unittest

from catchup.board import BOARD


class BoardTest(unittest.TestCase):
    def test_board_shape(self) -> None:
        self.assertEqual(BOARD.row_lengths, (5, 6, 7, 8, 9, 8, 7, 6, 5))
        self.assertEqual(BOARD.cell_count, 61)
        self.assertEqual(tuple(len(row) for row in BOARD.rows), BOARD.row_lengths)

    def test_neighbors_are_symmetric(self) -> None:
        for cell, neighbors in enumerate(BOARD.neighbors):
            self.assertLessEqual(len(neighbors), 6)
            for neighbor in neighbors:
                self.assertIn(cell, BOARD.neighbors[neighbor])

    def test_center_has_six_neighbors(self) -> None:
        center = BOARD.index(0, 0)
        self.assertEqual(len(BOARD.neighbors[center]), 6)


if __name__ == "__main__":
    unittest.main()
