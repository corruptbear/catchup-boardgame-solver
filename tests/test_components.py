import unittest

from catchup.board import BOARD
from catchup.components import PLAYER_ONE, PLAYER_TWO, ComponentTracker


class NonIterableOwners(list):
    def __iter__(self):
        raise AssertionError("components() should not scan owners")


class ComponentTrackerTest(unittest.TestCase):
    def test_claim_merges_adjacent_same_color_cells(self) -> None:
        tracker = ComponentTracker()
        center = BOARD.index(0, 0)
        neighbor = BOARD.index(1, 0)

        tracker.claim(PLAYER_ONE, center)
        self.assertEqual(tracker.group_sizes(PLAYER_ONE), (1,))

        tracker.claim(PLAYER_ONE, neighbor)
        self.assertEqual(tracker.group_sizes(PLAYER_ONE), (2,))
        self.assertEqual(len(tracker.components(PLAYER_ONE)), 1)

    def test_opposing_cells_do_not_merge(self) -> None:
        tracker = ComponentTracker()
        center = BOARD.index(0, 0)
        neighbor = BOARD.index(1, 0)

        tracker.claim(PLAYER_ONE, center)
        tracker.claim(PLAYER_TWO, neighbor)

        self.assertEqual(tracker.group_sizes(PLAYER_ONE), (1,))
        self.assertEqual(tracker.group_sizes(PLAYER_TWO), (1,))

    def test_bridge_cell_merges_multiple_existing_groups(self) -> None:
        tracker = ComponentTracker()
        left = BOARD.index(-1, 0)
        right = BOARD.index(1, 0)
        bridge = BOARD.index(0, 0)

        tracker.claim(PLAYER_ONE, left)
        tracker.claim(PLAYER_ONE, right)
        self.assertEqual(tracker.group_sizes(PLAYER_ONE), (1, 1))
        self.assertEqual(len(tracker.roots[PLAYER_ONE]), 2)

        tracker.claim(PLAYER_ONE, bridge)

        self.assertEqual(tracker.group_sizes(PLAYER_ONE), (3,))
        self.assertEqual(len(tracker.roots[PLAYER_ONE]), 1)
        self.assertEqual(tracker.components(PLAYER_ONE)[0].cells, tuple(sorted((left, bridge, right))))

    def test_bridge_cell_merges_more_than_two_existing_groups(self) -> None:
        tracker = ComponentTracker()
        bridge = BOARD.index(0, 0)
        spokes = (
            BOARD.index(1, 0),
            BOARD.index(0, -1),
            BOARD.index(-1, 1),
        )

        for cell in spokes:
            tracker.claim(PLAYER_ONE, cell)

        self.assertEqual(tracker.group_sizes(PLAYER_ONE), (1, 1, 1))
        self.assertEqual(len(tracker.roots[PLAYER_ONE]), 3)

        tracker.claim(PLAYER_ONE, bridge)

        self.assertEqual(tracker.group_sizes(PLAYER_ONE), (4,))
        self.assertEqual(len(tracker.roots[PLAYER_ONE]), 1)
        self.assertEqual(tracker.components(PLAYER_ONE)[0].cells, tuple(sorted((*spokes, bridge))))

    def test_components_uses_incremental_membership_not_owner_scan(self) -> None:
        tracker = ComponentTracker()
        center = BOARD.index(0, 0)
        neighbor = BOARD.index(1, 0)
        tracker.claim(PLAYER_ONE, center)
        tracker.claim(PLAYER_ONE, neighbor)
        tracker.owners = NonIterableOwners(tracker.owners)

        components = tracker.components(PLAYER_ONE)

        self.assertEqual(len(components), 1)
        self.assertEqual(components[0].size, 2)
        self.assertEqual(components[0].cells, tuple(sorted((center, neighbor))))

    def test_copy_is_independent(self) -> None:
        tracker = ComponentTracker()
        center = BOARD.index(0, 0)
        neighbor = BOARD.index(1, 0)
        tracker.claim(PLAYER_ONE, center)

        clone = tracker.copy()
        clone.claim(PLAYER_ONE, neighbor)

        self.assertEqual(tracker.group_sizes(PLAYER_ONE), (1,))
        self.assertEqual(clone.group_sizes(PLAYER_ONE), (2,))

    def test_empty_count_is_preserved_across_claims_and_copy(self) -> None:
        tracker = ComponentTracker()
        tracker.claim(PLAYER_ONE, BOARD.index(0, 0))
        tracker.claim(PLAYER_TWO, BOARD.index(1, 0))

        clone = tracker.copy()
        clone.claim(PLAYER_ONE, BOARD.index(0, 1))

        self.assertEqual(tracker.empty_count(), BOARD.cell_count - 2)
        self.assertEqual(clone.empty_count(), BOARD.cell_count - 3)

    def test_claiming_occupied_cell_fails(self) -> None:
        tracker = ComponentTracker()
        center = BOARD.index(0, 0)
        tracker.claim(PLAYER_ONE, center)

        with self.assertRaises(ValueError):
            tracker.claim(PLAYER_TWO, center)


if __name__ == "__main__":
    unittest.main()
