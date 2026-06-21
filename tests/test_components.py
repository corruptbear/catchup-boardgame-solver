import unittest

from catchup.board import BOARD
from catchup.components import EMPTY, PLAYER_ONE, PLAYER_TWO, ComponentTracker


class NonIterableCellOwners(list):
    def __iter__(self):
        raise AssertionError("components() should not scan cell_owners")


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
        tracker.cell_owners = NonIterableCellOwners(tracker.cell_owners)

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
        center = BOARD.index(0, 0)
        neighbor = BOARD.index(1, 0)
        next_cell = BOARD.index(0, 1)
        tracker.claim(PLAYER_ONE, center)
        tracker.claim(PLAYER_TWO, neighbor)

        clone = tracker.copy()
        clone.claim(PLAYER_ONE, next_cell)

        self.assertEqual(tracker.empty_count(), BOARD.cell_count - 2)
        self.assertEqual(clone.empty_count(), BOARD.cell_count - 3)
        self.assertNotIn(center, tracker.empty_cells)
        self.assertNotIn(neighbor, tracker.empty_cells)
        self.assertIn(next_cell, tracker.empty_cells)
        self.assertNotIn(next_cell, clone.empty_cells)

    def test_empty_cells_initializes_from_existing_cell_owners(self) -> None:
        cell_owners = [EMPTY] * BOARD.cell_count
        cell_owners[0] = PLAYER_ONE
        cell_owners[1] = PLAYER_TWO

        tracker = ComponentTracker(cell_owners=cell_owners)

        self.assertEqual(tracker.empty_count(), BOARD.cell_count - 2)
        self.assertNotIn(0, tracker.empty_cells)
        self.assertNotIn(1, tracker.empty_cells)
        self.assertIn(2, tracker.empty_cells)

    def test_existing_cell_owners_rebuilds_claimed_components_before_empty_boundaries(self) -> None:
        cell_owners = [EMPTY] * BOARD.cell_count
        blue_cells = (BOARD.index(0, 0), BOARD.index(1, 0))
        white_cell = BOARD.index(0, -1)
        for cell in blue_cells:
            cell_owners[cell] = PLAYER_ONE
        cell_owners[white_cell] = PLAYER_TWO

        tracker = ComponentTracker(cell_owners=cell_owners)
        empty_blue_roots = {
            root
            for region in tracker.empty_components()
            for root in region.blue_roots
        }

        self.assertEqual(tracker.group_sizes(PLAYER_ONE), (2,))
        self.assertEqual(tracker.components(PLAYER_ONE)[0].cells, tuple(sorted(blue_cells)))
        self.assertEqual(len(empty_blue_roots), 1)
        self.assertEqual(tracker.sizes[PLAYER_ONE][next(iter(empty_blue_roots))], 2)

    def test_empty_components_initially_track_whole_board(self) -> None:
        tracker = ComponentTracker()

        regions = tracker.empty_components()

        self.assertEqual(len(regions), 1)
        self.assertEqual(regions[0].size, BOARD.cell_count)
        self.assertEqual(regions[0].cells, tuple(range(BOARD.cell_count)))
        self.assertEqual(regions[0].blue_roots, ())
        self.assertEqual(regions[0].white_roots, ())

    def test_empty_region_splits_after_bridge_cell_claim(self) -> None:
        center = BOARD.index(0, 0)
        spokes = (
            BOARD.index(1, 0),
            BOARD.index(0, -1),
            BOARD.index(-1, 1),
        )
        cell_owners = [PLAYER_TWO] * BOARD.cell_count
        for cell in (*spokes, center):
            cell_owners[cell] = EMPTY
        tracker = ComponentTracker(cell_owners=cell_owners)

        before = tracker.empty_components()
        self.assertEqual(len(before), 1)
        self.assertEqual(before[0].size, 4)

        tracker.claim(PLAYER_ONE, center)
        regions = tracker.empty_components()

        self.assertEqual(tuple(region.size for region in regions), (1, 1, 1))
        self.assertEqual({region.root for region in regions}, set(spokes))
        for spoke in spokes:
            self.assertEqual(tracker.empty_component_of[spoke], spoke)
        for region in regions:
            self.assertEqual(region.blue_roots, (center,))
            self.assertTrue(region.white_roots)

    def test_empty_adjacency_updates_when_claimed_components_merge(self) -> None:
        tracker = ComponentTracker()
        left = BOARD.index(-1, 0)
        bridge = BOARD.index(0, 0)
        right = BOARD.index(1, 0)

        tracker.claim(PLAYER_ONE, left)
        tracker.claim(PLAYER_ONE, right)
        before_blue_roots = {
            root
            for region in tracker.empty_components()
            for root in region.blue_roots
        }
        self.assertEqual(before_blue_roots, {left, right})

        tracker.claim(PLAYER_ONE, bridge)
        final_root = tracker.components(PLAYER_ONE)[0].root
        after_blue_roots = {
            root
            for region in tracker.empty_components()
            for root in region.blue_roots
        }

        self.assertEqual(tracker.group_sizes(PLAYER_ONE), (3,))
        self.assertEqual(after_blue_roots, {final_root})
        self.assertNotIn(left if left != final_root else right, after_blue_roots)

    def test_reachable_group_bounds_initially_allow_whole_board(self) -> None:
        tracker = ComponentTracker()

        self.assertEqual(tracker.reachable_group_bounds(PLAYER_ONE), (BOARD.cell_count,))
        self.assertEqual(tracker.reachable_group_bounds(PLAYER_TWO), (BOARD.cell_count,))

    def test_reachable_group_bounds_group_claimed_and_empty_components(self) -> None:
        center = BOARD.index(0, 0)
        spokes = (
            BOARD.index(1, 0),
            BOARD.index(0, -1),
            BOARD.index(-1, 1),
        )
        cell_owners = [PLAYER_TWO] * BOARD.cell_count
        cell_owners[center] = PLAYER_ONE
        for cell in spokes:
            cell_owners[cell] = EMPTY
        tracker = ComponentTracker(cell_owners=cell_owners)

        self.assertEqual(tracker.reachable_group_bounds(PLAYER_ONE), (4,))

    def test_claiming_occupied_cell_fails(self) -> None:
        tracker = ComponentTracker()
        center = BOARD.index(0, 0)
        tracker.claim(PLAYER_ONE, center)

        with self.assertRaises(ValueError):
            tracker.claim(PLAYER_TWO, center)


if __name__ == "__main__":
    unittest.main()
