# Component Tracking

The engine tracks both claimed components and empty regions incrementally. This
supports faster search and makes early-win upper bounds easier to compute later.

## Claimed Components

Claimed cells are tracked with one union-find forest:

```python
parents[cell] -> parent cell, or -1 if empty
sizes[root] -> component size
roots[player] -> set[root]
size_histogram[player][size] -> number of this player's active groups of size
max_group_size[player] -> largest active group size
```

`cell_owners[cell]` identifies which player owns a claimed cell, so `parents`
and `sizes` do not need a player dimension. `roots[player]` is kept for scoring
and boundary updates. `size_histogram` makes score-vector rebuilds avoid sorting
component roots.

When a player claims a cell:

1. The cell becomes a new one-cell component.
2. The tracker checks the claimed cell's same-color neighbors.
3. Same-color neighboring components are merged.

This handles bridge moves naturally. If one claimed cell connects three existing
same-color groups, the tracker unions through each same-color neighbor until
only one component remains.

## Empty Regions

Empty cells are tracked as connected regions:

```python
cell_owners[cell] == EMPTY -> True if the cell is empty
_empty_count -> number of empty cells
empty_component_of[cell] -> empty region root, or -1
empty_component_cells[root] -> set[cell]
```

When a cell is claimed, `cell_owners[cell]` is changed from `EMPTY` to the
claiming player and `_empty_count` is decremented. The old empty region that
contained that cell may split, so the tracker runs:

```python
_split_empty_region_after_claim(old_empty_root, claimed_cell)
```

That method first flood-fills the old region's remaining cells to see whether
the claim actually split the region. If all remaining cells are still connected
and the old root cell was not claimed, the tracker keeps the existing empty
region record and only refreshes its boundary links.

When the empty region vanishes, splits, or needs a new root because the old root
cell was claimed, the method:

1. Unregisters the old empty region.
2. Registers the already flood-filled first replacement region.
3. Flood-fills any remaining cells into additional replacement regions.
4. Registers each additional replacement region.

Only the old empty region is rebuilt. Unrelated empty regions are not scanned.

## Boundary Adjacency

Each empty region also tracks which claimed components touch it:

```python
empty_adjacency[empty_root][player] -> set[touching_claimed_root]
```

There is also a reverse map:

```python
claimed_adjacent_empty[player][touching_claimed_root] -> set[empty_region_root]
```

This lets an empty region answer:

- Which Blue groups could expand into this region?
- Which White groups could expand into this region?

The UI displays this as `#root(size)` for each adjacent claimed group.

If claimed components merge, the tracker updates empty-region boundary
references from the merged claimed root to the remaining root.

## Operation Costs

Let:

```text
n = number of board cells
d = maximum hex degree, at most 6
c = number of claimed components for a player
e = number of empty regions
k = size of the old empty region containing a claimed cell
alpha(n) = inverse Ackermann function from union-find
```

Basic queries:

```text
owner(cell)                  O(1)
is_empty(cell)               average O(1)
empty_count()                O(1)
empty_component_of[cell]     O(1)
```

Claimed components:

```text
_find(cell)                   amortized O(alpha(n))
group_sizes(player)           O(m + c)
largest_group_size(player)    O(1)
components(player)            O(n * alpha(n) + c log c + output size)
```

Here `m` is `max_group_size[player]`, at most `n`. `group_sizes(player)` scans
the size histogram downward and emits each size according to its count, so it no
longer sorts roots.

`_union(player, first, second)` still takes `player` because it updates
`roots[player]` and that player's boundary references, but the underlying
union-find arrays are playerless. It is more expensive than plain union-find
because the tracker also maintains empty-region boundary data.

It does:

1. Find the two current claimed-component roots.
2. Attach the smaller claimed component to the larger one.
3. Decrement the two old component-size histogram counts and increment the
   merged-size count.
4. Update every empty region that used to reference the merged claimed root
   so it now references the remaining claimed root.

So if the merged claimed root touches `m` empty regions, the cost is:

```text
_union(...)  O(alpha(n) + m)
```

Example: if Blue group root `12` is merged into Blue group root `21`, and root
`12` touched empty regions `{0, 35}`, the union updates 2 empty-region
references. It does not maintain any cell-list metadata for inspection.

`components(player)` is inspection/debug output, not an MCTS hot-path method. It
materializes component cells on demand by scanning `cell_owners` and finding the
root for each of that player's cells.

Empty regions:

```text
empty_components()            O(e log e + output size)
_split_empty_region_after_claim(...)  O(k * d * alpha(n) + r)
```

`_split_empty_region_after_claim(old_empty_root, claimed_cell)` only rebuilds
the old empty region that contained the claimed cell, and avoids a full metadata
rebuild when the old region stays connected with the same root.

It does:

1. Flood-fill the remaining cells in the old region.
2. If there is no split and the old root is still empty, remove only the claimed
   cell and refresh that region's boundary links.
3. Otherwise remove the old empty-region record.
4. Register the replacement empty region or regions.
5. Update reverse links from the touching claimed component roots back to the new empty
   region roots.

The flood-fill collects each replacement region's touching claimed-component
roots while it walks the empty cells, so replacement registration does not scan
the same cells a second time just to compute adjacency.

Here `r` is the number of claimed-component/empty-region links removed and
created during that split. For example, if the old empty region touched 2 Blue
groups and 1 White group, and after the split the replacement regions together
touch 3 Blue groups and 2 White groups, then `r = 2 + 1 + 3 + 2 = 8`.

Because `d <= 6` and `alpha(n)` is effectively constant, this is usually
described as:

```text
O(k + r)
```

where `k` is only the affected old empty region, not the whole board.
