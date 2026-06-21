# Component Tracking

The engine tracks both claimed components and empty regions incrementally. This
supports faster search and makes early-win upper bounds easier to compute later.

## Claimed Components

Claimed cells are tracked separately for each player with union-find:

```python
parents[player][cell] -> parent cell
sizes[player][root] -> component size
roots[player] -> set[root]
component_members[player][root] -> immutable linked membership tree
```

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
empty_cells -> set[cell]
empty_component_of[cell] -> empty region root, or -1
empty_component_cells[root] -> set[cell]
```

When a cell is claimed, it is removed from `empty_cells`. The old empty region
that contained that cell may split, so the tracker runs:

```python
_split_empty_region_after_claim(old_empty_root, claimed_cell)
```

That method:

1. Unregisters the old empty region.
2. Removes the claimed cell from that region's cell set.
3. Flood-fills the remaining cells into replacement empty regions.
4. Registers each replacement region.

Only the old empty region is rebuilt. Unrelated empty regions are not scanned.

## Boundary Adjacency

Each empty region also tracks which claimed components touch it:

```python
empty_adjacency[empty_root][player] -> set[claimed_component_root]
```

There is also a reverse map:

```python
claimed_adjacent_empty[player][claimed_root] -> set[empty_region_root]
```

This lets an empty region answer:

- Which Blue groups could expand into this region?
- Which White groups could expand into this region?

The UI displays this as `#root(size)` for each adjacent claimed group.

If claimed components merge, the tracker links their membership trees and updates
empty-region boundary references from the merged claimed root to the remaining
root.

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
_find(player, cell)           amortized O(alpha(n))
group_sizes(player)           O(c log c)
largest_group_size(player)    O(c)
components(player)            O(c log c + output size)
```

`_union(player, first, second)` is more expensive than plain union-find because
the tracker also maintains empty-region boundary data.

It does:

1. Find the two current claimed-component roots.
2. Attach the smaller claimed component to the larger one.
3. Link the two claimed-component membership trees. This keeps access to all
   cells without moving them during the union.
4. Update every empty region that used to reference the merged claimed root
   so it now references the remaining claimed root.

So if the merged claimed root touches `m` empty regions, the cost is:

```text
_union(...)  O(alpha(n) + m)
```

Example: if Blue group root `12` is merged into Blue group root `21`, and root
`12` touched empty regions `{0, 35}`, the union links root `12`'s membership
tree under root `21` and updates 2 empty-region references. It does not copy all
cells from root `12` into a new set.

`components(player)` materializes component cells by walking each membership
tree when inspection/debug output needs actual cell lists.

Empty regions:

```text
empty_components()            O(e log e + output size)
_split_empty_region_after_claim(...)  O(k * d * alpha(n) + r)
```

`_split_empty_region_after_claim(old_empty_root, claimed_cell)` only rebuilds
the old empty region that contained the claimed cell.

It does:

1. Remove the old empty-region record.
2. Remove the newly claimed cell from that region's cell set.
3. Flood-fill the remaining cells in that old region to find the replacement
   empty regions.
4. For each replacement region, collect neighboring claimed Blue/White component
   roots.
5. Update reverse links from those claimed component roots back to the new empty
   region roots.

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
