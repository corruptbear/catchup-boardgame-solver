# Catchup Game Analysis

These notes collect strategy observations about Catchup itself, separate from
implementation details.

## Claiming More Cells Is Usually Good

A basic observation is that, in most positions, the current player should claim
as many cells as legally possible. Extra claimed cells usually give direct
benefits:

```text
more occupied territory
more chances to connect own groups
more chances to block opponent connections
fewer empty cells left for the opponent
```

So an early finish is not usually neutral. Passing up a legal claim often gives
away board control.

## Why Finish Early At All?

The important exception is the Catchup turn-size rule. After a turn, if the
player increased the global largest connected group, the opponent may get up to
three cells on the next turn. If the player did not increase the global largest
group, the opponent gets up to two cells.

That means an extra claim can be bad when it unavoidably increases the global
largest connected group size and gives the opponent a three-cell turn. In that
situation, finishing early may be better than taking a cell that triggers the
opponent's larger response.

So the rough strategic rule is:

```text
claim as much as possible
unless every useful extra claim increases the global largest group size
and giving the opponent three cells is worse than stopping now
```

## Reading The Global-Largest Tax

The cost of growing depends on who currently owns the global largest group.

If the opponent owns the largest group, you can often grow freely until you
match it. Your group may get bigger without increasing the global largest size,
so the opponent does not get the three-cell response just because you caught up
to the existing benchmark.

If you own the largest group, extending it is costly unless the move is
tactically valuable. Any increase to that group can raise the global largest
size and hand the opponent a larger turn.

Merging two own groups can still be worth the tax if the jump is large enough.
For example, connecting two medium groups may create such a strong component
that giving the opponent three cells is acceptable.

A harmless isolated claim is often better than finishing early if it avoids
increasing the global largest size. It still takes territory, reduces future
empty space, and may create later connection threats without paying the
three-cell-turn tax immediately.

This also matters for rollout design. A rollout policy that gives one-cell,
two-cell, and three-cell turns equal probability may be strategically unnatural:
it makes early finishing much more common than it should be in many positions.
A better random rollout policy probably needs to strongly prefer claiming the
maximum number of cells, while still allowing early finish when claiming more
would trigger a strategically bad increase to the global largest group size.
