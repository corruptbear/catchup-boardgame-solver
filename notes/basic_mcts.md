# Basic MCTS

This project currently uses plain Monte Carlo Tree Search, without a neural
network. MCTS is used to pick the next legal action for the current player in
the current board position.

In Catchup, a turn may claim one or more empty cells. The exact number depends
on whether the previous turn increased the largest connected group, so the
engine represents a real turn as smaller internal decisions. At each internal
decision, the player can either claim another legal empty cell or finish the
turn if at least one cell has already been claimed. Because of this, one real
multi-cell turn may be split into several internal actions, and the tree chooses
one internal action at a time.

The search tree is built over those internal actions. A numbered action means
"claim this empty cell." The special `FINISH` action means "stop claiming cells
and pass the turn to the other player." For example, one real turn that claims
cells 12 and 18 is represented as:

```text
12 -> 18 -> FINISH
```

A single real turn can therefore appear as multiple internal actions in the
search tree. The engine canonicalizes multi-cell turns by requiring claimed
cell indices to increase within the turn. That way, claiming cells `{12, 18}`
is represented only as `12 -> 18 -> FINISH`, not also as
`18 -> 12 -> FINISH`. Therefore the search tree will never contain the branch
`18 -> 12 -> FINISH`; after choosing `18`, cell `12` is no longer legal within
that same turn.

## What A Node Stores

Each MCTS node represents one game state after some action sequence.

**Value perspective rule:** `total_value` is always stored from the player-to-move
perspective at that node, not always from Blue's perspective and not always from
the root player's perspective.

It stores:

```text
state               game state at this node
parent              previous tree node
action_from_parent  action that created this node
children            expanded child nodes
visits              how many simulations passed through this node
total_value         accumulated rollout result from this node's player view
untried_actions     legal actions not expanded yet
```

`mean_value()` is:

```text
total_value / visits
```

The value is from the player-to-move perspective at that node.

Example:

```text
ROOT
player to move: Blue
visits: 10
total_value: -2
```

This means 10 simulations backed up through the root, and their results summed
to `-2` from Blue's perspective. One possible result mix is:

```text
4 Blue wins   -> +4
6 Blue losses -> -6
total_value   = -2
mean_value    = -0.2
```

A non-root node may have a different player to move:

```text
ROOT --action #58--> CHILD

CHILD
player to move: White
visits: 3
total_value: +1
mean_value: +0.333
```

Here `+1` is good for White, because the child stores value from White's
perspective. Whenever UCT compares a node's children, each child value is
converted to the parent node's player-to-move perspective:

```text
child mean from White perspective = +0.333
same child from Blue parent's view = -0.333
```

The root is just one place where this happens. The same conversion is used at
every fully expanded non-terminal node during the selection walk.

So `total_value` and `mean_value()` always belong to the node's player to move.
When a parent compares children, each child value is interpreted from the
parent player's perspective.

In pseudocode:

```text
child_value_for_parent =
    child.mean_value()
    if child.state.current_player == parent.state.current_player
    else -child.mean_value()
```

## Search Loop

The root runs a fixed number of simulations, for example 1000. Each simulation
starts from the root and has four phases:

```text
selection -> expansion -> simulation -> backpropagation
```

After the loop, the chosen action is the root child with the most visits, with
converted mean value used as a tiebreaker.

## 1. Selection

Selection builds one path starting at the root. At each node, including the
root:

```text
terminal node             -> stop
node with untried actions -> stop; expansion will add one child here
fully expanded node       -> use UCT to choose one child, then continue
```

The selected node is the last node in this path. In loop form:

```text
start at root
while current node is non-terminal and fully expanded:
    use UCT to choose one child
    move to that child
stop at a node with untried actions, or at a terminal node
```

UCT is not only a root rule. It is used at every fully expanded non-terminal
node along the path. The root is just the first possible decision point. UCT
does not score the current node itself; it scores that node's children as the
candidate actions.

For each child, the score is UCT:

```text
score = exploitation + exploration

exploitation = child's mean value from the current node player's view
exploration  = C * sqrt(log(parent visits) / child visits)
```

Current `C` is `1.4`. This is the standard UCT exploration constant used by
many simple MCTS implementations, not a tuned Catchup-specific value yet.

Numeric example:

```text
parent visits = 10
child visits  = 3
child mean from parent player's view = 0.4
C = 1.4

score = 0.4 + 1.4 * sqrt(log(10) / 3)
      = 0.4 + 1.4 * 0.876
      = 1.626
```

If a child has zero visits, its score is treated as positive infinity, so every
expanded child gets tried before the search relies on averages. If multiple
children have positive-infinity score, they are tied for best score, and one of
them is chosen randomly. The same random tie-break is used for any other equal
best scores.

## 2. Expansion

Expansion happens at the node where selection stops. This node is the selected
node for the current simulation. If it still has untried legal actions,
expansion creates one child from it.

Steps:

1. Pick one random action from `untried_actions`.
2. Remove that action from `untried_actions`.
3. Copy the selected node's state.
4. Apply the action to the copy.
5. Create a child node for the resulting state.
6. Add the child to the parent's `children`.
7. Continue the simulation from this new child.

Expansion must not mutate the selected node's state, because that node may be
visited again in later simulations. Instead, it copies the selected state first,
then applies the chosen action to the copy. That mutated copy is the child
state, and the tree stores it for future selection, expansion, and rollout
starts.

If the selected node is terminal, expansion is skipped. Otherwise, the selected
node should have at least one untried action, so expansion creates one child.

## 3. Simulation (Rollout)

Simulation, also called rollout or playout, estimates the value of the newly
selected/expanded node.

The current C++ solver does:

```text
copy the node state once
while the copied state is not terminal:
    get legal actions
    choose one uniformly at random
    apply it in place
return the terminal state
```

The rollout needs a copy because random playout actions are not added to the
tree. They are only used to estimate the selected node's value. If rollout
mutated the stored tree node directly, the node would no longer represent the
position it is supposed to store. The rollout copy is disposable: it can be
mutated until terminal and then thrown away.

Undo is another way to protect the stored tree node. People use undo when
copying the whole state is expensive but each action only changes a small amount
of state. An undo-based rollout applies random actions in place, records enough
deltas to restore the state, reads the terminal result, and then undoes back to
the rollout start state. In this repo, that undo experiment was slower than
copying once at rollout start.

Rollout stops when the game is terminal. The obvious terminal case is a full
board: there are no empty cells left, so the final component-size comparison can
decide the winner.

The engine can also stop earlier when the result is already proven. For each
player, it estimates the largest connected groups that player could still
possibly make using the remaining empty cells, without crossing the opponent's
stones. If even that optimistic future cannot catch the opponent's current
groups, the winner is decided before the board is full.

The rollout result is:

```text
1   if the queried player won
-1  if the queried player lost
0   for tie
```

A single random rollout from a newly created non-terminal node is noisy. Basic
MCTS reduces that noise by revisiting the node's subtree. When node A is first
created as a non-terminal child, the rollout starts directly from A. Later, if
UCT chooses A again, selection continues inside A, expands one of A's
descendants, and rolls out from that descendant. Backpropagation still updates A
because A is on the selected path.

So a node's value is not based only on rollouts that start exactly at that node:

```text
node visits = all simulations whose selected path passed through this node
node value  = rollout results backed up through this node
```

## 4. Backpropagation

Backpropagation walks back through the path from the rollout node to the root.

For every node on that path:

```text
node.visits += 1
node.total_value += terminal_result_for(node.state.current_player)
```

`terminal_result_for(player)` reads the final winner from the rollout terminal
state and returns `+1` if `player` won, `-1` if `player` lost, and `0` for a tie.

Example: if the terminal state is a Blue win, then the backed-up result is:

```text
+1 for nodes where Blue is to move
-1 for nodes where White is to move
```

This lets every node store value from its own player-to-move perspective.
During selection, child values are converted back into the parent player's
perspective before applying UCT.

In many strictly alternating-turn games, implementations can backpropagate by
flipping the sign at every tree level. Catchup's search tree cannot safely use
that shortcut. A real Catchup turn may contain multiple factored subactions by
the same player, so a parent and child node can have the same player to move.
Instead of assuming depth parity, backpropagation asks for the terminal result
from each node's actual `state.current_player`.

## Choosing The Move

After all simulations finish, the root action is chosen from root children.

The current policy is:

```text
highest visit count wins
if tied, higher converted mean value from the root player's perspective wins
if still tied, choose randomly
```

Visit count is preferred because MCTS is primarily using the tree policy to
allocate more simulations to more promising actions.

To inspect whether the root player is likely winning, look at the root's
average value and each root child's mean value after converting it to the root
player's perspective: positive is favorable, negative is unfavorable, and near
zero is mixed or still uncertain.

## Worked Example

Suppose the current position has three legal root actions:

```text
A, B, C
```

When the root node is first created, it stores the current position:

```text
root.children = empty
root.untried_actions = [A, B, C]
```

Simulation 1 starts at the root. The root still has untried actions, so
selection stops immediately at the root. Expansion chooses one untried action,
say `A`, copies the root state, applies `A` to the copy, and stores the result
as child node `A`.

```text
ROOT
└── A
```

The rollout starts from node `A`, plays random actions until terminal, and
backpropagation updates both `A` and `ROOT`.

Simulation 2 starts again at the root. The root still has untried actions, so
selection again stops at the root. Expansion might choose `C` this time:

```text
ROOT
├── A
└── C
```

After rollout and backpropagation, simulation 3 can expand the remaining root
action `B`:

```text
ROOT
├── A
├── B
└── C
```

Now the root is fully expanded. Future simulations do not simply finish all of
`A` before trying `B` or `C`. They use UCT at the root to choose among the root
children. For example:

```text
simulation 4: ROOT uses UCT and selects A
```

When ROOT calculates UCT, it compares child values from ROOT's
player-to-move perspective. If a child stores value from a different player's
perspective, that value is converted before the UCT score is computed.

If node `A` still has untried actions, selection stops at `A`, and expansion
adds one child below `A`, say `A1`:

```text
ROOT
├── A
│   └── A1
├── B
└── C
```

The rollout starts from `A1`, and backpropagation updates:

```text
A1 -> A -> ROOT
```

Later, if UCT selects `A` again and `A` is fully expanded, selection continues
inside `A`. UCT then chooses among `A`'s children:

```text
a later simulation: ROOT uses UCT and selects A
                    A is fully expanded
                    A uses UCT and selects A1
```

So the selection walk is recursive. It starts at the root, but every fully
expanded node on the path uses the same rule: compare its children with UCT,
move to the chosen child, and continue until reaching a terminal node or a node
with an untried action.

Terminal nodes are handled as a special stop case. Suppose a later selection
walk reaches child `B`, and `B` is already terminal:

```text
ROOT -> B
```

Selection stops at `B`. Expansion is skipped because terminal nodes have no
legal actions to expand. Simulation is also unnecessary because the result is
already known. Backpropagation immediately backs up `B`'s terminal result along
the selected path:

```text
B -> ROOT
```

## Python And C++ Differences

The following differences are relevant to understanding performance and
correctness of the four phases above.

The Python MCTS has an optional transposition table. It can reuse a node when
two action orders reach the same canonical game state.

The C++ solver currently does not use a transposition table. Its speed comes
from:

```text
compiled code
TrackedState
incremental claimed component tracking
incremental empty-region tracking
bit-mask metadata
copy once per rollout, then mutate in place
```

We tested real undo for C++ rollouts. It was slower because logging every
changed field cost more than copying one compact `TrackedState` at rollout
start.
