# PUCT MCTS

This note records the separate PUCT search path added beside the current random
rollout MCTS. It is intentionally separate from `basic_mcts.md` for now. Later
we can merge the shared parts and keep only the differences.

PUCT uses the same high-level tree loop:

```text
selection -> expansion -> evaluation -> backpropagation
```

The important difference is that each action has a prior probability:

```text
P(action)
```

In AlphaZero-style search, this prior comes from a neural network. This project
can currently run PUCT with either a flat prior or a hand-written heuristic
prior:

```text
prior=flat       -> P(action) = 1 / number_of_legal_actions
prior=heuristic  -> P(action) = normalized heuristic score for each legal action
```

The heuristic is still much simpler than a trained policy network, but it gives
the tree a first guess about which actions deserve earlier expansion and more
exploration pressure.

## Node And Edge Data

The basic random MCTS stores children as:

```text
action -> child node
```

PUCT stores two related lists:

```text
action_edges -> all legal action edges, each with action, child pointer, prior
children     -> materialized child edges only
```

An `action_edges` child pointer may be empty. That means the action is visible
to PUCT selection, but the resulting child position has not been materialized
yet. `children` keeps the older public meaning: only child nodes that actually
exist.

Each materialized node stores:

```text
state
parent
action_from_parent
children
visits
total_value
```

## Heuristic Prior

`HeuristicActionPrior(state)` scores each legal action from the current
position, then normalizes those scores into probabilities. For a claim action,
the current heuristic checks only local neighbors:

```text
adjacent own components      -> bonus for growing or connecting own groups
adjacent opponent components -> small contact bonus, larger near bigger groups
isolated claim               -> small bonus
global-largest record break  -> penalty unless the move connects own groups
```

The opponent-component bonus is not a real blocking heuristic. It does not yet
test whether the move cuts an opponent reachable region, prevents opponent group
merges, or lowers the opponent's reachable-region upper bound.

`FINISH` gets a higher prior only when there are no legal claims or no claim
that avoids increasing the global largest group.

The arena exposes prior and rollout as separate PUCT knobs:

```text
puct:N:prior=flat:rollout=flat
puct:N:prior=flat:rollout=biased
puct:N:prior=heuristic:rollout=flat
puct:N:prior=heuristic:rollout=biased
```

## Search Loop

The value perspective rule is the same as basic MCTS: a node's raw
`total_value / visits` is from that node's player-to-move perspective. When a
parent compares children, the child value is converted to the parent's
player-to-move perspective.

### 1. Selection

Selection starts at the root. At each node, including the root, apply these
cases in order:

```text
terminal node          -> selection ends here
non-terminal node      -> use PUCT to choose one legal action edge
edge has no child yet  -> selection ends after materializing that child
edge already has child -> continue selection from that child
```

So selection can compare every legal action edge as soon as the node has been
initialized, even when many of those edges have no child node yet.

At a non-terminal node, PUCT chooses an action edge by:

```text
score = Q + exploration

Q           = child mean value from the current node player's view, or 0 if unvisited
exploration = C * P(action) * sqrt(parent visits) / (1 + edge visits)
```

Current `C` is `1.4`.

Compared with UCT:

```text
UCT:
exploration = C * sqrt(log(parent visits) / child visits)

PUCT:
exploration = C * P(action) * sqrt(parent visits) / (1 + child visits)
```

So PUCT explicitly uses the action prior. A high-prior edge gets more
exploration pressure; a low-prior edge gets less. If an edge has no child yet,
its visit count is 0 and its `Q` is 0.

With flat priors, PUCT is similar in spirit to UCT, but not identical. It uses
`sqrt(parent visits)` instead of `sqrt(log(parent visits))`, and the prior term
scales exploration.

### 2. Expansion

When a node is initialized, PUCT creates one edge for every legal action and
stores each action's prior. It does not copy/apply every action immediately.

Expansion still materializes at most one new child position per simulation. If
selection chooses an edge with no child yet, expansion:

```text
1. Copy the parent state.
2. Apply the selected edge's action to the copy.
3. Create the child node from that new state.
4. Initialize that child node's own legal action edges and priors.
5. Start rollout/evaluation from the new child.
```

If selection ended at a terminal node, expansion is skipped and there is no
rollout. The terminal result is backed up directly.

The difference from the previous PUCT implementation is:

```text
Before:
selection stops at node S because S has untried actions
expansion chooses one untried action
create child for that action
rollout from child
backup

Now:
node S already has all legal action edges
selection at S scores those edges with PUCT
if selected edge has no child:
    create child for that edge
    rollout from child
    backup
```

Implication for search behavior:

```text
Before:
first visits at a node were forced by prior rank
high-prior action, then next-highest-prior action, then next-highest...

Now:
first visits are chosen by the full PUCT score
a high-prior action can receive multiple visits before a lower-prior action
if its Q plus exploration remains better
```

So the change is small structurally, but it matters most at low-visit nodes.
The search is less like a prior-sorted expansion queue and more like normal
PUCT edge selection from the first revisit of a node.

Example:

```text
P(A) = 0.70
P(B) = 0.20
P(C) = 0.10

edges at node S:
A -> child=null, prior=0.70
B -> child=null, prior=0.20
C -> child=null, prior=0.10
```

The first selection from `S` may choose `A`, because its prior gives it the
largest exploration term. After expansion and rollout:

```text
A -> child exists, visits=1, Q=rollout result
B -> child=null, visits=0, Q=0
C -> child=null, visits=0, Q=0
```

The next selection from `S` compares all three edges again. It may choose `A`
again if `A`'s value plus exploration is best, or it may choose an unvisited
edge if the prior/exploration term makes that edge best.

If priors are flat:

```text
P(A) = P(B) = P(C)
```

then zero-visit edges tie until rollout values and visit counts start to
separate them.

### 3. Evaluation

The current C++ PUCT implementation still uses rollout for leaf value. The
rollout policy is configurable:

```text
rollout=flat    -> choose uniformly from legal actions
rollout=biased  -> sample from HeuristicActionPrior(state) weights
```

For heuristic-biased rollout:

```text
copy the leaf state
while not terminal:
    score legal actions with HeuristicActionPrior(state)
    sample one action randomly from those weights
    apply action
back up the terminal result
```

This keeps rollouts stochastic, but avoids treating all legal actions as equally
plausible during the rollout. Plain `mcts:N` remains the uniform-rollout
baseline. The PUCT code also keeps a separate `flat_random_rollout()` function
beside `biased_random_rollout()` so we can switch or compare them without
mixing the two policies.

Later PUCT evaluators can replace rollout:

```text
HeuristicEvaluator:
    priors(state)
    value(state)

NeuralEvaluator:
    neural_net(state) -> priors, value
```

AlphaZero-style neural PUCT usually does not use random rollout. At a new leaf,
the network supplies both:

```text
P(action) for legal actions
V(state) as the leaf value
```

Then `V(state)` is backed up through the selected path.

### 4. Backpropagation

Backpropagation is currently the same as basic MCTS:

```text
for each node on the path, from leaf to root:
    visits += 1
    total_value += terminal_result_for(node.player_to_move)
```

If a future evaluator returns a non-terminal value instead of a terminal
rollout result, the backup rule stays the same in shape, but backs up that leaf
value from the correct player perspective.

## Current Status

The current PUCT path is available through:

```bash
catchup/cpp/build/catchup_mcts --engine puct --puct-prior heuristic --puct-rollout biased ...
```

The default engine remains:

```text
random
```

The current PUCT implementation has:

```text
configurable flat or heuristic priors
all legal action edges visible at node initialization
PUCT edge selection
lazy child-state materialization
configurable flat or heuristic-biased random rollout leaf evaluation
```

The next meaningful strength improvements are to tune the heuristic prior,
add a heuristic leaf value, or replace both with a neural policy/value model.
