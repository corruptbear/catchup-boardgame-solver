# Neural Self-Play

This note describes the iterative neural self-play workflow, replay-buffer
sampling, and exploration controls.

## Bootstrap Data

The C++ generator is `catchup/cpp/self_play.cpp`.
For bootstrap and neural self-play data, pass `--early-win false` so terminal
metadata comes from a full board. Interactive UI search can keep early wins
enabled.

Example command:

```sh
catchup/cpp/build/catchup_self_play --games 50 --simulations 10000 --threads 12 --early-win false --out data/bootstrap/shard_0001_50g_10k.jsonl
```

Normal data generation omits `--seed`, so each run uses fresh randomness. Use
`--seed` only for reproducible debugging.

Each JSONL line is one internal Catchup action position. A real Catchup turn can
contain more than one internal action because the player may claim more than one
cell before finishing the turn.

The sample has:

```text
state          position before the chosen internal action
policy_target 62 numbers, one per action
value_target  final result from state's current_player perspective
terminal      final game summary
meta          generator metadata for debugging
```

The 62 actions are:

```text
0..60  claim that board cell
61     finish the current turn
```

`policy_target` is the teacher search visit distribution at that position.

`value_target` is:

```text
+1  the player to move in this saved state eventually won
-1  the player to move in this saved state eventually lost
```

The current bootstrap directory has 30 shards:

```text
data/bootstrap/shard_0001_50g_10k.jsonl
...
data/bootstrap/shard_0030_50g_10k.jsonl
```

That is 1500 self-play games and 79384 saved positions.

## State Fields

The saved `state` contains:

```text
owners               61 entries: -1 empty, 0 Blue, 1 White
current_player       0 Blue or 1 White
selected_this_turn   cells already claimed in the current unfinished turn
claimed_this_turn    number of cells already claimed in this turn
max_claims           most cells this player may claim this turn
turn_start_largest   global largest connected group size at turn start
opening_turn         whether this is the first move special case
legal_mask           62 true/false entries for legal actions
```

The model input is built in `sample_to_arrays()` in
`catchup/training/torch_policy_value.py`.

The current input is 310 numbers:

```text
244 cell-state numbers
 62 legal-action numbers
  4 game-state numbers
---
310 total
```

The 244 cell-state numbers are four 0/1 features for each of the 61 cells:

```text
empty_cell              1.0 if owners[cell] == -1, else 0.0
current_player_cell     1.0 if owners[cell] == current_player, else 0.0
opponent_cell           1.0 if owners[cell] == opponent, else 0.0
selected_this_turn_cell 1.0 if cell is in selected_this_turn, else 0.0
```

Each feature is stored as a floating-point `0.0` or `1.0`.

The 62 legal-action numbers are copied from `legal_mask`.

The 4 game-state numbers are:

```text
claimed_this_turn / 3.0
max_claims / 3.0
turn_start_largest / 61.0
opening_turn as 1.0 or 0.0
```

`current_player` remains in the saved JSON because it defines the relative cell
planes and the value-target perspective. It is not passed as its own scalar
input. The board planes already tell the model which stones belong to the player
to move and which belong to the opponent; adding an absolute Blue/White scalar
encouraged side-specific shortcuts.

The model does not currently receive component sizes, empty-region data,
reachable-region bounds, or explicit group adjacency as separate inputs.

## Symmetry Augmentation

The loader is `catchup/training/data_loader.py`.

It can apply the 12 symmetries of the hex board:

```text
6 rotations
6 reflected rotations
```

Current supervised bootstrap training uses `--symmetry-copies 3`: every raw
sample is kept, and each augmentable raw sample gets three additional randomly
transformed symmetry views for that epoch.

Only turn-boundary samples are augmented, where:

```text
selected_this_turn == []
```

Mid-turn samples are left unchanged because internal actions are canonicalized
by increasing cell index. A rotation or reflection can change that order and
make the transformed mid-turn sample invalid.

## Neural Training Loop

The iterative loop is:

```text
1. load the previous MLX model
2. generate neural-PUCT self-play games
3. write one JSONL shard of training samples
4. replay-train the next PyTorch checkpoint from the saved shard window
5. export that checkpoint to MLX for the next generation
```

The C++ self-play generator is step 2. It uses neural PUCT to choose actions and
writes one training sample for each saved position in each completed game.
Batching is an evaluator detail; see `model_export_and_inference.md`.

Use the exploration controls in `Neural Self-Play Sample Generation` for root
noise and played-action sampling.

## Losses And Metrics

The training loss is:

```text
policy loss + value_weight * value loss
```

Current `value_weight` default is `1.0`.

Policy loss compares the model's action probabilities to the teacher visit
distribution.

Value loss is mean squared error against `value_target`.

When training uses AdamW weight decay, that regularization is applied by the
optimizer during parameter updates. The reported loss still only contains the
policy and value terms above.

Reported metrics:

```text
policy_top1     whether the model's highest-scored action matches the
                highest-visit teacher action

value_accuracy  whether sign(model value) matches sign(value_target)
```

`policy_top1` is easy to read but incomplete. If the teacher assigns similar
visit counts to several actions, predicting the second-best action can be
reasonable even though `policy_top1` counts it as wrong.

## Replay Buffer Coverage

Keep this section operational. For replay training, count raw positions, not
augmented views. Symmetry augmentation changes the view of a sampled position;
it does not create a new independent game position.

Use these controls:

```text
K          replay window size, in self-play generations
N          saved positions in a shard, roughly equal across shards
C          target lifetime coverage per raw position
B          raw positions sampled for one training update after generation
age        generation age, where 0 is newest
gamma      recency decay for age weighting
```

Training budget:

```text
B = C * N
training_batches = ceil(B / batch_size)
```

During warm-up, when fewer than `K` generations exist:

```text
loaded_generations = min(K, available_generations)
B = C * N * (loaded_generations / K)
```

Sampling:

```text
1. choose a generation using generation_probability(age)
2. choose one raw position uniformly inside that generation
3. apply one random legal symmetry view when augmentation is enabled
```

Age weights:

```text
age_weight(age) = gamma ^ age
generation_probability(age) =
    age_weight(age) / sum_{i=0}^{loaded_generations - 1} age_weight(i)
```

Warm-up generation rates for `K = 5`, `gamma = 0.8`:

```text
loop 1:   100.00%
loop 2:    44.44%   55.56%
loop 3:    26.23%   32.79%   40.98%
loop 4:    17.34%   21.68%   27.10%   33.88%
loop 5:    12.18%   15.23%   19.04%   23.80%   29.75%
```

Warm-up generation rates for `K = 10`, `gamma = 0.85`:

```text
loop 1:   100.00%
loop 2:    45.95%   54.05%
loop 3:    28.09%   33.04%   38.87%
loop 4:    19.27%   22.67%   26.67%   31.38%
loop 5:    14.08%   16.56%   19.48%   22.92%   26.96%
loop 6:    10.69%   12.57%   14.79%   17.40%   20.47%   24.08%
loop 7:     8.33%    9.80%   11.52%   13.56%   15.95%   18.77%   22.08%
loop 8:     6.61%    7.78%    9.15%   10.76%   12.66%   14.90%   17.53%   20.62%
loop 9:     5.32%    6.26%    7.36%    8.66%   10.19%   11.99%   14.10%   16.59%   19.52%
loop 10:    4.33%    5.09%    5.99%    7.04%    8.29%    9.75%   11.47%   13.49%   15.88%   18.68%
```

Do not overwrite older checkpoints. Each neural generation should write a new
checkpoint name.

## Neural Self-Play Sample Generation

Generate neural self-play data with:

```sh
catchup/cpp/build/catchup_self_play --teacher neural-puct --model data/models/gnn_policy_value_30shards_3sym_20ep_aoti_mps_b32.pt2 --games 50 --simulations 100 --threads 12 --neural-batch-size 32 --early-win false --out data/neural_self_play/example_50g.jsonl
```

### Exploration

For neural self-play data, there are two exploration controls near each root:
root Dirichlet noise before search, and visit-count temperature after search.

Root Dirichlet noise changes the policy prior used by PUCT at that root:

```text
noisy_prior(action) =
    (1 - effective_epsilon) * model_prior(action)
    + effective_epsilon * dirichlet_noise(action)
```

The effective root noise weight is:

```text
root_noise_epsilon
* (legal_action_count / root_noise_reference_actions) ^ root_noise_action_power
* (empty_cells / 61) ^ root_noise_empty_power
```

The current defaults are:

```text
root_noise_epsilon                  0.25
root_dirichlet_total_concentration  10.0
root_noise_reference_actions        61
root_noise_action_power             0.5
root_noise_empty_power              1.0
```

So the opening gets substantial, spiky exploration noise, while late-game roots
get much less noise.

After PUCT finishes its simulations, the self-play generator samples the played
action from child visit counts:

```text
tau = max(visit_temperature_min, empty_cells / 61)
played_action_probability(action) proportional to visits(action) ^ (1 / tau)
```

With the default `visit_temperature_min = 0.05`, the opening has `tau = 1.0`,
so moves are sampled proportional to visits. As the board fills, `tau` shrinks:
`tau = 0.5` means weights are squared, `tau = 0.25` means weights use the fourth
power, and `tau = 0.05` is very close to choosing the highest-visit action.

The policy target written to the sample is still the raw normalized visit count:

```text
policy_target(action) = visits(action) / sum(visits)
```
