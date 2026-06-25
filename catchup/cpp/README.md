# Catchup C++ Solver

This directory contains the C++ MCTS solver used by the Python UI when the
binary is available, plus the C++ headless arena for strength checks. It
implements the same Catchup game state rules as the Python engine, but runs
rollouts much faster.

## Build

From the repo root:

```sh
make -C catchup/cpp
```

Or from this directory:

```sh
make
```

The binary is written to:

```text
catchup/cpp/build/catchup_mcts
catchup/cpp/build/catchup_arena
catchup/cpp/build/catchup_self_play
```

`catchup_arena` links against the PyTorch C++ libraries for AOTInductor neural
PUCT, and against MLX for native Apple Silicon neural PUCT.

The Python bridge in `catchup/cpp_solver.py` looks for that binary by default.
You can override the path with:

```sh
CATCHUP_MCTS_BINARY=/path/to/catchup_mcts
```

## Clean

```sh
make -C catchup/cpp clean
```

## Inputs And Output

The binary is usually called by Python, not by hand. It expects the full game
state through command-line flags such as `--owners`, `--selected`,
`--current-player`, and `--simulations`.

It writes one JSON object to stdout:

```json
{
  "action": 12,
  "player": 0,
  "simulations": 1000,
  "state_mode": "tracked",
  "choices": [
    {"action": 12, "visits": 300, "value": 0.42}
  ]
}
```

In `choices`, `value` is the child mean value converted to the root player's
perspective.

## Arena

The C++ arena runs whole games in one process, avoiding the Python arena's
per-move solver subprocess overhead.

From the repo root:

```sh
catchup/cpp/build/catchup_arena --agent-a puct:1000:prior=heuristic:rollout=biased --agent-b mcts:1000 --pairs 20 --seed 1
```

Use `--threads N` to run independent game pairs in parallel. If omitted, the
arena uses the machine's hardware thread count, capped by the number of pairs.
The game records stay deterministic because each pair has fixed seeds and fixed
output slots.

Arena options:

```text
--pairs N        play N paired matches, meaning 2N total games
--seed N         base random seed
--threads N      worker threads; default is hardware thread count capped by pairs
--max-actions N  abort a game after N internal actions
--agent-a-action-selection max|sample
--agent-b-action-selection max|sample
                 max chooses the highest-visit root child; sample samples by visits
                 default is sample for both agents in neural-puct vs neural-puct,
                 max for both agents otherwise
--neural-backend aoti|mlx
                 neural evaluator backend; default aoti
--neural-device cpu|mps   device for neural AOTI inference; default mps
--neural-batch-size N     fixed neural eval batch size; default 32
--neural-batch-wait-ms N  max wait to gather a neural batch; default 0.5
--json           print full JSON records instead of the text summary
```

Supported arena agents are:

```text
mcts:N                                      C++ UCT MCTS with uniform rollout
puct:N:prior=flat:rollout=flat             PUCT with flat prior and uniform rollout
puct:N:prior=flat:rollout=biased           PUCT with flat prior and heuristic-biased rollout
puct:N:prior=heuristic:rollout=flat        PUCT with heuristic prior and uniform rollout
puct:N:prior=heuristic:rollout=biased      PUCT with heuristic prior and heuristic-biased rollout
neural-puct:N:MODEL                        neural PUCT using AOTI .pt2 or MLX .safetensors
```

When either arena agent is `neural-puct`, the arena shares a batched evaluator
across game worker threads. With `--neural-backend aoti`, the model package
must be exported for the same batch size passed with `--neural-batch-size`.
With `--neural-backend mlx`, use an MLX safetensors weight file converted from
the PyTorch checkpoint.

Add `--json` for full game records.

## Self-Play Data Generation

The C++ self-play generator is the preferred path for bootstrap policy/value
data because it runs whole games in one process and calls PUCT directly.
Each JSONL row includes the position, policy target, value target, and terminal
game summary for the completed self-play game.

Tiny smoke run:

```sh
catchup/cpp/build/catchup_self_play --games 2 --simulations 100 --out data/bootstrap_smoke.jsonl
```

Parallel run on a 12-core machine:

```sh
catchup/cpp/build/catchup_self_play --games 50 --simulations 10000 --threads 12 --out data/bootstrap_50g_10k.jsonl
```

Options:

```text
--games N                 number of self-play games
--simulations N           PUCT simulations per internal action
--threads N               worker threads; default is hardware thread count capped by games
--out PATH                JSONL output path
--teacher MODE            puct or neural-puct; default puct
--model PATH              AOTI package or MLX safetensors file for neural-puct teacher
--neural-backend aoti|mlx neural evaluator backend; default aoti
--neural-device cpu|mps   device for neural AOTI inference; default mps
--neural-batch-size N     fixed neural eval batch size; default 32
--neural-batch-wait-ms N  max wait to gather a batch; default 0.5
--profile-neural-batch 1  print neural batch timing stats to stderr
--root-noise-epsilon N    opening neural root noise weight; default 0.25
--root-dirichlet-total-concentration N
                          total root Dirichlet concentration; default 10.0
--root-noise-reference-actions N
                          legal-action count where epsilon is unchanged; default 61
--root-noise-action-power N
                          epsilon scales by (legal/reference)^power; default 0.5
--root-noise-empty-power N
                          epsilon also scales by (empty_cells/61)^power; default 1.0
--visit-temperature-min N
                          action sampling uses tau=max(N, empty_cells/61); default 0.05
--puct-prior MODE         flat or heuristic; default heuristic
--puct-rollout M          flat or biased; default biased
--max-actions N           abort a game after N internal actions
--seed N                  optional reproducibility seed; omit for normal data generation
```

For neural self-play, export a fixed-batch AOTInductor package and use the same
batch size in the generator. Neural self-play mixes Dirichlet noise into the
root priors of each search by default. The per-action Dirichlet alpha is
`total_concentration / legal_action_count`. The effective noise weight is:

```text
epsilon
* (legal_action_count / reference_actions) ^ action_power
* (empty_cells / 61) ^ empty_power
```

```sh
python3.10 -m catchup.training.export_aoti --checkpoint data/models/gnn_policy_value_30shards_3sym_20ep.pt --exported-program data/models/gnn_policy_value_30shards_3sym_20ep_exported_b32.pt2 --package data/models/gnn_policy_value_30shards_3sym_20ep_aoti_mps_b32.pt2 --device mps --batch-size 32
catchup/cpp/build/catchup_self_play --teacher neural-puct --model data/models/gnn_policy_value_30shards_3sym_20ep_aoti_mps_b32.pt2 --games 50 --simulations 100 --threads 12 --neural-batch-size 32 --out data/neural_self_play_smoke.jsonl
```

For MLX, convert the PyTorch checkpoint to safetensors and use
`--neural-backend mlx`:

```sh
python3.10 -m catchup.training.export_mlx_weights --checkpoint data/models/directional_cnn_h64_noplayer_iter_0008_npuct100cont_replay.pt --out data/models/directional_cnn_h64_noplayer_iter_0008_npuct100cont_replay_mlx.safetensors
catchup/cpp/build/catchup_self_play --teacher neural-puct --neural-backend mlx --model data/models/directional_cnn_h64_noplayer_iter_0008_npuct100cont_replay_mlx.safetensors --games 50 --simulations 100 --threads 50 --neural-batch-size 128 --out data/neural_self_play_mlx_smoke.jsonl
```
