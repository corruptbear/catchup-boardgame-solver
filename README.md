# Catchup

This repo contains a playable local implementation of the board game Catchup,
plus baseline search players. The browser UI is served by Python. When the C++
MCTS binary is built, the UI uses it for suggestions by default and falls back
to Python only if the binary is missing.

## Layout

```text
catchup/
  board.py          board coordinates and neighbor topology
  components.py     incremental component tracking
  game.py           Catchup rules and game state
  solvers.py        Python random players and MCTS fallback
  training/         self-play data generation for policy/value experiments
  ui_server.py      local browser UI server
  static/           HTML/CSS/JS for the UI
  cpp/              C++ MCTS solver and headless arena
tests/              unit tests
notes/              design notes and implementation explanations
```

## Build The C++ Solver

From the repo root:

```sh
make -C catchup/cpp
```

This creates:

```text
catchup/cpp/build/catchup_mcts
catchup/cpp/build/catchup_arena
catchup/cpp/build/catchup_self_play
```

The Python server looks for that binary automatically. To use a different
binary path:

```sh
CATCHUP_MCTS_BINARY=/path/to/catchup_mcts python3 -m catchup.ui_server
```

## Start The UI Server

From the repo root:

```sh
python3 -m catchup.ui_server --host 127.0.0.1 --port 8000
```

Then open:

```text
http://127.0.0.1:8000/
```

You can choose another port:

```sh
python3 -m catchup.ui_server --port 8768
```

## Run A Headless Arena

Build the C++ solver first:

```sh
make -C catchup/cpp
```

Then compare two C++ search agents with paired colors:

```sh
catchup/cpp/build/catchup_arena --agent-a puct:1000:prior=heuristic:rollout=biased --agent-b mcts:1000 --pairs 20 --seed 1
```

Use `--threads N` to run independent game pairs in parallel:

```sh
catchup/cpp/build/catchup_arena --agent-a puct:1000:prior=heuristic:rollout=biased --agent-b mcts:1000 --pairs 20 --seed 1 --threads 8
```

If `--threads` is omitted, the arena uses the machine's hardware thread count,
capped by the number of pairs.

Arena options:

```text
--pairs N        play N paired matches, meaning 2N total games
--seed N         base random seed
--threads N      worker threads; default is hardware thread count capped by pairs
--max-actions N  abort a game after N internal actions
--json           print full JSON records instead of the text summary
```

Supported arena agents are:

```text
mcts:N                                      C++ UCT MCTS with uniform rollout
puct:N:prior=flat:rollout=flat             PUCT with flat prior and uniform rollout
puct:N:prior=flat:rollout=biased           PUCT with flat prior and heuristic-biased rollout
puct:N:prior=heuristic:rollout=flat        PUCT with heuristic prior and uniform rollout
puct:N:prior=heuristic:rollout=biased      PUCT with heuristic prior and heuristic-biased rollout
neural-puct:N:MODEL.pt2                    neural PUCT using an AOTInductor package
```

Use `--json` to emit the full game records and summary as JSON.

## Generate Bootstrap Training Data

Build the C++ solver first:

```sh
make -C catchup/cpp
```

Then run a tiny C++ self-play smoke generation:

```sh
catchup/cpp/build/catchup_self_play --games 2 --simulations 100 --out data/bootstrap_smoke.jsonl
```

The generator uses PUCT with heuristic priors and biased rollouts by default.
Each JSONL row stores a state snapshot, a 62-action policy target from root
visit counts, and the final value target from that state's player-to-move
perspective. It also stores terminal game metadata: winner, final component
sizes, filled cells, and completed turns.

Use `--threads N` to run independent self-play games in parallel. On a 12-core
machine:

```sh
catchup/cpp/build/catchup_self_play --games 12 --simulations 100 --threads 12 --out data/bootstrap_parallel_smoke.jsonl
```

For real bootstrap data, use a larger budget only after the smoke output looks
right, for example:

```sh
catchup/cpp/build/catchup_self_play --games 50 --simulations 10000 --threads 12 --out data/bootstrap_50g_10k.jsonl
```

Normal data generation omits `--seed`, so each run uses fresh randomness.
Pass `--seed N` only when you intentionally want reproducible debugging output.

During training, load shards with symmetry augmentation in the Python loader:

```python
from catchup.training.data_loader import iter_training_samples

samples = iter_training_samples(
    "data/bootstrap/shard_0001_50g_10k.jsonl",
    augment_symmetry=True,
    symmetry_copies=3,
)
```

Train a small PyTorch policy/value net with `python3.10`:

```sh
python3.10 -m catchup.training.torch_policy_value --data-glob 'data/bootstrap/shard_*_50g_10k.jsonl' --validation-shards 3 --epochs 3 --batch-size 1024 --hidden-size 128 --symmetry-copies 3 --device mps --out data/models/small_policy_value_30shards_3sym.pt --metrics-out data/models/small_policy_value_30shards_3sym_metrics.json
```

Try the small graph policy/value net:

```sh
python3.10 -m catchup.training.torch_policy_value --architecture gnn --gnn-layers 4 --data-glob 'data/bootstrap/shard_*_50g_10k.jsonl' --validation-shards 3 --epochs 3 --batch-size 1024 --hidden-size 128 --symmetry-copies 3 --device mps --out data/models/gnn_policy_value_30shards_3sym_3ep.pt --metrics-out data/models/gnn_policy_value_30shards_3sym_3ep_metrics.json
```

Run the Python neural PUCT prototype from the empty board:

```sh
python3.10 -m catchup.neural_puct --checkpoint data/models/small_policy_value_30shards_3sym.pt --simulations 100 --device mps
```

Export a trained PyTorch model for the C++ neural arena:

```sh
python3.10 -m catchup.training.export_aoti --checkpoint data/models/gnn_policy_value_30shards_3sym_20ep.pt --exported-program data/models/gnn_policy_value_30shards_3sym_20ep_exported.pt2 --package data/models/gnn_policy_value_30shards_3sym_20ep_aoti_mps.pt2 --device mps
```

Run the C++ neural PUCT arena agent:

```sh
catchup/cpp/build/catchup_arena --agent-a neural-puct:100:data/models/gnn_policy_value_30shards_3sym_20ep_aoti_mps.pt2 --agent-b mcts:1000 --pairs 5 --threads 1 --seed 1
```

Validate the neural agents in the existing arena:

```sh
python3.10 -m catchup.arena --agent-a neural-greedy:data/models/small_policy_value_30shards_3sym.pt:device=mps --agent-b random --pairs 20 --seed 1
python3.10 -m catchup.arena --agent-a neural-puct:20:data/models/small_policy_value_30shards_3sym.pt:device=mps --agent-b random --pairs 10 --seed 1
```

## Run Tests

```sh
PYTHONDONTWRITEBYTECODE=1 python3 -m unittest discover -s tests
```
