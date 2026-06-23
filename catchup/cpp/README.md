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
```

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
--json           print full JSON records instead of the text summary
```

Supported arena agents are:

```text
mcts:N                                      C++ UCT MCTS with uniform rollout
puct:N:prior=flat:rollout=flat             PUCT with flat prior and uniform rollout
puct:N:prior=flat:rollout=biased           PUCT with flat prior and heuristic-biased rollout
puct:N:prior=heuristic:rollout=flat        PUCT with heuristic prior and uniform rollout
puct:N:prior=heuristic:rollout=biased      PUCT with heuristic prior and heuristic-biased rollout
```

Add `--json` for full game records.
