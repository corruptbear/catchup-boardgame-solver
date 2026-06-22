# Catchup C++ Solver

This directory contains the C++ MCTS solver used by the Python UI when the
binary is available. It implements the same Catchup game state rules as the
Python engine, but runs rollouts much faster.

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
