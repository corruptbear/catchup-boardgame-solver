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
```

Use `--json` to emit the full game records and summary as JSON.

## Run Tests

```sh
PYTHONDONTWRITEBYTECODE=1 python3 -m unittest discover -s tests
```
