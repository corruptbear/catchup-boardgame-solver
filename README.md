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
  cpp/              C++ MCTS solver used by the UI
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

## Run Tests

```sh
PYTHONDONTWRITEBYTECODE=1 python3 -m unittest discover -s tests
```
