"""Small local web UI for manually playing Catchup."""

from __future__ import annotations

import argparse
import json
import mimetypes
from http import HTTPStatus
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from threading import Lock
from typing import Any

from .board import BOARD
from .components import PLAYER_ONE, PLAYER_TWO
from .game import FINISH, GameState


STATIC_DIR = Path(__file__).with_name("static")
PLAYER_NAMES = {
    PLAYER_ONE: "Blue",
    PLAYER_TWO: "White",
}


def state_payload(state: GameState, message: str = "") -> dict[str, Any]:
    """Return a JSON-serializable snapshot for the browser UI."""

    legal_actions = ui_legal_actions(state)
    winner = None
    if state.is_terminal():
        winner_id = state.winner()
        winner = PLAYER_NAMES[winner_id] if winner_id is not None else "Tie"

    return {
        "board": {
            "cell_count": state.board.cell_count,
            "cells": [
                {
                    "index": index,
                    "q": q,
                    "r": r,
                    "owner": state.tracker.owner(index),
                }
                for index, (q, r) in enumerate(state.board.coords)
            ],
        },
        "current_player": state.current_player,
        "current_player_name": PLAYER_NAMES[state.current_player],
        "completed_turns": state.completed_turns,
        "empty_count": state.tracker.empty_count(),
        "finish_action": FINISH,
        "legal_actions": legal_actions,
        "max_claims": state.max_claims,
        "message": message,
        "opening_turn": state.opening_turn,
        "players": [
            {
                "id": PLAYER_ONE,
                "name": PLAYER_NAMES[PLAYER_ONE],
                "group_sizes": state.group_sizes(PLAYER_ONE),
                "largest_group": state.tracker.largest_group_size(PLAYER_ONE),
            },
            {
                "id": PLAYER_TWO,
                "name": PLAYER_NAMES[PLAYER_TWO],
                "group_sizes": state.group_sizes(PLAYER_TWO),
                "largest_group": state.tracker.largest_group_size(PLAYER_TWO),
            },
        ],
        "selected": state.selected,
        "terminal": state.is_terminal(),
        "turn_start_largest": state.turn_start_largest,
        "winner": winner,
    }


def ui_legal_actions(state: GameState) -> tuple[int, ...]:
    """Return legal browser clicks, hiding canonical-order internals.

    The engine canonicalizes multi-cell turns by requiring increasing cell
    indices. For human play, order should not matter, so the UI accepts any
    empty cell during a partial turn and the session later replays the turn in
    sorted order.
    """

    engine_actions = state.legal_actions()
    if not state.selected or len(state.selected) >= state.max_claims:
        return engine_actions

    actions = [FINISH]
    actions.extend(
        cell
        for cell in range(state.board.cell_count)
        if state.tracker.is_empty(cell)
    )
    return tuple(actions)


class GameSession:
    """Server-side state for one local self-play board."""

    def __init__(self) -> None:
        self._lock = Lock()
        self._state = GameState.new()
        self._history: list[GameState] = []
        self._message = "New game."

    def payload(self) -> dict[str, Any]:
        with self._lock:
            return state_payload(self._state, self._message)

    def apply_action(self, action: int) -> dict[str, Any]:
        with self._lock:
            before_player = self._state.current_player
            self._history.append(self._state.copy())
            try:
                if action in self._state.legal_actions():
                    self._state = self._state.apply_action(action)
                elif self._can_apply_out_of_order_cell(action):
                    self._state = self._apply_out_of_order_cell(action)
                else:
                    raise ValueError(f"illegal action: {action}")
            except Exception:
                self._history.pop()
                raise

            if action == FINISH:
                self._message = f"{PLAYER_NAMES[before_player]} finished the turn."
            elif self._state.current_player != before_player:
                self._message = f"{PLAYER_NAMES[before_player]} claimed cell {action} and finished the turn."
            else:
                self._message = f"{PLAYER_NAMES[before_player]} claimed cell {action}."
            return state_payload(self._state, self._message)

    def reset(self) -> dict[str, Any]:
        with self._lock:
            self._state = GameState.new()
            self._history = []
            self._message = "New game."
            return state_payload(self._state, self._message)

    def undo(self) -> dict[str, Any]:
        with self._lock:
            if self._history:
                self._state = self._history.pop()
                self._message = "Undid the last action."
            else:
                self._message = "Nothing to undo."
            return state_payload(self._state, self._message)

    def _can_apply_out_of_order_cell(self, action: int) -> bool:
        if action == FINISH or not self._state.selected:
            return False
        if len(self._state.selected) >= self._state.max_claims:
            return False
        try:
            return self._state.tracker.is_empty(action)
        except ValueError:
            return False

    def _apply_out_of_order_cell(self, action: int) -> GameState:
        base = self._current_turn_base()
        cells = tuple(sorted((*self._state.selected, action)))
        state = base.copy()
        for cell in cells:
            state = state.apply_action(cell)
        return state

    def _current_turn_base(self) -> GameState:
        for candidate in reversed(self._history):
            if (
                not candidate.selected
                and candidate.current_player == self._state.current_player
                and candidate.completed_turns == self._state.completed_turns
            ):
                return candidate
        raise ValueError("could not find the start of the current turn")


class CatchupRequestHandler(BaseHTTPRequestHandler):
    session = GameSession()

    def do_GET(self) -> None:
        if self.path in ("/", "/index.html"):
            self._send_file(STATIC_DIR / "index.html")
        elif self.path == "/app.css":
            self._send_file(STATIC_DIR / "app.css")
        elif self.path == "/app.js":
            self._send_file(STATIC_DIR / "app.js")
        elif self.path == "/api/state":
            self._send_json(self.session.payload())
        else:
            self.send_error(HTTPStatus.NOT_FOUND)

    def do_POST(self) -> None:
        try:
            if self.path == "/api/action":
                data = self._read_json()
                self._send_json(self.session.apply_action(int(data["action"])))
            elif self.path == "/api/reset":
                self._send_json(self.session.reset())
            elif self.path == "/api/undo":
                self._send_json(self.session.undo())
            else:
                self.send_error(HTTPStatus.NOT_FOUND)
        except (KeyError, TypeError, ValueError) as exc:
            self._send_json({"error": str(exc), "state": self.session.payload()}, HTTPStatus.BAD_REQUEST)

    def log_message(self, format: str, *args: Any) -> None:
        return

    def _read_json(self) -> dict[str, Any]:
        content_length = int(self.headers.get("Content-Length", "0"))
        body = self.rfile.read(content_length)
        if not body:
            return {}
        return json.loads(body.decode("utf-8"))

    def _send_json(self, payload: dict[str, Any], status: HTTPStatus = HTTPStatus.OK) -> None:
        body = json.dumps(payload).encode("utf-8")
        self.send_response(status)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def _send_file(self, path: Path) -> None:
        if not path.is_file() or path.parent != STATIC_DIR:
            self.send_error(HTTPStatus.NOT_FOUND)
            return
        body = path.read_bytes()
        content_type = mimetypes.guess_type(path.name)[0] or "application/octet-stream"
        self.send_response(HTTPStatus.OK)
        self.send_header("Content-Type", content_type)
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)


def run_server(host: str, port: int) -> None:
    CatchupRequestHandler.session = GameSession()
    server = ThreadingHTTPServer((host, port), CatchupRequestHandler)
    print(f"Catchup UI running at http://{host}:{port}/")
    server.serve_forever()


def main() -> None:
    parser = argparse.ArgumentParser(description="Run the local Catchup self-play UI.")
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=8000)
    args = parser.parse_args()
    run_server(args.host, args.port)


if __name__ == "__main__":
    main()
