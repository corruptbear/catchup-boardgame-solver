"""Prototype neural PUCT player for Catchup."""

from __future__ import annotations

import argparse
import json
import math
import random
from dataclasses import dataclass, field
from pathlib import Path
from typing import Protocol

from .game import FINISH, GameState


ACTION_COUNT = FINISH + 1


@dataclass(frozen=True)
class PuctEvaluation:
    """Policy priors and value from the evaluated state's player perspective."""

    priors: tuple[float, ...]
    value: float


class StateEvaluator(Protocol):
    def evaluate(self, state: GameState) -> PuctEvaluation:
        """Return priors over actions and value from ``state.current_player``."""


@dataclass(slots=True)
class NeuralPuctEdge:
    action: int
    prior: float
    child: "NeuralPuctNode | None" = None


@dataclass(slots=True)
class NeuralPuctNode:
    state: GameState
    parent: "NeuralPuctNode | None" = None
    action_from_parent: int | None = None
    visits: int = 0
    total_value: float = 0.0
    edges: list[NeuralPuctEdge] = field(default_factory=list)
    children: dict[int, "NeuralPuctNode"] = field(default_factory=dict)

    def mean_value(self) -> float:
        return 0.0 if self.visits == 0 else self.total_value / self.visits

    def is_expanded(self) -> bool:
        return bool(self.edges) or self.state.is_terminal()


@dataclass
class NeuralPuctPlayer:
    """Small Python neural PUCT prototype."""

    evaluator: StateEvaluator
    simulations: int = 100
    exploration_weight: float = 1.4
    rng: random.Random = field(default_factory=random.Random)

    def choose_action(self, state: GameState) -> int:
        root = self.search(state)
        return self._best_root_action(root)

    def search(self, state: GameState) -> NeuralPuctNode:
        if state.is_terminal():
            raise ValueError("cannot search from a terminal state")

        root = NeuralPuctNode(state.copy())
        for _ in range(self.simulations):
            path, leaf_value, leaf_player = self._select_and_evaluate(root)
            self._backpropagate(path, leaf_value, leaf_player)
        return root

    def _select_and_evaluate(
        self,
        root: NeuralPuctNode,
    ) -> tuple[list[NeuralPuctNode], float, int]:
        path = [root]
        node = root
        while True:
            if node.state.is_terminal():
                return path, node.state.result_for(node.state.current_player), node.state.current_player

            if not node.is_expanded():
                evaluation = self.evaluator.evaluate(node.state)
                self._initialize_edges(node, evaluation.priors)
                return path, evaluation.value, node.state.current_player

            edge = self._select_edge(node)
            if edge.child is None:
                child = NeuralPuctNode(
                    node.state.apply_action(edge.action),
                    parent=node,
                    action_from_parent=edge.action,
                )
                edge.child = child
                node.children[edge.action] = child
            node = edge.child
            path.append(node)

    def _initialize_edges(self, node: NeuralPuctNode, priors: tuple[float, ...]) -> None:
        actions = node.state.legal_actions()
        total = sum(priors[action] for action in actions)
        node.edges = [
            NeuralPuctEdge(action=action, prior=priors[action] / total)
            for action in actions
        ]

    def _select_edge(self, node: NeuralPuctNode) -> NeuralPuctEdge:
        parent_sqrt = math.sqrt(max(node.visits, 1))
        best_score = -math.inf
        best_edges: list[NeuralPuctEdge] = []
        for edge in node.edges:
            child = edge.child
            child_visits = 0 if child is None else child.visits
            exploitation = 0.0 if child is None else self._mean_value_for_player(
                child,
                node.state.current_player,
            )
            exploration = (
                self.exploration_weight
                * edge.prior
                * parent_sqrt
                / (1 + child_visits)
            )
            score = exploitation + exploration
            if score > best_score:
                best_score = score
                best_edges = [edge]
            elif score == best_score:
                best_edges.append(edge)
        return self.rng.choice(best_edges)

    @staticmethod
    def _backpropagate(
        path: list[NeuralPuctNode],
        leaf_value: float,
        leaf_player: int,
    ) -> None:
        for node in reversed(path):
            node.visits += 1
            if node.state.current_player == leaf_player:
                node.total_value += leaf_value
            else:
                node.total_value -= leaf_value

    @staticmethod
    def _mean_value_for_player(node: NeuralPuctNode, player: int) -> float:
        value = node.mean_value()
        return value if node.state.current_player == player else -value

    @staticmethod
    def _best_root_action(root: NeuralPuctNode) -> int:
        return int(root_choices(root)[0]["action"])


class TorchPolicyValueEvaluator:
    """Evaluate states with a checkpoint from ``torch_policy_value.py``."""

    def __init__(self, checkpoint: Path, device: str = "auto") -> None:
        import torch

        from .training.torch_policy_value import (
            build_model_from_metadata,
            normalize_model_state_dict,
            sample_to_arrays,
        )

        self.torch = torch
        self.sample_to_arrays = sample_to_arrays
        self.device = self._resolve_device(device)
        payload = torch.load(checkpoint, map_location="cpu")
        metadata = payload["metadata"]
        self.model = build_model_from_metadata(metadata)
        self.model.load_state_dict(normalize_model_state_dict(payload["model_state"]))
        self.model.to(self.device)
        self.model.eval()

    def evaluate(self, state: GameState) -> PuctEvaluation:
        import torch

        sample = {
            "state": state_payload(state),
            "policy_target": [0.0] * ACTION_COUNT,
            "value_target": 0.0,
        }
        features, _, _ = self.sample_to_arrays(sample)
        with torch.no_grad():
            feature_tensor = torch.from_numpy(features).unsqueeze(0).to(self.device)
            logits, value = self.model(feature_tensor)
            legal_actions = state.legal_actions()
            legal_logits = logits[0, list(legal_actions)]
            legal_probs = torch.softmax(legal_logits, dim=0).detach().cpu().tolist()

        priors = [0.0] * ACTION_COUNT
        for action, probability in zip(legal_actions, legal_probs):
            priors[action] = float(probability)
        return PuctEvaluation(tuple(priors), float(value.item()))

    def _resolve_device(self, device: str):
        if device == "auto":
            if self.torch.backends.mps.is_available():
                return self.torch.device("mps")
            return self.torch.device("cpu")
        return self.torch.device(device)


def state_payload(state: GameState) -> dict[str, object]:
    legal_mask = [False] * ACTION_COUNT
    for action in state.legal_actions():
        legal_mask[action] = True
    return {
        "owners": list(state.tracker.cell_owners),
        "current_player": state.current_player,
        "selected_this_turn": list(state.selected),
        "claimed_this_turn": len(state.selected),
        "max_claims": state.max_claims,
        "turn_start_largest": state.turn_start_largest,
        "opening_turn": state.opening_turn,
        "legal_mask": legal_mask,
    }


def root_choices(root: NeuralPuctNode) -> list[dict[str, float | int]]:
    choices = []
    for edge in root.edges:
        child = edge.child
        visits = 0 if child is None else child.visits
        value = 0.0 if child is None else NeuralPuctPlayer._mean_value_for_player(
            child,
            root.state.current_player,
        )
        choices.append({
            "action": edge.action,
            "visits": visits,
            "value": value,
            "prior": edge.prior,
        })
    return sorted(
        choices,
        key=lambda choice: (
            -int(choice["visits"]),
            -float(choice["prior"]),
            int(choice["action"]),
        ),
    )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--simulations", type=int, default=100)
    parser.add_argument("--seed", type=int, default=1)
    parser.add_argument("--device", choices=("auto", "mps", "cpu"), default="auto")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    evaluator = TorchPolicyValueEvaluator(args.checkpoint, args.device)
    player = NeuralPuctPlayer(
        evaluator,
        simulations=args.simulations,
        rng=random.Random(args.seed),
    )
    root = player.search(GameState.new())
    choices = root_choices(root)
    print(json.dumps({
        "action": choices[0]["action"],
        "player": root.state.current_player,
        "simulations": args.simulations,
        "engine": "python-neural-puct",
        "choices": choices,
    }))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
