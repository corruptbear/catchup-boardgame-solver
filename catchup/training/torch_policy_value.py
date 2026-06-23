"""Train a small PyTorch policy/value network from Catchup JSONL shards."""

from __future__ import annotations

import argparse
import glob
import json
import random
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, TensorDataset

from ..board import BOARD
from ..game import FINISH, PLAYER_ONE, PLAYER_TWO
from .data_loader import iter_training_samples


CELL_COUNT = FINISH
ACTION_COUNT = FINISH + 1
FEATURE_COUNT = CELL_COUNT * 4 + ACTION_COUNT + 5
CELL_FEATURE_COUNT = 4
GLOBAL_FEATURE_COUNT = 5
LEGAL_MASK_OFFSET = CELL_COUNT * CELL_FEATURE_COUNT
SCALAR_OFFSET = LEGAL_MASK_OFFSET + ACTION_COUNT
MLP_ARCHITECTURE = "mlp"
GNN_ARCHITECTURE = "gnn"


@dataclass(frozen=True)
class TrainConfig:
    data_glob: str = "data/bootstrap/shard_*_50g_10k.jsonl"
    validation_shards: int = 3
    epochs: int = 3
    batch_size: int = 1024
    hidden_size: int = 128
    architecture: str = MLP_ARCHITECTURE
    gnn_layers: int = 4
    learning_rate: float = 0.001
    value_weight: float = 1.0
    symmetry_copies: int = 3
    seed: int = 1
    device: str = "auto"
    out: Path = Path("data/models/small_policy_value.pt")
    metrics_out: Path = Path("data/models/small_policy_value_metrics.json")


class PolicyValueNet(nn.Module):
    def __init__(self, input_size: int = FEATURE_COUNT, hidden_size: int = 128) -> None:
        super().__init__()
        self.trunk = nn.Sequential(
            nn.Linear(input_size, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, hidden_size),
            nn.ReLU(),
        )
        self.policy_head = nn.Linear(hidden_size, ACTION_COUNT)
        self.value_head = nn.Linear(hidden_size, 1)

    def forward(self, features: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        hidden = self.trunk(features)
        policy_logits = self.policy_head(hidden)
        value = torch.tanh(self.value_head(hidden)).squeeze(-1)
        return policy_logits, value


class GraphMessageLayer(nn.Module):
    def __init__(self, hidden_size: int) -> None:
        super().__init__()
        self.self_linear = nn.Linear(hidden_size, hidden_size)
        self.neighbor_linear = nn.Linear(hidden_size, hidden_size)
        self.norm = nn.LayerNorm(hidden_size)

    def forward(self, hidden: torch.Tensor, neighbor_mean: torch.Tensor) -> torch.Tensor:
        update = F.relu(self.self_linear(hidden) + self.neighbor_linear(neighbor_mean))
        return self.norm(hidden + update)


class GraphPolicyValueNet(nn.Module):
    """Small graph network over the fixed 61-cell hex board."""

    def __init__(self, hidden_size: int = 128, gnn_layers: int = 4) -> None:
        super().__init__()
        cell_input_size = CELL_FEATURE_COUNT + 1 + GLOBAL_FEATURE_COUNT + 1
        self.cell_encoder = nn.Linear(cell_input_size, hidden_size)
        self.message_layers = nn.ModuleList(
            GraphMessageLayer(hidden_size) for _ in range(gnn_layers)
        )
        self.global_encoder = nn.Linear(GLOBAL_FEATURE_COUNT + 1, hidden_size)
        self.claim_policy_scorer = nn.Linear(hidden_size, 1)
        self.finish_policy_scorer = nn.Sequential(
            nn.Linear(hidden_size * 2, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, 1),
        )
        self.value_head = nn.Sequential(
            nn.Linear(hidden_size * 2, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, 1),
        )
        self.register_buffer("neighbor_matrix", _neighbor_matrix())

    def forward(self, features: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        batch_size = features.shape[0]
        cell_flags = features[:, :LEGAL_MASK_OFFSET].reshape(
            batch_size,
            CELL_FEATURE_COUNT,
            CELL_COUNT,
        ).transpose(1, 2)
        legal_mask = features[:, LEGAL_MASK_OFFSET:SCALAR_OFFSET]
        legal_claims = legal_mask[:, :CELL_COUNT].unsqueeze(-1)
        finish_legal = legal_mask[:, FINISH].unsqueeze(-1)
        scalars = features[:, SCALAR_OFFSET:]
        global_features = torch.cat((scalars, finish_legal), dim=1)
        broadcast_global = global_features.unsqueeze(1).expand(-1, CELL_COUNT, -1)

        hidden = F.relu(self.cell_encoder(torch.cat(
            (cell_flags, legal_claims, broadcast_global),
            dim=2,
        )))
        for layer in self.message_layers:
            neighbor_mean = torch.matmul(self.neighbor_matrix, hidden)
            hidden = layer(hidden, neighbor_mean)

        pooled = hidden.mean(dim=1)
        global_hidden = F.relu(self.global_encoder(global_features))
        combined = torch.cat((pooled, global_hidden), dim=1)

        claim_logits = self.claim_policy_scorer(hidden).squeeze(-1)
        finish_logit = self.finish_policy_scorer(combined)
        policy_logits = torch.cat((claim_logits, finish_logit), dim=1)
        value = torch.tanh(self.value_head(combined)).squeeze(-1)
        return policy_logits, value


def _neighbor_matrix() -> torch.Tensor:
    matrix = torch.zeros((CELL_COUNT, CELL_COUNT), dtype=torch.float32)
    for cell, neighbors in enumerate(BOARD.neighbors):
        weight = 1.0 / len(neighbors)
        for neighbor in neighbors:
            matrix[cell, neighbor] = weight
    return matrix


def build_model(
    architecture: str,
    *,
    hidden_size: int,
    gnn_layers: int,
) -> nn.Module:
    if architecture == MLP_ARCHITECTURE:
        return PolicyValueNet(hidden_size=hidden_size)
    if architecture == GNN_ARCHITECTURE:
        return GraphPolicyValueNet(hidden_size=hidden_size, gnn_layers=gnn_layers)
    raise ValueError(f"unknown architecture: {architecture}")


def build_model_from_metadata(metadata: dict[str, Any]) -> nn.Module:
    architecture = str(metadata.get("architecture", MLP_ARCHITECTURE))
    return build_model(
        architecture,
        hidden_size=int(metadata["hidden_size"]),
        gnn_layers=int(metadata.get("gnn_layers", TrainConfig.gnn_layers)),
    )


def normalize_model_state_dict(state_dict: dict[str, Any]) -> dict[str, Any]:
    renamed_prefixes = (
        ("claim_policy_head.", "claim_policy_scorer."),
        ("finish_policy_head.", "finish_policy_scorer."),
    )
    normalized: dict[str, Any] = {}
    for key, value in state_dict.items():
        normalized_key = key
        for old_prefix, new_prefix in renamed_prefixes:
            if key.startswith(old_prefix):
                normalized_key = new_prefix + key[len(old_prefix):]
                break
        normalized[normalized_key] = value
    return normalized


def sample_to_arrays(sample: dict[str, Any]) -> tuple[np.ndarray, np.ndarray, np.float32]:
    """Convert one JSONL sample into feature, policy target, and value target."""

    state = sample["state"]
    owners = np.asarray(state["owners"], dtype=np.int8)
    current_player = int(state["current_player"])
    opponent = PLAYER_TWO if current_player == PLAYER_ONE else PLAYER_ONE

    selected = np.zeros(CELL_COUNT, dtype=np.float32)
    for cell in state["selected_this_turn"]:
        selected[cell] = 1.0

    features = np.empty(FEATURE_COUNT, dtype=np.float32)
    cursor = 0
    for plane in (
        owners == -1,
        owners == current_player,
        owners == opponent,
    ):
        features[cursor:cursor + CELL_COUNT] = plane.astype(np.float32)
        cursor += CELL_COUNT
    features[cursor:cursor + CELL_COUNT] = selected
    cursor += CELL_COUNT
    features[cursor:cursor + ACTION_COUNT] = np.asarray(state["legal_mask"], dtype=np.float32)
    cursor += ACTION_COUNT
    features[cursor:] = np.asarray(
        [
            current_player,
            state["claimed_this_turn"] / 3.0,
            state["max_claims"] / 3.0,
            state["turn_start_largest"] / CELL_COUNT,
            1.0 if state["opening_turn"] else 0.0,
        ],
        dtype=np.float32,
    )
    return (
        features,
        np.asarray(sample["policy_target"], dtype=np.float32),
        np.float32(sample["value_target"]),
    )


def materialize_dataset(
    paths: Iterable[Path],
    *,
    augment_symmetry: bool,
    symmetry_copies: int,
    rng: random.Random,
) -> TensorDataset:
    features: list[np.ndarray] = []
    policies: list[np.ndarray] = []
    values: list[np.float32] = []
    for sample in iter_training_samples(
        paths,
        augment_symmetry=augment_symmetry,
        symmetry_copies=symmetry_copies,
        rng=rng,
    ):
        feature, policy, value = sample_to_arrays(sample)
        features.append(feature)
        policies.append(policy)
        values.append(value)

    return TensorDataset(
        torch.from_numpy(np.stack(features)),
        torch.from_numpy(np.stack(policies)),
        torch.from_numpy(np.asarray(values, dtype=np.float32)),
    )


def policy_value_loss(
    policy_logits: torch.Tensor,
    value: torch.Tensor,
    policy_target: torch.Tensor,
    value_target: torch.Tensor,
    value_weight: float,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    policy_loss = -(policy_target * F.log_softmax(policy_logits, dim=1)).sum(dim=1).mean()
    value_loss = F.mse_loss(value, value_target)
    return policy_loss + value_weight * value_loss, policy_loss, value_loss


@torch.no_grad()
def evaluate(
    model: nn.Module,
    dataset: TensorDataset,
    *,
    batch_size: int,
    device: torch.device,
    value_weight: float,
) -> dict[str, float]:
    model.eval()
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=False)
    totals = {
        "loss": 0.0,
        "policy_loss": 0.0,
        "value_loss": 0.0,
        "policy_top1": 0.0,
        "value_accuracy": 0.0,
    }
    count = 0
    for features, policy_target, value_target in loader:
        features = features.to(device)
        policy_target = policy_target.to(device)
        value_target = value_target.to(device)
        policy_logits, value = model(features)
        loss, policy_loss, value_loss = policy_value_loss(
            policy_logits,
            value,
            policy_target,
            value_target,
            value_weight,
        )
        batch_size_actual = features.shape[0]
        target_action = policy_target.argmax(dim=1)
        predicted_action = policy_logits.argmax(dim=1)
        value_prediction = torch.sign(value)
        count += batch_size_actual
        totals["loss"] += float(loss.item()) * batch_size_actual
        totals["policy_loss"] += float(policy_loss.item()) * batch_size_actual
        totals["value_loss"] += float(value_loss.item()) * batch_size_actual
        totals["policy_top1"] += float((predicted_action == target_action).float().mean().item()) * batch_size_actual
        totals["value_accuracy"] += float((value_prediction == value_target).float().mean().item()) * batch_size_actual
    return {name: total / count for name, total in totals.items()}


def train(config: TrainConfig) -> dict[str, Any]:
    set_seeds(config.seed)
    paths = [Path(path) for path in sorted(glob.glob(config.data_glob))]
    if len(paths) <= config.validation_shards:
        raise ValueError("not enough shards for requested validation split")

    train_paths = paths[:-config.validation_shards]
    validation_paths = paths[-config.validation_shards:]
    device = resolve_device(config.device)
    augmentation_rng = random.Random(config.seed)

    print(json.dumps({
        "device": str(device),
        "train_shards": len(train_paths),
        "validation_shards": len(validation_paths),
    }, sort_keys=True), flush=True)

    validation_dataset = materialize_dataset(
        validation_paths,
        augment_symmetry=False,
        symmetry_copies=1,
        rng=augmentation_rng,
    )
    model = build_model(
        config.architecture,
        hidden_size=config.hidden_size,
        gnn_layers=config.gnn_layers,
    ).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=config.learning_rate)
    history: list[dict[str, Any]] = []

    for epoch in range(1, config.epochs + 1):
        started = time.time()
        train_dataset = materialize_dataset(
            train_paths,
            augment_symmetry=True,
            symmetry_copies=config.symmetry_copies,
            rng=augmentation_rng,
        )
        loader = DataLoader(
            train_dataset,
            batch_size=config.batch_size,
            shuffle=True,
        )
        model.train()
        for features, policy_target, value_target in loader:
            features = features.to(device)
            policy_target = policy_target.to(device)
            value_target = value_target.to(device)
            optimizer.zero_grad(set_to_none=True)
            policy_logits, value = model(features)
            loss, _, _ = policy_value_loss(
                policy_logits,
                value,
                policy_target,
                value_target,
                config.value_weight,
            )
            loss.backward()
            optimizer.step()

        train_metrics = evaluate(
            model,
            train_dataset,
            batch_size=config.batch_size,
            device=device,
            value_weight=config.value_weight,
        )
        validation_metrics = evaluate(
            model,
            validation_dataset,
            batch_size=config.batch_size,
            device=device,
            value_weight=config.value_weight,
        )
        record = {
            "epoch": epoch,
            "seconds": round(time.time() - started, 3),
            "train_samples": len(train_dataset),
            "validation_samples": len(validation_dataset),
            "train": train_metrics,
            "validation": validation_metrics,
        }
        history.append(record)
        print(json.dumps(record, sort_keys=True), flush=True)

    save_checkpoint(model, config, train_paths, validation_paths, history, device)
    return {
        "config": config,
        "train_paths": train_paths,
        "validation_paths": validation_paths,
        "history": history,
        "device": device,
    }


def resolve_device(device: str) -> torch.device:
    if device == "auto":
        if torch.backends.mps.is_available():
            return torch.device("mps")
        return torch.device("cpu")
    return torch.device(device)


def set_seeds(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)


def save_checkpoint(
    model: PolicyValueNet,
    config: TrainConfig,
    train_paths: list[Path],
    validation_paths: list[Path],
    history: list[dict[str, Any]],
    device: torch.device,
) -> None:
    config.out.parent.mkdir(parents=True, exist_ok=True)
    config.metrics_out.parent.mkdir(parents=True, exist_ok=True)
    metadata = {
        "input_size": FEATURE_COUNT,
        "hidden_size": config.hidden_size,
        "architecture": config.architecture,
        "gnn_layers": config.gnn_layers,
        "action_count": ACTION_COUNT,
        "train_paths": [str(path) for path in train_paths],
        "validation_paths": [str(path) for path in validation_paths],
        "symmetry_copies": config.symmetry_copies,
        "epochs": config.epochs,
        "batch_size": config.batch_size,
        "learning_rate": config.learning_rate,
        "value_weight": config.value_weight,
        "seed": config.seed,
        "device": str(device),
    }
    torch.save(
        {
            "model_state": model.state_dict(),
            "metadata": metadata,
            "history": history,
        },
        config.out,
    )
    config.metrics_out.write_text(
        json.dumps({"metadata": metadata, "history": history}, indent=2, sort_keys=True),
        encoding="utf-8",
    )


def parse_args(argv: list[str] | None = None) -> TrainConfig:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-glob", default=TrainConfig.data_glob)
    parser.add_argument("--validation-shards", type=int, default=TrainConfig.validation_shards)
    parser.add_argument("--epochs", type=int, default=TrainConfig.epochs)
    parser.add_argument("--batch-size", type=int, default=TrainConfig.batch_size)
    parser.add_argument("--hidden-size", type=int, default=TrainConfig.hidden_size)
    parser.add_argument(
        "--architecture",
        choices=(MLP_ARCHITECTURE, GNN_ARCHITECTURE),
        default=TrainConfig.architecture,
    )
    parser.add_argument("--gnn-layers", type=int, default=TrainConfig.gnn_layers)
    parser.add_argument("--learning-rate", type=float, default=TrainConfig.learning_rate)
    parser.add_argument("--value-weight", type=float, default=TrainConfig.value_weight)
    parser.add_argument("--symmetry-copies", type=int, default=TrainConfig.symmetry_copies)
    parser.add_argument("--seed", type=int, default=TrainConfig.seed)
    parser.add_argument("--device", choices=("auto", "mps", "cpu"), default=TrainConfig.device)
    parser.add_argument("--out", type=Path, default=TrainConfig.out)
    parser.add_argument("--metrics-out", type=Path, default=TrainConfig.metrics_out)
    args = parser.parse_args(argv)
    return TrainConfig(
        data_glob=args.data_glob,
        validation_shards=args.validation_shards,
        epochs=args.epochs,
        batch_size=args.batch_size,
        hidden_size=args.hidden_size,
        architecture=args.architecture,
        gnn_layers=args.gnn_layers,
        learning_rate=args.learning_rate,
        value_weight=args.value_weight,
        symmetry_copies=args.symmetry_copies,
        seed=args.seed,
        device=args.device,
        out=args.out,
        metrics_out=args.metrics_out,
    )


def main(argv: list[str] | None = None) -> int:
    train(parse_args(argv))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
