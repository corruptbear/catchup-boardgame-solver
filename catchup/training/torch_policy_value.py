"""Train a small PyTorch policy/value network from Catchup JSONL shards."""

from __future__ import annotations

import argparse
import glob
import json
import math
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
from ..symmetry import SYMMETRIES
from .data_loader import (
    can_augment_with_symmetry,
    iter_jsonl,
    iter_training_samples,
    transform_sample,
)


CELL_COUNT = FINISH
ACTION_COUNT = FINISH + 1
FEATURE_COUNT = CELL_COUNT * 4 + ACTION_COUNT + 4
CELL_FEATURE_COUNT = 4
GLOBAL_FEATURE_COUNT = 4
LEGAL_MASK_OFFSET = CELL_COUNT * CELL_FEATURE_COUNT
SCALAR_OFFSET = LEGAL_MASK_OFFSET + ACTION_COUNT
MLP_ARCHITECTURE = "mlp"
GNN_ARCHITECTURE = "gnn"
DIRECTIONAL_CNN_ARCHITECTURE = "directional-cnn"
DIRECTIONAL_CNN_TANH_MARGIN_ARCHITECTURE = "directional-cnn-tanh-margin"
DIRECTIONAL_CNN_DUAL_VALUE_ARCHITECTURE = "directional-cnn-dual-value"
WIN_LOSS_VALUE_TARGET = "win-loss"
TANH_MARGIN_VALUE_TARGET = "tanh-margin-scale6"
DUAL_VALUE_TARGET = "win-loss+tanh-margin-scale6"
TANH_MARGIN_SCALE = 6.0
GRID_WIDTH = BOARD.radius * 2 + 1
GRID_CELL_COUNT = GRID_WIDTH * GRID_WIDTH


@dataclass(frozen=True)
class TrainConfig:
    data_glob: str = "data/bootstrap/shard_*_50g_10k.jsonl"
    replay_data_glob: str | None = None
    validation_data_glob: str | None = None
    init_checkpoint: Path | None = None
    validation_shards: int = 3
    epochs: int = 3
    train_batches: int | None = None
    batch_size: int = 1024
    hidden_size: int = 128
    architecture: str = MLP_ARCHITECTURE
    gnn_layers: int = 4
    cnn_layers: int = 4
    learning_rate: float = 0.001
    optimizer: str = "adam"
    weight_decay: float = 0.0
    value_weight: float = 1.0
    margin_value_weight: float = 1.0
    symmetry_copies: int = 3
    seed: int = 1
    device: str = "auto"
    replay_window_generations: int = 5
    replay_gamma: float = 0.8
    target_lifetime_coverage: float = 2.0
    inspect_value_targets: bool = False
    out: Path = Path("data/models/small_policy_value.pt")
    metrics_out: Path = Path("data/models/small_policy_value_metrics.json")


@dataclass(frozen=True)
class ReplayGeneration:
    path: Path
    samples: tuple[dict[str, Any], ...]


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


class HexDirectionalBlock(nn.Module):
    """Direction-specific convolution over the six axial hex-neighbor directions."""

    def __init__(self, hidden_size: int) -> None:
        super().__init__()
        self.self_linear = nn.Conv2d(hidden_size, hidden_size, kernel_size=1)
        self.neighbor_linear = nn.Conv2d(hidden_size * 6, hidden_size, kernel_size=1)
        self.register_buffer("direction_matrices", _direction_matrices())

    def forward(self, hidden: torch.Tensor, valid_mask: torch.Tensor) -> torch.Tensor:
        batch_size, hidden_size, _, _ = hidden.shape
        flat = hidden.reshape(batch_size, hidden_size, GRID_CELL_COUNT)
        directional_neighbors = []
        for direction in range(6):
            directional_neighbors.append(
                torch.matmul(flat, self.direction_matrices[direction])
                .reshape(batch_size, hidden_size, GRID_WIDTH, GRID_WIDTH)
            )
        neighbor_stack = torch.cat(directional_neighbors, dim=1)
        update = F.relu(self.self_linear(hidden) + self.neighbor_linear(neighbor_stack))
        return (hidden + update) * valid_mask


class HexDirectionalCnnPolicyValueNet(nn.Module):
    """Directional CNN over the six axial directions of the radius-4 hex board."""

    def __init__(
        self,
        hidden_size: int = 128,
        cnn_layers: int = 4,
        *,
        dual_value: bool = False,
    ) -> None:
        super().__init__()
        self.dual_value = dual_value
        input_channels = CELL_FEATURE_COUNT + 1 + GLOBAL_FEATURE_COUNT + 1
        self.input_projection = nn.Conv2d(input_channels, hidden_size, kernel_size=1)
        self.blocks = nn.ModuleList(HexDirectionalBlock(hidden_size) for _ in range(cnn_layers))
        self.global_encoder = nn.Linear(GLOBAL_FEATURE_COUNT + 1, hidden_size)
        self.claim_policy_scorer = nn.Conv2d(hidden_size, 1, kernel_size=1)
        self.finish_policy_scorer = nn.Sequential(
            nn.Linear(hidden_size * 2, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, 1),
        )
        if dual_value:
            self.win_value_head = self._make_value_head(hidden_size)
            self.margin_value_head = self._make_value_head(hidden_size)
        else:
            self.value_head = self._make_value_head(hidden_size)
        cell_to_grid = _cell_to_grid_matrix()
        self.register_buffer("cell_to_grid", cell_to_grid)
        self.register_buffer("grid_to_cell", cell_to_grid.transpose(0, 1))
        self.register_buffer("cell_grid_indices", _cell_grid_indices(), persistent=False)
        self.register_buffer("valid_mask", _valid_grid_mask())
        self.reset_parameters()

    @staticmethod
    def _make_value_head(hidden_size: int) -> nn.Sequential:
        return nn.Sequential(
            nn.Linear(hidden_size * 2, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, 1),
        )

    def forward(self, features: torch.Tensor) -> tuple[torch.Tensor, ...]:
        batch_size = features.shape[0]
        cell_flags = features[:, :LEGAL_MASK_OFFSET].reshape(
            batch_size,
            CELL_FEATURE_COUNT,
            CELL_COUNT,
        )
        legal_mask = features[:, LEGAL_MASK_OFFSET:SCALAR_OFFSET]
        legal_claims = legal_mask[:, :CELL_COUNT].unsqueeze(1)
        finish_legal = legal_mask[:, FINISH].unsqueeze(-1)
        scalars = features[:, SCALAR_OFFSET:]
        global_features = torch.cat((scalars, finish_legal), dim=1)

        cell_features = torch.cat((cell_flags, legal_claims), dim=1)
        cell_grid = torch.matmul(cell_features, self.cell_to_grid).reshape(
            batch_size,
            CELL_FEATURE_COUNT + 1,
            GRID_WIDTH,
            GRID_WIDTH,
        )
        global_grid = global_features.unsqueeze(-1).unsqueeze(-1).expand(
            -1,
            -1,
            GRID_WIDTH,
            GRID_WIDTH,
        )
        valid_mask = self.valid_mask.to(dtype=features.dtype)
        hidden = F.relu(self.input_projection(torch.cat((cell_grid, global_grid), dim=1)))
        hidden = hidden * valid_mask
        for block in self.blocks:
            hidden = block(hidden, valid_mask)

        pooled = (hidden * valid_mask).sum(dim=(2, 3)) / CELL_COUNT
        global_hidden = F.relu(self.global_encoder(global_features))
        combined = torch.cat((pooled, global_hidden), dim=1)

        claim_grid = self.claim_policy_scorer(hidden).reshape(batch_size, GRID_CELL_COUNT)
        # AOTInductor on MPS miscompiled this final one-hot matmul for the
        # directional CNN policy head. Direct indexing is the same gather and
        # keeps the exported package aligned with eager PyTorch.
        claim_logits = claim_grid.index_select(1, self.cell_grid_indices)
        finish_logit = self.finish_policy_scorer(combined)
        policy_logits = torch.cat((claim_logits, finish_logit), dim=1)
        if self.dual_value:
            win_value = torch.tanh(self.win_value_head(combined)).squeeze(-1)
            margin_value = torch.tanh(self.margin_value_head(combined)).squeeze(-1)
            return policy_logits, win_value, margin_value
        value = torch.tanh(self.value_head(combined)).squeeze(-1)
        return policy_logits, value

    def reset_parameters(self) -> None:
        for module in self.modules():
            if isinstance(module, (nn.Conv2d, nn.Linear)):
                nn.init.kaiming_normal_(module.weight, nonlinearity="relu")
                if module.bias is not None:
                    nn.init.zeros_(module.bias)

        nn.init.normal_(self.claim_policy_scorer.weight, mean=0.0, std=0.01)
        nn.init.zeros_(self.claim_policy_scorer.bias)
        nn.init.normal_(self.finish_policy_scorer[-1].weight, mean=0.0, std=0.01)
        nn.init.zeros_(self.finish_policy_scorer[-1].bias)
        value_heads = (
            (self.win_value_head, self.margin_value_head)
            if self.dual_value
            else (self.value_head,)
        )
        for value_head in value_heads:
            nn.init.normal_(value_head[-1].weight, mean=0.0, std=0.01)
            nn.init.zeros_(value_head[-1].bias)


def _neighbor_matrix() -> torch.Tensor:
    matrix = torch.zeros((CELL_COUNT, CELL_COUNT), dtype=torch.float32)
    for cell, neighbors in enumerate(BOARD.neighbors):
        weight = 1.0 / len(neighbors)
        for neighbor in neighbors:
            matrix[cell, neighbor] = weight
    return matrix


def _grid_index(q: int, r: int) -> int:
    row = r + BOARD.radius
    column = q + BOARD.radius
    return row * GRID_WIDTH + column


def _cell_to_grid_matrix() -> torch.Tensor:
    matrix = torch.zeros((CELL_COUNT, GRID_CELL_COUNT), dtype=torch.float32)
    for cell, (q, r) in enumerate(BOARD.coords):
        matrix[cell, _grid_index(q, r)] = 1.0
    return matrix


def _cell_grid_indices() -> torch.Tensor:
    return torch.tensor(
        [_grid_index(q, r) for q, r in BOARD.coords],
        dtype=torch.long,
    )


def _valid_grid_mask() -> torch.Tensor:
    mask = torch.zeros((1, 1, GRID_WIDTH, GRID_WIDTH), dtype=torch.float32)
    for q, r in BOARD.coords:
        mask[0, 0, r + BOARD.radius, q + BOARD.radius] = 1.0
    return mask


def _direction_matrices() -> torch.Tensor:
    directions = ((1, 0), (-1, 0), (0, 1), (0, -1), (1, -1), (-1, 1))
    matrices = torch.zeros((6, GRID_CELL_COUNT, GRID_CELL_COUNT), dtype=torch.float32)
    for direction_index, (dq, dr) in enumerate(directions):
        for q, r in BOARD.coords:
            neighbor = BOARD.coord_to_index.get((q + dq, r + dr))
            if neighbor is None:
                continue
            source_q, source_r = BOARD.coords[neighbor]
            matrices[
                direction_index,
                _grid_index(source_q, source_r),
                _grid_index(q, r),
            ] = 1.0
    return matrices


def build_model(
    architecture: str,
    *,
    hidden_size: int,
    gnn_layers: int,
    cnn_layers: int,
) -> nn.Module:
    if architecture == MLP_ARCHITECTURE:
        return PolicyValueNet(hidden_size=hidden_size)
    if architecture == GNN_ARCHITECTURE:
        return GraphPolicyValueNet(hidden_size=hidden_size, gnn_layers=gnn_layers)
    if architecture in (
        DIRECTIONAL_CNN_ARCHITECTURE,
        DIRECTIONAL_CNN_TANH_MARGIN_ARCHITECTURE,
    ):
        return HexDirectionalCnnPolicyValueNet(hidden_size=hidden_size, cnn_layers=cnn_layers)
    if architecture == DIRECTIONAL_CNN_DUAL_VALUE_ARCHITECTURE:
        return HexDirectionalCnnPolicyValueNet(
            hidden_size=hidden_size,
            cnn_layers=cnn_layers,
            dual_value=True,
        )
    raise ValueError(f"unknown architecture: {architecture}")


def value_target_kind_for_architecture(architecture: str) -> str:
    if architecture == DIRECTIONAL_CNN_DUAL_VALUE_ARCHITECTURE:
        return DUAL_VALUE_TARGET
    if architecture == DIRECTIONAL_CNN_TANH_MARGIN_ARCHITECTURE:
        return TANH_MARGIN_VALUE_TARGET
    return WIN_LOSS_VALUE_TARGET


def build_model_from_metadata(metadata: dict[str, Any]) -> nn.Module:
    architecture = str(metadata.get("architecture", MLP_ARCHITECTURE))
    return build_model(
        architecture,
        hidden_size=int(metadata["hidden_size"]),
        gnn_layers=int(metadata.get("gnn_layers", TrainConfig.gnn_layers)),
        cnn_layers=int(metadata.get("cnn_layers", TrainConfig.cnn_layers)),
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


def sample_to_arrays(
    sample: dict[str, Any],
    *,
    value_target_kind: str = WIN_LOSS_VALUE_TARGET,
) -> tuple[np.ndarray, np.ndarray, np.float32 | np.ndarray]:
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
        value_targets_from_sample(sample, value_target_kind),
    )


def value_targets_from_sample(
    sample: dict[str, Any],
    value_target_kind: str,
) -> np.float32 | np.ndarray:
    if value_target_kind == DUAL_VALUE_TARGET:
        return np.asarray(
            [
                value_target_from_sample(sample, WIN_LOSS_VALUE_TARGET),
                value_target_from_sample(sample, TANH_MARGIN_VALUE_TARGET),
            ],
            dtype=np.float32,
        )
    return value_target_from_sample(sample, value_target_kind)


def value_target_from_sample(
    sample: dict[str, Any],
    value_target_kind: str,
) -> np.float32:
    if value_target_kind == WIN_LOSS_VALUE_TARGET:
        return np.float32(sample["value_target"])
    if value_target_kind == TANH_MARGIN_VALUE_TARGET:
        return terminal_tanh_margin_value_target(sample)
    if value_target_kind == DUAL_VALUE_TARGET:
        raise ValueError("use value_targets_from_sample for dual value targets")
    raise ValueError(f"unknown value target kind: {value_target_kind}")


def terminal_tanh_margin_value_target(sample: dict[str, Any]) -> np.float32:
    """Return tanh(raw / 6) from the saved state's current-player perspective."""

    current_player = int(sample["state"]["current_player"])
    terminal = sample["terminal"]
    blue_sizes = terminal["blue_group_sizes"]
    white_sizes = terminal["white_group_sizes"]

    for rank in range(max(len(blue_sizes), len(white_sizes))):
        blue_size = blue_sizes[rank] if rank < len(blue_sizes) else 0
        white_size = white_sizes[rank] if rank < len(white_sizes) else 0
        diff = blue_size - white_size
        if diff:
            if current_player == PLAYER_TWO:
                diff = -diff
            raw = diff * (0.5 ** rank)
            return np.float32(math.tanh(raw / TANH_MARGIN_SCALE))

    raise RuntimeError("terminal Catchup sample has equal component vectors")


def materialize_dataset(
    paths: Iterable[Path],
    *,
    augment_symmetry: bool,
    symmetry_copies: int,
    rng: random.Random,
    value_target_kind: str,
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
        feature, policy, value = sample_to_arrays(
            sample,
            value_target_kind=value_target_kind,
        )
        features.append(feature)
        policies.append(policy)
        values.append(value)

    return TensorDataset(
        torch.from_numpy(np.stack(features)),
        torch.from_numpy(np.stack(policies)),
        torch.from_numpy(np.asarray(values, dtype=np.float32)),
    )


def load_replay_generations(paths: Iterable[Path]) -> list[ReplayGeneration]:
    generations: list[ReplayGeneration] = []
    for path in paths:
        generations.append(ReplayGeneration(path, tuple(iter_jsonl(path))))
    return generations


def replay_age_weights(generation_count: int, gamma: float) -> list[float]:
    """Return oldest-to-newest truncated geometric weights."""

    return [
        gamma ** (generation_count - 1 - generation_index)
        for generation_index in range(generation_count)
    ]


def replay_train_batches(config: TrainConfig, generations: list[ReplayGeneration]) -> int:
    if config.train_batches is not None:
        return config.train_batches

    newest_size = len(generations[-1].samples)
    warmup_fraction = min(1.0, len(generations) / config.replay_window_generations)
    sampled_positions = (
        config.target_lifetime_coverage
        * newest_size
        * warmup_fraction
    )
    return max(1, math.ceil(sampled_positions / config.batch_size))


def sample_replay_batch(
    generations: list[ReplayGeneration],
    weights: list[float],
    *,
    batch_size: int,
    rng: random.Random,
    augment_symmetry: bool,
    value_target_kind: str,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, list[int]]:
    features: list[np.ndarray] = []
    policies: list[np.ndarray] = []
    values: list[np.float32] = []
    generation_counts = [0] * len(generations)

    generation_indexes = range(len(generations))
    for _ in range(batch_size):
        generation_index = rng.choices(generation_indexes, weights=weights, k=1)[0]
        generation = generations[generation_index]
        sample = rng.choice(generation.samples)
        if augment_symmetry and can_augment_with_symmetry(sample):
            sample = transform_sample(sample, rng.choice(SYMMETRIES))
        feature, policy, value = sample_to_arrays(
            sample,
            value_target_kind=value_target_kind,
        )
        features.append(feature)
        policies.append(policy)
        values.append(value)
        generation_counts[generation_index] += 1

    return (
        torch.from_numpy(np.stack(features)),
        torch.from_numpy(np.stack(policies)),
        torch.from_numpy(np.asarray(values, dtype=np.float32)),
        generation_counts,
    )


def policy_value_loss(
    policy_logits: torch.Tensor,
    value_outputs: tuple[torch.Tensor, ...],
    policy_target: torch.Tensor,
    value_target: torch.Tensor,
    value_weight: float,
    margin_value_weight: float,
) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
    policy_loss = -(policy_target * F.log_softmax(policy_logits, dim=1)).sum(dim=1).mean()
    if len(value_outputs) == 1:
        target = value_target
        if target.ndim == 2:
            target = target[:, 0]
        value_loss = F.mse_loss(value_outputs[0], target)
        return (
            policy_loss + value_weight * value_loss,
            {
                "policy_loss": policy_loss,
                "value_loss": value_loss,
            },
        )
    if len(value_outputs) == 2:
        if value_target.ndim != 2 or value_target.shape[1] != 2:
            raise ValueError("dual value model requires two value targets")
        win_value_loss = F.mse_loss(value_outputs[0], value_target[:, 0])
        margin_value_loss = F.mse_loss(value_outputs[1], value_target[:, 1])
        value_loss = win_value_loss + margin_value_loss
        return (
            policy_loss
            + value_weight * win_value_loss
            + margin_value_weight * margin_value_loss,
            {
                "policy_loss": policy_loss,
                "value_loss": value_loss,
                "win_value_loss": win_value_loss,
                "margin_value_loss": margin_value_loss,
            },
        )
    raise ValueError("model must return one or two value outputs")


def split_model_outputs(outputs: tuple[torch.Tensor, ...]) -> tuple[torch.Tensor, tuple[torch.Tensor, ...]]:
    if len(outputs) < 2:
        raise ValueError("model must return policy logits and at least one value")
    return outputs[0], tuple(outputs[1:])


def first_value_target(value_target: torch.Tensor) -> torch.Tensor:
    return value_target if value_target.ndim == 1 else value_target[:, 0]


def make_optimizer(config: TrainConfig, model: nn.Module) -> torch.optim.Optimizer:
    if config.optimizer == "adam":
        return torch.optim.Adam(model.parameters(), lr=config.learning_rate)
    if config.optimizer == "adamw":
        return torch.optim.AdamW(
            model.parameters(),
            lr=config.learning_rate,
            weight_decay=config.weight_decay,
        )
    raise ValueError(f"unknown optimizer: {config.optimizer}")


@torch.no_grad()
def evaluate(
    model: nn.Module,
    dataset: TensorDataset,
    *,
    batch_size: int,
    device: torch.device,
    value_weight: float,
    margin_value_weight: float,
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
        policy_logits, value_outputs = split_model_outputs(model(features))
        loss, losses = policy_value_loss(
            policy_logits,
            value_outputs,
            policy_target,
            value_target,
            value_weight,
            margin_value_weight,
        )
        batch_size_actual = features.shape[0]
        target_action = policy_target.argmax(dim=1)
        predicted_action = policy_logits.argmax(dim=1)
        value_prediction = torch.sign(value_outputs[0])
        value_target_sign = torch.sign(first_value_target(value_target))
        count += batch_size_actual
        totals["loss"] += float(loss.item()) * batch_size_actual
        for name, metric in losses.items():
            totals[name] = totals.get(name, 0.0) + float(metric.item()) * batch_size_actual
        totals["policy_top1"] += float((predicted_action == target_action).float().mean().item()) * batch_size_actual
        totals["value_accuracy"] += (
            float((value_prediction == value_target_sign).float().mean().item())
            * batch_size_actual
        )
    return {name: total / count for name, total in totals.items()}


def train(config: TrainConfig) -> dict[str, Any]:
    if config.replay_data_glob is not None:
        return train_replay(config)

    set_seeds(config.seed)
    paths = [Path(path) for path in sorted(glob.glob(config.data_glob))]
    if len(paths) <= config.validation_shards:
        raise ValueError("not enough shards for requested validation split")

    train_paths = paths[:-config.validation_shards]
    validation_paths = paths[-config.validation_shards:]
    device = resolve_device(config.device)
    augmentation_rng = random.Random(config.seed)
    value_target_kind = value_target_kind_for_architecture(config.architecture)

    print(json.dumps({
        "device": str(device),
        "train_shards": len(train_paths),
        "validation_shards": len(validation_paths),
        "value_target_kind": value_target_kind,
    }, sort_keys=True), flush=True)

    validation_dataset = materialize_dataset(
        validation_paths,
        augment_symmetry=False,
        symmetry_copies=1,
        rng=augmentation_rng,
        value_target_kind=value_target_kind,
    )
    model = build_model(
        config.architecture,
        hidden_size=config.hidden_size,
        gnn_layers=config.gnn_layers,
        cnn_layers=config.cnn_layers,
    ).to(device)
    load_initial_checkpoint(model, config.init_checkpoint)
    optimizer = make_optimizer(config, model)
    history: list[dict[str, Any]] = []

    for epoch in range(1, config.epochs + 1):
        started = time.time()
        train_dataset = materialize_dataset(
            train_paths,
            augment_symmetry=True,
            symmetry_copies=config.symmetry_copies,
            rng=augmentation_rng,
            value_target_kind=value_target_kind,
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
            policy_logits, value_outputs = split_model_outputs(model(features))
            loss, _ = policy_value_loss(
                policy_logits,
                value_outputs,
                policy_target,
                value_target,
                config.value_weight,
                config.margin_value_weight,
            )
            loss.backward()
            optimizer.step()

        train_metrics = evaluate(
            model,
            train_dataset,
            batch_size=config.batch_size,
            device=device,
            value_weight=config.value_weight,
            margin_value_weight=config.margin_value_weight,
        )
        validation_metrics = evaluate(
            model,
            validation_dataset,
            batch_size=config.batch_size,
            device=device,
            value_weight=config.value_weight,
            margin_value_weight=config.margin_value_weight,
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


def train_replay(config: TrainConfig) -> dict[str, Any]:
    set_seeds(config.seed)
    if (
        config.init_checkpoint is not None
        and config.out.resolve() == config.init_checkpoint.resolve()
    ):
        raise ValueError("replay training output must not overwrite the init checkpoint")

    all_paths = [Path(path) for path in sorted(glob.glob(config.replay_data_glob or ""))]
    if not all_paths:
        raise ValueError("replay data glob matched no files")
    replay_paths = all_paths[-config.replay_window_generations:]
    generations = load_replay_generations(replay_paths)
    weights = replay_age_weights(len(generations), config.replay_gamma)
    train_batches = replay_train_batches(config, generations)
    device = resolve_device(config.device)
    rng = random.Random(config.seed)
    value_target_kind = value_target_kind_for_architecture(config.architecture)

    validation_paths = (
        [Path(path) for path in sorted(glob.glob(config.validation_data_glob))]
        if config.validation_data_glob is not None
        else []
    )
    validation_dataset = (
        materialize_dataset(
            validation_paths,
            augment_symmetry=False,
            symmetry_copies=1,
            rng=rng,
            value_target_kind=value_target_kind,
        )
        if validation_paths
        else None
    )

    model = build_model(
        config.architecture,
        hidden_size=config.hidden_size,
        gnn_layers=config.gnn_layers,
        cnn_layers=config.cnn_layers,
    ).to(device)
    load_initial_checkpoint(model, config.init_checkpoint)
    optimizer = make_optimizer(config, model)

    generation_counts = [0] * len(generations)
    totals: dict[str, float] = {"loss": 0.0}
    sampled_positions = train_batches * config.batch_size
    started = time.time()

    print(json.dumps({
        "device": str(device),
        "mode": "replay",
        "replay_generations": len(generations),
        "replay_paths": [str(generation.path) for generation in generations],
        "replay_generation_sizes": [len(generation.samples) for generation in generations],
        "replay_age_weights": weights,
        "train_batches": train_batches,
        "sampled_positions": sampled_positions,
        "value_target_kind": value_target_kind,
    }, sort_keys=True), flush=True)

    model.train()
    for _ in range(train_batches):
        features, policy_target, value_target, batch_counts = sample_replay_batch(
            generations,
            weights,
            batch_size=config.batch_size,
            rng=rng,
            augment_symmetry=True,
            value_target_kind=value_target_kind,
        )
        features = features.to(device)
        policy_target = policy_target.to(device)
        value_target = value_target.to(device)
        optimizer.zero_grad(set_to_none=True)
        policy_logits, value_outputs = split_model_outputs(model(features))
        loss, losses = policy_value_loss(
            policy_logits,
            value_outputs,
            policy_target,
            value_target,
            config.value_weight,
            config.margin_value_weight,
        )
        loss.backward()
        optimizer.step()

        totals["loss"] += float(loss.item()) * config.batch_size
        for name, metric in losses.items():
            totals[name] = totals.get(name, 0.0) + float(metric.item()) * config.batch_size
        for index, count in enumerate(batch_counts):
            generation_counts[index] += count

    record: dict[str, Any] = {
        "seconds": round(time.time() - started, 3),
        "train_batches": train_batches,
        "sampled_positions": sampled_positions,
        "generation_sample_counts": generation_counts,
        "generation_sample_rates": [
            count / sampled_positions
            for count in generation_counts
        ],
        "train": {
            name: total / sampled_positions
            for name, total in totals.items()
        },
    }
    if validation_dataset is not None:
        record["validation_samples"] = len(validation_dataset)
        record["validation"] = evaluate(
            model,
            validation_dataset,
            batch_size=config.batch_size,
            device=device,
            value_weight=config.value_weight,
            margin_value_weight=config.margin_value_weight,
        )
    print(json.dumps(record, sort_keys=True), flush=True)

    history = [record]
    save_checkpoint(
        model,
        config,
        replay_paths,
        validation_paths,
        history,
        device,
    )
    return {
        "config": config,
        "train_paths": replay_paths,
        "validation_paths": validation_paths,
        "history": history,
        "device": device,
    }


def target_distribution_paths(config: TrainConfig) -> list[Path]:
    glob_pattern = config.replay_data_glob if config.replay_data_glob is not None else config.data_glob
    paths = [Path(path) for path in sorted(glob.glob(glob_pattern or ""))]
    if config.replay_data_glob is not None:
        return paths[-config.replay_window_generations:]
    return paths


def inspect_value_target_distribution(config: TrainConfig) -> dict[str, Any]:
    paths = target_distribution_paths(config)
    value_target_kind = value_target_kind_for_architecture(config.architecture)
    values: list[np.ndarray] = []
    for sample in iter_training_samples(
        paths,
        augment_symmetry=False,
        symmetry_copies=1,
        rng=random.Random(config.seed),
    ):
        values.append(np.asarray(value_targets_from_sample(sample, value_target_kind)).reshape(-1))

    target_names = (
        ["win_loss", "tanh_margin_scale6"]
        if value_target_kind == DUAL_VALUE_TARGET
        else [value_target_kind]
    )
    array = (
        np.stack(values).astype(np.float32)
        if values
        else np.empty((0, len(target_names)), dtype=np.float32)
    )
    quantile_points = (0.0, 0.01, 0.05, 0.25, 0.5, 0.75, 0.95, 0.99, 1.0)

    def summarize_target(column: np.ndarray) -> dict[str, Any]:
        return {
            "count": int(column.size),
            "negative": int((column < 0).sum()),
            "zero": int((column == 0).sum()),
            "positive": int((column > 0).sum()),
            "min": float(column.min()) if column.size else None,
            "max": float(column.max()) if column.size else None,
            "mean": float(column.mean()) if column.size else None,
            "std": float(column.std()) if column.size else None,
            "mean_abs": float(np.abs(column).mean()) if column.size else None,
            "quantiles": {
                f"{point:.2f}": float(np.quantile(column, point))
                for point in quantile_points
            } if column.size else {},
            "abs_quantiles": {
                f"{point:.2f}": float(np.quantile(np.abs(column), point))
                for point in quantile_points
            } if column.size else {},
        }

    return {
        "paths": [str(path) for path in paths],
        "value_target_kind": value_target_kind,
        "targets": {
            name: summarize_target(array[:, index])
            for index, name in enumerate(target_names)
        },
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


def load_initial_checkpoint(model: nn.Module, checkpoint: Path | None) -> None:
    if checkpoint is None:
        return
    payload = torch.load(checkpoint, map_location="cpu")
    model.load_state_dict(normalize_model_state_dict(payload["model_state"]))


def save_checkpoint(
    model: nn.Module,
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
        "value_target_kind": value_target_kind_for_architecture(config.architecture),
        "gnn_layers": config.gnn_layers,
        "cnn_layers": config.cnn_layers,
        "action_count": ACTION_COUNT,
        "train_paths": [str(path) for path in train_paths],
        "validation_paths": [str(path) for path in validation_paths],
        "init_checkpoint": str(config.init_checkpoint) if config.init_checkpoint is not None else None,
        "symmetry_copies": config.symmetry_copies,
        "epochs": config.epochs,
        "train_batches": config.train_batches,
        "batch_size": config.batch_size,
        "learning_rate": config.learning_rate,
        "optimizer": config.optimizer,
        "weight_decay": config.weight_decay,
        "value_weight": config.value_weight,
        "margin_value_weight": config.margin_value_weight,
        "seed": config.seed,
        "device": str(device),
        "replay_data_glob": config.replay_data_glob,
        "replay_window_generations": config.replay_window_generations,
        "replay_gamma": config.replay_gamma,
        "target_lifetime_coverage": config.target_lifetime_coverage,
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
    parser.add_argument("--replay-data-glob", default=TrainConfig.replay_data_glob)
    parser.add_argument("--validation-data-glob", default=TrainConfig.validation_data_glob)
    parser.add_argument("--init-checkpoint", type=Path, default=TrainConfig.init_checkpoint)
    parser.add_argument("--validation-shards", type=int, default=TrainConfig.validation_shards)
    parser.add_argument("--epochs", type=int, default=TrainConfig.epochs)
    parser.add_argument("--train-batches", type=int, default=TrainConfig.train_batches)
    parser.add_argument("--batch-size", type=int, default=TrainConfig.batch_size)
    parser.add_argument("--hidden-size", type=int, default=TrainConfig.hidden_size)
    parser.add_argument(
        "--architecture",
        choices=(
            MLP_ARCHITECTURE,
            GNN_ARCHITECTURE,
            DIRECTIONAL_CNN_ARCHITECTURE,
            DIRECTIONAL_CNN_TANH_MARGIN_ARCHITECTURE,
            DIRECTIONAL_CNN_DUAL_VALUE_ARCHITECTURE,
        ),
        default=TrainConfig.architecture,
    )
    parser.add_argument("--gnn-layers", type=int, default=TrainConfig.gnn_layers)
    parser.add_argument("--cnn-layers", type=int, default=TrainConfig.cnn_layers)
    parser.add_argument("--learning-rate", type=float, default=TrainConfig.learning_rate)
    parser.add_argument("--optimizer", choices=("adam", "adamw"), default=TrainConfig.optimizer)
    parser.add_argument("--weight-decay", type=float, default=TrainConfig.weight_decay)
    parser.add_argument("--value-weight", type=float, default=TrainConfig.value_weight)
    parser.add_argument(
        "--margin-value-weight",
        type=float,
        default=TrainConfig.margin_value_weight,
    )
    parser.add_argument("--symmetry-copies", type=int, default=TrainConfig.symmetry_copies)
    parser.add_argument("--seed", type=int, default=TrainConfig.seed)
    parser.add_argument("--device", choices=("auto", "mps", "cpu"), default=TrainConfig.device)
    parser.add_argument("--replay-window-generations", type=int, default=TrainConfig.replay_window_generations)
    parser.add_argument("--replay-gamma", type=float, default=TrainConfig.replay_gamma)
    parser.add_argument(
        "--target-lifetime-coverage",
        type=float,
        default=TrainConfig.target_lifetime_coverage,
    )
    parser.add_argument(
        "--inspect-value-targets",
        action="store_true",
        default=TrainConfig.inspect_value_targets,
    )
    parser.add_argument("--out", type=Path, default=TrainConfig.out)
    parser.add_argument("--metrics-out", type=Path, default=TrainConfig.metrics_out)
    args = parser.parse_args(argv)
    return TrainConfig(
        data_glob=args.data_glob,
        replay_data_glob=args.replay_data_glob,
        validation_data_glob=args.validation_data_glob,
        init_checkpoint=args.init_checkpoint,
        validation_shards=args.validation_shards,
        epochs=args.epochs,
        train_batches=args.train_batches,
        batch_size=args.batch_size,
        hidden_size=args.hidden_size,
        architecture=args.architecture,
        gnn_layers=args.gnn_layers,
        cnn_layers=args.cnn_layers,
        learning_rate=args.learning_rate,
        optimizer=args.optimizer,
        weight_decay=args.weight_decay,
        value_weight=args.value_weight,
        margin_value_weight=args.margin_value_weight,
        symmetry_copies=args.symmetry_copies,
        seed=args.seed,
        device=args.device,
        replay_window_generations=args.replay_window_generations,
        replay_gamma=args.replay_gamma,
        target_lifetime_coverage=args.target_lifetime_coverage,
        inspect_value_targets=args.inspect_value_targets,
        out=args.out,
        metrics_out=args.metrics_out,
    )


def main(argv: list[str] | None = None) -> int:
    config = parse_args(argv)
    if config.inspect_value_targets:
        print(json.dumps(inspect_value_target_distribution(config), indent=2, sort_keys=True))
        return 0
    train(config)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
