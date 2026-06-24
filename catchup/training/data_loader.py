"""JSONL training-data loading and symmetry augmentation."""

from __future__ import annotations

import json
import random
from collections.abc import Iterable, Iterator, Sequence
from pathlib import Path
from typing import Any

from ..symmetry import (
    SYMMETRIES,
    Symmetry,
    transform_action_values,
    transform_cell_values,
    transform_cells,
)


PathLike = str | Path
Sample = dict[str, Any]


def iter_jsonl(paths: PathLike | Iterable[PathLike]) -> Iterator[Sample]:
    """Yield raw JSONL samples from one path or several paths."""

    for path in _normalize_paths(paths):
        with Path(path).open(encoding="utf-8") as handle:
            for line in handle:
                yield json.loads(line)


def can_augment_with_symmetry(sample: Sample) -> bool:
    """Return whether symmetry augmentation is clean for this sample."""

    return not sample["state"]["selected_this_turn"]


def transform_sample(
    sample: Sample,
    symmetry: Symmetry,
    *,
    require_turn_boundary: bool = True,
) -> Sample:
    """Return one symmetry-transformed copy of a training sample.

    By default this only accepts turn-boundary samples. Mid-turn samples are
    awkward because internal Catchup actions are canonicalized by increasing
    cell index, and a geometric symmetry can change that ordering.
    """

    if require_turn_boundary and not can_augment_with_symmetry(sample):
        raise ValueError("symmetry augmentation is only safe at turn boundaries")

    transformed = dict(sample)
    state = dict(sample["state"])
    state["owners"] = transform_cell_values(state["owners"], symmetry)
    state["legal_mask"] = transform_action_values(state["legal_mask"], symmetry)
    state["selected_this_turn"] = transform_cells(state["selected_this_turn"], symmetry)
    transformed["state"] = state
    transformed["policy_target"] = transform_action_values(
        sample["policy_target"],
        symmetry,
    )
    return transformed


def iter_training_samples(
    paths: PathLike | Iterable[PathLike],
    *,
    augment_symmetry: bool = False,
    symmetry_copies: int = 1,
    rng: random.Random | None = None,
) -> Iterator[Sample]:
    """Yield raw samples, plus optional random symmetry views."""

    random_source = rng if rng is not None else random
    for sample in iter_jsonl(paths):
        yield sample
        if augment_symmetry and can_augment_with_symmetry(sample):
            for _ in range(symmetry_copies):
                yield transform_sample(sample, random_source.choice(SYMMETRIES))


def all_symmetry_variants(sample: Sample) -> Iterator[Sample]:
    """Yield the 12 transformed variants of one turn-boundary sample."""

    for symmetry in SYMMETRIES:
        yield transform_sample(sample, symmetry)


def _normalize_paths(paths: PathLike | Iterable[PathLike]) -> Sequence[PathLike]:
    if isinstance(paths, (str, Path)):
        return (paths,)
    return tuple(paths)
