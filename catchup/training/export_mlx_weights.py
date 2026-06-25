"""Convert a Catchup PyTorch policy/value checkpoint to MLX safetensors."""

from __future__ import annotations

import argparse
from pathlib import Path

import mlx.core as mx
import torch


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", required=True, type=Path)
    parser.add_argument("--out", required=True, type=Path)
    args = parser.parse_args()

    checkpoint = torch.load(args.checkpoint, map_location="cpu")
    state = checkpoint["model_state"]
    arrays = {
        name: mx.array(tensor.detach().cpu().numpy())
        for name, tensor in state.items()
    }
    metadata = {
        key: str(value)
        for key, value in checkpoint.get("metadata", {}).items()
        if isinstance(value, (str, int, float, bool))
    }
    args.out.parent.mkdir(parents=True, exist_ok=True)
    mx.save_safetensors(str(args.out), arrays, metadata)
    print(args.out)


if __name__ == "__main__":
    main()
