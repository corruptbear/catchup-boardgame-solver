"""Export a trained Catchup policy/value checkpoint as an AOTInductor package."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import torch

from .torch_policy_value import (
    FEATURE_COUNT,
    build_model_from_metadata,
    normalize_model_state_dict,
)


def export_aoti(
    checkpoint: Path,
    *,
    exported_program: Path,
    package: Path,
    device: str,
) -> None:
    payload = torch.load(checkpoint, map_location="cpu")
    model = build_model_from_metadata(payload["metadata"])
    model.load_state_dict(normalize_model_state_dict(payload["model_state"]))
    model.eval()

    torch_device = torch.device(device)
    model.to(torch_device)
    example = torch.zeros((1, FEATURE_COUNT), dtype=torch.float32, device=torch_device)

    with torch.no_grad():
        exported = torch.export.export(model, (example,), strict=False)
        torch.export.save(exported, exported_program)
        try:
            torch._inductor.aoti_compile_and_package(exported, package_path=package)
        except AssertionError:
            if not package.exists():
                raise
            print(
                "warning: AOTInductor raised AssertionError after writing package",
                file=sys.stderr,
            )

    if not package.exists():
        raise RuntimeError(f"AOTInductor did not write package: {package}")

    print({
        "checkpoint": str(checkpoint),
        "exported_program": str(exported_program),
        "package": str(package),
        "device": device,
    })


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--exported-program", type=Path, required=True)
    parser.add_argument("--package", type=Path, required=True)
    parser.add_argument("--device", choices=("cpu", "mps"), default="mps")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    export_aoti(
        args.checkpoint,
        exported_program=args.exported_program,
        package=args.package,
        device=args.device,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
