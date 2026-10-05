from __future__ import annotations

import argparse
import json
from typing import Any

from pfm.core.execution import ExecutionContext, execution_context_from_env
from pfm.core.serialization import to_jsonable


def print_json(payload: Any) -> None:
    print(json.dumps(to_jsonable(payload), indent=2, sort_keys=True))


def add_runtime_arguments(
    parser: argparse.ArgumentParser, *, device: bool = True, checkpoint_output: bool = False,
) -> None:
    parser.add_argument("--data-root", default="data",
                        help="Parent of mnist/, cifar10/, and imagenet1k-ilsvrc2012/.")
    parser.add_argument("--output-dir", default="outputs",
                        help=("New final output directory with --checkpoint; output workspace with --variant-root."
                              if checkpoint_output else
                              "Output workspace containing experiments/ and frozen inputs/."))
    parser.add_argument("--assets-root", default="assets",
                        help="Directory containing classifier and FID reference assets.")
    parser.add_argument("--cache-root", default=".cache/pfm",
                        help="Cache directory for PyTorch weights and FID reference computation.")
    if device:
        parser.add_argument("--device", choices=("cuda", "cpu"), default="cuda",
                            help="Use CPU explicitly for small examples; training defaults to CUDA.")


def context_from_args(args: argparse.Namespace, *, distributed: bool = False) -> ExecutionContext:
    context = execution_context_from_env(
        data_root=args.data_root, output_dir=args.output_dir,
        assets_root=args.assets_root, cache_root=args.cache_root,
        device=getattr(args, "device", "cpu"),
    )
    if context.world_size > 1 and not distributed:
        raise ValueError("This command is a single-process consumer; launch without torchrun.")
    import torch
    torch.hub.set_dir(str(context.runtime.cache_root / "torch"))
    return context


def require_cpu_fp32(context: ExecutionContext, config: Any, *, category: str) -> None:
    """Reject a CUDA BF16 protocol rather than misreporting CPU FP32 execution."""
    if context.device != "cpu":
        return
    if category == "train":
        precision, override = config.mixed_precision, "mixed_precision=none"
    elif category == "sample" or config.profile == "generated_common":
        precision, override = config.sampling.precision, "sampling.precision=fp32"
    else:
        precision, override = config.precision, "precision=fp32"
    if precision not in {"fp32", "none"}:
        raise ValueError(f"CPU execution uses FP32; select it explicitly with --set {override}")
