from __future__ import annotations

import random
from contextlib import nullcontext
from typing import ContextManager

import numpy as np
import torch

from pfm.core.config import ExperimentConfig, validate_mixed_precision
from pfm.core.execution import ExecutionContext


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def execution_device(context: ExecutionContext) -> torch.device:
    if context.device == "cpu":
        if context.world_size != 1:
            raise ValueError("CPU execution supports one process; distributed training requires CUDA")
        return torch.device("cpu")
    if torch.cuda.is_available():
        visible_devices = torch.cuda.device_count()
        if context.local_rank >= visible_devices:
            raise RuntimeError("LOCAL_RANK exceeds the visible CUDA device count")
        device = torch.device("cuda", context.local_rank)
        torch.cuda.set_device(device)
        return device
    raise RuntimeError("CUDA requested but unavailable; use --device cpu for small examples")


def autocast_context(config: ExperimentConfig, device: torch.device) -> ContextManager[None]:
    mode = validate_mixed_precision(config.mixed_precision)
    if mode == "none" or device.type != "cuda":
        return nullcontext()
    return torch.amp.autocast("cuda", dtype=torch.bfloat16)


def apply_runtime_settings(config: ExperimentConfig, device: torch.device) -> None:
    if device.type != "cuda":
        return
    torch.backends.cuda.matmul.allow_tf32 = bool(config.allow_tf32)
    torch.backends.cudnn.allow_tf32 = bool(config.allow_tf32)
    torch.backends.cudnn.benchmark = bool(config.cudnn_benchmark)


def prepare_compiled_inference() -> None:
    """Keep compiled backward buffers reusable for multiple input VJPs.

    This process-level compiler policy must precede the first compiled forward
    and backward. Consumer processes can alternate ODE, posterior and SDE calls;
    their compiled cache must remain valid when SDE retains a graph for probes.
    Training runs in a separate process and keeps its single-backward policy.
    The private switch is part of our pinned PyTorch runtime, not a user knob.
    """
    import torch._functorch.config as aot_config

    aot_config.donated_buffer = False
