from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

import torch
import torch.distributed as dist
from torch import nn

from pfm.core.runtime import execution_device
from pfm.core.execution import ExecutionContext


@dataclass(frozen=True)
class DistributedContext:
    enabled: bool
    rank: int
    local_rank: int
    world_size: int
    device: torch.device
    backend: Optional[str] = None

    @property
    def is_main_process(self) -> bool:
        return self.rank == 0


def distributed_context_from_job(
    context: ExecutionContext,
    *,
    initialize: bool = True,
) -> DistributedContext:
    device = execution_device(context)
    if dist.is_initialized():
        if context.world_size == 1:
            raise RuntimeError("Existing distributed process group is incompatible with world_size==1")
        _validate_initialized_process_group(context)
    elif context.world_size == 1:
        return DistributedContext(False, context.rank, context.local_rank, context.world_size, device, None)
    elif initialize:
        dist.init_process_group(backend="nccl", init_method="env://", device_id=device)
    else:
        raise RuntimeError("initialize=False requires an existing initialized NCCL process group")
    return DistributedContext(True, context.rank, context.local_rank, context.world_size, device, "nccl")


def _validate_initialized_process_group(context: ExecutionContext) -> None:
    backend = dist.get_backend()
    if backend != "nccl":
        raise RuntimeError(f"Existing distributed process group must use NCCL, got {backend!r}")
    rank = dist.get_rank()
    if int(rank) != int(context.rank):
        raise RuntimeError(f"Existing distributed process group rank {rank} does not match execution context rank {context.rank}")
    world_size = dist.get_world_size()
    if int(world_size) != int(context.world_size):
        raise RuntimeError(
            f"Existing distributed process group world_size {world_size} does not match "
            f"execution context world_size {context.world_size}"
        )


def wrap_model_for_distributed(model: nn.Module, context: DistributedContext) -> nn.Module:
    if not context.enabled:
        return model
    from torch.nn.parallel import DistributedDataParallel

    if context.device.type != "cuda":
        raise RuntimeError("Distributed training requires a CUDA device")
    return DistributedDataParallel(model, device_ids=[context.local_rank], output_device=context.local_rank)


def unwrap_model(model: nn.Module) -> nn.Module:
    return model.module if hasattr(model, "module") else model


def barrier(context: DistributedContext) -> None:
    if not context.enabled:
        return
    if not dist.is_available() or not dist.is_initialized():
        raise RuntimeError("enabled distributed context requires an initialized process group")
    dist.barrier(device_ids=[context.local_rank])


def cleanup_distributed() -> None:
    if dist.is_available() and dist.is_initialized():
        dist.destroy_process_group()
