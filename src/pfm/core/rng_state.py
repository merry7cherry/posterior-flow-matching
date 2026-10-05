from __future__ import annotations

import random
from collections.abc import Mapping
from numbers import Integral, Real
from typing import Any

import numpy as np
import torch
import torch.distributed as dist

from pfm.core.distributed import DistributedContext


RNG_STATE_REQUIRED_KEYS = frozenset({"python", "numpy", "torch_cpu", "torch_cuda"})


def validate_rng_state_payload(
    state: object,
    *,
    context: str,
) -> dict[str, Any]:
    if not isinstance(state, Mapping):
        raise ValueError(f"{context} RNG state must be a mapping")
    actual_keys = set(state)
    unknown = sorted(actual_keys - RNG_STATE_REQUIRED_KEYS)
    if unknown:
        raise ValueError(f"{context} RNG state has unknown fields: {unknown}")
    missing = sorted(RNG_STATE_REQUIRED_KEYS - actual_keys)
    if missing:
        raise ValueError(f"{context} RNG state is missing fields: {missing}")
    _validate_python_rng_state(state["python"], context=context)
    _validate_numpy_rng_state(state["numpy"], context=context)
    _validate_cpu_rng_state(state["torch_cpu"], context=context)
    torch_cuda = state["torch_cuda"]
    if torch_cuda is not None:
        if not isinstance(torch_cuda, list) or not torch_cuda:
            raise ValueError(
                f"{context} RNG state torch_cuda must be None or a non-empty list of 1-D uint8 tensors"
            )
        for index, cuda_state in enumerate(torch_cuda):
            _validate_rng_tensor(cuda_state, field_name=f"torch_cuda[{index}]", context=context)
    return dict(state)


def _validate_python_rng_state(value: object, *, context: str) -> None:
    try:
        random.Random().setstate(value)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"{context} RNG state python is invalid") from exc


def _validate_numpy_rng_state(value: object, *, context: str) -> None:
    if not isinstance(value, tuple) or len(value) != 5:
        raise ValueError(f"{context} RNG state numpy is invalid")
    bit_generator, keys, position, has_gauss, cached_gaussian = value
    keys_array = np.asarray(keys)
    if keys_array.ndim != 1 or not np.issubdtype(keys_array.dtype, np.integer):
        raise ValueError(f"{context} RNG state numpy is invalid")
    if isinstance(position, bool) or not isinstance(position, Integral):
        raise ValueError(f"{context} RNG state numpy is invalid")
    if isinstance(has_gauss, bool) or not isinstance(has_gauss, Integral):
        raise ValueError(f"{context} RNG state numpy is invalid")
    if isinstance(cached_gaussian, bool) or not isinstance(cached_gaussian, Real):
        raise ValueError(f"{context} RNG state numpy is invalid")
    try:
        np.random.RandomState().set_state(
            (
                bit_generator,
                keys_array.astype(np.uint32, copy=False),
                int(position),
                int(has_gauss),
                float(cached_gaussian),
            )
        )
    except (TypeError, ValueError) as exc:
        raise ValueError(f"{context} RNG state numpy is invalid") from exc


def _validate_rng_tensor(value: object, *, field_name: str, context: str) -> None:
    if not isinstance(value, torch.Tensor) or value.dtype != torch.uint8 or value.dim() != 1:
        raise ValueError(f"{context} RNG state {field_name} must be a 1-D uint8 tensor")


def _validate_cpu_rng_state(value: object, *, context: str) -> None:
    _validate_rng_tensor(value, field_name="torch_cpu", context=context)
    try:
        torch.Generator(device="cpu").set_state(value)
    except (RuntimeError, TypeError) as exc:
        raise ValueError(f"{context} RNG state torch_cpu is not restorable") from exc


def _validate_cuda_rng_states_for_restore(states: list[torch.Tensor], *, context: str) -> None:
    if not torch.cuda.is_available():
        raise ValueError(f"{context} RNG state torch_cuda cannot be validated because CUDA is unavailable")
    device_count = torch.cuda.device_count()
    if len(states) != device_count:
        raise ValueError(
            f"{context} RNG state torch_cuda contains {len(states)} device states, "
            f"but {device_count} CUDA devices are visible"
        )
    for index, cuda_state in enumerate(states):
        try:
            torch.Generator(device=f"cuda:{index}").set_state(cuda_state)
        except (RuntimeError, TypeError) as exc:
            raise ValueError(f"{context} RNG state torch_cuda[{index}] is not restorable") from exc


def validate_rng_state_payload_for_restore(
    state: object,
    *,
    context: str,
    require_cuda: bool,
) -> dict[str, Any]:
    if type(require_cuda) is not bool:
        raise TypeError("require_cuda must be a boolean")
    validated = validate_rng_state_payload(state, context=context)
    torch_cuda = validated["torch_cuda"]
    if require_cuda:
        if torch_cuda is None:
            raise ValueError(f"{context} RNG state torch_cuda is required for CUDA training")
        _validate_cuda_rng_states_for_restore(torch_cuda, context=context)
    elif torch_cuda is not None:
        raise ValueError(f"{context} RNG state torch_cuda cannot be restored for CPU training")
    return validated


def validate_rng_state_list(
    states: object,
    *,
    world_size: int,
    context: str,
) -> list[dict[str, Any]]:
    if not isinstance(states, list):
        raise ValueError(f"{context} rng_state_by_rank must be a list")
    if len(states) != int(world_size):
        raise ValueError(f"{context} world size {len(states)} does not match current world size {int(world_size)}")
    return [
        validate_rng_state_payload(
            state,
            context=f"{context} rank {rank}",
        )
        for rank, state in enumerate(states)
    ]


def capture_rng_state() -> dict[str, Any]:
    numpy_state = np.random.get_state()
    state: dict[str, Any] = {
        "python": random.getstate(),
        "numpy": (
            numpy_state[0],
            numpy_state[1].tolist(),
            int(numpy_state[2]),
            int(numpy_state[3]),
            float(numpy_state[4]),
        ),
        "torch_cpu": torch.random.get_rng_state(),
        "torch_cuda": torch.cuda.get_rng_state_all() if torch.cuda.is_available() else None,
    }
    return state


def restore_rng_state(state: dict[str, Any], *, require_cuda: bool) -> None:
    state = validate_rng_state_payload_for_restore(
        state,
        context="restore",
        require_cuda=require_cuda,
    )
    random.setstate(state["python"])
    numpy_state = state["numpy"]
    np.random.set_state(
        (
            numpy_state[0],
            np.asarray(numpy_state[1], dtype=np.uint32),
            int(numpy_state[2]),
            int(numpy_state[3]),
            float(numpy_state[4]),
        )
    )
    torch.random.set_rng_state(state["torch_cpu"])
    cuda_state = state["torch_cuda"]
    if require_cuda:
        assert isinstance(cuda_state, list)
        torch.cuda.set_rng_state_all(cuda_state)


def gather_rng_state_for_checkpoint(context: DistributedContext) -> list[dict[str, Any]] | None:
    local_state = capture_rng_state()
    if not context.enabled:
        return [local_state]
    if not (dist.is_available() and dist.is_initialized()):
        raise RuntimeError("Distributed RNG gathering requires an initialized process group")
    gathered = [None for _ in range(context.world_size)] if context.is_main_process else None
    dist.gather_object(local_state, object_gather_list=gathered, dst=0)
    if context.is_main_process:
        assert gathered is not None
        return [state for state in gathered if state is not None]
    return None


def rng_state_for_rank(checkpoint: dict[str, Any], rank: int, world_size: int) -> dict[str, Any]:
    states = validate_rng_state_list(
        checkpoint["rng_state_by_rank"],
        world_size=int(world_size),
        context="resume checkpoint",
    )
    return states[int(rank)]
