from __future__ import annotations

import hashlib

import torch


RANDOM_PROTOCOL = "pfm-indexed-brownian-probes-v1"
_BLOCK_SIZE = 64


def indexed_step_random(
    sample_indices: torch.Tensor, sample_shape: tuple[int, ...], *, seed: int,
    step: int, stream: str, probe: int = 0, device: torch.device,
) -> torch.Tensor:
    """CPU-generated fixed blocks keep draws invariant to consumer batching.

    Initialization retains the existing generation service protocol. Brownian
    and derivative streams never consume its generator or one another's state.
    """
    if stream not in {"brownian", "derivative"}:
        raise ValueError("unknown sampling random stream")
    indices = sample_indices.detach().to(device="cpu", dtype=torch.long)
    if indices.ndim != 1 or torch.any(indices < 0):
        raise ValueError("sample_indices must be one-dimensional and nonnegative")
    output = torch.empty((indices.numel(), *sample_shape), dtype=torch.float32)
    blocks = torch.div(indices, _BLOCK_SIZE, rounding_mode="floor")
    for block in torch.unique(blocks).tolist():
        key = f"{RANDOM_PROTOCOL}:{seed}:{stream}:{step}:{probe}:{block}".encode("ascii")
        block_seed = int.from_bytes(hashlib.sha256(key).digest()[:8], "little") % (2**63)
        generator = torch.Generator(device="cpu").manual_seed(block_seed)
        shape = (_BLOCK_SIZE, *sample_shape)
        if stream == "brownian":
            values = torch.randn(shape, generator=generator)
        else:
            values = torch.randint(0, 2, shape, generator=generator, dtype=torch.int8).float().mul_(2).sub_(1)
        mask = blocks == block
        output[mask] = values[indices[mask] % _BLOCK_SIZE]
    return output.to(device=device)
