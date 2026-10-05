from __future__ import annotations

from collections.abc import Iterator
from dataclasses import dataclass
import math

import torch

from pfm.core.config import ExperimentConfig
from pfm.core.sampling_config import SamplingConfig
from pfm.core.validation import require_positive_integer
from pfm.evaluation.checkpoint_loading import (
    BoundEpochCheckpoint,
    load_model_from_bound_checkpoint,
)
from pfm.evaluation.generation_metadata import INITIAL_NOISE_BLOCK_SIZE
from pfm.sampling.samplers import SampleResult, sample


@dataclass(frozen=True)
class GenerationRequest:
    labels: torch.Tensor
    sampling: SamplingConfig
    batch_size: int
    seed: int
    uncertainty_capture_t: float | None = None


@dataclass(frozen=True)
class GeneratedChunk:
    start_index: int
    labels: torch.Tensor
    sample_result: SampleResult
    peak_memory_allocated_bytes: int = 0


def iter_bound_generation(
    bound_checkpoint: BoundEpochCheckpoint,
    request: GenerationRequest,
    *,
    device: torch.device,
) -> Iterator[GeneratedChunk]:
    if not isinstance(bound_checkpoint, BoundEpochCheckpoint):
        raise TypeError("bound_checkpoint must be a BoundEpochCheckpoint")
    if not isinstance(request, GenerationRequest):
        raise TypeError("request must be a GenerationRequest")
    if not isinstance(device, torch.device):
        raise TypeError("device must be a torch.device")
    _validate_generation_request(request)

    model, variant, config = load_model_from_bound_checkpoint(
        bound_checkpoint,
        device,
        sampling=request.sampling,
    )
    return iter_loaded_generation(model, variant, config, request, device=device)


def iter_loaded_generation(
    model: torch.nn.Module,
    variant: str,
    config: ExperimentConfig,
    request: GenerationRequest,
    *,
    device: torch.device,
) -> Iterator[GeneratedChunk]:
    if not isinstance(model, torch.nn.Module):
        raise TypeError("model must be a torch module")
    if not isinstance(config, ExperimentConfig):
        raise TypeError("config must be an ExperimentConfig")
    if not isinstance(request, GenerationRequest):
        raise TypeError("request must be a GenerationRequest")
    labels, batch_size, capture_t = _validate_generation_request(
        request
    )
    return _iter_loaded_model_chunks(
        model=model,
        variant=variant,
        config=config,
        labels=labels,
        sampling=request.sampling,
        batch_size=batch_size,
        device=device,
        seed=request.seed,
        uncertainty_capture_t=None if capture_t is None else float(capture_t),
    )


def _validate_generation_request(
    request: GenerationRequest,
) -> tuple[torch.Tensor, int, float | None]:
    if not isinstance(request, GenerationRequest):
        raise TypeError("request must be a GenerationRequest")
    labels = _snapshot_labels(request.labels)
    if not isinstance(request.sampling, SamplingConfig):
        raise TypeError("bound generation request sampling must be a SamplingConfig")
    batch_size = require_positive_integer(
        request.batch_size,
        "batch_size",
        context="bound generation request",
    )
    if isinstance(request.seed, bool) or not isinstance(request.seed, int):
        raise TypeError("bound generation request seed must be an integer")
    if not 0 <= request.seed < 2**32:
        raise ValueError("bound generation request seed must lie in [0, 2**32)")
    capture = request.uncertainty_capture_t
    if capture is not None and (
        isinstance(capture, bool) or not isinstance(capture, (int, float))
    ):
        raise TypeError("bound generation request uncertainty_capture_t must be numeric or None")
    if capture is not None:
        h = request.sampling.terminal_time / request.sampling.steps
        if not math.isfinite(capture) or capture < 0 or capture >= request.sampling.terminal_time:
            raise ValueError("bound generation request capture time must precede terminal_time")
        if abs(round(capture / h) * h - capture) > 1e-12:
            raise ValueError("bound generation request capture time must lie on the sampling grid")
    return labels, batch_size, (
        None if capture is None else float(capture)
    )


def _snapshot_labels(labels: object) -> torch.Tensor:
    if not isinstance(labels, torch.Tensor):
        raise TypeError("bound generation request labels must be a torch.Tensor")
    if labels.ndim != 1:
        raise ValueError("bound generation request labels must be one-dimensional")
    if labels.numel() == 0:
        raise ValueError("bound generation request labels must contain at least one item")
    if labels.dtype not in {
        torch.uint8,
        torch.int8,
        torch.int16,
        torch.int32,
        torch.int64,
    }:
        raise ValueError("bound generation request labels must use an integer dtype")
    return labels.detach().to(device="cpu", dtype=torch.long).clone()


@torch.no_grad()
def _iter_loaded_model_chunks(
    *,
    model: torch.nn.Module,
    variant: str,
    labels: torch.Tensor,
    config: ExperimentConfig,
    sampling: SamplingConfig,
    batch_size: int,
    device: torch.device,
    seed: int,
    uncertainty_capture_t: float | None,
) -> Iterator[GeneratedChunk]:
    resolved_batch_size = require_positive_integer(
        batch_size,
        "batch_size",
        context="generation sampling",
    )
    for start in range(0, labels.numel(), resolved_batch_size):
        label_chunk_cpu = labels[start : start + resolved_batch_size]
        label_chunk = label_chunk_cpu.to(device)
        sample_indices = torch.arange(
            start,
            start + int(label_chunk_cpu.numel()),
            dtype=torch.long,
        )
        noise_chunk = _indexed_initial_noise(config, sample_indices, seed)
        capture_kwargs = (
            {"uncertainty_capture_t": uncertainty_capture_t}
            if uncertainty_capture_t is not None
            else {}
        )
        if device.type == "cuda":
            torch.cuda.synchronize(device)
            torch.cuda.reset_peak_memory_stats(device)
        sample_result = sample(
            model,
            variant,
            label_chunk,
            config,
            sampling,
            device,
            initial_noise=noise_chunk,
            sample_indices=sample_indices,
            seed=seed,
            **capture_kwargs,
        )
        if device.type == "cuda":
            torch.cuda.synchronize(device)
        yield GeneratedChunk(
            start_index=int(start),
            labels=label_chunk_cpu.detach().cpu().clone(),
            sample_result=sample_result,
            peak_memory_allocated_bytes=(
                int(torch.cuda.max_memory_allocated(device)) if device.type == "cuda" else 0
            ),
        )


def _indexed_initial_noise(
    config: ExperimentConfig,
    sample_indices: torch.Tensor,
    seed: int,
    dtype: torch.dtype = torch.float32,
) -> torch.Tensor:
    indices = sample_indices.detach().cpu().to(torch.long).flatten()
    shape = (
        int(indices.numel()),
        int(config.image_channels),
        int(config.image_size),
        int(config.image_size),
    )
    output = torch.empty(shape, dtype=dtype)
    if indices.numel() == 0:
        return output

    block_ids = torch.div(indices, INITIAL_NOISE_BLOCK_SIZE, rounding_mode="floor")
    for block_id in torch.unique(block_ids).tolist():
        resolved_block_id = int(block_id)
        selected = block_ids == resolved_block_id
        selected_positions = selected.nonzero(as_tuple=False).flatten()
        offsets = indices[selected_positions] - resolved_block_id * INITIAL_NOISE_BLOCK_SIZE
        block = _noise_block(config, resolved_block_id, seed, dtype=dtype)
        output[selected_positions] = block[offsets]
    return output


def _noise_block(
    config: ExperimentConfig,
    block_id: int,
    seed: int,
    dtype: torch.dtype = torch.float32,
) -> torch.Tensor:
    generator = torch.Generator(device="cpu")
    generator.manual_seed(_chunk_seed(seed, block_id))
    shape = (
        int(INITIAL_NOISE_BLOCK_SIZE),
        int(config.image_channels),
        int(config.image_size),
        int(config.image_size),
    )
    return float(config.noise_scale) * torch.randn(shape, generator=generator, dtype=dtype)


def _chunk_seed(seed: int, start_index: int) -> int:
    return int((int(seed) + 1_000_003 * int(start_index)) % (2**63 - 1))
