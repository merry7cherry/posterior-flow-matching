from __future__ import annotations

import inspect
from dataclasses import fields
from pathlib import Path

import pytest
import torch

from pfm.core.sampling_config import SamplingConfig

import pfm.evaluation as evaluation
from pfm.core.config import ExperimentConfig
from pfm.evaluation import generation_service
from pfm.evaluation.checkpoint_loading import BoundEpochCheckpoint
from pfm.evaluation.generation_service import (
    GeneratedChunk,
    GenerationRequest,
    iter_bound_generation,
    iter_loaded_generation,
)
from pfm.sampling.samplers import SampleResult


def _bound_checkpoint(tmp_path: Path) -> BoundEpochCheckpoint:
    epoch_dir = tmp_path / "epoch_0001"
    return BoundEpochCheckpoint(
        epoch_id="epoch_0001",
        epoch_dir=epoch_dir,
        checkpoint_path=epoch_dir / "checkpoint.pt",
        manifest_path=epoch_dir / "checkpoint_manifest.json",
        manifest={},
        config_identity={},
        checkpoint_runtime_identity={
            "project_name": "pfm",
            "git_commit": "a" * 40,
            "rank": 0,
            "world_size": 1,
        },
        checkpoint_sha256="b" * 64,
    )


def _config() -> ExperimentConfig:
    return ExperimentConfig(
        dataset="mnist",
        image_channels=1,
        image_size=4,
        num_classes=3,
        noise_scale=1.25,
    )


def _fake_sample(
    model,
    variant,
    labels,
    config,
    sampling,
    device,
    sample_indices=None,
    seed=0,
    initial_noise=None,

):
    del model, variant, config, sample_indices, seed
    assert initial_noise is not None
    return SampleResult(
        samples=initial_noise.to(device),
        endpoint_var=None,
        velocity_var=None,
        variance_available=False,
        sampling_steps=sampling.steps,
        actual_nfe=sampling.steps + 7,
        sampling=sampling,
        actual_vjp=0,
        numerical_summary={},
        elapsed_seconds=0.1,
        uncertainty_source_step=None,
        uncertainty_source_t=None,
        uncertainty_source_state=None,


    )


def test_generation_service_has_exact_public_contract() -> None:
    assert [field.name for field in fields(GenerationRequest)] == [
        "labels",
        "sampling",
        "batch_size",
        "seed",
        "uncertainty_capture_t",
    ]
    assert [field.name for field in fields(GeneratedChunk)] == [
        "start_index",
        "labels",
        "sample_result",
        "peak_memory_allocated_bytes",
    ]
    signature = inspect.signature(iter_bound_generation)
    assert list(signature.parameters) == ["bound_checkpoint", "request", "device"]
    assert signature.parameters["bound_checkpoint"].kind is inspect.Parameter.POSITIONAL_OR_KEYWORD
    assert signature.parameters["request"].kind is inspect.Parameter.POSITIONAL_OR_KEYWORD
    assert signature.parameters["device"].kind is inspect.Parameter.KEYWORD_ONLY
    assert evaluation.__all__ == [
        "GeneratedChunk",
        "GenerationRequest",
        "iter_bound_generation",
        "iter_loaded_generation",
    ]
    assert evaluation.GenerationRequest is GenerationRequest
    assert evaluation.GeneratedChunk is GeneratedChunk
    assert evaluation.iter_bound_generation is iter_bound_generation
    assert evaluation.iter_loaded_generation is iter_loaded_generation


def test_generation_service_loads_once_and_preserves_chunk_order(
    monkeypatch,
    tmp_path,
) -> None:
    bound_checkpoint = _bound_checkpoint(tmp_path)
    device = torch.device("cpu")
    load_calls = []

    def fake_load(selected, selected_device, *, sampling):
        load_calls.append((selected, selected_device))
        return torch.nn.Identity(), "velocity_fm", _config()

    monkeypatch.setattr(generation_service, "load_model_from_bound_checkpoint", fake_load)
    monkeypatch.setattr(generation_service, "sample", _fake_sample)
    labels = torch.tensor([2, 0, 1, 2, 1], dtype=torch.long)
    request = GenerationRequest(
        labels=labels,
        sampling=SamplingConfig(steps=3),
        batch_size=2,
        seed=123,
    )

    chunks = list(iter_bound_generation(bound_checkpoint, request, device=device))

    assert load_calls == [(bound_checkpoint, device)]
    assert [
        (chunk.start_index, chunk.labels.tolist())
        for chunk in chunks
    ] == [
        (0, [2, 0]),
        (2, [1, 2]),
        (4, [1]),
    ]
    assert all(isinstance(chunk, GeneratedChunk) for chunk in chunks)
    assert [chunk.sample_result.actual_nfe for chunk in chunks] == [10, 10, 10]
    assert [chunk.sample_result.sampling_steps for chunk in chunks] == [3, 3, 3]


def test_generation_service_is_batch_size_invariant(
    monkeypatch,
    tmp_path,
) -> None:
    bound_checkpoint = _bound_checkpoint(tmp_path)
    monkeypatch.setattr(
        generation_service,
        "load_model_from_bound_checkpoint",
        lambda *args, **kwargs: (torch.nn.Identity(), "velocity_fm", _config()),
    )
    monkeypatch.setattr(generation_service, "sample", _fake_sample)
    labels = torch.tensor([0, 1, 2, 0, 1, 2, 0], dtype=torch.long)

    def generated(batch_size: int) -> torch.Tensor:
        request = GenerationRequest(
            labels=labels,
            sampling=SamplingConfig(steps=2),
            batch_size=batch_size,
            seed=991,
        )
        chunks = list(
            iter_bound_generation(
                bound_checkpoint,
                request,
                device=torch.device("cpu"),
            )
        )
        return torch.cat(
            [chunk.sample_result.samples.detach().cpu() for chunk in chunks],
            dim=0,
        )

    batch_two = generated(2)
    batch_seven = generated(7)

    assert torch.equal(batch_two, batch_seven)


def test_generation_service_snapshots_the_label_plan_before_iteration(
    monkeypatch,
    tmp_path,
) -> None:
    monkeypatch.setattr(
        generation_service,
        "load_model_from_bound_checkpoint",
        lambda *args, **kwargs: (torch.nn.Identity(), "velocity_fm", _config()),
    )
    monkeypatch.setattr(generation_service, "sample", _fake_sample)
    labels = torch.tensor([0, 1, 2], dtype=torch.long)
    iterator = iter_bound_generation(
        _bound_checkpoint(tmp_path),
        GenerationRequest(labels, SamplingConfig(steps=2), 2, 7),
        device=torch.device("cpu"),
    )

    labels.fill_(2)
    chunks = list(iterator)

    assert torch.cat([chunk.labels for chunk in chunks]).tolist() == [0, 1, 2]


@pytest.mark.parametrize(
    ("bound_checkpoint", "generation_request", "device", "message"),
    [
        (object(), GenerationRequest(torch.tensor([0]), SamplingConfig(steps=1), 1, 0), torch.device("cpu"), "bound_checkpoint"),
        (None, object(), torch.device("cpu"), "bound_checkpoint"),
        (None, None, "cpu", "bound_checkpoint"),
    ],
)
def test_generation_service_rejects_invalid_boundary_types_without_loading(
    monkeypatch,
    bound_checkpoint,
    generation_request,
    device,
    message,
) -> None:
    monkeypatch.setattr(
        generation_service,
        "load_model_from_bound_checkpoint",
        lambda *args, **kwargs: (_ for _ in ()).throw(
            AssertionError("invalid service input reached checkpoint loading")
        ),
    )

    with pytest.raises(TypeError, match=message):
        list(iter_bound_generation(bound_checkpoint, generation_request, device=device))


@pytest.mark.parametrize(
    ("generation_request", "error_type", "message"),
    [
        (GenerationRequest([0], SamplingConfig(steps=1), 1, 0), TypeError, "labels"),
        (GenerationRequest(torch.tensor([], dtype=torch.long), SamplingConfig(steps=1), 1, 0), ValueError, "labels"),
        (GenerationRequest(torch.tensor([[0]], dtype=torch.long), SamplingConfig(steps=1), 1, 0), ValueError, "one-dimensional"),
        (GenerationRequest(torch.tensor([0.0]), SamplingConfig(steps=1), 1, 0), ValueError, "integer"),
        (GenerationRequest(torch.tensor([0]), 0, 1, 0), TypeError, "sampling"),
        (GenerationRequest(torch.tensor([0]), True, 1, 0), TypeError, "sampling"),
        (GenerationRequest(torch.tensor([0]), SamplingConfig(steps=1), 0, 0), ValueError, "batch_size"),
        (GenerationRequest(torch.tensor([0]), SamplingConfig(steps=1), True, 0), ValueError, "batch_size"),
        (GenerationRequest(torch.tensor([0]), SamplingConfig(steps=1), 1, True), TypeError, "seed"),
        (
            GenerationRequest(
                torch.tensor([0]),
                SamplingConfig(steps=1),
                1,
                0,
                uncertainty_capture_t=True,
            ),
            TypeError,
            "uncertainty_capture_t",
        ),
    ],
)
def test_generation_service_rejects_invalid_request_before_loading(
    monkeypatch,
    tmp_path,
    generation_request,
    error_type,
    message,
) -> None:
    monkeypatch.setattr(
        generation_service,
        "load_model_from_bound_checkpoint",
        lambda *args, **kwargs: (_ for _ in ()).throw(
            AssertionError("invalid request reached checkpoint loading")
        ),
    )

    with pytest.raises(error_type, match=message):
        list(
            iter_bound_generation(
                _bound_checkpoint(tmp_path),
                generation_request,
                device=torch.device("cpu"),
            )
        )


def test_generation_service_rejects_non_device_before_loading(monkeypatch, tmp_path) -> None:
    monkeypatch.setattr(
        generation_service,
        "load_model_from_bound_checkpoint",
        lambda *args, **kwargs: (_ for _ in ()).throw(
            AssertionError("invalid device reached checkpoint loading")
        ),
    )
    request = GenerationRequest(torch.tensor([0]), SamplingConfig(steps=1), 1, 0)

    with pytest.raises(TypeError, match="device"):
        list(iter_bound_generation(_bound_checkpoint(tmp_path), request, device="cpu"))
