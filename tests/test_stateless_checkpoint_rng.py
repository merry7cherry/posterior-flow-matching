from __future__ import annotations

import hashlib
import inspect
from pathlib import Path

import pytest
import torch

from pfm.core import checkpointing as checkpointing_module
from pfm.core import rng_state as rng_state_module
from pfm.core.config import (
    ExperimentConfig,
    experiment_config_to_mapping,
    shared_training_protocol_sha256,
)
from pfm.core.rng_state import capture_rng_state
from pfm.evaluation import checkpoint_loading
from pfm.training.checkpointing import write_checkpoint_manifest


def _saved_rng_state(*, rank: int, cuda_device_count: int) -> dict[str, object]:
    state = capture_rng_state()
    cuda_states = []
    for device_index in range(cuda_device_count):
        generator = torch.Generator(device="cpu")
        generator.manual_seed(rank * 100 + device_index)
        cuda_states.append(generator.get_state())
    state["torch_cuda"] = cuda_states
    return state


def _write_two_gpu_checkpoint(job_context) -> Path:
    config = ExperimentConfig(
        batch_size=2,
        time_proj_dim=16,
        class_proj_dim=16,
        unet_base_channels=8,
        image_size=28,
        num_classes=10,
    )
    training_config = experiment_config_to_mapping(config)
    producer_identity = {
        "project_name": "pfm",
        "git_commit": "b" * 40,
        "rank": 0,
        "world_size": 2,
    }
    variant_root = (
        job_context.runtime.experiments_root / "two_gpu_producer" / "velocity_fm"
    )
    checkpoint_path = variant_root / "epochs" / "epoch_0001" / "checkpoint.pt"
    checkpoint_path.parent.mkdir(parents=True)
    torch.save(
        {
            "variant": "velocity_fm",
            "training_config": training_config,
            "epoch": 1,
            "step": 1,
            "loss": 0.0,
            "model_state": {},
            "ema_model_state": {},
            "optimizer_state": {},
            "rng_state_by_rank": [
                _saved_rng_state(rank=rank, cuda_device_count=2)
                for rank in range(2)
            ],
            "distributed_world_size": 2,
            "runtime_identity": producer_identity,
            "dataset_scientific_identity_sha256": "d" * 64,
        },
        checkpoint_path,
    )
    write_checkpoint_manifest(
        path=checkpoint_path.with_name("checkpoint_manifest.json"),
        checkpoint_path=checkpoint_path,
        task_id=variant_root.parent.name,
        variant_id=variant_root.name,
        epoch=1,
        step=1,
        loss=0.0,
        checkpoint_sha256=hashlib.sha256(checkpoint_path.read_bytes()).hexdigest(),
        config_identity={
            "dataset": config.dataset,
            "architecture": config.architecture,
            "variant": "velocity_fm",
            "runtime_identity": producer_identity,
            "dataset_scientific_identity_sha256": "d" * 64,
            "shared_training_protocol_sha256": shared_training_protocol_sha256(config),
        },
    )
    return variant_root


def _install_consumer_cuda_topology(
    monkeypatch: pytest.MonkeyPatch,
    *,
    visible_cuda_devices: int,
) -> None:
    monkeypatch.setattr(
        torch.cuda,
        "is_available",
        lambda: visible_cuda_devices > 0,
    )
    monkeypatch.setattr(torch.cuda, "device_count", lambda: visible_cuda_devices)
    monkeypatch.setattr(
        rng_state_module,
        "_validate_cuda_rng_states_for_restore",
        lambda *args, **kwargs: pytest.fail(
            "stateless checkpoint consumers must not probe CUDA RNG restorability"
        ),
    )


@pytest.mark.parametrize("visible_cuda_devices", [0, 1])
def test_resolve_binds_two_gpu_checkpoint_without_rng_restore_probe(
    execution_context,
    monkeypatch,
    visible_cuda_devices,
):
    variant_root = _write_two_gpu_checkpoint(execution_context)
    _install_consumer_cuda_topology(
        monkeypatch,
        visible_cuda_devices=visible_cuda_devices,
    )

    bound = checkpoint_loading.resolve_bound_epoch_checkpoint(
        variant_root,
        1,
    )

    assert bound.checkpoint_runtime_identity["world_size"] == 2


@pytest.mark.parametrize("visible_cuda_devices", [0, 1])
def test_shared_consumer_loader_reads_two_gpu_checkpoint_without_rng_restore_probe(
    execution_context,
    monkeypatch,
    visible_cuda_devices,
):
    variant_root = _write_two_gpu_checkpoint(execution_context)
    _install_consumer_cuda_topology(
        monkeypatch,
        visible_cuda_devices=visible_cuda_devices,
    )
    bound = checkpoint_loading.resolve_bound_epoch_checkpoint(
        variant_root,
        1,
    )
    checkpoint = checkpoint_loading.load_bound_training_checkpoint(bound)

    assert checkpoint["distributed_world_size"] == 2
    assert checkpoint["runtime_identity"]["world_size"] == 2


def test_rng_loading_contract_has_no_compatibility_validation_policy() -> None:
    assert not hasattr(rng_state_module, "CudaRngValidation")
    for function in (
        rng_state_module.validate_rng_state_payload,
        rng_state_module.validate_rng_state_list,
        checkpointing_module.validate_training_checkpoint_payload,
        checkpointing_module.load_training_checkpoint_with_sha256,
        checkpoint_loading.resolve_bound_epoch_checkpoint,
        checkpoint_loading.load_bound_training_checkpoint,
    ):
        assert "cuda_rng_validation" not in inspect.signature(function).parameters

    require_cuda = inspect.signature(
        rng_state_module.restore_rng_state
    ).parameters["require_cuda"]
    assert require_cuda.kind is inspect.Parameter.KEYWORD_ONLY
    assert require_cuda.default is inspect.Parameter.empty
