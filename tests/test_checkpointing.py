import hashlib
import json
from dataclasses import replace
from pathlib import Path

import torch
import pytest

from pfm.core import checkpointing as checkpointing_module
from pfm.training import checkpointing as training_checkpointing_module
from pfm.evaluation import checkpoint_loading as evaluation_checkpointing_module
from pfm.core.atomic_io import write_torch_atomic
from pfm.core.checkpointing import (
    load_training_checkpoint_with_sha256,
    validate_runtime_identity_payload,
    validate_checkpoint_manifest,
    validate_training_checkpoint_payload,
)
from pfm.training.checkpointing import (
    build_training_checkpoint,
    write_checkpoint_manifest,
)
from pfm.core.config import (
    ExperimentConfig,
    experiment_config_to_mapping,
    shared_training_protocol_sha256,
)
from pfm.core.layout import training_run_layout
from pfm.core.rng_state import capture_rng_state
from conftest import temporary_runtime


def _key_complete_rng_state(rank=0):
    state = capture_rng_state()
    generator = torch.Generator(device="cpu")
    generator.manual_seed(rank)
    state["torch_cpu"] = generator.get_state()
    state["torch_cuda"] = None
    return state


def _runtime_identity(rank=0, world_size=1):
    return {
        "project_name": "pfm",
        "git_commit": "a" * 40,
        "rank": rank,
        "world_size": world_size,
    }


def _config_identity(
    *,
    config=None,
    variant="velocity_fm",
    runtime_identity=None,
    dataset_scientific_identity_sha256="d" * 64,
):
    resolved_config = ExperimentConfig() if config is None else config
    return {
        "dataset": resolved_config.dataset,
        "architecture": resolved_config.architecture,
        "variant": variant,
        "runtime_identity": (
            _runtime_identity() if runtime_identity is None else runtime_identity
        ),
        "dataset_scientific_identity_sha256": dataset_scientific_identity_sha256,
        "shared_training_protocol_sha256": shared_training_protocol_sha256(
            resolved_config
        ),
    }


def valid_manifest_dict(**overrides):
    manifest = {
        "task_id": "task_ab12",
        "variant_id": "velocity_fm",
        "epoch": 3,
        "epoch_id": "epoch_0003",
        "step": 12,
        "loss": 0.25,
        "checkpoint": "epochs/epoch_0003/checkpoint.pt",
        "checkpoint_sha256": hashlib.sha256(b"placeholder").hexdigest(),
        "config_identity": _config_identity(),
    }
    manifest.update(overrides)
    return manifest


def _valid_training_checkpoint_payload(**overrides):
    config = ExperimentConfig()
    checkpoint = {
        "variant": "velocity_fm",
        "training_config": experiment_config_to_mapping(config),
        "epoch": 3,
        "step": 12,
        "loss": 0.25,
        "model_state": {"weight": torch.tensor([1.0])},
        "ema_model_state": {"weight": torch.tensor([2.0])},
        "optimizer_state": {"state": {}, "param_groups": []},
        "rng_state_by_rank": [_key_complete_rng_state(rank=0)],
        "distributed_world_size": 1,
        "runtime_identity": _runtime_identity(),
        "dataset_scientific_identity_sha256": "d" * 64,
    }
    checkpoint.update(overrides)
    return checkpoint


def valid_checkpoint_dict(**overrides):
    return _valid_training_checkpoint_payload(**overrides)


def test_current_checkpoint_contract_requires_ema_state() -> None:
    checkpoint = _valid_training_checkpoint_payload()
    checkpoint.pop("ema_model_state")

    with pytest.raises(ValueError, match="missing fields.*ema_model_state"):
        validate_training_checkpoint_payload(checkpoint)


def test_current_checkpoint_contract_rejects_open_metadata_bag() -> None:
    checkpoint = _valid_training_checkpoint_payload(
        metadata={"self_conditioning_training_policy": "training_sampled"}
    )

    with pytest.raises(ValueError, match="unknown fields.*metadata"):
        validate_training_checkpoint_payload(checkpoint)


def test_atomic_torch_save_writes_loadable_file(tmp_path):
    path = tmp_path / "checkpoint.pt"
    write_torch_atomic(path, {"x": torch.tensor([1])})

    payload = torch.load(path, map_location="cpu")
    assert payload["x"].tolist() == [1]
    assert not (tmp_path / "checkpoint.pt.tmp").exists()


def test_training_checkpoint_loader_hashes_the_same_open_file_after_atomic_replacement(
    tmp_path,
    monkeypatch,
):
    checkpoint_path = tmp_path / "checkpoint.pt"
    replacement_path = tmp_path / "replacement.pt"
    original_payload = _valid_training_checkpoint_payload(
        model_state={"version": "original"}
    )
    replacement_payload = _valid_training_checkpoint_payload(
        model_state={"version": "replacement"}
    )
    torch.save(original_payload, checkpoint_path)
    torch.save(replacement_payload, replacement_path)
    original_bytes = checkpoint_path.read_bytes()
    replacement_bytes = replacement_path.read_bytes()
    original_torch_load = torch.load

    def replace_path_after_load(handle, *args, **kwargs):
        payload = original_torch_load(handle, *args, **kwargs)
        replacement_path.replace(checkpoint_path)
        return payload

    monkeypatch.setattr(checkpointing_module.torch, "load", replace_path_after_load)

    payload, digest = checkpointing_module.load_training_checkpoint_with_sha256(
        checkpoint_path
    )

    assert payload["model_state"] == {"version": "original"}
    assert digest == hashlib.sha256(original_bytes).hexdigest()
    assert checkpoint_path.read_bytes() == replacement_bytes


def test_training_checkpoint_uses_explicit_training_config_key():
    model = torch.nn.Linear(1, 1)
    optimizer = torch.optim.AdamW(model.parameters())
    config = ExperimentConfig()
    payload = build_training_checkpoint(
        model=model,
        optimizer=optimizer,
        variant="velocity_fm",
        config=config,
        epoch=1,
        step=2,
        loss=0.5,
        ema_model=model,
        distributed_world_size=1,
        rng_state_by_rank=[_key_complete_rng_state(rank=0)],
        runtime_identity=_runtime_identity(),
        dataset_scientific_identity_sha256="d" * 64,
    )

    assert "training_config" in payload
    assert "config" not in payload
    assert "metric_names" not in payload["training_config"]


def test_training_checkpoint_has_exact_current_state_contract():
    model = torch.nn.Linear(1, 1)
    optimizer = torch.optim.AdamW(model.parameters())
    config = ExperimentConfig()
    payload = build_training_checkpoint(
        model=model,
        optimizer=optimizer,
        variant="pfm_delta",
        config=config,
        epoch=1,
        step=2,
        loss=0.5,
        ema_model=model,
        distributed_world_size=1,
        rng_state_by_rank=[_key_complete_rng_state(rank=0)],
        runtime_identity=_runtime_identity(),
        dataset_scientific_identity_sha256="d" * 64,
    )

    assert "metadata" not in payload
    assert payload["model_state"] == model.state_dict()
    assert payload["ema_model_state"] == model.state_dict()


def test_training_checkpoint_builder_requires_ema_model() -> None:
    model = torch.nn.Linear(1, 1)
    optimizer = torch.optim.AdamW(model.parameters())

    with pytest.raises(TypeError, match="ema_model"):
        build_training_checkpoint(
            model=model,
            optimizer=optimizer,
            variant="velocity_fm",
            config=ExperimentConfig(),
            epoch=1,
            step=2,
            loss=0.5,
            ema_model=None,
            distributed_world_size=1,
            rng_state_by_rank=[_key_complete_rng_state(rank=0)],
            runtime_identity=_runtime_identity(),
            dataset_scientific_identity_sha256="d" * 64,
        )


def test_checkpoint_model_consumer_loads_ema_state_only(monkeypatch) -> None:
    raw_model = torch.nn.Linear(1, 1)
    ema_model = torch.nn.Linear(1, 1)
    with torch.no_grad():
        raw_model.weight.fill_(1.0)
        raw_model.bias.fill_(1.0)
        ema_model.weight.fill_(2.0)
        ema_model.bias.fill_(2.0)
    checkpoint = _valid_training_checkpoint_payload(
        model_state=raw_model.state_dict(),
        ema_model_state=ema_model.state_dict(),
    )
    loaded_model = torch.nn.Linear(1, 1)
    monkeypatch.setattr(
        evaluation_checkpointing_module,
        "load_bound_training_checkpoint",
        lambda _: checkpoint,
    )
    monkeypatch.setattr(
        evaluation_checkpointing_module,
        "build_model",
        lambda *_: loaded_model,
    )
    monkeypatch.setattr(
        evaluation_checkpointing_module,
        "apply_runtime_settings",
        lambda *_: None,
    )

    model, variant, _ = evaluation_checkpointing_module.load_model_from_bound_checkpoint(
        object(),
        torch.device("cpu"),
    )

    assert variant == "velocity_fm"
    assert torch.equal(model.weight, ema_model.weight)
    assert torch.equal(model.bias, ema_model.bias)


def test_training_checkpoint_build_requires_rng_state_by_rank():
    model = torch.nn.Linear(1, 1)
    optimizer = torch.optim.AdamW(model.parameters())
    config = ExperimentConfig()

    with pytest.raises(ValueError, match="rng_state_by_rank is required"):
        build_training_checkpoint(
            model=model,
            optimizer=optimizer,
            variant="velocity_fm",
            config=config,
            epoch=1,
            step=2,
            loss=0.5,
            ema_model=model,
            distributed_world_size=1,
            rng_state_by_rank=None,
            runtime_identity=_runtime_identity(),
            dataset_scientific_identity_sha256="d" * 64,
        )


def test_training_checkpoint_build_rejects_rng_state_count_mismatch():
    model = torch.nn.Linear(1, 1)
    optimizer = torch.optim.AdamW(model.parameters())
    config = ExperimentConfig()

    with pytest.raises(ValueError, match="world size"):
        build_training_checkpoint(
            model=model,
            optimizer=optimizer,
            variant="velocity_fm",
            config=config,
            epoch=1,
            step=2,
            loss=0.5,
            ema_model=model,
            distributed_world_size=2,
            rng_state_by_rank=[{"rank": 0}],
            runtime_identity=_runtime_identity(world_size=2),
            dataset_scientific_identity_sha256="d" * 64,
        )


def test_training_checkpoint_build_rejects_malformed_rank_rng_state():
    model = torch.nn.Linear(1, 1)
    optimizer = torch.optim.AdamW(model.parameters())
    config = ExperimentConfig()

    with pytest.raises(ValueError, match="training checkpoint rank 0 RNG state must be a mapping"):
        build_training_checkpoint(
            model=model,
            optimizer=optimizer,
            variant="velocity_fm",
            config=config,
            epoch=1,
            step=2,
            loss=0.5,
            ema_model=model,
            distributed_world_size=1,
            rng_state_by_rank=[None],
            runtime_identity=_runtime_identity(),
            dataset_scientific_identity_sha256="d" * 64,
        )


@pytest.mark.parametrize("distributed_world_size", ["1", 1.0, True])
def test_training_checkpoint_build_rejects_non_integral_distributed_world_size(distributed_world_size):
    model = torch.nn.Linear(1, 1)
    optimizer = torch.optim.AdamW(model.parameters())
    config = ExperimentConfig()

    with pytest.raises(ValueError, match="distributed_world_size"):
        build_training_checkpoint(
            model=model,
            optimizer=optimizer,
            variant="velocity_fm",
            config=config,
            epoch=1,
            step=2,
            loss=0.5,
            ema_model=model,
            distributed_world_size=distributed_world_size,
            rng_state_by_rank=[{"rank": 0}],
            runtime_identity=_runtime_identity(),
            dataset_scientific_identity_sha256="d" * 64,
        )


@pytest.mark.parametrize(
    ("field_name", "invalid_value", "match"),
    [
        ("variant", 123, "variant"),
        ("variant", True, "variant"),
        ("variant", "", "variant"),
        ("epoch", "3", "epoch"),
        ("epoch", True, "epoch"),
        ("step", "12", "step"),
        ("step", True, "step"),
        ("training_config", None, "training_config"),
        ("model_state", None, "model_state"),
        ("optimizer_state", None, "optimizer_state"),
        ("ema_model_state", None, "ema_model_state"),
    ],
)
def test_training_checkpoint_payload_rejects_invalid_current_contract_fields(field_name, invalid_value, match):
    checkpoint = _valid_training_checkpoint_payload()
    checkpoint[field_name] = invalid_value

    with pytest.raises(ValueError, match=match):
        validate_training_checkpoint_payload(checkpoint)


@pytest.mark.parametrize(
    ("overrides", "match"),
    [
        ({"variant": 123}, "variant"),
        ({"variant": True}, "variant"),
        ({"variant": ""}, "variant"),
        ({"epoch": "1"}, "epoch"),
        ({"epoch": True}, "epoch"),
        ({"step": "2"}, "step"),
        ({"step": True}, "step"),
    ],
)
def test_training_checkpoint_build_rejects_invalid_current_contract_inputs(overrides, match):
    model = torch.nn.Linear(1, 1)
    optimizer = torch.optim.AdamW(model.parameters())
    config = ExperimentConfig()
    kwargs = {
        "model": model,
        "optimizer": optimizer,
        "variant": "velocity_fm",
        "config": config,
        "epoch": 1,
        "step": 2,
        "loss": 0.5,
        "ema_model": model,
        "distributed_world_size": 1,
        "rng_state_by_rank": [_key_complete_rng_state(rank=0)],
        "runtime_identity": _runtime_identity(),
        "dataset_scientific_identity_sha256": "d" * 64,
    }
    kwargs.update(overrides)

    with pytest.raises(ValueError, match=match):
        build_training_checkpoint(**kwargs)


def test_training_checkpoint_payload_rejects_unknown_old_field():
    checkpoint = _valid_training_checkpoint_payload()
    checkpoint["checkpoint_version"] = 1

    with pytest.raises(ValueError, match="unknown fields"):
        validate_training_checkpoint_payload(checkpoint)


def test_training_checkpoint_payload_rejects_legacy_rng_state():
    checkpoint = _valid_training_checkpoint_payload()
    checkpoint.pop("rng_state_by_rank")
    checkpoint["rng_state"] = {"python": "old"}

    with pytest.raises(ValueError, match="missing fields|unknown fields"):
        validate_training_checkpoint_payload(checkpoint)


def test_training_checkpoint_payload_requires_distributed_world_size():
    checkpoint = _valid_training_checkpoint_payload()
    checkpoint.pop("distributed_world_size")

    with pytest.raises(ValueError, match="missing fields"):
        validate_training_checkpoint_payload(checkpoint)


def test_checkpoint_requires_runtime_identity():
    checkpoint = _valid_training_checkpoint_payload()
    checkpoint.pop("runtime_identity")

    with pytest.raises(ValueError, match="missing fields"):
        validate_training_checkpoint_payload(checkpoint)


def test_checkpoint_runtime_identity_has_exact_keys():
    checkpoint = _valid_training_checkpoint_payload()
    checkpoint["runtime_identity"] = _runtime_identity() | {"local_rank": 0}

    with pytest.raises(ValueError, match="runtime_identity"):
        validate_training_checkpoint_payload(checkpoint)


def test_runtime_identity_accepts_unknown_source_revision_for_installed_package():
    identity = _runtime_identity() | {"git_commit": None}

    assert validate_runtime_identity_payload(identity, context="training checkpoint") == identity


@pytest.mark.parametrize(
    "identity",
    [
        _runtime_identity() | {"git_commit": "A" * 40},
        _runtime_identity() | {"project_name": " pfm"},
        _runtime_identity() | {"project_name": "pfm "},
    ],
)
def test_runtime_identity_rejects_noncanonical_commit_and_project_name(identity):
    with pytest.raises(ValueError, match="lowercase 40-character|single path component"):
        validate_runtime_identity_payload(identity, context="training checkpoint")


def test_training_checkpoint_payload_rejects_invalid_runtime_shapes():
    checkpoint = _valid_training_checkpoint_payload(rng_state_by_rank={"rank": 0})

    with pytest.raises(ValueError, match="rng_state_by_rank"):
        validate_training_checkpoint_payload(checkpoint)

    checkpoint = _valid_training_checkpoint_payload(distributed_world_size=0)

    with pytest.raises(ValueError, match="distributed_world_size"):
        validate_training_checkpoint_payload(checkpoint)


def test_training_checkpoint_payload_rejects_rng_state_count_mismatch():
    checkpoint = _valid_training_checkpoint_payload()
    checkpoint["distributed_world_size"] = 2
    checkpoint["runtime_identity"] = _runtime_identity(world_size=2)
    checkpoint["rng_state_by_rank"] = [_key_complete_rng_state(rank=0)]
    with pytest.raises(ValueError, match="world size"):
        validate_training_checkpoint_payload(checkpoint)


def test_training_checkpoint_payload_rejects_non_mapping_rank_rng_state():
    checkpoint = _valid_training_checkpoint_payload(rng_state_by_rank=[None])

    with pytest.raises(ValueError, match="rank 0 RNG state must be a mapping"):
        validate_training_checkpoint_payload(checkpoint)


def test_training_checkpoint_payload_accepts_rng_state_count_matching_world_size():
    checkpoint = _valid_training_checkpoint_payload()
    checkpoint["distributed_world_size"] = 2
    checkpoint["runtime_identity"] = _runtime_identity(world_size=2)
    checkpoint["rng_state_by_rank"] = [_key_complete_rng_state(rank=0), _key_complete_rng_state(rank=1)]
    validate_training_checkpoint_payload(checkpoint)


@pytest.mark.parametrize("loss", [float("nan"), float("inf"), float("-inf"), True])
def test_training_checkpoint_payload_rejects_invalid_loss(loss):
    checkpoint = _valid_training_checkpoint_payload(loss=loss)

    with pytest.raises(ValueError, match="training checkpoint field 'loss'"):
        validate_training_checkpoint_payload(checkpoint)


def test_load_training_checkpoint_rejects_nonfinite_loss(tmp_path):
    checkpoint_path = tmp_path / "checkpoint.pt"
    torch.save(_valid_training_checkpoint_payload(loss=float("inf")), checkpoint_path)

    with pytest.raises(ValueError, match="training checkpoint field 'loss'"):
        load_training_checkpoint_with_sha256(checkpoint_path)


def test_load_training_checkpoint_validates_payload(tmp_path):
    checkpoint_path = tmp_path / "checkpoint.pt"
    torch.save(_valid_training_checkpoint_payload(), checkpoint_path)

    loaded, _ = load_training_checkpoint_with_sha256(checkpoint_path)

    assert loaded["variant"] == "velocity_fm"


def test_training_checkpoint_loader_structurally_validates_cuda_rng_without_cuda_runtime(
    tmp_path,
    monkeypatch,
):
    checkpoint_path = tmp_path / "checkpoint.pt"
    checkpoint = _valid_training_checkpoint_payload()
    checkpoint["rng_state_by_rank"][0]["torch_cuda"] = [
        _key_complete_rng_state(rank=1)["torch_cpu"]
    ]
    torch.save(checkpoint, checkpoint_path)
    monkeypatch.setattr(torch.cuda, "is_available", lambda: False)

    loaded, _ = checkpointing_module.load_training_checkpoint_with_sha256(checkpoint_path)

    assert loaded["runtime_identity"] == checkpoint["runtime_identity"]


def test_load_training_checkpoint_rejects_non_mapping_payload(tmp_path):
    checkpoint_path = tmp_path / "checkpoint.pt"
    torch.save(["not", "a", "mapping"], checkpoint_path)

    with pytest.raises(ValueError, match="mapping"):
        load_training_checkpoint_with_sha256(checkpoint_path)


def test_checkpoint_manifest_points_to_sibling_epoch_checkpoint(tmp_path):
    checkpoint_path = tmp_path / "task_ab12" / "velocity_fm" / "epochs" / "epoch_0003" / "checkpoint.pt"
    checkpoint_path.parent.mkdir(parents=True)
    checkpoint_path.write_bytes(b"checkpoint")

    checkpoint_manifest = write_checkpoint_manifest(
        path=checkpoint_path.with_name("checkpoint_manifest.json"),
        checkpoint_path=checkpoint_path,
        task_id="task_ab12",
        variant_id="velocity_fm",
        epoch=3,
        step=12,
        loss=0.25,
        checkpoint_sha256=hashlib.sha256(checkpoint_path.read_bytes()).hexdigest(),
        config_identity=_config_identity(),
    )
    assert checkpoint_manifest["epoch_id"] == "epoch_0003"
    assert set(checkpoint_manifest) == {
        "task_id",
        "variant_id",
        "epoch",
        "epoch_id",
        "step",
        "loss",
        "checkpoint",
        "checkpoint_sha256",
        "config_identity",
    }
    assert checkpoint_manifest["checkpoint"] == "checkpoint.pt"


@pytest.mark.parametrize("loss", [float("nan"), float("inf"), float("-inf")])
def test_checkpoint_manifest_rejects_nonfinite_loss(tmp_path, loss):
    checkpoint_path = tmp_path / "checkpoint.pt"
    checkpoint_path.write_bytes(b"checkpoint")

    with pytest.raises(ValueError, match="loss"):
        write_checkpoint_manifest(
            path=tmp_path / "checkpoint_manifest.json",
            checkpoint_path=checkpoint_path,
            task_id="task_ab12",
            variant_id="velocity_fm",
            epoch=1,
            step=1,
            loss=loss,
            checkpoint_sha256=hashlib.sha256(checkpoint_path.read_bytes()).hexdigest(),
            config_identity=_config_identity(),
        )


def test_checkpoint_manifest_rejects_unknown_fields(tmp_path):
    manifest_root = tmp_path
    checkpoint = manifest_root / "epochs" / "epoch_0001" / "checkpoint.pt"
    checkpoint.parent.mkdir(parents=True)
    checkpoint.write_bytes(b"placeholder")
    manifest = {
        "task_id": "task",
        "variant_id": "pfm_full",
        "epoch": 1,
        "epoch_id": "epoch_0001",
        "step": 10,
        "loss": 0.5,
        "checkpoint": "epochs/epoch_0001/checkpoint.pt",
        "config_identity": {
            "dataset": "mnist",
            "architecture": "conditional_unet",
            "variant": "pfm_full",
        },
        "checkpoint_version": 2,
    }

    with pytest.raises(ValueError, match="unknown checkpoint manifest fields"):
        validate_checkpoint_manifest(manifest, manifest_root=manifest_root)


def test_checkpoint_manifest_rejects_epoch_id_mismatch(tmp_path):
    checkpoint = tmp_path / "epochs" / "epoch_0001" / "checkpoint.pt"
    checkpoint.parent.mkdir(parents=True)
    checkpoint.write_bytes(b"placeholder")
    manifest = valid_manifest_dict(
        epoch=1,
        epoch_id="epoch_9999",
        checkpoint="epochs/epoch_0001/checkpoint.pt",
    )

    with pytest.raises(ValueError, match="epoch_id must match epoch"):
        validate_checkpoint_manifest(manifest, manifest_root=tmp_path)


def test_checkpoint_manifest_requires_config_identity(tmp_path):
    manifest = {
        "task_id": "task",
        "variant_id": "pfm_full",
        "epoch": 1,
        "epoch_id": "epoch_0001",
        "step": 10,
        "loss": 0.5,
        "checkpoint": "epochs/epoch_0001/checkpoint.pt",
    }

    with pytest.raises(ValueError, match="missing checkpoint manifest fields"):
        validate_checkpoint_manifest(manifest, manifest_root=tmp_path)


def test_checkpoint_manifest_rejects_absolute_checkpoint_path(tmp_path):
    manifest = valid_manifest_dict(checkpoint=str(tmp_path / "checkpoint.pt"))

    with pytest.raises(ValueError, match="checkpoint path must be relative"):
        validate_checkpoint_manifest(manifest, manifest_root=tmp_path)


def test_checkpoint_manifest_resolves_relative_path_once(tmp_path):
    checkpoint = tmp_path / "epochs" / "epoch_0001" / "checkpoint.pt"
    checkpoint.parent.mkdir(parents=True)
    checkpoint.write_bytes(b"placeholder")
    manifest = valid_manifest_dict(epoch=1, epoch_id="epoch_0001", checkpoint="epochs/epoch_0001/checkpoint.pt")

    assert validate_checkpoint_manifest(manifest, manifest_root=tmp_path) == checkpoint


def test_resume_rejects_noncanonical_latest_alias(tmp_path):
    layout = training_run_layout(
        temporary_runtime(tmp_path),
        task_id="task_ab12",
        variant_id="velocity_fm",
    )

    with pytest.raises(ValueError, match="canonical recovery path"):
        training_checkpointing_module.load_canonical_resume_checkpoint(
            layout,
            layout.variant_root / "latest.json",
        )


def test_recovery_checkpoint_advances_latest_and_removes_old_generation(tmp_path):
    layout = training_run_layout(
        temporary_runtime(tmp_path),
        task_id="task_ab12",
        variant_id="velocity_fm",
    )
    for epoch in (1, 2):
        step = epoch * 4
        training_checkpointing_module.publish_recovery_checkpoint(
            layout=layout,
            checkpoint_payload=_valid_training_checkpoint_payload(
                epoch=epoch, step=step, loss=0.25 / epoch
            ),
            variant="velocity_fm",
            epoch=epoch,
            step=step,
            loss=0.25 / epoch,
            config_identity=_config_identity(),
        )

    checkpoint_path, checkpoint = (
        training_checkpointing_module.load_canonical_resume_checkpoint(
            layout, layout.recovery_latest_path
        )
    )
    generations = [path for path in layout.recovery_generations_root.iterdir()]
    assert checkpoint["epoch"] == 2
    assert checkpoint_path.is_file()
    assert len(generations) == 1


def _publish_epoch(layout, epoch: int):
    step = epoch * 4
    loss = 0.25 / epoch
    payload = _valid_training_checkpoint_payload(epoch=epoch, step=step, loss=loss)
    return training_checkpointing_module.publish_training_checkpoint(
        layout=layout,
        checkpoint_payload=payload,
        variant="velocity_fm",
        epoch=epoch,
        step=step,
        loss=loss,
        config_identity=_config_identity(),
    )


def test_checkpoint_publication_validates_payload_before_writing(tmp_path, monkeypatch):
    layout = training_run_layout(
        temporary_runtime(tmp_path),
        task_id="task_ab12",
        variant_id="velocity_fm",
    )
    payload = _valid_training_checkpoint_payload(epoch=1, step=4, loss=0.25)
    payload.pop("ema_model_state")

    def fail_if_called(*args, **kwargs):
        raise AssertionError("invalid checkpoint must not be written")

    monkeypatch.setattr(training_checkpointing_module, "write_torch_atomic", fail_if_called)

    with pytest.raises(ValueError, match="missing fields.*ema_model_state"):
        training_checkpointing_module.publish_training_checkpoint(
            layout=layout,
            checkpoint_payload=payload,
            variant="velocity_fm",
            epoch=1,
            step=4,
            loss=0.25,
            config_identity=_config_identity(),
        )

    assert not layout.epochs_root.exists()


def test_checkpoint_publication_does_not_read_back_checkpoint_or_manifest(
    tmp_path,
    monkeypatch,
):
    layout = training_run_layout(
        temporary_runtime(tmp_path),
        task_id="task_ab12",
        variant_id="velocity_fm",
    )

    def fail_if_called(*args, **kwargs):
        raise AssertionError("producer must not read back files it just wrote")

    monkeypatch.setattr(training_checkpointing_module.torch, "load", fail_if_called)
    monkeypatch.setattr(
        training_checkpointing_module,
        "load_checkpoint_manifest",
        fail_if_called,
    )

    checkpoint_path = _publish_epoch(layout, 1)

    assert checkpoint_path.is_file()
    assert layout.checkpoint_manifest_path(1).is_file()


def test_training_checkpoint_publication_resolves_output_root_alias(tmp_path):
    physical_runs_root = tmp_path / "physical-runs"
    physical_runs_root.mkdir()
    lexical_runs_root = tmp_path / "lexical-runs"
    lexical_runs_root.symlink_to(physical_runs_root, target_is_directory=True)
    runtime = temporary_runtime(tmp_path)
    runtime = replace(runtime, runs_root=lexical_runs_root)
    layout = training_run_layout(
        runtime,
        task_id="task_ab12",
        variant_id="velocity_fm",
    )

    checkpoint_path = _publish_epoch(layout, 1)

    assert checkpoint_path == (
        physical_runs_root.resolve()
        / "experiments/task_ab12/velocity_fm/epochs/epoch_0001/checkpoint.pt"
    )
    assert layout.checkpoint_manifest_path(1).is_file()
    assert json.loads(layout.checkpoint_manifest_path(1).read_text(encoding="utf-8"))[
        "checkpoint"
    ] == "checkpoint.pt"


@pytest.mark.parametrize(
    "fault_boundary",
    [
        "write_torch_atomic",
        "write_checkpoint_manifest",
        "_publish_epoch_directory",
    ],
)
def test_checkpoint_publication_cleans_only_unpublished_staging_on_pre_rename_failure(
    tmp_path,
    monkeypatch,
    fault_boundary,
):
    layout = training_run_layout(
        temporary_runtime(tmp_path),
        task_id="task_ab12",
        variant_id="velocity_fm",
    )
    _publish_epoch(layout, 1)

    def fail(*args, **kwargs):
        raise OSError(f"injected {fault_boundary} failure")

    monkeypatch.setattr(training_checkpointing_module, fault_boundary, fail)

    with pytest.raises(OSError, match="injected"):
        _publish_epoch(layout, 2)

    assert not layout.epoch_dir(2).exists()
    assert not list(layout.epochs_root.glob(".epoch_0002.staging-*"))
    assert layout.checkpoint_path(1).is_file()
    assert layout.checkpoint_manifest_path(1).is_file()

def test_immutable_checkpoint_publication_refuses_existing_epoch_without_clobber(tmp_path):
    layout = training_run_layout(
        temporary_runtime(tmp_path),
        task_id="task_ab12",
        variant_id="velocity_fm",
    )
    _publish_epoch(layout, 1)
    checkpoint_before = layout.checkpoint_path(1).read_bytes()
    manifest_before = layout.checkpoint_manifest_path(1).read_bytes()

    with pytest.raises(FileExistsError, match="epoch"):
        _publish_epoch(layout, 1)

    assert layout.checkpoint_path(1).read_bytes() == checkpoint_before
    assert layout.checkpoint_manifest_path(1).read_bytes() == manifest_before
    assert not list(layout.epochs_root.glob(".epoch_0001.staging-*"))
