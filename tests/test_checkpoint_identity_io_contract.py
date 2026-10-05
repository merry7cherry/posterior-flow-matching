from __future__ import annotations

import hashlib
import json
from copy import deepcopy

import pytest
import torch

from conftest import temporary_runtime
from pfm.core.checkpointing import (
    validate_checkpoint_config_identity,
    validate_checkpoint_config_identity_binding,
)
from pfm.core.config import ExperimentConfig, shared_training_protocol_sha256
from pfm.core.layout import training_run_layout
from pfm.core.rng_state import capture_rng_state
from pfm.evaluation import checkpoint_loading
from pfm.training import checkpointing as checkpointing_module
from pfm.training.checkpointing import build_training_checkpoint


def _runtime_identity() -> dict[str, object]:
    return {
        "project_name": "pfm",
        "git_commit": "a" * 40,
        "rank": 0,
        "world_size": 1,
    }


def _checkpoint_case(tmp_path) -> dict[str, object]:
    config = ExperimentConfig()
    model = torch.nn.Linear(1, 1)
    optimizer = torch.optim.AdamW(model.parameters())
    rng_state = capture_rng_state()
    rng_state["torch_cuda"] = None
    payload = build_training_checkpoint(
        model=model,
        optimizer=optimizer,
        variant="velocity_fm",
        config=config,
        epoch=1,
        step=4,
        loss=0.25,
        ema_model=model,
        distributed_world_size=1,
        runtime_identity=_runtime_identity(),
        dataset_scientific_identity_sha256="d" * 64,
        rng_state_by_rank=[rng_state],
    )
    return {
        "layout": training_run_layout(
            temporary_runtime(tmp_path),
            task_id="task_ab12",
            variant_id="velocity_fm",
        ),
        "payload": payload,
        "config_identity": {
            "dataset": config.dataset,
            "architecture": config.architecture,
            "variant": "velocity_fm",
            "runtime_identity": _runtime_identity(),
            "dataset_scientific_identity_sha256": "d" * 64,
            "shared_training_protocol_sha256": shared_training_protocol_sha256(config),
        },
    }


def _publish(case: dict[str, object], config_identity: dict[str, object] | None = None):
    payload = case["payload"]
    assert isinstance(payload, dict)
    return checkpointing_module.publish_training_checkpoint(
        layout=case["layout"],
        checkpoint_payload=payload,
        variant="velocity_fm",
        epoch=1,
        step=4,
        loss=0.25,
        config_identity=(
            deepcopy(case["config_identity"])
            if config_identity is None
            else config_identity
        ),
    )


def _replace_published_config_identity(
    case: dict[str, object],
    config_identity: dict[str, object],
) -> None:
    layout = case["layout"]
    manifest_path = layout.checkpoint_manifest_path(1)
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    manifest["config_identity"] = deepcopy(config_identity)
    manifest_path.write_text(json.dumps(manifest) + "\n", encoding="utf-8")


def _exercise_boundary(
    case: dict[str, object],
    *,
    boundary: str,
    config_identity: dict[str, object],
) -> None:
    if boundary == "publication":
        _publish(case, config_identity)
        return

    _publish(case)
    _replace_published_config_identity(case, config_identity)
    layout = case["layout"]
    if boundary == "explicit_resume":
        checkpointing_module.load_canonical_resume_checkpoint(
            layout,
            layout.checkpoint_manifest_path(1),
        )
        return
    if boundary == "evaluation":
        bound = checkpoint_loading.resolve_bound_epoch_checkpoint(
            layout.variant_root,
            1,
        )
        checkpoint_loading.load_bound_training_checkpoint(bound)
        return
    raise AssertionError(f"unknown checkpoint boundary: {boundary}")


def _invalid_identity(case: dict[str, object], mutation: str) -> dict[str, object]:
    identity = deepcopy(case["config_identity"])
    if mutation == "missing_key":
        identity.pop("shared_training_protocol_sha256")
    elif mutation == "unknown_key":
        identity["checkpoint_schema"] = "legacy"
    elif mutation == "invalid_type":
        identity["dataset"] = 7
    elif mutation == "empty_value":
        identity["architecture"] = ""
    elif mutation == "invalid_digest":
        identity["shared_training_protocol_sha256"] = "not-a-sha256"
    elif mutation == "invalid_runtime_identity":
        runtime_identity = dict(identity["runtime_identity"])
        runtime_identity.pop("rank")
        identity["runtime_identity"] = runtime_identity
    else:
        raise AssertionError(f"unknown identity mutation: {mutation}")
    return identity


@pytest.mark.parametrize(
    "mutation",
    (
        "missing_key",
        "unknown_key",
        "invalid_type",
        "empty_value",
        "invalid_digest",
        "invalid_runtime_identity",
    ),
)
def test_checkpoint_config_identity_validator_rejects_each_invalid_shape(
    tmp_path,
    mutation,
):
    case = _checkpoint_case(tmp_path)
    invalid_identity = _invalid_identity(case, mutation)

    with pytest.raises(ValueError):
        validate_checkpoint_config_identity(
            invalid_identity,
            context="checkpoint identity unit test",
        )


def _mismatched_identity(case: dict[str, object], field_name: str) -> dict[str, object]:
    identity = deepcopy(case["config_identity"])
    mismatched_values: dict[str, object] = {
        "dataset": "cifar10",
        "architecture": "edm_unet",
        "variant": "endpoint_fm",
        "runtime_identity": {
            **_runtime_identity(),
            "git_commit": "b" * 40,
        },
        "dataset_scientific_identity_sha256": "e" * 64,
        "shared_training_protocol_sha256": "e" * 64,
    }
    identity[field_name] = mismatched_values[field_name]
    return identity


@pytest.mark.parametrize(
    "field_name",
    (
        "dataset",
        "architecture",
        "variant",
        "runtime_identity",
        "dataset_scientific_identity_sha256",
        "shared_training_protocol_sha256",
    ),
)
def test_checkpoint_config_identity_binding_rejects_each_mismatched_value(
    tmp_path,
    field_name,
):
    case = _checkpoint_case(tmp_path)
    mismatched_identity = _mismatched_identity(case, field_name)
    validated_identity = validate_checkpoint_config_identity(
        mismatched_identity,
        context="checkpoint identity unit test",
    )

    with pytest.raises(ValueError):
        validate_checkpoint_config_identity_binding(
            validated_identity,
            case["payload"],
            context="checkpoint identity unit test",
        )


@pytest.mark.parametrize(
    ("boundary", "invalid_identity_factory"),
    (
        pytest.param(
            "publication",
            lambda case: _invalid_identity(case, "missing_key"),
            id="publication",
        ),
        pytest.param(
            "explicit_resume",
            lambda case: _mismatched_identity(case, "variant"),
            id="explicit-resume",
        ),
        pytest.param(
            "evaluation",
            lambda case: _invalid_identity(case, "invalid_digest"),
            id="evaluation",
        ),
    ),
)
def test_checkpoint_boundaries_reject_invalid_config_identity(
    tmp_path,
    boundary,
    invalid_identity_factory,
):
    case = _checkpoint_case(tmp_path)

    with pytest.raises(ValueError):
        _exercise_boundary(
            case,
            boundary=boundary,
            config_identity=invalid_identity_factory(case),
        )


def _record_completed_sha256(monkeypatch):
    original_sha256 = hashlib.sha256
    completed: list[str] = []

    class RecordingSHA256:
        def __init__(self, data=b"", *args, **kwargs):
            self._delegate = original_sha256(data, *args, **kwargs)
            self._recorded = False

        def update(self, data):
            return self._delegate.update(data)

        def _record(self, value: str) -> None:
            if not self._recorded:
                completed.append(value)
                self._recorded = True

        def digest(self):
            value = self._delegate.digest()
            self._record(value.hex())
            return value

        def hexdigest(self):
            value = self._delegate.hexdigest()
            self._record(value)
            return value

        def copy(self):
            copied = object.__new__(type(self))
            copied._delegate = self._delegate.copy()
            copied._recorded = False
            return copied

        def __getattr__(self, name):
            return getattr(self._delegate, name)

    monkeypatch.setattr(hashlib, "sha256", RecordingSHA256)
    return original_sha256, completed


@pytest.mark.parametrize(
    "operation",
    ("publication", "explicit_resume", "evaluation"),
)
def test_checkpoint_operation_performs_one_full_sha256_traversal(
    tmp_path,
    monkeypatch,
    operation,
):
    case = _checkpoint_case(tmp_path)
    layout = case["layout"]
    if operation != "publication":
        checkpoint_path = _publish(case)

    original_sha256, completed = _record_completed_sha256(monkeypatch)
    if operation == "publication":
        checkpoint_path = _publish(case)
    elif operation == "explicit_resume":
        checkpointing_module.load_canonical_resume_checkpoint(
            layout,
            layout.checkpoint_manifest_path(1),
        )
    elif operation == "evaluation":
        bound = checkpoint_loading.resolve_bound_epoch_checkpoint(
            layout.variant_root,
            1,
        )
        checkpoint_loading.load_bound_training_checkpoint(bound)
    else:
        raise AssertionError(f"unknown checkpoint operation: {operation}")

    checkpoint_sha256 = original_sha256(checkpoint_path.read_bytes()).hexdigest()
    assert completed.count(checkpoint_sha256) == 1


@pytest.mark.parametrize(
    "operation",
    ("explicit_resume", "evaluation"),
)
def test_checkpoint_consumers_reject_bytes_that_do_not_match_manifest(
    tmp_path,
    operation,
):
    case = _checkpoint_case(tmp_path)
    layout = case["layout"]
    checkpoint_path = _publish(case)
    checkpoint = torch.load(checkpoint_path, map_location="cpu")
    checkpoint["model_state"]["tampered"] = torch.tensor([1.0])
    torch.save(checkpoint, checkpoint_path)

    with pytest.raises(ValueError, match="SHA-256|digest"):
        if operation == "explicit_resume":
            checkpointing_module.load_canonical_resume_checkpoint(
                layout,
                layout.checkpoint_manifest_path(1),
            )
        else:
            bound = checkpoint_loading.resolve_bound_epoch_checkpoint(
                layout.variant_root,
                1,
            )
            checkpoint_loading.load_bound_training_checkpoint(bound)
