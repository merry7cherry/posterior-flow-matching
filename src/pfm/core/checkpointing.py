from __future__ import annotations

import hashlib
import hmac
import json
import re
from collections.abc import Mapping
from pathlib import Path

import torch

from pfm.core.config import checkpoint_config_from_mapping, shared_training_protocol_sha256
from pfm.core.layout import epoch_id
from pfm.core.rng_state import validate_rng_state_list
from pfm.core.validation import (
    require_finite_number,
    require_non_negative_integer,
    require_positive_integer,
)


CHECKPOINT_MANIFEST_KEYS = frozenset(
    {
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
)

CHECKPOINT_CONFIG_IDENTITY_KEYS = frozenset(
    {
        "dataset",
        "architecture",
        "variant",
        "runtime_identity",
        "dataset_scientific_identity_sha256",
        "shared_training_protocol_sha256",
    }
)

TRAINING_CHECKPOINT_REQUIRED_KEYS = frozenset(
    {
        "variant",
        "training_config",
        "epoch",
        "step",
        "loss",
        "model_state",
        "ema_model_state",
        "optimizer_state",
        "rng_state_by_rank",
        "distributed_world_size",
        "runtime_identity",
        "dataset_scientific_identity_sha256",
    }
)
RUNTIME_IDENTITY_REQUIRED_KEYS = frozenset(
    {"project_name", "git_commit", "rank", "world_size"}
)


def require_sha256(value: object, context: str) -> str:
    if (
        not isinstance(value, str)
        or len(value) != 64
        or any(character not in "0123456789abcdef" for character in value)
    ):
        raise ValueError(f"{context} SHA-256 must be 64 lowercase hexadecimal characters")
    return value


def _require_non_empty_string(value: object, field_name: str, *, context: str) -> str:
    if not isinstance(value, str) or value == "":
        raise ValueError(f"{context} field '{field_name}' must be a non-empty string")
    return value


def _require_mapping(value: object, field_name: str, *, context: str) -> Mapping[str, object]:
    if not isinstance(value, Mapping):
        raise ValueError(f"{context} field '{field_name}' must be a mapping")
    return value


def validate_runtime_identity_payload(identity: object, *, context: str) -> dict[str, object]:
    if not isinstance(identity, Mapping):
        raise ValueError(f"{context} runtime_identity must be a mapping")
    actual_keys = set(identity)
    unknown = sorted(actual_keys - RUNTIME_IDENTITY_REQUIRED_KEYS)
    if unknown:
        raise ValueError(f"{context} runtime_identity has unknown fields: {unknown}")
    missing = sorted(RUNTIME_IDENTITY_REQUIRED_KEYS - actual_keys)
    if missing:
        raise ValueError(f"{context} runtime_identity is missing fields: {missing}")
    project_name = identity["project_name"]
    git_commit = identity["git_commit"]
    rank = identity["rank"]
    world_size = identity["world_size"]
    if (
        not isinstance(project_name, str)
        or project_name in {"", ".", ".."}
        or project_name != project_name.strip()
        or "/" in project_name
        or "\\" in project_name
    ):
        raise ValueError(
            f"{context} runtime_identity project_name must be a non-empty single path component"
        )
    if git_commit is not None and (
        not isinstance(git_commit, str)
        or len(git_commit) != 40
        or any(character not in "0123456789abcdef" for character in git_commit)
    ):
        raise ValueError(
            f"{context} runtime_identity git_commit must be a lowercase 40-character hex commit"
        )
    if type(rank) is not int or rank < 0:
        raise ValueError(f"{context} runtime_identity rank must be a non-negative integer")
    if type(world_size) is not int or world_size < 1:
        raise ValueError(f"{context} runtime_identity world_size must be an integer >= 1")
    if rank >= world_size:
        raise ValueError(f"{context} runtime_identity rank must be less than world_size")
    return dict(identity)


def validate_checkpoint_config_identity(
    identity: object,
    *,
    context: str,
) -> dict[str, object]:
    if not isinstance(identity, Mapping):
        raise ValueError(f"{context} config_identity must be a mapping")
    actual_keys = set(identity)
    unknown = sorted(actual_keys - CHECKPOINT_CONFIG_IDENTITY_KEYS)
    if unknown:
        raise ValueError(f"{context} config_identity has unknown fields: {unknown}")
    missing = sorted(CHECKPOINT_CONFIG_IDENTITY_KEYS - actual_keys)
    if missing:
        raise ValueError(f"{context} config_identity is missing fields: {missing}")
    validated = {
        field_name: _require_non_empty_string(
            identity[field_name],
            field_name,
            context=f"{context} config_identity",
        )
        for field_name in ("dataset", "architecture", "variant")
    }
    validated["runtime_identity"] = validate_runtime_identity_payload(
        identity["runtime_identity"],
        context=f"{context} config_identity",
    )
    validated["dataset_scientific_identity_sha256"] = require_sha256(
        identity["dataset_scientific_identity_sha256"],
        f"{context} config_identity dataset scientific identity",
    )
    validated["shared_training_protocol_sha256"] = require_sha256(
        identity["shared_training_protocol_sha256"],
        f"{context} config_identity shared training protocol",
    )
    return validated


def validate_checkpoint_config_identity_binding(
    identity: Mapping[str, object],
    checkpoint: Mapping[str, object],
    *,
    context: str,
) -> dict[str, object]:
    """Bind an identity already validated at its manifest/publication boundary."""
    validated = dict(identity)
    training_config = _require_mapping(
        checkpoint["training_config"],
        "training_config",
        context=context,
    )
    for field_name in ("dataset", "architecture"):
        if training_config.get(field_name) != validated[field_name]:
            raise ValueError(
                f"{context} config_identity {field_name} does not match "
                "checkpoint training_config"
            )
    if checkpoint["variant"] != validated["variant"]:
        raise ValueError(f"{context} config_identity variant does not match checkpoint")
    if checkpoint["runtime_identity"] != validated["runtime_identity"]:
        raise ValueError(f"{context} config_identity runtime_identity does not match checkpoint")
    if not hmac.compare_digest(
        str(checkpoint["dataset_scientific_identity_sha256"]),
        str(validated["dataset_scientific_identity_sha256"]),
    ):
        raise ValueError(
            f"{context} config_identity dataset scientific identity does not match checkpoint"
        )
    config = checkpoint_config_from_mapping(training_config)
    expected_protocol_sha256 = shared_training_protocol_sha256(
        config, legacy_edm=config.architecture == "edm_ddpmpp" and "edm_compile" not in training_config,
    )
    if not hmac.compare_digest(
        expected_protocol_sha256,
        str(validated["shared_training_protocol_sha256"]),
    ):
        raise ValueError(
            f"{context} config_identity shared training protocol does not match checkpoint"
        )
    return validated


def validate_training_checkpoint_payload(
    checkpoint: Mapping[str, object],
) -> None:
    if not isinstance(checkpoint, Mapping):
        raise ValueError("training checkpoint payload must be a mapping")
    actual_keys = set(checkpoint)
    unknown = sorted(actual_keys - TRAINING_CHECKPOINT_REQUIRED_KEYS)
    if unknown:
        raise ValueError(f"training checkpoint has unknown fields: {unknown}")
    missing = sorted(TRAINING_CHECKPOINT_REQUIRED_KEYS - actual_keys)
    if missing:
        raise ValueError(f"training checkpoint has missing fields: {missing}")

    require_finite_number(
        checkpoint["loss"],
        "field 'loss'",
        context="training checkpoint",
    )
    _require_non_empty_string(
        checkpoint["variant"],
        "variant",
        context="training checkpoint",
    )
    require_positive_integer(checkpoint["epoch"], "epoch", context="training checkpoint")
    require_non_negative_integer(checkpoint["step"], "step", context="training checkpoint")
    _require_mapping(
        checkpoint["training_config"],
        "training_config",
        context="training checkpoint",
    )
    _require_mapping(checkpoint["model_state"], "model_state", context="training checkpoint")
    _require_mapping(
        checkpoint["ema_model_state"],
        "ema_model_state",
        context="training checkpoint",
    )
    _require_mapping(
        checkpoint["optimizer_state"],
        "optimizer_state",
        context="training checkpoint",
    )
    distributed_world_size = checkpoint["distributed_world_size"]
    if type(distributed_world_size) is not int or distributed_world_size < 1:
        raise ValueError(
            "training checkpoint field 'distributed_world_size' must be an integer >= 1"
        )
    runtime_identity = validate_runtime_identity_payload(
        checkpoint["runtime_identity"],
        context="training checkpoint",
    )
    require_sha256(
        checkpoint["dataset_scientific_identity_sha256"],
        "training checkpoint dataset scientific identity",
    )
    if int(runtime_identity["world_size"]) != distributed_world_size:
        raise ValueError(
            "training checkpoint runtime_identity world_size must match distributed_world_size"
        )
    validate_rng_state_list(
        checkpoint["rng_state_by_rank"],
        world_size=distributed_world_size,
        context="training checkpoint",
    )


def validate_checkpoint_dataset_binding(
    checkpoint: Mapping[str, object],
    current_scientific_identity_sha256: str,
) -> None:
    """Compare the compact binding without touching model or optimizer state."""
    if not isinstance(checkpoint, Mapping):
        raise ValueError("checkpoint dataset binding must be read from a mapping")
    expected = require_sha256(
        checkpoint.get("dataset_scientific_identity_sha256"),
        "checkpoint dataset scientific identity",
    )
    current = require_sha256(
        current_scientific_identity_sha256,
        "current dataset scientific identity",
    )
    if not hmac.compare_digest(expected, current):
        raise ValueError("checkpoint dataset scientific identity mismatch")


def load_training_checkpoint_with_sha256(
    path: str | Path,
) -> tuple[dict[str, object], str]:
    """Deserialize and hash the same open checkpoint file descriptor."""
    checkpoint_path = Path(path)
    digest = hashlib.sha256()
    with checkpoint_path.open("rb") as handle:
        payload = torch.load(handle, map_location="cpu")
        handle.seek(0)
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    if not isinstance(payload, Mapping):
        raise ValueError("training checkpoint payload must be a mapping")
    checkpoint = dict(payload)
    validate_training_checkpoint_payload(checkpoint)
    return checkpoint, digest.hexdigest()


def load_checkpoint_manifest(path: str | Path) -> dict[str, object]:
    manifest_path = Path(path)
    if not manifest_path.exists():
        raise FileNotFoundError(f"checkpoint manifest does not exist: {manifest_path}")
    payload = json.loads(manifest_path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError("checkpoint manifest must contain a JSON object")
    validate_checkpoint_manifest(payload, manifest_root=manifest_path.parent)
    return payload


def validate_checkpoint_manifest(manifest: Mapping[str, object], *, manifest_root: Path) -> Path:
    actual_keys = set(manifest)
    unknown = sorted(actual_keys - CHECKPOINT_MANIFEST_KEYS)
    if unknown:
        raise ValueError(f"unknown checkpoint manifest fields: {unknown}")
    missing = sorted(CHECKPOINT_MANIFEST_KEYS - actual_keys)
    if missing:
        raise ValueError(f"missing checkpoint manifest fields: {missing}")

    if not isinstance(manifest["checkpoint"], str) or not manifest["checkpoint"]:
        raise ValueError("checkpoint manifest field 'checkpoint' is missing or invalid")
    validate_checkpoint_config_identity(manifest["config_identity"], context="checkpoint manifest")
    if not isinstance(manifest["task_id"], str) or not manifest["task_id"]:
        raise ValueError("checkpoint manifest field 'task_id' is missing or invalid")
    if not isinstance(manifest["variant_id"], str) or not manifest["variant_id"]:
        raise ValueError("checkpoint manifest field 'variant_id' is missing or invalid")
    if type(manifest["epoch"]) is not int or manifest["epoch"] < 1:
        raise ValueError("checkpoint manifest field 'epoch' must be a positive integer")
    if not isinstance(manifest["epoch_id"], str) or not manifest["epoch_id"]:
        raise ValueError("checkpoint manifest field 'epoch_id' is missing or invalid")
    expected_epoch_id = epoch_id(manifest["epoch"])
    if manifest["epoch_id"] != expected_epoch_id:
        raise ValueError(
            f"checkpoint manifest epoch_id must match epoch: expected {expected_epoch_id!r}, "
            f"got {manifest['epoch_id']!r}"
        )
    if type(manifest["step"]) is not int or manifest["step"] < 0:
        raise ValueError("checkpoint manifest field 'step' must be a non-negative integer")
    require_finite_number(manifest["loss"], "loss", context="checkpoint manifest")

    checkpoint_path = _resolve_manifest_checkpoint(manifest_root, manifest)
    require_sha256(manifest["checkpoint_sha256"], "checkpoint manifest checkpoint")
    if not checkpoint_path.is_file():
        raise FileNotFoundError(f"checkpoint manifest target does not exist: {checkpoint_path}")
    return checkpoint_path


def _resolve_manifest_checkpoint(
    manifest_root: Path,
    manifest: Mapping[str, object],
) -> Path:
    checkpoint_path = Path(str(manifest["checkpoint"]))
    if checkpoint_path.is_absolute():
        raise ValueError("checkpoint path must be relative")
    if ".." in checkpoint_path.parts:
        raise ValueError("checkpoint path must stay within the manifest root")
    return (manifest_root / checkpoint_path).resolve()
