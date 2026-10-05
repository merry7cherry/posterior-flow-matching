from __future__ import annotations

import hashlib
import hmac
import json
import os
import shutil
import tempfile
from collections.abc import Mapping
from pathlib import Path

import torch

from pfm.core.atomic_io import write_json_atomic, write_torch_atomic
from pfm.core.checkpointing import (
    load_checkpoint_manifest,
    load_training_checkpoint_with_sha256,
    require_sha256,
    validate_checkpoint_config_identity,
    validate_checkpoint_config_identity_binding,
    validate_runtime_identity_payload,
    validate_training_checkpoint_payload,
)
from pfm.core.config import experiment_config_to_mapping
from pfm.core.layout import epoch_id
from pfm.core.rng_state import validate_rng_state_list
from pfm.core.execution import require_canonical_path
from pfm.core.validation import (
    require_finite_number,
    require_non_negative_integer,
    require_positive_integer,
)


RECOVERY_LATEST_SCHEMA = "pfm_training_recovery_latest_v1"
RECOVERY_LATEST_KEYS = frozenset(
    {"schema", "manifest", "epoch", "step", "checkpoint_sha256"}
)


def validate_resume_runtime_identity(
    checkpoint: Mapping[str, object],
    current_runtime_identity: Mapping[str, object],
) -> None:
    checkpoint_identity = validate_runtime_identity_payload(
        checkpoint["runtime_identity"],
        context="resume checkpoint",
    )
    current_identity = validate_runtime_identity_payload(
        current_runtime_identity,
        context="current job",
    )
    for field_name in ("project_name", "git_commit"):
        if checkpoint_identity[field_name] != current_identity[field_name]:
            raise ValueError(
                f"resume checkpoint runtime_identity {field_name} differs from current job"
            )


def build_training_checkpoint(
    *,
    model: torch.nn.Module,
    optimizer: torch.optim.Optimizer,
    variant: str,
    config,
    epoch: int,
    step: int,
    loss: float,
    ema_model: torch.nn.Module,
    distributed_world_size: int,
    runtime_identity: Mapping[str, object],
    dataset_scientific_identity_sha256: str,
    rng_state_by_rank: list[dict[str, object]] | None = None,
) -> dict[str, object]:
    validated_variant = _require_non_empty_string(
        variant,
        "variant",
        context="training checkpoint",
    )
    validated_epoch = require_positive_integer(
        epoch,
        "epoch",
        context="training checkpoint",
    )
    validated_step = require_non_negative_integer(
        step,
        "step",
        context="training checkpoint",
    )
    if not isinstance(ema_model, torch.nn.Module):
        raise TypeError("ema_model must be a torch module")
    resolved_world_size = require_positive_integer(
        distributed_world_size,
        "distributed_world_size",
        context="training checkpoint",
    )
    if rng_state_by_rank is None:
        raise ValueError("rng_state_by_rank is required when building a training checkpoint")
    validated_rng_state_by_rank = validate_rng_state_list(
        rng_state_by_rank,
        world_size=resolved_world_size,
        context="training checkpoint",
    )
    validated_runtime_identity = validate_runtime_identity_payload(
        runtime_identity,
        context="training checkpoint",
    )
    validated_dataset_identity_sha256 = require_sha256(
        dataset_scientific_identity_sha256,
        "training checkpoint dataset scientific identity",
    )
    if int(validated_runtime_identity["world_size"]) != resolved_world_size:
        raise ValueError(
            "training checkpoint runtime_identity world_size must match distributed_world_size"
        )

    payload: dict[str, object] = {
        "variant": validated_variant,
        "training_config": experiment_config_to_mapping(config),
        "epoch": validated_epoch,
        "step": validated_step,
        "loss": require_finite_number(
            loss,
            "field 'loss'",
            context="training checkpoint",
        ),
        "model_state": model.state_dict(),
        "ema_model_state": ema_model.state_dict(),
        "optimizer_state": optimizer.state_dict(),
        "rng_state_by_rank": validated_rng_state_by_rank,
        "distributed_world_size": resolved_world_size,
        "runtime_identity": validated_runtime_identity,
        "dataset_scientific_identity_sha256": validated_dataset_identity_sha256,
    }
    return payload


def write_checkpoint_manifest(
    *,
    path: str | Path,
    checkpoint_path: str | Path,
    task_id: str,
    variant_id: str,
    epoch: int,
    step: int,
    loss: float,
    checkpoint_sha256: str,
    config_identity: Mapping[str, object],
) -> dict[str, object]:
    payload = _manifest_payload(
        manifest_root=Path(path).parent,
        checkpoint_path=checkpoint_path,
        task_id=task_id,
        variant_id=variant_id,
        epoch=epoch,
        step=step,
        loss=loss,
        checkpoint_sha256=checkpoint_sha256,
        config_identity=config_identity,
    )
    write_json_atomic(path, payload)
    return payload


def publish_training_checkpoint(
    *,
    layout,
    checkpoint_payload: Mapping[str, object],
    variant: str,
    epoch: int,
    step: int,
    loss: float,
    config_identity: Mapping[str, object],
) -> Path:
    """Publish one immutable epoch while the caller owns the variant lifecycle."""
    (
        validated_checkpoint,
        validated_config_identity,
        validated_epoch,
        validated_step,
        validated_loss,
    ) = _validate_checkpoint_publication_inputs(
        checkpoint_payload=checkpoint_payload,
        variant=variant,
        epoch=epoch,
        step=step,
        loss=loss,
        config_identity=config_identity,
    )

    epochs_root = layout.epochs_root
    epochs_root.mkdir(parents=True, exist_ok=True)
    target_epoch_dir = layout.epoch_dir(validated_epoch)
    if os.path.lexists(target_epoch_dir):
        raise FileExistsError(f"canonical epoch directory already exists: {target_epoch_dir}")
    staging_root = Path(
        tempfile.mkdtemp(
            prefix=f".{epoch_id(validated_epoch)}.staging-",
            dir=epochs_root,
        )
    )
    staged_checkpoint = staging_root / "checkpoint.pt"
    staged_manifest = staging_root / "checkpoint_manifest.json"
    published = False
    try:
        write_torch_atomic(staged_checkpoint, validated_checkpoint)
        staged_checkpoint_sha256 = _sha256_file(staged_checkpoint)
        write_checkpoint_manifest(
            path=staged_manifest,
            checkpoint_path=staged_checkpoint,
            task_id=layout.task_id,
            variant_id=layout.variant_id,
            epoch=validated_epoch,
            step=validated_step,
            loss=validated_loss,
            checkpoint_sha256=staged_checkpoint_sha256,
            config_identity=validated_config_identity,
        )
        _publish_epoch_directory(staging_root, target_epoch_dir)
        published = True
        return target_epoch_dir / "checkpoint.pt"
    finally:
        if not published and staging_root.exists():
            shutil.rmtree(staging_root, ignore_errors=True)


def publish_recovery_checkpoint(
    *,
    layout,
    checkpoint_payload: Mapping[str, object],
    variant: str,
    epoch: int,
    step: int,
    loss: float,
    config_identity: Mapping[str, object],
) -> Path:
    """Atomically advance the one-epoch recovery pointer without exposing it to consumers."""
    (
        validated_checkpoint,
        validated_config_identity,
        validated_epoch,
        validated_step,
        validated_loss,
    ) = _validate_checkpoint_publication_inputs(
        checkpoint_payload=checkpoint_payload,
        variant=variant,
        epoch=epoch,
        step=step,
        loss=loss,
        config_identity=config_identity,
    )
    generations_root = layout.recovery_generations_root
    generations_root.mkdir(parents=True, exist_ok=True)
    staging_root = Path(
        tempfile.mkdtemp(
            prefix=f".{epoch_id(validated_epoch)}.staging-",
            dir=generations_root,
        )
    )
    generation_name = staging_root.name.removeprefix(".").replace(".staging-", "-")
    generation_root = generations_root / generation_name
    staged_checkpoint = staging_root / "checkpoint.pt"
    staged_manifest = staging_root / "checkpoint_manifest.json"
    published = False
    try:
        write_torch_atomic(staged_checkpoint, validated_checkpoint)
        checkpoint_sha256 = _sha256_file(staged_checkpoint)
        write_checkpoint_manifest(
            path=staged_manifest,
            checkpoint_path=staged_checkpoint,
            task_id=layout.task_id,
            variant_id=layout.variant_id,
            epoch=validated_epoch,
            step=validated_step,
            loss=validated_loss,
            checkpoint_sha256=checkpoint_sha256,
            config_identity=validated_config_identity,
        )
        os.rename(staging_root, generation_root)
        published = True
        manifest_path = generation_root / "checkpoint_manifest.json"
        _publish_recovery_latest_pointer(layout, manifest_path)
        _remove_superseded_recovery_generations(
            generations_root,
            keep=generation_root,
        )
        return generation_root / "checkpoint.pt"
    finally:
        if not published and staging_root.exists():
            shutil.rmtree(staging_root, ignore_errors=True)


def publish_recovery_pointer_to_immutable_checkpoint(
    *,
    layout,
    epoch: int,
) -> None:
    """Advance latest recovery to an already-published immutable milestone."""
    _publish_recovery_latest_pointer(
        layout,
        layout.checkpoint_manifest_path(epoch),
    )
    _remove_superseded_recovery_generations(
        layout.recovery_generations_root,
        keep=None,
    )


def load_canonical_resume_checkpoint(
    layout,
    reference: str | Path,
) -> tuple[Path, dict[str, object]]:
    reference_path = Path(os.path.abspath(Path(reference)))
    if reference_path.name == "latest.json":
        canonical_latest = require_canonical_path(
            reference_path,
            layout.variant_root,
            name="resume recovery reference",
        )
        if canonical_latest != layout.recovery_latest_path:
            raise ValueError("resume latest.json must use the canonical recovery path")
        manifest_path, latest = _load_recovery_latest_pointer(layout)
        checkpoint_path, checkpoint, manifest = _load_resume_manifest(
            layout,
            manifest_path,
            require_immutable=False,
        )
        for field_name in ("epoch", "step", "checkpoint_sha256"):
            if latest[field_name] != manifest[field_name]:
                raise ValueError(
                    f"recovery latest {field_name} does not match checkpoint manifest"
                )
        return checkpoint_path, checkpoint
    if reference_path.name != "checkpoint_manifest.json":
        raise ValueError(
            "resume reference must be checkpoint_manifest.json or canonical recovery/latest.json"
        )
    try:
        canonical_reference = require_canonical_path(
            reference_path,
            layout.variant_root,
            name="resume reference",
        )
    except ValueError as exc:
        raise ValueError(
            "resume reference must be within the same canonical variant root"
        ) from exc
    checkpoint_path, checkpoint, _ = _load_resume_manifest(
        layout,
        canonical_reference,
        require_immutable=True,
    )
    return checkpoint_path, checkpoint


def _load_resume_manifest(
    layout,
    manifest_path: Path,
    *,
    require_immutable: bool,
) -> tuple[Path, dict[str, object], dict[str, object]]:
    manifest = load_checkpoint_manifest(manifest_path)
    if manifest["task_id"] != layout.task_id:
        raise ValueError("resume manifest task_id does not match canonical task")
    if manifest["variant_id"] != layout.variant_id:
        raise ValueError("resume manifest variant_id does not match canonical variant")
    manifest_epoch = manifest["epoch"]
    if type(manifest_epoch) is not int or manifest_epoch < 1:
        raise ValueError("resume manifest epoch must be a positive integer")
    expected_epoch_id = epoch_id(manifest_epoch)
    if manifest["epoch_id"] != expected_epoch_id:
        raise ValueError("resume manifest epoch_id does not match epoch")
    expected_manifest_path = layout.checkpoint_manifest_path(manifest_epoch)
    if require_immutable:
        if manifest_path != expected_manifest_path:
            raise ValueError(
                "resume checkpoint_manifest.json must use the canonical epoch directory"
            )
        expected_checkpoint_path = layout.checkpoint_path(manifest_epoch)
    else:
        canonical_manifest = require_canonical_path(
            manifest_path,
            layout.variant_root,
            name="recovery checkpoint manifest",
        )
        immutable = canonical_manifest == expected_manifest_path
        under_recovery = canonical_manifest.parent.parent == layout.recovery_generations_root
        if not immutable and not under_recovery:
            raise ValueError(
                "recovery latest must bind an immutable epoch or recovery generation"
            )
        expected_checkpoint_path = canonical_manifest.with_name("checkpoint.pt")
    resolved_checkpoint_path = _manifest_checkpoint_path(manifest_path, manifest)
    if resolved_checkpoint_path != expected_checkpoint_path.resolve():
        raise ValueError(
            "resume manifest checkpoint does not bind its canonical checkpoint"
        )
    checkpoint, actual_checkpoint_sha256 = load_training_checkpoint_with_sha256(
        expected_checkpoint_path
    )
    if not hmac.compare_digest(
        actual_checkpoint_sha256,
        str(manifest["checkpoint_sha256"]),
    ):
        raise ValueError("resume checkpoint bytes do not match producer manifest SHA-256")
    for field_name in ("epoch", "step", "loss"):
        if checkpoint[field_name] != manifest[field_name]:
            raise ValueError(f"resume checkpoint {field_name} does not match manifest")
    validate_checkpoint_config_identity_binding(
        manifest["config_identity"],
        checkpoint,
        context="resume manifest",
    )
    return expected_checkpoint_path, checkpoint, manifest


def _validate_checkpoint_publication_inputs(
    *,
    checkpoint_payload: Mapping[str, object],
    variant: str,
    epoch: int,
    step: int,
    loss: float,
    config_identity: Mapping[str, object],
) -> tuple[dict[str, object], dict[str, object], int, int, float]:
    validate_training_checkpoint_payload(checkpoint_payload)
    validated_checkpoint = dict(checkpoint_payload)
    validated_config_identity = validate_checkpoint_config_identity(
        config_identity,
        context="checkpoint publication",
    )
    validate_checkpoint_config_identity_binding(
        validated_config_identity,
        validated_checkpoint,
        context="checkpoint publication",
    )
    validated_variant = _require_non_empty_string(
        variant,
        "variant",
        context="checkpoint publication",
    )
    validated_epoch = require_positive_integer(
        epoch,
        "epoch",
        context="checkpoint publication",
    )
    validated_step = require_non_negative_integer(
        step,
        "step",
        context="checkpoint publication",
    )
    validated_loss = require_finite_number(
        loss,
        "loss",
        context="checkpoint publication",
    )
    for field_name, expected in (
        ("variant", validated_variant),
        ("epoch", validated_epoch),
        ("step", validated_step),
        ("loss", validated_loss),
    ):
        if validated_checkpoint[field_name] != expected:
            raise ValueError(
                f"checkpoint publication field '{field_name}' does not match checkpoint"
            )
    return (
        validated_checkpoint,
        validated_config_identity,
        validated_epoch,
        validated_step,
        validated_loss,
    )


def _publish_recovery_latest_pointer(layout, manifest_path: Path) -> None:
    manifest = load_checkpoint_manifest(manifest_path)
    canonical_manifest = require_canonical_path(
        manifest_path,
        layout.variant_root,
        name="recovery checkpoint manifest",
    )
    relative_manifest = canonical_manifest.relative_to(layout.variant_root).as_posix()
    write_json_atomic(
        layout.recovery_latest_path,
        {
            "schema": RECOVERY_LATEST_SCHEMA,
            "manifest": relative_manifest,
            "epoch": manifest["epoch"],
            "step": manifest["step"],
            "checkpoint_sha256": manifest["checkpoint_sha256"],
        },
    )


def _load_recovery_latest_pointer(layout) -> tuple[Path, dict[str, object]]:
    path = layout.recovery_latest_path
    if not path.is_file():
        raise FileNotFoundError(f"recovery latest pointer does not exist: {path}")
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError("recovery latest pointer must contain a JSON object")
    unknown = sorted(set(payload) - RECOVERY_LATEST_KEYS)
    missing = sorted(RECOVERY_LATEST_KEYS - set(payload))
    if unknown:
        raise ValueError(f"unknown recovery latest fields: {unknown}")
    if missing:
        raise ValueError(f"missing recovery latest fields: {missing}")
    if payload["schema"] != RECOVERY_LATEST_SCHEMA:
        raise ValueError("recovery latest schema is invalid")
    manifest = payload["manifest"]
    if (
        not isinstance(manifest, str)
        or manifest == ""
        or Path(manifest).is_absolute()
        or ".." in Path(manifest).parts
    ):
        raise ValueError("recovery latest manifest must be a safe relative path")
    require_positive_integer(
        payload["epoch"],
        "epoch",
        context="recovery latest",
    )
    require_non_negative_integer(
        payload["step"],
        "step",
        context="recovery latest",
    )
    require_sha256(
        payload["checkpoint_sha256"],
        "recovery latest checkpoint",
    )
    manifest_path = require_canonical_path(
        layout.variant_root / manifest,
        layout.variant_root,
        name="recovery latest manifest",
    )
    return manifest_path, payload


def _remove_superseded_recovery_generations(
    generations_root: Path,
    *,
    keep: Path | None,
) -> None:
    if not generations_root.exists():
        return
    for candidate in generations_root.iterdir():
        if keep is not None and candidate == keep:
            continue
        if candidate.is_dir() and not candidate.is_symlink():
            shutil.rmtree(candidate, ignore_errors=True)


def _publish_epoch_directory(staging_root: Path, target_epoch_dir: Path) -> None:
    if os.path.lexists(target_epoch_dir):
        raise FileExistsError(f"canonical epoch directory already exists: {target_epoch_dir}")
    os.rename(staging_root, target_epoch_dir)


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _manifest_payload(
    *,
    manifest_root: Path,
    checkpoint_path: str | Path,
    task_id: str,
    variant_id: str,
    epoch: int,
    step: int,
    loss: float,
    checkpoint_sha256: str,
    config_identity: Mapping[str, object],
) -> dict[str, object]:
    return {
        "task_id": str(task_id),
        "variant_id": str(variant_id),
        "epoch": int(epoch),
        "epoch_id": epoch_id(epoch),
        "step": int(step),
        "loss": require_finite_number(
            loss,
            "loss",
            context="checkpoint manifest",
        ),
        "checkpoint": _relative_checkpoint_path(
            checkpoint_path,
            manifest_root=manifest_root,
        ),
        "checkpoint_sha256": require_sha256(
            checkpoint_sha256,
            "checkpoint manifest checkpoint",
        ),
        "config_identity": validate_checkpoint_config_identity(
            config_identity,
            context="checkpoint manifest",
        ),
    }


def _manifest_checkpoint_path(
    manifest_path: Path,
    manifest: Mapping[str, object],
) -> Path:
    return (manifest_path.parent / str(manifest["checkpoint"])).resolve()


def _relative_checkpoint_path(checkpoint_path: str | Path, *, manifest_root: Path) -> str:
    root = manifest_root.resolve()
    checkpoint = Path(checkpoint_path)
    if checkpoint.is_absolute():
        resolved_checkpoint = checkpoint.resolve()
    else:
        resolved_checkpoint = (root / checkpoint).resolve()
    relative = Path(os.path.relpath(resolved_checkpoint, root))
    if relative.is_absolute() or ".." in relative.parts:
        raise ValueError("checkpoint path must stay within the manifest root")
    return relative.as_posix()


def _require_non_empty_string(value: object, field_name: str, *, context: str) -> str:
    if not isinstance(value, str) or value == "":
        raise ValueError(f"{context} field '{field_name}' must be a non-empty string")
    return value
