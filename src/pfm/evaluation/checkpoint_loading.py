from __future__ import annotations

import hashlib
import hmac
import os
from collections.abc import Mapping
from dataclasses import dataclass, replace
from pathlib import Path
from typing import Any

import torch

from pfm.artifacts.consumer_artifact import make_relative_reference
from pfm.core.checkpointing import (
    load_checkpoint_manifest,
    load_training_checkpoint_with_sha256,
    validate_checkpoint_config_identity_binding,
)
from pfm.core.config import (
    ExperimentConfig,
    checkpoint_config_from_mapping,
)
from pfm.core.layout import epoch_id
from pfm.core.sampling_config import SamplingConfig
from pfm.core.inference_config import InferenceExecutionConfig
from pfm.core.runtime import apply_runtime_settings, prepare_compiled_inference
from pfm.core.execution import ExecutionContext, RuntimePaths, require_canonical_path
from pfm.data import identity as dataset_identity
from pfm.data.datasets import validate_dataset_manifest
from pfm.models.factory import build_model, configure_jit_execution


@dataclass(frozen=True)
class BoundEpochCheckpoint:
    epoch_id: str
    epoch_dir: Path
    checkpoint_path: Path
    manifest_path: Path | None
    manifest: dict[str, Any]
    config_identity: dict[str, object]
    checkpoint_runtime_identity: dict[str, object]
    checkpoint_sha256: str


def require_canonical_variant_root(
    variant_root: str | Path,
    experiments_root: str | Path,
) -> Path:
    canonical_experiments_root = Path(os.path.abspath(Path(experiments_root)))
    try:
        root = require_canonical_path(
            variant_root,
            canonical_experiments_root,
            name="variant_root",
        )
    except ValueError as exc:
        raise ValueError(
            f"variant_root must stay within {canonical_experiments_root}"
        ) from exc
    if len(root.relative_to(canonical_experiments_root).parts) != 2:
        raise ValueError(
            "variant_root must be exactly task_id/variant_id below experiments_root"
        )
    return root


def resolve_bound_epoch_checkpoint(
    variant_root: str | Path,
    epoch: int,
) -> BoundEpochCheckpoint:
    root = Path(os.path.abspath(Path(variant_root)))
    clean_epoch_id = epoch_id(epoch)
    epoch_dir = require_canonical_path(
        root / "epochs" / clean_epoch_id,
        root,
        name="epoch_dir",
    )
    checkpoint_path = require_canonical_path(
        epoch_dir / "checkpoint.pt",
        epoch_dir,
        name="checkpoint_path",
    )
    manifest_path = require_canonical_path(
        epoch_dir / "checkpoint_manifest.json",
        epoch_dir,
        name="manifest_path",
    )
    if not checkpoint_path.is_file():
        raise FileNotFoundError(f"checkpoint does not exist: {checkpoint_path}")
    if not manifest_path.is_file():
        raise FileNotFoundError(f"checkpoint_manifest.json is required: {manifest_path}")

    manifest = load_checkpoint_manifest(manifest_path)
    manifest_checkpoint_path = (manifest_path.parent / str(manifest["checkpoint"])).resolve()
    config_identity = dict(manifest["config_identity"])
    if manifest_checkpoint_path.resolve() != checkpoint_path.resolve():
        raise ValueError(
            f"manifest checkpoint {manifest_checkpoint_path} does not match epoch checkpoint {checkpoint_path}"
        )
    if manifest["variant_id"] != root.name:
        raise ValueError(
            f"checkpoint manifest variant_id {manifest['variant_id']!r} "
            f"does not match variant directory {root.name!r}: {manifest_path}"
        )
    if manifest["task_id"] != root.parent.name:
        raise ValueError(
            f"checkpoint manifest task_id {manifest['task_id']!r} "
            f"does not match task directory {root.parent.name!r}: {manifest_path}"
        )
    if manifest["epoch_id"] != clean_epoch_id:
        raise ValueError(
            f"checkpoint manifest epoch_id {manifest['epoch_id']!r} does not match epoch directory {clean_epoch_id!r}"
        )
    if type(manifest["epoch"]) is not int:
        raise ValueError("checkpoint manifest epoch must be an integer")
    if type(manifest["step"]) is not int:
        raise ValueError("checkpoint manifest step must be an integer")
    return BoundEpochCheckpoint(
        epoch_id=clean_epoch_id,
        epoch_dir=epoch_dir,
        checkpoint_path=checkpoint_path,
        manifest_path=manifest_path,
        manifest=dict(manifest),
        config_identity=config_identity,
        checkpoint_runtime_identity=dict(config_identity["runtime_identity"]),
        checkpoint_sha256=str(manifest["checkpoint_sha256"]),
    )


def load_bound_training_checkpoint(
    bound_checkpoint: BoundEpochCheckpoint,
) -> dict[str, object]:
    if not isinstance(bound_checkpoint, BoundEpochCheckpoint):
        raise TypeError("bound_checkpoint must be a BoundEpochCheckpoint")
    checkpoint, actual_sha256 = load_training_checkpoint_with_sha256(
        bound_checkpoint.checkpoint_path
    )
    if not hmac.compare_digest(actual_sha256, bound_checkpoint.checkpoint_sha256):
        raise ValueError(
            "bound checkpoint digest does not match the checkpoint bytes selected during resolution"
        )
    manifest = bound_checkpoint.manifest
    validate_checkpoint_config_identity_binding(
        bound_checkpoint.config_identity,
        checkpoint,
        context="bound checkpoint",
    )
    for field_name in ("epoch", "step", "loss"):
        if checkpoint[field_name] != manifest[field_name]:
            raise ValueError(f"bound checkpoint {field_name} does not match manifest")
    return checkpoint


def validate_bound_checkpoint_dataset(
    bound_checkpoint: BoundEpochCheckpoint,
    runtime: RuntimePaths,
) -> dict[str, object]:
    """Fail on current prepared-dataset mismatch before checkpoint deserialization."""
    if not isinstance(bound_checkpoint, BoundEpochCheckpoint):
        raise TypeError("bound_checkpoint must be a BoundEpochCheckpoint")
    config_identity = bound_checkpoint.config_identity
    dataset = str(config_identity["dataset"])
    expected_digest = str(config_identity["dataset_scientific_identity_sha256"])
    manifest = validate_dataset_manifest(runtime.dataset_root(dataset), dataset)
    current_digest = manifest.get("scientific_identity_sha256")
    if current_digest != expected_digest:
        raise ValueError("checkpoint dataset scientific identity mismatch")
    return manifest


def build_checkpoint_consumer_binding(
    bound_checkpoint: BoundEpochCheckpoint,
    *,
    profile: str,
    consumer_config_path: str | Path,
    job_context: ExecutionContext,
    asset_paths: Mapping[str, str | Path] | None = None,
) -> tuple[dict[str, object], dict[str, object]]:
    """Preflight one bound checkpoint and build the shared consumer identity."""
    if not isinstance(bound_checkpoint, BoundEpochCheckpoint):
        raise TypeError("bound_checkpoint must be a BoundEpochCheckpoint")
    if not isinstance(job_context, ExecutionContext):
        raise TypeError("job_context must be a ExecutionContext")
    config_path = Path(consumer_config_path)
    config_digest = _sha256_file(config_path)
    config_identity = bound_checkpoint.config_identity
    dataset = str(config_identity["dataset"])
    dataset_manifest = validate_bound_checkpoint_dataset(
        bound_checkpoint,
        job_context.runtime,
    )
    dataset_manifest_path = (
        job_context.runtime.dataset_root(dataset) / dataset_identity.DATASET_MANIFEST_NAME
    )
    assets = {
        str(name): make_relative_reference(
            owner="assets",
            path=Path(path),
            root=job_context.runtime.assets_root,
            sha256=_sha256_file(Path(path)),
        )
        for name, path in (asset_paths or {}).items()
    }
    binding = {
        "profile": str(profile),
        "method": str(bound_checkpoint.manifest["variant_id"]),
        "shared_training_protocol_sha256": config_identity[
            "shared_training_protocol_sha256"
        ],
        "dataset_scientific_identity_sha256": config_identity[
            "dataset_scientific_identity_sha256"
        ],
        "consumer_revision": job_context.git_commit,
        "checkpoint_revision": bound_checkpoint.checkpoint_runtime_identity[
            "git_commit"
        ],
        "checkpoint_progress": {
            "epoch": bound_checkpoint.manifest["epoch"],
            "step": bound_checkpoint.manifest["step"],
        },
        "checkpoint": make_relative_reference(
            owner="runs",
            path=bound_checkpoint.checkpoint_path,
            root=job_context.runtime.runs_root,
            sha256=bound_checkpoint.checkpoint_sha256,
        ),
        "dataset": make_relative_reference(
            owner="datasets",
            path=dataset_manifest_path,
            root=job_context.runtime.datasets_root,
            sha256=_sha256_file(dataset_manifest_path),
        ),
        "config": make_relative_reference(
            owner="runs" if config_path.is_relative_to(job_context.runtime.runs_root / "inputs") else "project",
            path=config_path,
            root=job_context.runtime.runs_root if config_path.is_relative_to(job_context.runtime.runs_root / "inputs") else job_context.runtime.project_root,
            sha256=config_digest,
        ),
        "assets": assets,
    }
    return binding, dataset_manifest


def load_model_from_bound_checkpoint(
    bound_checkpoint: BoundEpochCheckpoint,
    device: torch.device,
    *,
    sampling: SamplingConfig | None = None,
) -> tuple[torch.nn.Module, str, ExperimentConfig]:
    checkpoint = load_bound_training_checkpoint(bound_checkpoint)
    config = checkpoint_config_from_mapping(checkpoint["training_config"])
    variant = checkpoint["variant"]
    model = build_checkpoint_inference_model(
        checkpoint, variant, config, device, sampling=sampling
    )
    return model, variant, config


def build_checkpoint_inference_model(
    checkpoint: Mapping[str, object], variant: str, config: ExperimentConfig,
    device: torch.device, *, sampling: SamplingConfig | None = None,
    execution: InferenceExecutionConfig | None = None,
) -> torch.nn.Module:
    """Strict-load the recorded module layout before selecting inference execution."""
    if sampling is not None and execution is not None:
        raise ValueError("specify sampling or execution, not both")
    selected = sampling.execution if sampling is not None else execution
    execution_config = config if selected is None else selected.apply(config)
    # torch.compile adds _orig_mod wrappers to serialized JiT module paths.
    # Keep the recorded layout for strict weight validation, then change only
    # the known execution wrappers. No state key rewriting or missing-key bypass.
    load_config = replace(execution_config, jit_compile=config.jit_compile)
    apply_runtime_settings(execution_config, device)
    if execution_config.jit_compile or execution_config.edm_compile:
        prepare_compiled_inference()
    model = build_model(variant, load_config, device)
    model.load_state_dict(checkpoint["ema_model_state"], strict=True)
    if config.architecture == "jit_transformer" and execution_config.jit_compile != config.jit_compile:
        configure_jit_execution(model, compile=execution_config.jit_compile)
    if hasattr(model, "config"):
        model.config = execution_config
    model.eval()
    return model


def _sha256_file(path: Path) -> str:
    if path.is_symlink() or not path.is_file():
        raise FileNotFoundError(f"critical binding file does not exist: {path}")
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def bind_checkpoint_file(path: str | Path) -> BoundEpochCheckpoint:
    """Bind a standalone training checkpoint without requiring its original run tree."""
    from pfm.core.config import shared_training_protocol_sha256
    path = Path(path).expanduser().resolve()
    checkpoint, digest = load_training_checkpoint_with_sha256(path)
    config = checkpoint_config_from_mapping(checkpoint["training_config"])
    identity = {
        "dataset": config.dataset,
        "architecture": config.architecture,
        "variant": checkpoint["variant"],
        "runtime_identity": checkpoint["runtime_identity"],
        "dataset_scientific_identity_sha256": checkpoint["dataset_scientific_identity_sha256"],
        "shared_training_protocol_sha256": shared_training_protocol_sha256(config),
    }
    progress = {key: checkpoint[key] for key in ("epoch", "step", "loss")}
    return BoundEpochCheckpoint(
        epoch_id=epoch_id(checkpoint["epoch"]), epoch_dir=path.parent,
        checkpoint_path=path, manifest_path=None, manifest=progress,
        config_identity=identity,
        checkpoint_runtime_identity=dict(checkpoint["runtime_identity"]),
        checkpoint_sha256=digest,
    )
