from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass
import hashlib
import json
from pathlib import Path
import shutil

import torch
from torch import nn
from tqdm import tqdm

from pfm.core.atomic_io import write_json_atomic, write_torch_atomic
from pfm.core.config import ExperimentConfig
from pfm.core.immutable_artifact import prepare_immutable_directory, publish_immutable_directory
from pfm.core.runtime import autocast_context, set_seed
from pfm.core.execution import RuntimePaths, ExecutionContext, classifier_asset_dir, runtime_identity
from pfm.core.validation import require_positive_integer
from pfm.data import identity as dataset_identity
from pfm.data.datasets import make_dataloaders
from pfm.data.identity import validate_dataset_manifest_payload
from pfm.models.classifier import (
    CLASSIFIER_ASSET_CHECKPOINT_NAME,
    CLASSIFIER_ASSET_MANIFEST_NAME,
    ClassifierConfig,
    FeatureClassifier,
    classifier_asset_manifest_payload,
    classifier_checkpoint_payload,
    classifier_config_to_mapping,
    evaluate_classifier_accuracy,
    validate_classifier_class_mapping,
)


@dataclass(frozen=True)
class _DatasetManifestBinding:
    path: Path
    payload: dict[str, object]
    contents: bytes


def train_classifier(
    config: ClassifierConfig,
    *,
    classifier_asset_id: str,
    job_context: ExecutionContext,
    device: torch.device,
    show_progress: bool = True,
) -> dict[str, object]:
    epochs = require_positive_integer(
        config.epochs,
        "epochs",
        context="classifier training",
    )
    classifier_config_to_mapping(config)
    target = classifier_asset_dir(job_context.runtime, config.dataset, classifier_asset_id)
    if target.exists():
        raise FileExistsError(f"immutable target already exists: {target}")
    dataset_binding = _capture_dataset_manifest(job_context.runtime, config.dataset)
    set_seed(config.seed)
    loader_config = ExperimentConfig(
        seed=config.seed,
        dataset=config.dataset,
        batch_size=config.batch_size,
        num_workers=config.num_workers,
        mixed_precision=config.mixed_precision,
        num_classes=config.num_classes,
        image_channels=config.image_channels,
        image_size=config.image_size,
    )
    data_root = job_context.runtime.dataset_root(config.dataset)
    train_loader, test_loader = make_dataloaders(loader_config, data_root)
    class_to_idx = _dataset_class_mapping(train_loader.dataset, config.num_classes)
    validation_mapping = _dataset_class_mapping(test_loader.dataset, config.num_classes)
    if validation_mapping != class_to_idx:
        raise ValueError("classifier training and validation class mappings must match")
    model = FeatureClassifier.from_config(config).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=config.lr, weight_decay=config.weight_decay)
    loss_fn = nn.CrossEntropyLoss()

    for epoch in range(1, epochs + 1):
        model.train()
        progress = tqdm(
            train_loader,
            desc=f"classifier epoch {epoch}/{epochs}",
            leave=False,
            disable=not show_progress,
        )
        for images, labels in progress:
            images = images.to(device, non_blocking=True)
            labels = labels.to(device, non_blocking=True)
            with autocast_context(loader_config, device):
                logits = model(images)
                loss = loss_fn(logits, labels)
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            optimizer.step()
            progress.set_postfix(loss=f"{float(loss.detach().cpu()):.4f}")

    metrics = evaluate_classifier_accuracy(model, test_loader, device)
    _save_classifier_asset_with_binding(
        job_context.runtime,
        config.dataset,
        classifier_asset_id,
        model,
        config,
        metrics,
        class_to_idx=class_to_idx,
        producer_runtime_identity=runtime_identity(job_context),
        dataset_binding=dataset_binding,
    )
    return metrics


def _save_classifier_asset_with_binding(
    runtime: RuntimePaths,
    dataset: str,
    classifier_asset_id: str,
    model: FeatureClassifier,
    config: ClassifierConfig,
    metrics: Mapping[str, object],
    *,
    class_to_idx: Mapping[str, int],
    producer_runtime_identity: Mapping[str, object],
    dataset_binding: _DatasetManifestBinding,
) -> Path:
    config_payload = classifier_config_to_mapping(config)
    if config_payload["dataset"] != dataset:
        raise ValueError("classifier asset dataset does not match classifier config dataset")
    expected_manifest_path = (
        runtime.dataset_root(dataset) / dataset_identity.DATASET_MANIFEST_NAME
    )
    if dataset_binding.path != expected_manifest_path:
        raise ValueError("classifier dataset manifest binding path does not match runtime dataset")
    training_split, validation_split = _dataset_splits(dataset_binding)
    target = classifier_asset_dir(runtime, dataset, classifier_asset_id)
    staging_dir, prepared_target = prepare_immutable_directory(target.parent, target.name)
    if prepared_target != target:
        raise ValueError("classifier asset target does not match canonical runtime path")
    checkpoint_path = staging_dir / CLASSIFIER_ASSET_CHECKPOINT_NAME
    manifest_path = staging_dir / CLASSIFIER_ASSET_MANIFEST_NAME
    try:
        checkpoint = classifier_checkpoint_payload(model)
        write_torch_atomic(checkpoint_path, checkpoint)
        manifest = classifier_asset_manifest_payload(
            classifier_asset_id=classifier_asset_id,
            dataset=dataset,
            config=config,
            class_to_idx=class_to_idx,
            training_split=training_split,
            validation_split=validation_split,
            dataset_scientific_identity_sha256=dataset_binding.payload[
                "scientific_identity_sha256"
            ],
            checkpoint_sha256=_sha256_file(checkpoint_path),
            producer_runtime_identity=producer_runtime_identity,
            validation_metrics=metrics,
        )
        write_json_atomic(manifest_path, manifest)
        _verify_dataset_manifest_binding(dataset_binding)
        return publish_immutable_directory(staging_dir, target)
    finally:
        if staging_dir.exists():
            shutil.rmtree(staging_dir, ignore_errors=True)


def _dataset_class_mapping(dataset: object, num_classes: int) -> dict[str, int]:
    class_to_idx = getattr(dataset, "class_to_idx", None)
    if class_to_idx is not None:
        if not isinstance(class_to_idx, Mapping):
            raise ValueError("classifier dataset class_to_idx must be a mapping")
        return validate_classifier_class_mapping(class_to_idx, num_classes)
    classes = getattr(dataset, "classes", None)
    if not isinstance(classes, Sequence) or isinstance(classes, (str, bytes)):
        raise ValueError("classifier dataset must provide class_to_idx or ordered classes")
    if len(classes) != num_classes:
        raise ValueError("classifier dataset ordered classes must contain every class")
    return validate_classifier_class_mapping(
        {str(class_name): index for index, class_name in enumerate(classes)},
        num_classes,
    )


def _dataset_splits(binding: _DatasetManifestBinding) -> tuple[str, str]:
    scientific_identity = binding.payload.get("scientific_identity")
    splits = (
        list(scientific_identity.get("splits", {}))
        if isinstance(scientific_identity, Mapping)
        else None
    )
    if (
        not isinstance(splits, list)
        or len(splits) != 2
        or any(not isinstance(split, str) or not split for split in splits)
    ):
        raise ValueError("classifier dataset manifest must provide two ordered split names")
    return splits[0], splits[1]


def _capture_dataset_manifest(runtime: RuntimePaths, dataset: str) -> _DatasetManifestBinding:
    data_root = runtime.dataset_root(dataset)
    path = data_root / dataset_identity.DATASET_MANIFEST_NAME
    if path.is_symlink():
        raise ValueError(f"dataset manifest must not be a symlink: {path}")
    contents = path.read_bytes()
    try:
        raw_payload = json.loads(contents)
    except (UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise ValueError(f"invalid dataset identity JSON: {path}") from exc
    payload = validate_dataset_manifest_payload(
        raw_payload,
        expected_dataset=dataset,
    )
    return _DatasetManifestBinding(
        path=path,
        payload=payload,
        contents=contents,
    )


def _verify_dataset_manifest_binding(binding: _DatasetManifestBinding) -> None:
    if binding.path.is_symlink() or binding.path.read_bytes() != binding.contents:
        raise ValueError("dataset manifest changed during classifier training")


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()
