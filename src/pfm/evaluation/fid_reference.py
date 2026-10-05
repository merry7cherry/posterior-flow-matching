from __future__ import annotations

import hashlib
import json
import shutil
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import Dataset
from torchvision import datasets, transforms

from pfm.core.atomic_io import atomic_output_path, write_json_atomic
from pfm.core.immutable_artifact import prepare_immutable_directory, publish_immutable_directory
from pfm.core.execution import ExecutionContext, fid_asset_dir, runtime_identity
from pfm.data import identity as dataset_identity
from pfm.data.datasets import center_crop_arr, validate_dataset_manifest
from pfm.evaluation.assets import (
    FID_ASSET_MANIFEST_NAME,
    FID_ASSET_STATS_NAME,
    FIDConfig,
    canonical_fid_preprocessing_identity,
    fid_asset_manifest_payload,
    fid_config_to_mapping,
    load_fid_config,
    validate_fid_asset_manifest_payload,
    validate_fid_config_asset_binding,
)
from pfm.evaluation.fid_metrics import (
    load_fid_statistics,
    require_jit_torch_fidelity,
    torch_fidelity_inception_weights_identity,
    validate_torch_fidelity_inception_weights,
)


_CIFAR_REFERENCE_SEED = 321


@dataclass(frozen=True)
class _DatasetManifestBinding:
    path: Path
    payload: dict[str, object]
    contents: bytes


def prepare_fid_asset(
    config: FIDConfig,
    *,
    fid_asset_id: str,
    job_context: ExecutionContext,
    device: torch.device,
) -> Path:
    config_payload = fid_config_to_mapping(config)
    asset_id = validate_fid_config_asset_binding(config, fid_asset_id)
    target = fid_asset_dir(job_context.runtime, asset_id)
    staging_dir, prepared_target = prepare_immutable_directory(target.parent, target.name)
    if prepared_target != target:
        raise ValueError("FID asset target does not match canonical runtime path")
    try:
        _require_fid_device(device)
        dataset_binding = _capture_dataset_manifest(
            job_context.runtime.dataset_root(str(config_payload["dataset"])),
            str(config_payload["dataset"]),
        )
        stats, actual_num_samples, preprocessing_identity = _compute_fid_statistics(
            config,
            job_context=job_context,
            device=device,
        )
        stats_path = staging_dir / FID_ASSET_STATS_NAME
        manifest_path = staging_dir / FID_ASSET_MANIFEST_NAME
        save_stats(stats_path, stats)
        validate_stats(stats_path)
        stats_sha256 = _sha256_file(stats_path)
        manifest = fid_asset_manifest_payload(
            fid_asset_id=asset_id,
            config=config,
            preprocessing_identity=preprocessing_identity,
            actual_num_samples=actual_num_samples,
            stats_sha256=stats_sha256,
            dataset_scientific_identity_sha256=dataset_binding.payload[
                "scientific_identity_sha256"
            ],
            producer_runtime_identity=runtime_identity(job_context),
        )
        write_json_atomic(manifest_path, manifest)
        _validate_staged_fid_asset(
            staging_dir,
            config=config,
            fid_asset_id=asset_id,
            dataset_binding=dataset_binding,
            expected_manifest=manifest,
        )
        _verify_dataset_manifest_binding(dataset_binding)
        return publish_immutable_directory(staging_dir, target)
    finally:
        if staging_dir.exists():
            shutil.rmtree(staging_dir, ignore_errors=True)


def _compute_fid_statistics(
    config: FIDConfig,
    *,
    job_context: ExecutionContext,
    device: torch.device,
) -> tuple[dict[str, np.ndarray], int, dict[str, object]]:
    data_root = job_context.runtime.dataset_root(config.dataset)
    if config.backend == "jit_imagenet":
        dataset = build_imagenet_train_dataset(data_root, config.image_size)
        actual_num_samples = len(dataset)
        if actual_num_samples != config.num_samples:
            raise ValueError(
                "FID dataset sample count does not match requested sample count: "
                f"expected {config.num_samples}, got {actual_num_samples}"
            )
        stats = compute_jit_fid_statistics(
            input1=dataset,
            cuda=True,
            batch_size=config.batch_size,
            cache_root=job_context.runtime.cache_root / "jit_fid_stats",
            verbose=True,
        )
        return stats, actual_num_samples, _bound_preprocessing_identity(config)

    dataset = build_cifar10_train_dataset(data_root)
    actual_num_samples = len(dataset)
    if actual_num_samples != config.num_samples:
        raise ValueError(
            "FID dataset sample count does not match requested sample count: "
            f"expected {config.num_samples}, got {actual_num_samples}"
        )
    stats = compute_jit_fid_statistics(
        input1=dataset,
        cuda=True,
        batch_size=config.batch_size,
        cache_root=job_context.runtime.cache_root / "cifar10_fid_stats",
        verbose=True,
    )
    return stats, actual_num_samples, _bound_preprocessing_identity(config)


def build_cifar10_train_dataset(data_root: Path):
    validate_dataset_manifest(data_root, "cifar10")
    return ImageOnlyDataset(
        datasets.CIFAR10(
            root=data_root,
            train=True,
            transform=transforms.PILToTensor(),
            download=False,
        )
    )


def _bound_preprocessing_identity(config: FIDConfig) -> dict[str, object]:
    identity = canonical_fid_preprocessing_identity(
        config.dataset,
        config.image_size,
        config.split,
        config.backend,
    )
    weights = torch_fidelity_inception_weights_identity()
    identity["feature_extractor_weights"] = {
        "filename": weights["filename"],
        "sha256": weights["sha256"],
    }
    return identity


def build_imagenet_train_dataset(data_root: Path, image_size: int):
    validate_dataset_manifest(data_root, "imagenet")
    train_root = data_root / "train"
    if not train_root.exists():
        raise FileNotFoundError(f"ImageNet train/ directory does not exist: {train_root}")
    transform = transforms.Compose(
        [
            transforms.Lambda(lambda image: center_crop_arr(image, int(image_size))),
            transforms.CenterCrop(int(image_size)),
            transforms.PILToTensor(),
        ]
    )
    return ImageOnlyDataset(datasets.ImageFolder(root=train_root, transform=transform))


class ImageOnlyDataset(Dataset):
    def __init__(self, dataset: datasets.ImageFolder) -> None:
        self.dataset = dataset

    def __len__(self) -> int:
        return len(self.dataset)

    def __getitem__(self, index: int) -> torch.Tensor:
        image, _ = self.dataset[index]
        return image


def compute_jit_fid_statistics(
    input1: object,
    cuda: bool,
    batch_size: int,
    cache_root: Path,
    verbose: bool,
) -> dict[str, np.ndarray]:
    weights_path = validate_torch_fidelity_inception_weights()
    require_jit_torch_fidelity()
    from torch_fidelity.defaults import DEFAULTS
    from torch_fidelity.metric_fid import fid_input_id_to_statistics_cached
    from torch_fidelity.utils import create_feature_extractor, resolve_feature_extractor, resolve_feature_layer_for_metric

    kwargs = dict(DEFAULTS)
    kwargs.update(
        {
            "input1": input1,
            "input2": None,
            "cuda": bool(cuda),
            "batch_size": int(batch_size),
            "isc": False,
            "fid": True,
            "kid": False,
            "prc": False,
            "ppl": False,
            "cache_root": str(cache_root),
            "cache": True,
            "verbose": bool(verbose),
            "feature_extractor_weights_path": str(weights_path),
        }
    )
    feature_extractor = resolve_feature_extractor(**kwargs)
    feature_layer = resolve_feature_layer_for_metric("fid", **kwargs)
    extractor = create_feature_extractor(feature_extractor, [feature_layer], **kwargs)
    stats = fid_input_id_to_statistics_cached(1, extractor, feature_layer, **kwargs)
    return {"mu": np.asarray(stats["mu"]), "sigma": np.asarray(stats["sigma"])}


def _validate_staged_fid_asset(
    staging_dir: Path,
    *,
    config: FIDConfig,
    fid_asset_id: str,
    dataset_binding: _DatasetManifestBinding,
    expected_manifest: dict[str, object],
) -> None:
    members = {path.name: path for path in staging_dir.iterdir()}
    if set(members) != {FID_ASSET_STATS_NAME, FID_ASSET_MANIFEST_NAME}:
        raise ValueError("staged FID asset must contain exactly statistics and manifest files")
    stats_path = members[FID_ASSET_STATS_NAME]
    manifest_path = members[FID_ASSET_MANIFEST_NAME]
    validate_stats(stats_path)
    load_fid_statistics(stats_path)
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    if manifest != expected_manifest:
        raise ValueError("staged FID asset manifest does not match constructed manifest")
    validate_fid_asset_manifest_payload(
        manifest,
        fid_asset_id=fid_asset_id,
        dataset=config.dataset,
        image_size=config.image_size,
        split=config.split,
        backend=config.backend,
        requested_num_samples=config.num_samples,
        stats_sha256=_sha256_file(stats_path),
        dataset_scientific_identity_sha256=dataset_binding.payload[
            "scientific_identity_sha256"
        ],
    )


def _capture_dataset_manifest(data_root: Path, dataset: str) -> _DatasetManifestBinding:
    path = data_root / dataset_identity.DATASET_MANIFEST_NAME
    if path.is_symlink():
        raise ValueError(f"dataset manifest must not be a symlink: {path}")
    payload = validate_dataset_manifest(data_root, dataset)
    contents = path.read_bytes()
    if json.loads(contents) != payload:
        raise ValueError("dataset manifest changed while FID asset binding was captured")
    return _DatasetManifestBinding(
        path=path,
        payload=payload,
        contents=contents,
    )


def _verify_dataset_manifest_binding(binding: _DatasetManifestBinding) -> None:
    if binding.path.is_symlink() or binding.path.read_bytes() != binding.contents:
        raise ValueError("dataset manifest changed during FID asset preparation")


def _require_fid_device(device: torch.device) -> None:
    if not isinstance(device, torch.device) or device.type != "cuda":
        raise ValueError("FID asset preparation requires an explicit CUDA device")


def save_stats(output_path: Path, stats: dict[str, np.ndarray]) -> None:
    with atomic_output_path(output_path) as tmp_path:
        np.savez(tmp_path, mu=stats["mu"], sigma=stats["sigma"])


def validate_stats(path: Path) -> None:
    with np.load(path, allow_pickle=False) as data:
        mu = data["mu"]
        sigma = data["sigma"]
    if mu.shape != (2048,) or sigma.shape != (2048, 2048):
        raise ValueError(f"Unexpected stats shapes: mu={mu.shape}, sigma={sigma.shape}")
    if not np.isfinite(mu).all() or not np.isfinite(sigma).all():
        raise ValueError(f"Stats contain non-finite values: {path}")


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()
