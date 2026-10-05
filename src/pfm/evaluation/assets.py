from __future__ import annotations

from collections.abc import Mapping
from dataclasses import asdict, dataclass, fields
import hashlib
import hmac
import io
import json
from pathlib import Path
from typing import Any

import torch

from pfm.core.checkpointing import validate_runtime_identity_payload
from pfm.core.identities import validate_dataset_identity
from pfm.core.layout import validate_path_component
from pfm.core.execution import RuntimePaths, classifier_asset_dir, fid_asset_dir
from pfm.core.yaml_config import load_yaml_mapping
from pfm.data import identity as dataset_identity
from pfm.data.datasets import validate_dataset_manifest
from pfm.models.classifier import (
    CLASSIFIER_ASSET_CHECKPOINT_NAME,
    CLASSIFIER_ASSET_MANIFEST_NAME,
    ClassifierConfig,
    FeatureClassifier,
    validate_classifier_asset_manifest_payload,
    validate_classifier_checkpoint_payload,
)


FID_ASSET_STATS_NAME = "fid_stats.npz"
FID_ASSET_MANIFEST_NAME = "fid_manifest.json"
FID_CONFIG_BINDINGS = {
    ("cifar10", 32, "train", "inception"): ("cifar10-inception-train-pfm-v4", 50_000),
    ("imagenet", 256, "train", "jit_imagenet"): ("imagenet256-jit-train-pfm-v4", 1_281_167),
}
_CIFAR_PREPROCESSING_IDENTITY = {
    "backend": "LTH14.torch-fidelity",
    "backend_commit": "bfe72b995473e4fee9351705d7b005f17442a4ca",
    "feature_extractor": "default_inception_v3",
    "feature_layer": 2048,
    "real_population": "official_cifar10_train_all_50000",
    "input_geometry": "original_32x32_rgb_no_resize",
    "generated_quantization": "round((clamp(x,-1,1)+1)*127.5)->uint8",
}
_JIT_PREPROCESSING_IDENTITY = {
    "dataset_transform": [
        "pfm.data.datasets.center_crop_arr",
        "torchvision.transforms.CenterCrop",
        "torchvision.transforms.PILToTensor",
    ],
    "feature_extractor": "LTH14.torch_fidelity.DEFAULTS",
    "feature_layer": "fid",
    "split_order": "torchvision.datasets.ImageFolder",
}
FID_ASSET_MANIFEST_KEYS = frozenset(
    {
        "fid_asset_id",
        "dataset",
        "image_size",
        "split",
        "backend",
        "preprocessing_identity",
        "requested_num_samples",
        "actual_num_samples",
        "stats_file",
        "stats_sha256",
        "dataset_scientific_identity_sha256",
        "producer_runtime_identity",
    }
)


@dataclass(frozen=True)
class FIDConfig:
    dataset: str
    image_size: int
    split: str
    backend: str
    num_samples: int
    batch_size: int


FID_CONFIG_KEYS = frozenset(field.name for field in fields(FIDConfig))


def load_fid_config(path: str | Path) -> FIDConfig:
    return fid_config_from_mapping(load_yaml_mapping(Path(path)))


def fid_config_from_mapping(raw: Mapping) -> FIDConfig:
    data = validate_fid_config_mapping(raw, context="FID config")
    identity = _fid_config_identity(data)
    expected_num_samples = FID_CONFIG_BINDINGS[identity][1]
    if data["num_samples"] != expected_num_samples:
        raise ValueError(
            "FID config binding requires "
            f"num_samples={expected_num_samples}, got {data['num_samples']}"
        )
    return FIDConfig(**data)


def fid_config_to_mapping(config: FIDConfig) -> dict[str, object]:
    if not isinstance(config, FIDConfig):
        raise TypeError("config must be a FIDConfig")
    return validate_fid_config_mapping(asdict(config), context="FID config")


def validate_fid_config_asset_binding(config: FIDConfig, fid_asset_id: str) -> str:
    config_payload = fid_config_to_mapping(config)
    identity = _fid_config_identity(config_payload)
    expected_asset_id, expected_num_samples = FID_CONFIG_BINDINGS[identity]
    asset_id = validate_path_component(fid_asset_id, "fid_asset_id")
    if asset_id != expected_asset_id:
        raise ValueError(
            "FID config asset binding requires "
            f"fid_asset_id={expected_asset_id!r}, got {asset_id!r}"
        )
    if config_payload["num_samples"] != expected_num_samples:
        raise ValueError(
            "FID config asset binding requires "
            f"num_samples={expected_num_samples}, got {config_payload['num_samples']}"
        )
    return asset_id


def canonical_fid_preprocessing_identity(
    dataset: str,
    image_size: int,
    split: str,
    backend: str,
) -> dict[str, object]:
    identity = _validate_fid_identity(
        dataset,
        image_size,
        split,
        backend,
        context="FID preprocessing identity",
    )
    canonical = (
        _CIFAR_PREPROCESSING_IDENTITY
        if identity == ("cifar10", 32, "train", "inception")
        else _JIT_PREPROCESSING_IDENTITY
    )
    return json.loads(json.dumps(canonical, allow_nan=False, sort_keys=True))


def validate_fid_config_mapping(
    value: object,
    *,
    context: str,
) -> dict[str, object]:
    if not isinstance(value, Mapping):
        raise ValueError(f"{context} YAML document must be a mapping")
    actual_keys = set(value)
    missing = sorted(FID_CONFIG_KEYS - actual_keys)
    unknown = sorted(actual_keys - FID_CONFIG_KEYS)
    if missing or unknown:
        details = []
        if missing:
            details.append(f"missing fields {missing}")
        if unknown:
            details.append(f"unknown fields {unknown}")
        raise ValueError(f"invalid {context}: {', '.join(details)}")
    data = dict(value)
    for name in ("image_size", "num_samples", "batch_size"):
        if isinstance(data[name], bool) or not isinstance(data[name], int) or data[name] < 1:
            raise ValueError(f"{context} {name} must be a positive integer")
    _fid_config_identity(data, context=context)
    return asdict(FIDConfig(**data))


def fid_asset_manifest_payload(
    *,
    fid_asset_id: str,
    config: FIDConfig,
    preprocessing_identity: Mapping[str, object],
    actual_num_samples: int,
    stats_sha256: str,
    dataset_scientific_identity_sha256: str,
    producer_runtime_identity: Mapping[str, object],
) -> dict[str, object]:
    config_payload = fid_config_to_mapping(config)
    asset_id = validate_fid_config_asset_binding(config, fid_asset_id)
    identity = _fid_config_identity(config_payload)
    payload: dict[str, object] = {
        "fid_asset_id": asset_id,
        "dataset": config_payload["dataset"],
        "image_size": config_payload["image_size"],
        "split": config_payload["split"],
        "backend": config_payload["backend"],
        "preprocessing_identity": _validated_preprocessing_identity(
            preprocessing_identity,
            identity=identity,
        ),
        "requested_num_samples": config_payload["num_samples"],
        "actual_num_samples": actual_num_samples,
        "stats_file": FID_ASSET_STATS_NAME,
        "stats_sha256": stats_sha256,
        "dataset_scientific_identity_sha256": dataset_scientific_identity_sha256,
        "producer_runtime_identity": dict(producer_runtime_identity),
    }
    return validate_fid_asset_manifest_payload(
        payload,
        fid_asset_id=asset_id,
        dataset=str(config_payload["dataset"]),
        image_size=int(config_payload["image_size"]),
        split=str(config_payload["split"]),
        backend=str(config_payload["backend"]),
        requested_num_samples=int(config_payload["num_samples"]),
        stats_sha256=stats_sha256,
        dataset_scientific_identity_sha256=dataset_scientific_identity_sha256,
    )


def validate_fid_asset_manifest_payload(
    value: object,
    *,
    fid_asset_id: str,
    dataset: str,
    image_size: int,
    split: str,
    backend: str,
    requested_num_samples: int,
    stats_sha256: str,
    dataset_scientific_identity_sha256: str,
) -> dict[str, object]:
    if not isinstance(value, Mapping):
        raise ValueError("FID asset manifest must contain a JSON object")
    manifest = dict(value)
    if set(manifest) != FID_ASSET_MANIFEST_KEYS:
        raise ValueError(
            "FID asset manifest keys must be exactly "
            f"{sorted(FID_ASSET_MANIFEST_KEYS)}"
        )
    expected_asset_id = validate_path_component(fid_asset_id, "fid_asset_id")
    identity = _validate_fid_identity(
        dataset,
        image_size,
        split,
        backend,
        context="requested FID asset",
    )
    canonical_asset_id, canonical_num_samples = FID_CONFIG_BINDINGS[identity]
    if expected_asset_id != canonical_asset_id:
        raise ValueError("requested FID asset id does not match canonical config asset binding")
    if manifest["fid_asset_id"] != expected_asset_id:
        raise ValueError("FID asset manifest id does not match requested asset id")
    for name, expected in (
        ("dataset", dataset),
        ("image_size", image_size),
        ("split", split),
        ("backend", backend),
    ):
        if manifest[name] != expected:
            raise ValueError(f"FID asset manifest {name} does not match requested {name}")
    _validate_fid_identity(
        manifest["dataset"],
        manifest["image_size"],
        manifest["split"],
        manifest["backend"],
        context="FID asset manifest",
    )
    _validated_preprocessing_identity(manifest["preprocessing_identity"], identity=identity)
    expected_count = _positive_integer(
        requested_num_samples,
        "expected sample count",
    )
    if expected_count != canonical_num_samples:
        raise ValueError("FID asset requested sample count does not match canonical config binding")
    manifest_requested_count = _positive_integer(
        manifest["requested_num_samples"],
        "requested sample count",
    )
    actual_count = _positive_integer(
        manifest["actual_num_samples"],
        "actual sample count",
    )
    if manifest_requested_count != expected_count or actual_count != expected_count:
        raise ValueError(
            "FID asset manifest requested/actual sample count does not match expected count"
        )
    if manifest["stats_file"] != FID_ASSET_STATS_NAME:
        raise ValueError("FID asset manifest stats_file must be fid_stats.npz")
    expected_stats_sha256 = _require_sha256(manifest["stats_sha256"], "FID statistics")
    current_stats_sha256 = _require_sha256(stats_sha256, "current FID statistics")
    if not hmac.compare_digest(expected_stats_sha256, current_stats_sha256):
        raise ValueError("FID asset statistics SHA-256 does not match manifest")
    expected_dataset_identity = _require_sha256(
        dataset_scientific_identity_sha256,
        "current dataset scientific identity",
    )
    manifest_dataset_identity = _require_sha256(
        manifest["dataset_scientific_identity_sha256"],
        "FID asset dataset scientific identity",
    )
    if not hmac.compare_digest(manifest_dataset_identity, expected_dataset_identity):
        raise ValueError(
            "FID asset dataset scientific identity does not match current manifest"
        )
    validate_runtime_identity_payload(
        manifest["producer_runtime_identity"],
        context="FID asset manifest",
    )
    return manifest


def validate_classifier_asset_manifest(
    value: object,
    *,
    runtime: RuntimePaths,
    dataset: str,
    classifier_asset_id: str,
) -> tuple[dict[str, object], dict[str, object]]:
    _, dataset_manifest = _bound_dataset_manifest(
        runtime,
        dataset,
    )
    return validate_classifier_asset_manifest_payload(
        value,
        dataset=dataset,
        classifier_asset_id=classifier_asset_id,
        dataset_scientific_identity_sha256=dataset_manifest[
            "scientific_identity_sha256"
        ],
        dataset_splits=list(dataset_manifest["scientific_identity"]["splits"]),
    )


def load_classifier_asset(
    runtime: RuntimePaths,
    dataset: str,
    classifier_asset_id: str,
    device: torch.device,
) -> tuple[FeatureClassifier, dict[str, Any]]:
    asset_dir = classifier_asset_dir(runtime, dataset, classifier_asset_id)
    manifest_path = asset_dir / CLASSIFIER_ASSET_MANIFEST_NAME
    if manifest_path.is_symlink():
        raise ValueError(f"classifier asset manifest must not be a symlink: {manifest_path}")
    if not manifest_path.is_file():
        raise FileNotFoundError(f"classifier asset manifest does not exist: {manifest_path}")
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    config, _ = validate_classifier_asset_manifest(
        manifest,
        runtime=runtime,
        dataset=dataset,
        classifier_asset_id=classifier_asset_id,
    )
    checkpoint_path = asset_dir / CLASSIFIER_ASSET_CHECKPOINT_NAME
    if checkpoint_path.is_symlink():
        raise ValueError(f"classifier asset checkpoint must not be a symlink: {checkpoint_path}")
    checkpoint_bytes = checkpoint_path.read_bytes()
    actual_sha256 = hashlib.sha256(checkpoint_bytes).hexdigest()
    if not hmac.compare_digest(actual_sha256, manifest["checkpoint_sha256"]):
        raise ValueError("classifier asset checkpoint SHA-256 does not match manifest")
    checkpoint = validate_classifier_checkpoint_payload(
        torch.load(io.BytesIO(checkpoint_bytes), map_location="cpu")
    )
    model = FeatureClassifier.from_config(ClassifierConfig(**config))
    model.load_state_dict(checkpoint["model_state"], strict=True)
    model.to(device)
    model.eval()
    return model, manifest


def load_fid_asset(
    runtime: RuntimePaths,
    fid_asset_id: str,
    *,
    dataset: str,
    image_size: int,
    backend: str,
    split: str,
) -> tuple[Path, dict[str, object]]:
    _validate_fid_identity(dataset, image_size, split, backend, context="requested FID asset")
    asset_dir = fid_asset_dir(runtime, fid_asset_id)
    if asset_dir.is_symlink():
        raise ValueError(f"FID asset directory must not be a symlink: {asset_dir}")
    if not asset_dir.exists():
        raise FileNotFoundError(f"FID asset directory does not exist: {asset_dir}")
    if not asset_dir.is_dir():
        raise NotADirectoryError(f"FID asset path is not a directory: {asset_dir}")
    manifest_path = asset_dir / FID_ASSET_MANIFEST_NAME
    stats_path = asset_dir / FID_ASSET_STATS_NAME
    for member in (manifest_path, stats_path):
        if member.is_symlink():
            raise ValueError(f"FID asset member must not be a symlink: {member}")
        if not member.is_file():
            raise ValueError(f"FID asset member must be a regular file: {member}")

    _, dataset_manifest = _bound_dataset_manifest(
        runtime,
        dataset,
    )
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    stats_sha256 = _stream_sha256(stats_path)
    validated_manifest = validate_fid_asset_manifest_payload(
        manifest,
        fid_asset_id=fid_asset_id,
        dataset=dataset,
        image_size=image_size,
        split=split,
        backend=backend,
        requested_num_samples=FID_CONFIG_BINDINGS[(dataset, image_size, split, backend)][1],
        stats_sha256=stats_sha256,
        dataset_scientific_identity_sha256=dataset_manifest[
            "scientific_identity_sha256"
        ],
    )
    return stats_path, validated_manifest


def _bound_dataset_manifest(
    runtime: RuntimePaths,
    dataset: str,
) -> tuple[Path, dict[str, object]]:
    data_root = runtime.dataset_root(dataset)
    manifest_path = data_root / dataset_identity.DATASET_MANIFEST_NAME
    if manifest_path.is_symlink():
        raise ValueError(f"dataset manifest must not be a symlink: {manifest_path}")
    manifest = validate_dataset_manifest(data_root, dataset)
    contents = manifest_path.read_bytes()
    if json.loads(contents) != manifest:
        raise ValueError("dataset manifest changed while classifier asset binding was read")
    return manifest_path, manifest


def _stream_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _fid_config_identity(
    value: Mapping[str, object],
    *,
    context: str = "FID config",
) -> tuple[str, int, str, str]:
    return _validate_fid_identity(
        value["dataset"],
        value["image_size"],
        value["split"],
        value["backend"],
        context=context,
    )


def _validate_fid_identity(
    dataset: object,
    image_size: object,
    split: object,
    backend: object,
    *,
    context: str,
) -> tuple[str, int, str, str]:
    if not isinstance(dataset, str):
        raise ValueError(f"{context} dataset must be a string")
    if isinstance(image_size, bool) or not isinstance(image_size, int):
        raise ValueError(f"{context} image_size must be an integer")
    if not isinstance(split, str):
        raise ValueError(f"{context} split must be a string")
    if not isinstance(backend, str):
        raise ValueError(f"{context} backend must be a string")
    identity = (dataset, image_size, split, backend)
    if identity not in FID_CONFIG_BINDINGS:
        raise ValueError(
            f"{context} binding is not one of the canonical FID dataset/image/split/backend bindings"
        )
    return identity


def _positive_integer(value: object, name: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value < 1:
        raise ValueError(f"FID asset manifest {name} must be a positive integer")
    return value


def _require_sha256(value: object, name: str) -> str:
    if (
        not isinstance(value, str)
        or len(value) != 64
        or any(character not in "0123456789abcdef" for character in value)
    ):
        raise ValueError(f"{name} SHA-256 must be 64 lowercase hex characters")
    return value


def _validated_preprocessing_identity(
    value: object,
    *,
    identity: tuple[str, int, str, str],
) -> dict[str, object]:
    if not isinstance(value, Mapping) or not value:
        raise ValueError("FID asset preprocessing identity must be a non-empty mapping")
    try:
        encoded = json.dumps(dict(value), allow_nan=False, sort_keys=True)
    except (TypeError, ValueError) as exc:
        raise ValueError("FID asset preprocessing identity must contain JSON values") from exc
    normalized = json.loads(encoded)
    if not isinstance(normalized, dict) or not normalized:
        raise ValueError("FID asset preprocessing identity must be a non-empty mapping")
    weights = normalized.pop("feature_extractor_weights", None)
    if not isinstance(weights, dict) or set(weights) != {"filename", "sha256"}:
        raise ValueError("FID asset preprocessing identity must bind Inception weights")
    if weights["filename"] != "weights-inception-2015-12-05-6726825d.pth":
        raise ValueError("FID asset preprocessing identity has unexpected Inception weights")
    _require_sha256(weights["sha256"], "FID asset Inception weights")
    expected = canonical_fid_preprocessing_identity(*identity)
    if normalized != expected:
        raise ValueError("FID asset preprocessing identity does not match canonical backend identity")
    normalized["feature_extractor_weights"] = weights
    return normalized
