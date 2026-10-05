from __future__ import annotations

from collections.abc import Mapping, Sequence
import hashlib
import hmac
import json
from pathlib import Path

import torch

from pfm.core.identities import validate_dataset_identity


DATASET_MANIFEST_NAME = "dataset_identity.json"
DATASET_SPLITS = {
    "mnist": ("train", "test"),
    "cifar10": ("train", "test"),
    "imagenet": ("train", "val"),
}
DATASET_SOURCES = {
    "mnist": {"name": "MNIST", "release": "official-idx"},
    "cifar10": {"name": "CIFAR-10", "release": "python"},
    "imagenet": {"name": "ImageNet", "release": "ILSVRC2012"},
}
IMAGENET_SPLIT_COUNTS = {"train": 1_281_167, "val": 50_000}
_IDENTITY_KEYS = frozenset(
    {"schema_version", "dataset", "source", "splits", "class_to_idx"}
)
_SPLIT_KEYS = frozenset({"count", "order_sha256"})
_MANIFEST_KEYS = frozenset(
    {"scientific_identity", "scientific_identity_sha256"}
)


def scientific_identity_from_datasets(
    dataset: str,
    datasets_by_split: Mapping[str, object],
    *,
    split_roots: Mapping[str, Path] | None = None,
) -> dict[str, object]:
    """Build compact identity from official ordering metadata, never image bytes."""
    dataset_name = validate_dataset_identity(dataset)
    expected_splits = DATASET_SPLITS[dataset_name]
    if set(datasets_by_split) != set(expected_splits):
        raise ValueError(
            f"{dataset_name} datasets must provide exactly {list(expected_splits)}"
        )

    class_to_idx: dict[str, int] | None = None
    split_identities: dict[str, object] = {}
    for split in expected_splits:
        split_dataset = datasets_by_split[split]
        current_mapping = _class_mapping(split_dataset)
        if class_to_idx is None:
            class_to_idx = current_mapping
        elif current_mapping != class_to_idx:
            raise ValueError(f"{dataset_name} split class mappings must match exactly")

        if dataset_name == "imagenet":
            if split_roots is None or split not in split_roots:
                raise ValueError("ImageNet scientific identity requires split roots")
            count, order_sha256 = _imagefolder_order_identity(
                split_dataset,
                Path(split_roots[split]),
            )
        else:
            count, order_sha256 = _labeled_order_identity(split_dataset)
        split_identities[split] = {
            "count": count,
            "order_sha256": order_sha256,
        }

    return validate_dataset_scientific_identity(
        {
            "schema_version": 1,
            "dataset": dataset_name,
            "source": DATASET_SOURCES[dataset_name],
            "splits": split_identities,
            "class_to_idx": class_to_idx,
        },
        expected_dataset=dataset_name,
    )


def dataset_manifest_payload(
    dataset: str,
    scientific_identity: Mapping[str, object],
) -> dict[str, object]:
    identity = validate_dataset_scientific_identity(
        scientific_identity,
        expected_dataset=dataset,
    )
    return {
        "scientific_identity": identity,
        "scientific_identity_sha256": scientific_identity_sha256(identity),
    }


def validate_dataset_manifest_payload(
    value: object,
    *,
    expected_dataset: str,
) -> dict[str, object]:
    if not isinstance(value, Mapping) or set(value) != _MANIFEST_KEYS:
        raise ValueError(
            "dataset manifest keys must be exactly " f"{sorted(_MANIFEST_KEYS)}"
        )
    identity = validate_dataset_scientific_identity(
        value["scientific_identity"],
        expected_dataset=expected_dataset,
    )
    expected_sha256 = _require_sha256(
        value["scientific_identity_sha256"],
        "dataset scientific identity",
    )
    actual_sha256 = scientific_identity_sha256(identity)
    if not hmac.compare_digest(expected_sha256, actual_sha256):
        raise ValueError("dataset scientific identity SHA-256 does not match identity")
    return {
        "scientific_identity": identity,
        "scientific_identity_sha256": actual_sha256,
    }


def validate_dataset_scientific_identity(
    value: object,
    *,
    expected_dataset: str | None = None,
) -> dict[str, object]:
    if not isinstance(value, Mapping) or set(value) != _IDENTITY_KEYS:
        raise ValueError(
            "dataset scientific identity keys must be exactly "
            f"{sorted(_IDENTITY_KEYS)}"
        )
    if value["schema_version"] != 1:
        raise ValueError("dataset scientific identity schema_version must be 1")
    dataset = validate_dataset_identity(value["dataset"])
    if expected_dataset is not None:
        requested = validate_dataset_identity(expected_dataset)
        if dataset != requested:
            raise ValueError(
                f"dataset scientific identity {dataset!r} does not match requested {requested!r}"
            )
    if value["source"] != DATASET_SOURCES[dataset]:
        raise ValueError(
            f"dataset scientific identity source must be {DATASET_SOURCES[dataset]!r}"
        )

    raw_splits = value["splits"]
    expected_splits = DATASET_SPLITS[dataset]
    if not isinstance(raw_splits, Mapping) or set(raw_splits) != set(expected_splits):
        raise ValueError(
            f"dataset scientific identity splits must be exactly {list(expected_splits)}"
        )
    splits: dict[str, object] = {}
    for split in expected_splits:
        raw = raw_splits[split]
        if not isinstance(raw, Mapping) or set(raw) != _SPLIT_KEYS:
            raise ValueError(
                f"dataset scientific identity split {split!r} keys must be exactly "
                f"{sorted(_SPLIT_KEYS)}"
            )
        count = raw["count"]
        if type(count) is not int or count < 1:
            raise ValueError(
                f"dataset scientific identity split {split!r} count must be positive"
            )
        if dataset == "imagenet" and count != IMAGENET_SPLIT_COUNTS[split]:
            raise ValueError(
                f"ImageNet {split} count must equal {IMAGENET_SPLIT_COUNTS[split]}; "
                f"got {count}"
            )
        splits[split] = {
            "count": count,
            "order_sha256": _require_sha256(
                raw["order_sha256"],
                f"dataset scientific identity split {split!r} order",
            ),
        }

    class_to_idx = _validate_class_mapping(value["class_to_idx"])
    return {
        "schema_version": 1,
        "dataset": dataset,
        "source": dict(DATASET_SOURCES[dataset]),
        "splits": splits,
        "class_to_idx": class_to_idx,
    }


def scientific_identity_sha256(identity: Mapping[str, object]) -> str:
    validated = validate_dataset_scientific_identity(identity)
    return hashlib.sha256(_canonical_json_bytes(validated)).hexdigest()


def _class_mapping(dataset: object) -> dict[str, int]:
    mapping = getattr(dataset, "class_to_idx", None)
    if mapping is not None:
        return _validate_class_mapping(mapping)
    classes = getattr(dataset, "classes", None)
    if not isinstance(classes, Sequence) or isinstance(classes, (str, bytes)):
        raise ValueError("prepared dataset must expose class_to_idx or ordered classes")
    return _validate_class_mapping(
        {str(class_name): index for index, class_name in enumerate(classes)}
    )


def _validate_class_mapping(value: object) -> dict[str, int]:
    if not isinstance(value, Mapping) or not value:
        raise ValueError("dataset scientific identity class_to_idx must be a nonempty mapping")
    result: dict[str, int] = {}
    for name, index in value.items():
        if not isinstance(name, str) or not name:
            raise ValueError("dataset class names must be nonempty strings")
        if type(index) is not int or index < 0:
            raise ValueError("dataset class indices must be non-negative integers")
        result[name] = index
    if sorted(result.values()) != list(range(len(result))):
        raise ValueError("dataset class indices must be contiguous from zero")
    return dict(sorted(result.items()))


def _labeled_order_identity(dataset: object) -> tuple[int, str]:
    try:
        count = len(dataset)  # type: ignore[arg-type]
    except TypeError as exc:
        raise ValueError("prepared dataset must expose a finite length") from exc
    targets = getattr(dataset, "targets", None)
    if isinstance(targets, torch.Tensor):
        targets = targets.detach().cpu().tolist()
    if not isinstance(targets, Sequence) or isinstance(targets, (str, bytes)):
        raise ValueError("prepared dataset must expose ordered targets")
    if count < 1 or len(targets) != count:
        raise ValueError("prepared dataset target count must match dataset length")
    digest = hashlib.sha256()
    for index, label in enumerate(targets):
        if isinstance(label, torch.Tensor) and label.ndim == 0:
            label = label.item()
        if type(label) is not int or label < 0:
            raise ValueError("prepared dataset targets must be non-negative integers")
        _update_order_digest(digest, index=index, label=label)
    return count, digest.hexdigest()


def _imagefolder_order_identity(dataset: object, split_root: Path) -> tuple[int, str]:
    samples = getattr(dataset, "samples", None)
    if not isinstance(samples, Sequence) or isinstance(samples, (str, bytes)):
        raise ValueError("ImageNet split must expose ordered ImageFolder samples")
    if len(samples) < 1:
        raise ValueError("ImageNet split must contain at least one sample")
    digest = hashlib.sha256()
    for index, sample in enumerate(samples):
        if not isinstance(sample, (list, tuple)) or len(sample) != 2:
            raise ValueError("ImageNet sample records must be path/label pairs")
        path, label = sample
        if type(label) is not int or label < 0:
            raise ValueError("ImageNet sample labels must be non-negative integers")
        try:
            relative_path = Path(path).relative_to(split_root).as_posix()
        except ValueError as exc:
            raise ValueError("ImageNet sample path escapes its split root") from exc
        _update_order_digest(
            digest,
            index=index,
            label=label,
            relative_path=relative_path,
        )
    return len(samples), digest.hexdigest()


def _update_order_digest(
    digest: object,
    *,
    index: int,
    label: int,
    relative_path: str | None = None,
) -> None:
    record: dict[str, object] = {"index": index, "label": label}
    if relative_path is not None:
        record["path"] = relative_path
    digest.update(_canonical_json_bytes(record) + b"\n")


def _canonical_json_bytes(value: object) -> bytes:
    return json.dumps(
        value,
        allow_nan=False,
        ensure_ascii=False,
        separators=(",", ":"),
        sort_keys=True,
    ).encode("utf-8")


def _require_sha256(value: object, context: str) -> str:
    if (
        not isinstance(value, str)
        or len(value) != 64
        or any(character not in "0123456789abcdef" for character in value)
    ):
        raise ValueError(f"{context} SHA-256 must be 64 lowercase hexadecimal characters")
    return value
