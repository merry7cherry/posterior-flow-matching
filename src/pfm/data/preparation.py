"""Explicit data preparation, separate from read-only training and evaluation."""
from __future__ import annotations

import json
from pathlib import Path

from torchvision import datasets

from pfm.core.atomic_io import write_json_exclusive
from pfm.core.identities import validate_dataset_identity
from pfm.data.identity import (
    DATASET_MANIFEST_NAME,
    DATASET_SPLITS,
    IMAGENET_SPLIT_COUNTS,
    dataset_manifest_payload,
    scientific_identity_from_datasets,
    validate_dataset_manifest_payload,
)


OFFICIAL_SPLIT_COUNTS = {
    "mnist": {"train": 60_000, "test": 10_000},
    "cifar10": {"train": 50_000, "test": 10_000},
    "imagenet": IMAGENET_SPLIT_COUNTS,
}
OFFICIAL_CLASS_COUNTS = {"mnist": 10, "cifar10": 10, "imagenet": 1000}


def prepare_dataset(dataset: str, root: str | Path, *, download: bool = False) -> Path:
    """Validate official split metadata and publish its relocatable identity.

    torchvision verifies MNIST/CIFAR archives when explicitly downloading.
    ImageNet must already have train/<class>/ and val/<class>/ directories;
    identity records its file ordering and labels, not image-content hashes.
    """
    dataset = validate_dataset_identity(dataset)
    root = Path(root).expanduser().resolve()
    if dataset == "imagenet":
        if download:
            raise ValueError("ImageNet is user-provided; --download is only for MNIST/CIFAR-10")
        split_roots = {split: root / split for split in DATASET_SPLITS[dataset]}
        for split_root in split_roots.values():
            if not split_root.is_dir():
                raise FileNotFoundError(f"ImageNet requires train/ and val/ class directories: {split_root}")
        prepared = {split: datasets.ImageFolder(root=path) for split, path in split_roots.items()}
    else:
        constructor = datasets.MNIST if dataset == "mnist" else datasets.CIFAR10
        prepared = {
            split: constructor(root=root, train=split == "train", download=download)
            for split in DATASET_SPLITS[dataset]
        }
        split_roots = None
    for split, values in prepared.items():
        expected = OFFICIAL_SPLIT_COUNTS[dataset][split]
        if len(values) != expected:
            raise ValueError(f"{dataset} {split} requires {expected} samples, found {len(values)}")
    identity = scientific_identity_from_datasets(dataset, prepared, split_roots=split_roots)
    if len(identity["class_to_idx"]) != OFFICIAL_CLASS_COUNTS[dataset]:
        raise ValueError(f"{dataset} requires exactly {OFFICIAL_CLASS_COUNTS[dataset]} classes")
    payload = dataset_manifest_payload(dataset, identity)
    destination = root / DATASET_MANIFEST_NAME
    if not write_json_exclusive(destination, payload):
        with destination.open(encoding="utf-8") as handle:
            existing = validate_dataset_manifest_payload(json.load(handle), expected_dataset=dataset)
        if existing != payload:
            raise ValueError(f"Existing dataset identity differs from prepared data: {destination}")
    return destination
