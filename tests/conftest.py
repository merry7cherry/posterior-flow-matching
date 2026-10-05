from __future__ import annotations

from pathlib import Path

import pytest

from pfm.core.atomic_io import write_json_atomic
from pfm.core.execution import ExecutionContext, RuntimePaths
from pfm.data import datasets as dataset_module
from pfm.data.identity import (
    DATASET_MANIFEST_NAME,
    DATASET_SOURCES,
    DATASET_SPLITS,
    IMAGENET_SPLIT_COUNTS,
    dataset_manifest_payload,
    scientific_identity_from_datasets,
)
from pfm.models.jit_config import JIT_MODEL_SPECS, JiTModelSpec


def dataset_scientific_identity(dataset: str, *, marker: str = "a") -> dict[str, object]:
    if len(marker) != 1 or marker not in "0123456789abcdef":
        raise ValueError("marker must be one lowercase hexadecimal character")
    return {
        "schema_version": 1,
        "dataset": dataset,
        "source": dict(DATASET_SOURCES[dataset]),
        "splits": {
            split: {
                "count": IMAGENET_SPLIT_COUNTS[split] if dataset == "imagenet" else 1,
                "order_sha256": marker * 64,
            }
            for split in DATASET_SPLITS[dataset]
        },
        "class_to_idx": {"class0": 0},
    }


def write_test_dataset_manifest(dataset: str, root: Path) -> Path:
    """Construct a Dataset identity only for a test-owned fixture tree."""
    root.mkdir(parents=True, exist_ok=True)
    if dataset == "mnist":
        constructor = dataset_module.datasets.MNIST
        transform = dataset_module.mnist_transform()
    elif dataset == "cifar10":
        constructor = dataset_module.datasets.CIFAR10
        transform = dataset_module.cifar10_transform()
    elif dataset == "imagenet":
        datasets_by_split = {
            split: dataset_module.datasets.ImageFolder(root=root / split)
            for split in DATASET_SPLITS[dataset]
        }
        identity = scientific_identity_from_datasets(
            dataset,
            datasets_by_split,
            split_roots={split: root / split for split in DATASET_SPLITS[dataset]},
        )
        manifest_path = root / DATASET_MANIFEST_NAME
        write_json_atomic(manifest_path, dataset_manifest_payload(dataset, identity))
        return manifest_path
    else:
        raise ValueError(f"unsupported test Dataset: {dataset}")

    datasets_by_split = {
        split: constructor(
            root=root,
            train=split == "train",
            transform=transform,
            download=False,
        )
        for split in DATASET_SPLITS[dataset]
    }
    identity = scientific_identity_from_datasets(dataset, datasets_by_split)
    manifest_path = root / DATASET_MANIFEST_NAME
    write_json_atomic(manifest_path, dataset_manifest_payload(dataset, identity))
    return manifest_path


def temporary_runtime(tmp_path: Path) -> RuntimePaths:
    return RuntimePaths(
        project_name="pfm",
        project_root=tmp_path / "project",
        datasets_root=tmp_path / "datasets",
        assets_root=tmp_path / "scratch" / "pfm" / "assets",
        runs_root=tmp_path / "runs",
        cache_root=tmp_path / "cache",
    )


@pytest.fixture
def runtime_paths(tmp_path: Path) -> RuntimePaths:
    return temporary_runtime(tmp_path)


@pytest.fixture
def execution_context(runtime_paths: RuntimePaths) -> ExecutionContext:
    return ExecutionContext(runtime_paths, "a" * 40, 0, 0, 1)


@pytest.fixture
def tiny_jit_spec(monkeypatch) -> JiTModelSpec:
    spec = JiTModelSpec("JiT-Tiny/4", 4, 32, 2, 4, 16, 0, 0, (8, 16))
    monkeypatch.setitem(JIT_MODEL_SPECS, spec.name, spec)
    return spec
