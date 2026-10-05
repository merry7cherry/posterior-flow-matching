from __future__ import annotations

import pytest

import pfm.data.datasets as datasets
from conftest import dataset_scientific_identity
from pfm.core.atomic_io import write_json_atomic
from pfm.core.config import ExperimentConfig
from pfm.data.datasets import make_datasets
from pfm.data.identity import DATASET_MANIFEST_NAME, dataset_manifest_payload


def test_training_dataset_requires_dataset_identity_before_constructor(
    monkeypatch,
    tmp_path,
) -> None:
    def fail_mnist_constructor(**kwargs):
        raise AssertionError(
            f"dataset constructor should not run before manifest validation: {kwargs}"
        )

    monkeypatch.setattr(datasets.datasets, "MNIST", fail_mnist_constructor)
    config = ExperimentConfig(dataset="mnist")
    with pytest.raises(FileNotFoundError, match="dataset_identity.json"):
        make_datasets(config, tmp_path / "mnist")


def test_dataset_manifest_payload_uses_exact_keys_and_splits() -> None:
    identity = dataset_scientific_identity("mnist")
    manifest = dataset_manifest_payload("mnist", identity)

    assert manifest["scientific_identity"] == identity
    assert len(manifest["scientific_identity_sha256"]) == 64
    assert set(manifest) == {"scientific_identity", "scientific_identity_sha256"}


def test_training_dataset_rejects_manifest_for_different_dataset(tmp_path) -> None:
    data_root = tmp_path / "mnist"
    write_json_atomic(
        data_root / DATASET_MANIFEST_NAME,
        dataset_manifest_payload(
            "cifar10",
            dataset_scientific_identity("cifar10"),
        ),
    )

    with pytest.raises(ValueError, match="does not match requested"):
        make_datasets(ExperimentConfig(dataset="mnist"), data_root)


@pytest.mark.parametrize(
    ("manifest", "message"),
    [
        (
            dataset_manifest_payload("mnist", dataset_scientific_identity("mnist"))
            | {"extra": "unexpected"},
            "keys must be exactly",
        ),
        (
            {"scientific_identity": dataset_scientific_identity("mnist")},
            "keys must be exactly",
        ),
        (
            {
                "scientific_identity": {
                    **dataset_scientific_identity("mnist"),
                    "splits": {
                        "train": {"count": 1, "order_sha256": "a" * 64},
                        "val": {"count": 1, "order_sha256": "a" * 64},
                    },
                },
                "scientific_identity_sha256": "a" * 64,
            },
            "splits must be",
        ),
        (
            {
                "scientific_identity": "not-an-object",
                "scientific_identity_sha256": "a" * 64,
            },
            "scientific identity keys",
        ),
    ],
)
def test_training_dataset_rejects_malformed_manifest_before_constructor(
    monkeypatch,
    tmp_path,
    manifest,
    message,
) -> None:
    def fail_mnist_constructor(**kwargs):
        raise AssertionError(
            f"dataset constructor should not run after malformed manifest: {kwargs}"
        )

    monkeypatch.setattr(datasets.datasets, "MNIST", fail_mnist_constructor)
    data_root = tmp_path / "mnist"
    write_json_atomic(data_root / DATASET_MANIFEST_NAME, manifest)

    with pytest.raises(ValueError, match=message):
        make_datasets(ExperimentConfig(dataset="mnist"), data_root)


def test_dataset_manifest_rejects_invalid_scientific_identity_digest(tmp_path) -> None:
    data_root = tmp_path / "imagenet"
    write_json_atomic(
        data_root / DATASET_MANIFEST_NAME,
        {
            "scientific_identity": dataset_scientific_identity("imagenet"),
            "scientific_identity_sha256": "not-a-digest",
        },
    )

    with pytest.raises(ValueError, match="SHA-256"):
        datasets.validate_dataset_manifest(data_root, "imagenet")
