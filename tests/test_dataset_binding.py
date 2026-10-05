from __future__ import annotations

import hashlib
import json

import pytest
import torch
from PIL import Image

from conftest import write_test_dataset_manifest
from pfm.core.checkpointing import validate_checkpoint_dataset_binding
from pfm.core.config import ExperimentConfig
from pfm.core.rng_state import capture_rng_state
from pfm.data.datasets import validate_dataset_manifest
from pfm.data.identity import DATASET_MANIFEST_NAME
from pfm.training.checkpointing import build_training_checkpoint


def _write_image(root, split: str, class_name: str, name: str, color: int) -> None:
    path = root / split / class_name / name
    path.parent.mkdir(parents=True, exist_ok=True)
    Image.new("RGB", (2, 2), color=(color, color, color)).save(path)


def _prepare_tiny_imagenet(root, *, reverse_train_names: bool = False, color: int = 0):
    train_names = ["a.png", "b.png"]
    if reverse_train_names:
        train_names = ["a.png", "c.png"]
    for index, name in enumerate(train_names):
        _write_image(root, "train", "class0", name, color + index)
    _write_image(root, "val", "class0", "v.png", color + 2)
    return write_test_dataset_manifest("imagenet", root)


def _runtime_identity():
    return {
        "project_name": "pfm",
        "git_commit": "a" * 40,
        "rank": 0,
        "world_size": 1,
    }


def _rng_state():
    state = capture_rng_state()
    state["torch_cuda"] = None
    return state


@pytest.fixture
def tiny_imagenet_counts(monkeypatch):
    monkeypatch.setattr(
        "pfm.data.identity.IMAGENET_SPLIT_COUNTS",
        {"train": 2, "val": 1},
    )


def test_prepared_dataset_identity_is_relocation_and_runtime_independent(
    tmp_path,
    tiny_imagenet_counts,
):
    first = tmp_path / "machine-a" / "imagenet"
    second = tmp_path / "machine-b" / "imagenet"
    first_manifest_path = _prepare_tiny_imagenet(first, color=10)
    second_manifest_path = _prepare_tiny_imagenet(second, color=90)

    first_manifest = validate_dataset_manifest(first, "imagenet")
    second_manifest = validate_dataset_manifest(second, "imagenet")

    assert first_manifest == second_manifest
    assert first_manifest_path.read_bytes() == second_manifest_path.read_bytes()
    assert set(first_manifest) == {
        "scientific_identity",
        "scientific_identity_sha256",
    }
    encoded = json.dumps(first_manifest, sort_keys=True)
    for forbidden in (str(first), str(second), "runtime_identity", "git_commit", "slurm_job_id"):
        assert forbidden not in encoded


def test_prepared_dataset_identity_binds_official_order_without_per_image_hashes(
    tmp_path,
    tiny_imagenet_counts,
):
    first = tmp_path / "first"
    changed = tmp_path / "changed"
    _prepare_tiny_imagenet(first)
    _prepare_tiny_imagenet(changed, reverse_train_names=True)

    first_manifest = validate_dataset_manifest(first, "imagenet")
    changed_manifest = validate_dataset_manifest(changed, "imagenet")

    assert first_manifest["scientific_identity_sha256"] != changed_manifest[
        "scientific_identity_sha256"
    ]
    identity = first_manifest["scientific_identity"]
    assert identity["splits"]["train"]["count"] == 2
    assert len(identity["splits"]["train"]["order_sha256"]) == 64
    assert "records" not in identity["splits"]["train"]
    assert "content_sha256" not in json.dumps(identity, sort_keys=True)
    assert len(json.dumps(first_manifest)) < 4096


def test_dataset_manifest_rejects_tampered_scientific_identity_before_dataset_load(
    tmp_path,
    tiny_imagenet_counts,
):
    root = tmp_path / "imagenet"
    manifest_path = _prepare_tiny_imagenet(root)
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    manifest["scientific_identity"]["splits"]["train"]["order_sha256"] = "f" * 64
    manifest_path.write_text(json.dumps(manifest) + "\n", encoding="utf-8")

    with pytest.raises(ValueError, match="scientific identity SHA-256"):
        validate_dataset_manifest(root, "imagenet")


def test_dataset_manifest_rejects_self_consistent_nonofficial_imagenet_count(
    tmp_path,
    tiny_imagenet_counts,
):
    root = tmp_path / "imagenet"
    manifest_path = _prepare_tiny_imagenet(root)
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    identity = manifest["scientific_identity"]
    identity["splits"]["train"]["count"] = 3
    encoded = json.dumps(
        identity,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=False,
    ).encode("utf-8")
    manifest["scientific_identity_sha256"] = hashlib.sha256(encoded).hexdigest()
    manifest_path.write_text(json.dumps(manifest) + "\n", encoding="utf-8")

    with pytest.raises(ValueError, match="ImageNet train count must equal 2; got 3"):
        validate_dataset_manifest(root, "imagenet")


def test_training_checkpoint_carries_and_validates_dataset_binding():
    model = torch.nn.Linear(1, 1)
    optimizer = torch.optim.AdamW(model.parameters())
    digest = "d" * 64
    payload = build_training_checkpoint(
        model=model,
        optimizer=optimizer,
        variant="velocity_fm",
        config=ExperimentConfig(dataset="mnist"),
        epoch=1,
        step=0,
        loss=0.5,
        ema_model=model,
        distributed_world_size=1,
        rng_state_by_rank=[_rng_state()],
        runtime_identity=_runtime_identity(),
        dataset_scientific_identity_sha256=digest,
    )

    assert payload["dataset_scientific_identity_sha256"] == digest
    validate_checkpoint_dataset_binding(payload, digest)
    with pytest.raises(ValueError, match="dataset scientific identity mismatch"):
        validate_checkpoint_dataset_binding(payload, "e" * 64)


def test_checkpoint_dataset_mismatch_fails_without_reading_model_state():
    class ExplodingMapping(dict):
        def __getitem__(self, key):
            if key == "model_state":
                raise AssertionError("model state must not be read before dataset binding")
            return super().__getitem__(key)

    checkpoint = ExplodingMapping(
        dataset_scientific_identity_sha256="a" * 64,
        model_state={"unused": torch.tensor(1)},
    )

    with pytest.raises(ValueError, match="dataset scientific identity mismatch"):
        validate_checkpoint_dataset_binding(checkpoint, "b" * 64)
