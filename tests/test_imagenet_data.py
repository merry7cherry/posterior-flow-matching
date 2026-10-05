from pathlib import Path

import numpy as np
import pytest
import torch
from PIL import Image
from torch.utils.data.distributed import DistributedSampler

from conftest import write_test_dataset_manifest
from pfm.core.config import ExperimentConfig
from pfm.data.datasets import (
    make_dataloaders,
    make_dataloaders_from_datasets,
    make_datasets,
    set_training_dataset_epoch,
)


def _runtime_identity():
    return {
        "project_name": "pfm",
        "git_commit": "a" * 40,
        "rank": 0,
        "world_size": 1,
    }


def _write_rgb(path: Path, value: tuple[int, int, int]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    Image.new("RGB", (20, 16), color=value).save(path)


def _write_asymmetric_rgb(path: Path, offset: int) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    array = np.zeros((16, 20, 3), dtype=np.uint8)
    array[:, :10, 0] = 32 + offset
    array[:, 10:, 2] = 224 - offset
    Image.fromarray(array, mode="RGB").save(path)


def _make_imagenet_tree(root: Path) -> None:
    for split in ("train", "val"):
        _write_rgb(root / split / "00000000" / "a.png", (255, 0, 0))
        _write_rgb(root / split / "00000001" / "b.png", (0, 255, 0))


def test_imagenet_datasets_use_imagefolder_train_val_layout(tmp_path, monkeypatch):
    data_root = tmp_path / "imagenet"
    _make_imagenet_tree(data_root)
    monkeypatch.setattr(
        "pfm.data.identity.IMAGENET_SPLIT_COUNTS",
        {"train": 2, "val": 2},
    )
    write_test_dataset_manifest("imagenet", data_root)
    config = ExperimentConfig(dataset="imagenet", image_channels=3, image_size=16)

    train, val = make_datasets(config, data_root)

    assert len(train) == 2
    assert len(val) == 2
    image, label = train[0]
    assert image.shape == (3, 16, 16)
    assert image.dtype == torch.float32
    assert image.min() >= -1.0
    assert image.max() <= 1.0
    assert label in {0, 1}


def test_imagenet_loader_does_not_download_and_preserves_no_drop_last(
    tmp_path,
    monkeypatch,
):
    data_root = tmp_path / "imagenet"
    _make_imagenet_tree(data_root)
    monkeypatch.setattr(
        "pfm.data.identity.IMAGENET_SPLIT_COUNTS",
        {"train": 2, "val": 2},
    )
    write_test_dataset_manifest("imagenet", data_root)
    config = ExperimentConfig(
        dataset="imagenet",
        image_channels=3,
        image_size=16,
        batch_size=3,
        num_workers=0,
        drop_last=False,
    )

    train_loader, val_loader = make_dataloaders(config, data_root)

    train_batch = next(iter(train_loader))
    val_batch = next(iter(val_loader))
    assert train_batch[0].shape[0] == 2
    assert val_batch[0].shape[0] == 2


def test_imagenet_loader_rejects_split_count_drift_after_preparation(
    tmp_path,
    monkeypatch,
):
    data_root = tmp_path / "imagenet"
    _make_imagenet_tree(data_root)
    monkeypatch.setattr(
        "pfm.data.identity.IMAGENET_SPLIT_COUNTS",
        {"train": 2, "val": 2},
    )
    write_test_dataset_manifest("imagenet", data_root)
    (data_root / "val" / "00000001" / "b.png").unlink()
    (data_root / "val" / "00000001").rmdir()
    config = ExperimentConfig(dataset="imagenet", image_channels=3, image_size=16)

    with pytest.raises(
        ValueError,
        match="ImageNet val must contain exactly 2 samples; found 1",
    ):
        make_datasets(config, data_root)


def test_imagenet_persistent_workers_reconstruct_epoch_augmentation_after_resume(
    tmp_path,
    monkeypatch,
):
    data_root = tmp_path / "imagenet"
    for split in ("train", "val"):
        for index in range(4):
            _write_asymmetric_rgb(
                data_root / split / f"{index:08d}" / "sample.png",
                offset=index,
            )
    monkeypatch.setattr(
        "pfm.data.identity.IMAGENET_SPLIT_COUNTS",
        {"train": 4, "val": 4},
    )
    write_test_dataset_manifest("imagenet", data_root)
    config = ExperimentConfig(
        dataset="imagenet",
        image_channels=3,
        image_size=16,
        batch_size=2,
        num_workers=1,
        drop_last=False,
        seed=17,
    )

    torch.manual_seed(1234)
    uninterrupted_dataset, validation_dataset = make_datasets(config, data_root)
    uninterrupted_sampler = DistributedSampler(
        uninterrupted_dataset,
        num_replicas=1,
        rank=0,
        shuffle=True,
        drop_last=False,
    )
    uninterrupted_loader, _ = make_dataloaders_from_datasets(
        uninterrupted_dataset,
        validation_dataset,
        config,
        train_sampler=uninterrupted_sampler,
    )
    set_training_dataset_epoch(uninterrupted_dataset, 1)
    uninterrupted_sampler.set_epoch(1)
    epoch_one = torch.cat([images for images, _ in uninterrupted_loader])
    checkpoint_rank_rng = torch.get_rng_state()
    set_training_dataset_epoch(uninterrupted_dataset, 2)
    uninterrupted_sampler.set_epoch(2)
    uninterrupted_epoch_two = torch.cat([images for images, _ in uninterrupted_loader])
    uninterrupted_rank_rng = torch.get_rng_state()

    resumed_dataset, resumed_validation_dataset = make_datasets(config, data_root)
    resumed_sampler = DistributedSampler(
        resumed_dataset,
        num_replicas=1,
        rank=0,
        shuffle=True,
        drop_last=False,
    )
    resumed_loader, _ = make_dataloaders_from_datasets(
        resumed_dataset,
        resumed_validation_dataset,
        config,
        train_sampler=resumed_sampler,
    )
    torch.set_rng_state(checkpoint_rank_rng)
    set_training_dataset_epoch(resumed_dataset, 2)
    resumed_sampler.set_epoch(2)
    resumed_epoch_two = torch.cat([images for images, _ in resumed_loader])

    assert not torch.equal(epoch_one, uninterrupted_epoch_two)
    assert torch.equal(uninterrupted_epoch_two, resumed_epoch_two)
    assert torch.equal(uninterrupted_rank_rng, torch.get_rng_state())
