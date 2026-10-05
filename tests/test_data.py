import numpy as np
import pytest
import torch
from PIL import Image
from torch.utils.data import TensorDataset

import pfm.data.datasets as data_module
from conftest import write_test_dataset_manifest
from pfm.core.config import ExperimentConfig


def _runtime_identity():
    return {
        "project_name": "pfm",
        "git_commit": "a" * 40,
        "rank": 0,
        "world_size": 1,
    }


class _FakeVisionDataset(TensorDataset):
    calls: list[dict[str, object]] = []
    classes = [str(index) for index in range(10)]

    def __init__(self, root, train, transform, download):
        self.calls.append({"root": root, "train": train, "download": download})
        images = torch.rand(4, 3, 32, 32) if train else torch.rand(2, 3, 32, 32)
        labels = torch.arange(images.shape[0]) % 10
        self.targets = labels.tolist()
        if transform is not None:
            images = torch.stack([transform(image) for image in images])
        super().__init__(images, labels)


def test_make_dataloaders_selects_cifar10_and_returns_rgb_batches(tmp_path, monkeypatch):
    _FakeVisionDataset.calls = []
    monkeypatch.setattr(data_module.datasets, "CIFAR10", _FakeVisionDataset)
    data_root = tmp_path / "cifar10"
    write_test_dataset_manifest("cifar10", data_root)
    config = ExperimentConfig(
        dataset="cifar10",
        image_channels=3,
        image_size=32,
        batch_size=2,
        num_workers=0,
    )

    train_loader, test_loader = data_module.make_dataloaders(config, data_root)
    images, labels = next(iter(train_loader))
    test_images, _ = next(iter(test_loader))

    assert images.shape == (2, 3, 32, 32)
    assert test_images.shape == (2, 3, 32, 32)
    assert labels.dtype == torch.long
    assert float(images.min()) >= -1.0
    assert float(images.max()) <= 1.0
    assert [call["download"] for call in _FakeVisionDataset.calls] == [
        False,
        False,
        False,
        False,
    ]


def test_make_dataloaders_rejects_unknown_dataset(tmp_path):
    config = ExperimentConfig(dataset="unknown")

    try:
        data_module.make_dataloaders(config, tmp_path / "unknown")
    except ValueError as error:
        assert "dataset must be exactly one of" in str(error)
    else:
        raise AssertionError("unknown dataset should be rejected")


def test_center_crop_arr_uses_adm_resize_then_center_crop():
    source = np.arange(8 * 4 * 3, dtype=np.uint8).reshape(4, 8, 3)
    image = Image.fromarray(source)

    cropped = data_module.center_crop_arr(image, 4)

    assert cropped.size == (4, 4)
    assert np.array_equal(np.array(cropped), source[:, 2:6, :])


def test_center_crop_arr_requires_pillow_resampling_api(monkeypatch):
    image_module = data_module.Image

    class ImageWithoutResampling:
        def __getattr__(self, name):
            if name == "Resampling":
                raise AttributeError(name)
            return getattr(image_module, name)

    monkeypatch.setattr(data_module, "Image", ImageWithoutResampling())
    image = Image.fromarray(np.zeros((4, 4, 3), dtype=np.uint8))

    with pytest.raises(AttributeError, match="Resampling"):
        data_module.center_crop_arr(image, 4)
