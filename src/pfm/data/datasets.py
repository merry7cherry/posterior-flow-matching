from __future__ import annotations

import json
import multiprocessing
from pathlib import Path
from typing import Callable

import numpy as np
import torch
from PIL import Image
from torch.utils.data import DataLoader, RandomSampler
from torchvision import datasets, transforms
from torchvision.transforms import functional as transform_functional

from pfm.core.config import ExperimentConfig
from pfm.core.identities import validate_dataset_identity
from pfm.data import identity as dataset_identity
from pfm.data.identity import validate_dataset_manifest_payload


DatasetFactory = Callable[[ExperimentConfig, Path], tuple[object, object]]
_UINT64_MASK = (1 << 64) - 1
_SPLITMIX_INCREMENT = 0x9E3779B97F4A7C15
_SPLITMIX_MULTIPLIER_1 = 0xBF58476D1CE4E5B9
_SPLITMIX_MULTIPLIER_2 = 0x94D049BB133111EB


class ToTensorIfNeeded:
    def __call__(self, x) -> torch.Tensor:
        if isinstance(x, torch.Tensor):
            return x.float()
        return transform_functional.to_tensor(x)


class ScaleToMinusOneOne:
    def __call__(self, x: torch.Tensor) -> torch.Tensor:
        return x * 2.0 - 1.0


class CenterCropArr:
    def __init__(self, image_size: int) -> None:
        self.image_size = int(image_size)

    def __call__(self, image: Image.Image) -> Image.Image:
        return center_crop_arr(image, self.image_size)


def image_transform() -> transforms.Compose:
    return transforms.Compose(
        [
            ToTensorIfNeeded(),
            ScaleToMinusOneOne(),
        ]
    )


def mnist_transform() -> transforms.Compose:
    return image_transform()


def cifar10_transform() -> transforms.Compose:
    return image_transform()


def imagenet_transform(image_size: int) -> transforms.Compose:
    return transforms.Compose(
        [
            CenterCropArr(image_size),
            transforms.ToTensor(),
            ScaleToMinusOneOne(),
        ]
    )


class EpochDeterministicImageFolder(datasets.ImageFolder):
    """ImageFolder with stateless, epoch-addressed 50% horizontal flips."""

    def __init__(self, *, root: Path, image_size: int, augmentation_seed: int) -> None:
        super().__init__(root=root, transform=imagenet_transform(image_size))
        self._augmentation_seed = int(augmentation_seed)
        self._epoch = multiprocessing.Value("q", 0, lock=False)

    def set_epoch(self, epoch: int) -> None:
        epoch_value = int(epoch)
        if epoch_value < 0:
            raise ValueError("dataset epoch must be non-negative")
        self._epoch.value = epoch_value

    def __getitem__(self, index: int) -> tuple[torch.Tensor, int]:
        epoch = int(self._epoch.value)
        image, label = super().__getitem__(index)
        if _stateless_horizontal_flip(
            seed=self._augmentation_seed,
            epoch=epoch,
            sample_index=int(index),
        ):
            image = transform_functional.hflip(image)
        return image, label


def set_training_dataset_epoch(dataset: object, epoch: int) -> None:
    set_epoch = getattr(dataset, "set_epoch", None)
    if set_epoch is not None:
        set_epoch(int(epoch))


def _stateless_horizontal_flip(*, seed: int, epoch: int, sample_index: int) -> bool:
    if epoch < 0 or sample_index < 0:
        raise ValueError("epoch and sample_index must be non-negative")
    value = (
        (int(seed) & _UINT64_MASK)
        + _SPLITMIX_INCREMENT * (int(epoch) + 1)
        + _SPLITMIX_MULTIPLIER_1 * (int(sample_index) + 1)
    ) & _UINT64_MASK
    return bool(_splitmix64(value) & 1)


def _splitmix64(value: int) -> int:
    value = ((value ^ (value >> 30)) * _SPLITMIX_MULTIPLIER_1) & _UINT64_MASK
    value = ((value ^ (value >> 27)) * _SPLITMIX_MULTIPLIER_2) & _UINT64_MASK
    value ^= value >> 31
    return value & _UINT64_MASK


def _dataloader_worker_generator(seed: int, *, stream: int) -> torch.Generator:
    generator = torch.Generator()
    generator.manual_seed((int(seed) + int(stream) * _SPLITMIX_INCREMENT) & _UINT64_MASK)
    return generator


def make_dataloaders(config: ExperimentConfig, data_root: str | Path) -> tuple[DataLoader, DataLoader]:
    train, test = make_datasets(config, data_root)
    return make_dataloaders_from_datasets(train, test, config)


def make_datasets(config: ExperimentConfig, data_root: str | Path) -> tuple[object, object]:
    dataset_name = validate_dataset_identity(config.dataset)
    factory = DATASET_REGISTRY.get(dataset_name)
    if factory is not None:
        prepared_root = Path(data_root)
        validate_dataset_manifest(prepared_root, dataset_name)
        return factory(config, prepared_root)
    raise ValueError(f"Unsupported dataset {config.dataset!r}; expected one of {DATASET_NAMES}")


def _make_mnist_datasets(config: ExperimentConfig, data_root: Path) -> tuple[object, object]:
    transform = mnist_transform()
    train = datasets.MNIST(root=data_root, train=True, transform=transform, download=False)
    test = datasets.MNIST(root=data_root, train=False, transform=transform, download=False)
    return train, test


def _make_cifar10_datasets(config: ExperimentConfig, data_root: Path) -> tuple[object, object]:
    transform = cifar10_transform()
    train = datasets.CIFAR10(root=data_root, train=True, transform=transform, download=False)
    test = datasets.CIFAR10(root=data_root, train=False, transform=transform, download=False)
    return train, test


def _make_imagenet_datasets(config: ExperimentConfig, data_root: Path) -> tuple[object, object]:
    train_root = data_root / "train"
    val_root = data_root / "val"
    if not train_root.exists() or not val_root.exists():
        raise FileNotFoundError(f"ImageNet data_root must contain train/ and val/: {data_root}")
    train = EpochDeterministicImageFolder(
        root=train_root,
        image_size=config.image_size,
        augmentation_seed=config.seed,
    )
    val = datasets.ImageFolder(root=val_root, transform=imagenet_transform(config.image_size))
    for split, dataset in {"train": train, "val": val}.items():
        sample_count = len(dataset)
        expected_count = dataset_identity.IMAGENET_SPLIT_COUNTS[split]
        if sample_count != expected_count:
            raise ValueError(
                f"ImageNet {split} must contain exactly {expected_count} samples; "
                f"found {sample_count}"
            )
    return train, val


DATASET_REGISTRY: dict[str, DatasetFactory] = {
    "mnist": _make_mnist_datasets,
    "cifar10": _make_cifar10_datasets,
    "imagenet": _make_imagenet_datasets,
}
DATASET_NAMES = tuple(sorted(DATASET_REGISTRY))


def validate_dataset_manifest(data_root: str | Path, dataset_name: str) -> dict[str, object]:
    data_root = Path(data_root)
    manifest_path = data_root / dataset_identity.DATASET_MANIFEST_NAME
    if not manifest_path.exists():
        raise FileNotFoundError(f"prepared dataset manifest not found: {manifest_path}")
    with manifest_path.open("r", encoding="utf-8") as handle:
        manifest = json.load(handle)
    if not isinstance(manifest, dict):
        raise ValueError(f"{manifest_path} must contain a JSON object")
    return validate_dataset_manifest_payload(
        manifest,
        expected_dataset=dataset_name,
    )


def make_dataloaders_from_datasets(
    train,
    test,
    config: ExperimentConfig,
    train_sampler=None,
    test_sampler=None,
    shuffle_train: bool = True,
) -> tuple[DataLoader, DataLoader]:
    if train_sampler is not None:
        shuffle_train = False
    elif shuffle_train:
        # Keep sample-order randomness on the rank RNG restored by checkpoints;
        # the DataLoader generator below is reserved for worker initialization.
        train_sampler = RandomSampler(train)
        shuffle_train = False
    train_loader = DataLoader(
        train,
        batch_size=config.batch_size,
        shuffle=bool(shuffle_train),
        sampler=train_sampler,
        num_workers=config.num_workers,
        pin_memory=torch.cuda.is_available(),
        drop_last=bool(config.drop_last),
        persistent_workers=config.num_workers > 0,
        generator=_dataloader_worker_generator(config.seed, stream=0),
    )
    test_loader = DataLoader(
        test,
        batch_size=config.batch_size,
        shuffle=False,
        sampler=test_sampler,
        num_workers=config.num_workers,
        pin_memory=torch.cuda.is_available(),
        persistent_workers=config.num_workers > 0,
        generator=_dataloader_worker_generator(config.seed, stream=1),
    )
    return train_loader, test_loader


def to_display_range(x: torch.Tensor) -> torch.Tensor:
    return (x.clamp(-1.0, 1.0) + 1.0) / 2.0


def center_crop_arr(image, image_size: int):
    """ADM/JiT center crop used for ImageNet pixel-space diffusion."""
    size = int(image_size)
    resampling = Image.Resampling
    while min(*image.size) >= 2 * size:
        image = image.resize(tuple(component // 2 for component in image.size), resample=resampling.BOX)
    scale = size / min(*image.size)
    image = image.resize(tuple(round(component * scale) for component in image.size), resample=resampling.BICUBIC)
    array = np.array(image)
    crop_y = (array.shape[0] - size) // 2
    crop_x = (array.shape[1] - size) // 2
    return Image.fromarray(array[crop_y : crop_y + size, crop_x : crop_x + size])
