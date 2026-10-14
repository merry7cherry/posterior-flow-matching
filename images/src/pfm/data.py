"""Native image preprocessing with explicit dataset paths."""
import multiprocessing
from pathlib import Path
import numpy as np
import torch
from PIL import Image
from torchvision import datasets, transforms
from torchvision.transforms import functional as transform_functional

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



def make_dataset(config, data_root, *, train, download=False):
    root = Path(data_root)
    if config.dataset in {"mnist", "cifar10"}:
        cls = datasets.MNIST if config.dataset == "mnist" else datasets.CIFAR10
        return cls(root=root, train=train, transform=image_transform(), download=download)
    if config.dataset == "imagenet":
        if train:
            return EpochDeterministicImageFolder(root=root / "train", image_size=config.image_size,
                                                 augmentation_seed=config.seed)
        return datasets.ImageFolder(root=root / "val", transform=imagenet_transform(config.image_size))
    raise ValueError(f"unknown dataset: {config.dataset}")


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
