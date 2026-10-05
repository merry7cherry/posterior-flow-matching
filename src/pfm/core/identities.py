from __future__ import annotations

from collections.abc import Mapping


CANONICAL_DATASETS = ("cifar10", "imagenet", "mnist")
CANONICAL_ARCHITECTURES = ("conditional_unet", "edm_ddpmpp", "jit_transformer")

# Facts about the supported Dataset readers, not experimental hyperparameters.
DATASET_FIXED_ATTRIBUTES = {
    "mnist": {"num_classes": 10, "image_channels": 1, "image_size": 28},
    "cifar10": {"num_classes": 10, "image_channels": 3, "image_size": 32},
    "imagenet": {"num_classes": 1000, "image_channels": 3},
}


def validate_dataset_attributes(dataset: str, attributes: Mapping[str, object]) -> None:
    fixed = DATASET_FIXED_ATTRIBUTES[validate_dataset_identity(dataset)]
    for field_name, expected in fixed.items():
        if attributes[field_name] != expected:
            raise ValueError(
                f"{dataset} requires {field_name}={expected}; it is a fixed Dataset attribute"
            )


def validate_dataset_identity(value: object) -> str:
    if not isinstance(value, str) or value not in CANONICAL_DATASETS:
        valid = ", ".join(CANONICAL_DATASETS)
        raise ValueError(f"dataset must be exactly one of: {valid}")
    return value


def validate_architecture_identity(value: object) -> str:
    if not isinstance(value, str) or value not in CANONICAL_ARCHITECTURES:
        valid = ", ".join(CANONICAL_ARCHITECTURES)
        raise ValueError(f"architecture must be exactly one of: {valid}")
    return value
