from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import asdict, dataclass, fields
import hashlib
import hmac
import json
import math
from pathlib import Path

import torch
import yaml
from torch import nn

from pfm.core.checkpointing import validate_runtime_identity_payload
from pfm.core.config import validate_mixed_precision


CLASSIFIER_CHECKPOINT_KEYS = frozenset({"model_state"})
CLASSIFIER_ASSET_CHECKPOINT_NAME = "classifier.pt"
CLASSIFIER_ASSET_MANIFEST_NAME = "classifier_manifest.json"
CLASSIFIER_DATASET_SHAPES = {
    "mnist": (10, 1, 28),
}
CLASSIFIER_PREPROCESSING_PROTOCOL = "pfm_classifier_preprocessing_v1"
CLASSIFIER_VALIDATION_METRIC_KEYS = frozenset(
    {
        "test_accuracy",
        "test_loss",
        "per_class_accuracy",
        "per_class_correct",
        "per_class_total",
    }
)
CLASSIFIER_ASSET_MANIFEST_KEYS = frozenset(
    {
        "classifier_asset_id",
        "paths",
        "dataset",
        "image_size",
        "num_classes",
        "training_split",
        "validation_split",
        "config",
        "optimizer",
        "preprocessing_identity",
        "class_to_idx",
        "class_mapping_sha256",
        "dataset_scientific_identity_sha256",
        "checkpoint_sha256",
        "producer_runtime_identity",
        "validation_metrics",
    }
)


@dataclass
class ClassifierConfig:
    seed: int = 123
    dataset: str = "mnist"
    batch_size: int = 512
    num_workers: int = 0
    mixed_precision: str = "none"
    epochs: int = 20
    lr: float = 1e-3
    weight_decay: float = 1e-4
    feature_dim: int = 256
    num_classes: int = 10
    image_channels: int = 1
    image_size: int = 28
    pool_size: int = 7
    conv_channels: tuple[int, int, int] = (64, 128, 128)
    norm: str = "group"


CLASSIFIER_CONFIG_KEYS = frozenset(field.name for field in fields(ClassifierConfig))


class FeatureClassifier(nn.Module):
    def __init__(
        self,
        feature_dim: int = 256,
        num_classes: int = 10,
        image_channels: int = 1,
        pool_size: int = 7,
        conv_channels: tuple[int, int, int] = (64, 128, 128),
        norm: str = "group",
    ) -> None:
        super().__init__()
        self.feature_dim = int(feature_dim)
        self.num_classes = int(num_classes)
        self.image_channels = int(image_channels)
        self.pool_size = int(pool_size)
        self.conv_channels = tuple(int(channel) for channel in conv_channels)
        self.norm = str(norm)
        self.features = self._make_feature_extractor(self.image_channels, self.conv_channels, self.norm)
        self.projection = nn.Sequential(
            nn.AdaptiveAvgPool2d((self.pool_size, self.pool_size)),
            nn.Flatten(),
            nn.Linear(self.conv_channels[-1] * self.pool_size * self.pool_size, self.feature_dim),
            nn.SiLU(),
        )
        self.classifier = nn.Linear(self.feature_dim, self.num_classes)

    @classmethod
    def from_config(cls, config: ClassifierConfig) -> "FeatureClassifier":
        return cls(
            feature_dim=config.feature_dim,
            num_classes=config.num_classes,
            image_channels=config.image_channels,
            pool_size=config.pool_size,
            conv_channels=config.conv_channels,
            norm=config.norm,
        )

    def forward_features(self, x: torch.Tensor) -> torch.Tensor:
        return self.projection(self.features(x))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.classifier(self.forward_features(x))

    @staticmethod
    def _make_feature_extractor(image_channels: int, conv_channels: tuple[int, int, int], norm: str) -> nn.Sequential:
        if len(conv_channels) != 3:
            raise ValueError("conv_channels must contain exactly three channel sizes")
        layers: list[nn.Module] = []
        in_channels = int(image_channels)
        for index, out_channels in enumerate(conv_channels):
            layers.append(nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1, bias=norm == "none"))
            layers.append(_make_norm(norm, out_channels))
            layers.append(nn.SiLU())
            if index < 2:
                layers.append(nn.MaxPool2d(2))
            in_channels = out_channels
        return nn.Sequential(*layers)


def load_classifier_config(path: str | Path) -> ClassifierConfig:
    config_path = Path(path)
    with config_path.open("r", encoding="utf-8") as handle:
        raw = yaml.safe_load(handle)
    if not isinstance(raw, Mapping):
        raise ValueError(f"{config_path} classifier config YAML document must be a mapping")
    return ClassifierConfig(**validate_classifier_config_mapping(raw, context="classifier config"))


def classifier_config_to_mapping(config: ClassifierConfig) -> dict[str, object]:
    if not isinstance(config, ClassifierConfig):
        raise TypeError("config must be a ClassifierConfig")
    return validate_classifier_config_mapping(asdict(config), context="classifier config")


def validate_classifier_config_mapping(
    value: object,
    *,
    context: str,
) -> dict[str, object]:
    if not isinstance(value, Mapping):
        raise ValueError(f"{context} config is missing")
    actual_keys = set(value)
    missing = sorted(CLASSIFIER_CONFIG_KEYS - actual_keys)
    unknown = sorted(actual_keys - CLASSIFIER_CONFIG_KEYS)
    if missing or unknown:
        details = []
        if missing:
            details.append(f"missing fields {missing}")
        if unknown:
            details.append(f"unknown fields {unknown}")
        raise ValueError(f"invalid {context} config: {', '.join(details)}")
    data = dict(value)
    data["mixed_precision"] = validate_mixed_precision(data["mixed_precision"])
    for name in (
        "batch_size",
        "epochs",
        "feature_dim",
        "num_classes",
        "image_channels",
        "image_size",
        "pool_size",
    ):
        if isinstance(data[name], bool) or not isinstance(data[name], int) or data[name] < 1:
            raise ValueError(f"{context} config {name} must be a positive integer")
    for name in ("seed", "num_workers"):
        if isinstance(data[name], bool) or not isinstance(data[name], int) or data[name] < 0:
            raise ValueError(f"{context} config {name} must be a non-negative integer")
    for name, allow_zero in (("lr", False), ("weight_decay", True)):
        raw = data[name]
        if isinstance(raw, bool) or not isinstance(raw, (int, float)) or not math.isfinite(raw):
            raise ValueError(f"{context} config {name} must be finite")
        if raw < 0 or (not allow_zero and raw == 0):
            raise ValueError(f"{context} config {name} is outside its valid range")
        data[name] = float(raw)
    channels = data["conv_channels"]
    if not isinstance(channels, (list, tuple)) or len(channels) != 3:
        raise ValueError(f"{context} config conv_channels must contain exactly three integers")
    if any(isinstance(channel, bool) or not isinstance(channel, int) or channel < 1 for channel in channels):
        raise ValueError(f"{context} config conv_channels must contain positive integers")
    data["conv_channels"] = tuple(channels)
    if not isinstance(data["norm"], str) or data["norm"] not in {"group", "batch", "none"}:
        raise ValueError(f"{context} config norm must be one of 'group', 'batch', or 'none'")
    dataset = data["dataset"]
    if not isinstance(dataset, str) or dataset not in CLASSIFIER_DATASET_SHAPES:
        raise ValueError(
            f"{context} config dataset must be one of: {', '.join(sorted(CLASSIFIER_DATASET_SHAPES))}"
        )
    shape = (data["num_classes"], data["image_channels"], data["image_size"])
    if shape != CLASSIFIER_DATASET_SHAPES[dataset]:
        raise ValueError(
            f"{context} config shape must be exactly {CLASSIFIER_DATASET_SHAPES[dataset]} for {dataset}"
        )
    return asdict(ClassifierConfig(**data))


def classifier_preprocessing_identity(dataset: str, image_size: int) -> dict[str, object]:
    shape = CLASSIFIER_DATASET_SHAPES.get(dataset)
    if shape is None:
        raise ValueError("classifier preprocessing dataset is unsupported")
    if isinstance(image_size, bool) or not isinstance(image_size, int) or image_size != shape[2]:
        raise ValueError("classifier preprocessing image size does not match dataset")

    def role(*, horizontal_flip_probability: float) -> dict[str, object]:
        return {
            "center_crop": None,
            "random_horizontal_flip_probability": horizontal_flip_probability,
        }

    return {
        "protocol": CLASSIFIER_PREPROCESSING_PROTOCOL,
        "dataset": dataset,
        "image_size": image_size,
        "tensor_conversion": "to_float_tensor",
        "normalization": {
            "source_range": [0.0, 1.0],
            "target_range": [-1.0, 1.0],
        },
        "training": role(horizontal_flip_probability=0.0),
        "validation": role(horizontal_flip_probability=0.0),
    }


def classifier_optimizer_identity(config: Mapping[str, object]) -> dict[str, object]:
    return {
        "type": "AdamW",
        "lr": float(config["lr"]),
        "weight_decay": float(config["weight_decay"]),
    }


def classifier_asset_manifest_payload(
    *,
    classifier_asset_id: str,
    dataset: str,
    config: ClassifierConfig,
    class_to_idx: Mapping[str, int],
    training_split: str,
    validation_split: str,
    dataset_scientific_identity_sha256: str,
    checkpoint_sha256: str,
    producer_runtime_identity: Mapping[str, object],
    validation_metrics: Mapping[str, object],
) -> dict[str, object]:
    config_payload = classifier_config_to_mapping(config)
    if config_payload["dataset"] != dataset:
        raise ValueError("classifier asset dataset does not match classifier config dataset")
    class_mapping = validate_classifier_class_mapping(class_to_idx, config.num_classes)
    metrics = validate_classifier_validation_metrics(validation_metrics, config.num_classes)
    resolved_training_split = _validated_split_name(training_split, "training")
    resolved_validation_split = _validated_split_name(validation_split, "validation")
    return {
        "classifier_asset_id": classifier_asset_id,
        "paths": {
            "checkpoint": CLASSIFIER_ASSET_CHECKPOINT_NAME,
            "manifest": CLASSIFIER_ASSET_MANIFEST_NAME,
        },
        "dataset": dataset,
        "image_size": config.image_size,
        "num_classes": config.num_classes,
        "training_split": resolved_training_split,
        "validation_split": resolved_validation_split,
        "config": config_payload,
        "optimizer": classifier_optimizer_identity(config_payload),
        "preprocessing_identity": classifier_preprocessing_identity(
            dataset,
            config.image_size,
        ),
        "class_to_idx": class_mapping,
        "class_mapping_sha256": _canonical_json_sha256(class_mapping),
        "dataset_scientific_identity_sha256": _require_sha256(
            dataset_scientific_identity_sha256,
            "dataset scientific identity",
        ),
        "checkpoint_sha256": _require_sha256(checkpoint_sha256, "checkpoint"),
        "producer_runtime_identity": validate_runtime_identity_payload(
            producer_runtime_identity,
            context="classifier asset manifest",
        ),
        "validation_metrics": metrics,
    }


def validate_classifier_asset_manifest_payload(
    value: object,
    *,
    dataset: str,
    classifier_asset_id: str,
    dataset_scientific_identity_sha256: str,
    dataset_splits: Sequence[str],
) -> tuple[dict[str, object], dict[str, object]]:
    if not isinstance(value, dict):
        raise ValueError("classifier asset manifest must contain a JSON object")
    if set(value) != CLASSIFIER_ASSET_MANIFEST_KEYS:
        raise ValueError(
            "classifier asset manifest keys must be exactly "
            f"{sorted(CLASSIFIER_ASSET_MANIFEST_KEYS)}"
        )
    if value["classifier_asset_id"] != classifier_asset_id:
        raise ValueError("classifier asset manifest id does not match requested asset id")
    if value["paths"] != {
        "checkpoint": CLASSIFIER_ASSET_CHECKPOINT_NAME,
        "manifest": CLASSIFIER_ASSET_MANIFEST_NAME,
    }:
        raise ValueError("classifier asset manifest paths do not match the canonical relative paths")
    if value["dataset"] != dataset:
        raise ValueError("classifier asset manifest dataset does not match requested dataset")
    config = validate_classifier_config_mapping(value["config"], context="classifier asset manifest")
    if config["dataset"] != dataset:
        raise ValueError("classifier asset manifest config dataset does not match requested dataset")
    if value["image_size"] != config["image_size"] or value["num_classes"] != config["num_classes"]:
        raise ValueError("classifier asset manifest shape identity does not match config")
    if (
        not isinstance(dataset_splits, (list, tuple))
        or len(dataset_splits) != 2
    ):
        raise ValueError("classifier asset dataset manifest must provide exactly two ordered splits")
    training_split = _validated_split_name(dataset_splits[0], "training")
    validation_split = _validated_split_name(dataset_splits[1], "validation")
    if value["training_split"] != training_split or value["validation_split"] != validation_split:
        raise ValueError("classifier asset manifest split identity is invalid")
    _validated_optimizer_identity(value["optimizer"], config=config)
    _validated_preprocessing_identity(
        value["preprocessing_identity"],
        dataset=dataset,
        image_size=int(config["image_size"]),
    )
    class_to_idx = validate_classifier_class_mapping(value["class_to_idx"], int(config["num_classes"]))
    class_mapping_sha256 = _require_sha256(value["class_mapping_sha256"], "class mapping")
    if not hmac.compare_digest(class_mapping_sha256, _canonical_json_sha256(class_to_idx)):
        raise ValueError("classifier asset manifest class mapping SHA-256 does not match mapping")
    expected_dataset_identity = _require_sha256(
        dataset_scientific_identity_sha256,
        "current dataset scientific identity",
    )
    dataset_identity = _require_sha256(
        value["dataset_scientific_identity_sha256"],
        "classifier asset dataset scientific identity",
    )
    if not hmac.compare_digest(dataset_identity, expected_dataset_identity):
        raise ValueError("classifier asset dataset scientific identity does not match current manifest")
    validate_runtime_identity_payload(
        value["producer_runtime_identity"],
        context="classifier asset manifest",
    )
    _require_sha256(value["checkpoint_sha256"], "checkpoint")
    metrics = validate_classifier_validation_metrics(
        value["validation_metrics"],
        int(config["num_classes"]),
    )
    return config, metrics


def classifier_checkpoint_payload(model: FeatureClassifier) -> dict[str, object]:
    return {"model_state": model.state_dict()}


def validate_classifier_checkpoint_payload(value: object) -> dict[str, object]:
    if not isinstance(value, Mapping):
        raise ValueError("classifier asset checkpoint must contain a mapping")
    checkpoint = dict(value)
    _validate_checkpoint_keys(checkpoint, context="classifier asset checkpoint")
    if not isinstance(checkpoint["model_state"], Mapping):
        raise ValueError("classifier asset checkpoint model_state must be a mapping")
    return checkpoint


def validate_classifier_validation_metrics(
    value: object,
    num_classes: int,
) -> dict[str, object]:
    if not isinstance(value, Mapping) or set(value) != CLASSIFIER_VALIDATION_METRIC_KEYS:
        raise ValueError(
            "classifier validation metrics keys must be exactly "
            f"{sorted(CLASSIFIER_VALIDATION_METRIC_KEYS)}"
        )
    accuracy = _finite_metric(value["test_accuracy"], "test_accuracy")
    loss = _finite_metric(value["test_loss"], "test_loss")
    if not 0.0 <= accuracy <= 1.0 or loss < 0.0:
        raise ValueError("classifier validation metrics accuracy/loss are outside valid ranges")
    per_class_accuracy = _metric_sequence(value["per_class_accuracy"], num_classes, "per_class_accuracy")
    if any(not 0.0 <= item <= 1.0 for item in per_class_accuracy):
        raise ValueError("classifier validation metrics per-class accuracy is outside [0, 1]")
    per_class_correct = _count_sequence(
        value["per_class_correct"],
        num_classes,
        "per_class_correct",
        allow_zero=True,
    )
    per_class_total = _count_sequence(
        value["per_class_total"],
        num_classes,
        "per_class_total",
        allow_zero=False,
    )
    for index, (class_accuracy, correct, total) in enumerate(
        zip(per_class_accuracy, per_class_correct, per_class_total, strict=True)
    ):
        if correct > total or not math.isclose(class_accuracy, correct / total, rel_tol=1e-9, abs_tol=1e-12):
            raise ValueError(f"classifier validation metrics class {index} counts do not match accuracy")
    if not math.isclose(
        accuracy,
        sum(per_class_correct) / sum(per_class_total),
        rel_tol=1e-9,
        abs_tol=1e-12,
    ):
        raise ValueError("classifier validation metrics overall accuracy does not match per-class counts")
    return {
        "test_accuracy": accuracy,
        "test_loss": loss,
        "per_class_accuracy": per_class_accuracy,
        "per_class_correct": per_class_correct,
        "per_class_total": per_class_total,
    }


@torch.no_grad()
def evaluate_classifier_accuracy(
    model: FeatureClassifier,
    data_loader,
    device: torch.device,
) -> dict[str, object]:
    model.eval()
    num_classes = int(model.num_classes)
    correct = 0
    total = 0
    loss_sum = 0.0
    per_class_correct = [0] * num_classes
    per_class_total = [0] * num_classes
    loss_fn = nn.CrossEntropyLoss(reduction="sum")
    for images, labels in data_loader:
        images = images.to(device)
        labels = labels.to(device)
        logits = model(images)
        loss_sum += float(loss_fn(logits, labels).detach().cpu())
        predictions = logits.argmax(dim=1)
        matches = predictions.eq(labels)
        correct += int(matches.sum().detach().cpu())
        total += int(labels.numel())
        label_counts = torch.bincount(labels.detach().cpu(), minlength=num_classes)
        correct_counts = torch.bincount(labels[matches].detach().cpu(), minlength=num_classes)
        if label_counts.numel() != num_classes or correct_counts.numel() != num_classes:
            raise ValueError("classifier evaluation labels are outside the configured class range")
        per_class_total = [
            current + int(batch_count)
            for current, batch_count in zip(per_class_total, label_counts.tolist(), strict=True)
        ]
        per_class_correct = [
            current + int(batch_count)
            for current, batch_count in zip(per_class_correct, correct_counts.tolist(), strict=True)
        ]
    if total <= 0:
        raise ValueError("classifier evaluation produced no samples")
    if any(class_total <= 0 for class_total in per_class_total):
        raise ValueError("classifier evaluation produced incomplete per-class metrics")
    return {
        "test_accuracy": correct / total,
        "test_loss": loss_sum / total,
        "per_class_accuracy": [
            class_correct / class_total
            for class_correct, class_total in zip(per_class_correct, per_class_total, strict=True)
        ],
        "per_class_correct": per_class_correct,
        "per_class_total": per_class_total,
    }


def _validate_checkpoint_keys(value: object, *, context: str) -> None:
    if not isinstance(value, Mapping) or set(value) != CLASSIFIER_CHECKPOINT_KEYS:
        raise ValueError(f"invalid {context} contract: expected keys {sorted(CLASSIFIER_CHECKPOINT_KEYS)}")


def _finite_metric(value: object, name: str) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ValueError(f"classifier validation metrics {name} must be numeric")
    resolved = float(value)
    if not math.isfinite(resolved):
        raise ValueError(f"classifier validation metrics {name} must be finite")
    return resolved


def _metric_sequence(value: object, length: int, name: str) -> list[float]:
    if not isinstance(value, (list, tuple)) or len(value) != length:
        raise ValueError(f"classifier validation metrics {name} must contain one value per class")
    return [_finite_metric(item, name) for item in value]


def _count_sequence(value: object, length: int, name: str, *, allow_zero: bool) -> list[int]:
    if not isinstance(value, (list, tuple)) or len(value) != length:
        raise ValueError(f"classifier validation metrics {name} must contain one value per class")
    counts = list(value)
    minimum = 0 if allow_zero else 1
    if any(isinstance(item, bool) or not isinstance(item, int) or item < minimum for item in counts):
        raise ValueError(f"classifier validation metrics {name} contains invalid counts")
    return counts


def _validated_split_name(value: object, role: str) -> str:
    if not isinstance(value, str) or not value or value != value.strip():
        raise ValueError(f"classifier asset {role} split must be a non-empty canonical string")
    return value


def _validated_preprocessing_identity(
    value: object,
    *,
    dataset: str,
    image_size: int,
) -> dict[str, object]:
    identity = _validated_json_mapping(value, "preprocessing identity")
    if identity != classifier_preprocessing_identity(dataset, image_size):
        raise ValueError(
            "classifier asset preprocessing identity does not match the authored protocol"
        )
    return identity


def _validated_optimizer_identity(
    value: object,
    *,
    config: Mapping[str, object],
) -> dict[str, object]:
    identity = _validated_json_mapping(value, "optimizer identity")
    if set(identity) != {"type", "lr", "weight_decay"}:
        raise ValueError("classifier asset optimizer identity keys are invalid")
    if identity["type"] != "AdamW":
        raise ValueError("classifier asset optimizer type must be AdamW")
    for name in ("lr", "weight_decay"):
        value = identity[name]
        if (
            isinstance(value, bool)
            or not isinstance(value, (int, float))
            or not math.isfinite(value)
            or float(value) != float(config[name])
        ):
            raise ValueError(
                f"classifier asset optimizer {name} must match classifier config"
            )
    return classifier_optimizer_identity(config)


def _validated_json_mapping(value: object, name: str) -> dict[str, object]:
    if not isinstance(value, Mapping):
        raise ValueError(f"classifier asset {name} must be a mapping")
    try:
        serialized = json.dumps(
            dict(value),
            sort_keys=True,
            separators=(",", ":"),
            ensure_ascii=False,
            allow_nan=False,
        )
    except (TypeError, ValueError) as exc:
        raise ValueError(f"classifier asset {name} must contain stable JSON values") from exc
    normalized = json.loads(serialized)
    if not isinstance(normalized, dict):
        raise ValueError(f"classifier asset {name} must be a JSON object")
    return normalized


def validate_classifier_class_mapping(value: object, num_classes: int) -> dict[str, int]:
    if not isinstance(value, Mapping):
        raise ValueError("classifier class mapping must be a mapping")
    mapping = dict(value)
    if len(mapping) != num_classes:
        raise ValueError("classifier class mapping must contain every class")
    if any(not isinstance(name, str) or not name for name in mapping):
        raise ValueError("classifier class mapping names must be non-empty strings")
    if any(isinstance(index, bool) or not isinstance(index, int) for index in mapping.values()):
        raise ValueError("classifier class mapping indices must be integers")
    if set(mapping.values()) != set(range(num_classes)):
        raise ValueError("classifier class mapping indices must be unique and contiguous")
    return mapping


def _canonical_json_sha256(value: object) -> str:
    encoded = json.dumps(
        value,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=False,
        allow_nan=False,
    ).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def _require_sha256(value: object, name: str) -> str:
    if not _is_sha256(value):
        raise ValueError(f"classifier asset {name} SHA-256 is invalid")
    return value


def _is_sha256(value: object) -> bool:
    return (
        isinstance(value, str)
        and len(value) == 64
        and all(character in "0123456789abcdef" for character in value)
    )


def _make_norm(norm: str, channels: int) -> nn.Module:
    if norm == "group":
        groups = min(8, channels)
        while channels % groups != 0:
            groups -= 1
        return nn.GroupNorm(groups, channels)
    if norm == "batch":
        return nn.BatchNorm2d(channels)
    if norm == "none":
        return nn.Identity()
    raise ValueError("norm must be one of 'group', 'batch', or 'none'")
