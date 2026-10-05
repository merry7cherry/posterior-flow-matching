from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable

from pfm.core.config import ExperimentConfig
from pfm.core.evaluation_config import (
    GeneratedCommonEvaluationConfig,
    MetricEvaluationConfig,
    PosteriorDiagnosticsEvaluationConfig,
)
from pfm.core.identities import validate_dataset_identity
from pfm.core.metric_names import (
    CONDITIONAL_ACCURACY_METRIC_NAME,
    FID_METRIC_NAME,
    GENERATED_METRIC_NAMES,
    INCEPTION_SCORE_METRIC_NAME,
    POSTERIOR_CALIBRATION_METRIC_NAME,
    POSTERIOR_CONTROLLED_METRIC_NAME,
    POSTERIOR_METRIC_NAMES,
    validate_metric_names,
)
from pfm.core.variant_metadata import requires_learned_variance


@dataclass(frozen=True)
class MetricProfile:
    dataset: str
    image_size: int
    metrics: tuple[str, ...]
    classifier_asset_id: str | None
    fid_asset_id: str | None
    fid_backend: str


def _dataset_image_identity(config: ExperimentConfig) -> tuple[str, int]:
    dataset = validate_dataset_identity(config.dataset)
    key = (dataset, int(config.image_size))
    if key not in {("mnist", 28), ("cifar10", 32), ("imagenet", 256)}:
        raise ValueError(f"no evaluation runtime exists for dataset/image_size {key!r}")
    return key


def resolve_metric_names(metric_names: Iterable[str], config: ExperimentConfig) -> tuple[str, ...]:
    selected = validate_metric_names(metric_names) or ()
    _dataset_image_identity(config)
    return selected


def validate_metric_request(
    metric_names: Iterable[str],
    config: ExperimentConfig,
    *,
    variant: str,
) -> tuple[str, ...]:
    selected = resolve_metric_names(metric_names, config)
    if any(metric in POSTERIOR_METRIC_NAMES for metric in selected):
        requires_learned_variance(variant, feature="posterior metrics")
    return selected


def validate_evaluation_asset_bindings(
    experiment_config: ExperimentConfig,
    evaluation_config: MetricEvaluationConfig,
) -> MetricProfile:
    dataset, image_size = _dataset_image_identity(experiment_config)
    selected = resolve_metric_names(evaluation_config.metric_names, experiment_config)
    if isinstance(evaluation_config, PosteriorDiagnosticsEvaluationConfig):
        if any(metric not in POSTERIOR_METRIC_NAMES for metric in selected):
            raise ValueError("posterior_diagnostics may contain only posterior metrics")
        return MetricProfile(dataset, image_size, selected, None, None, "")
    if not isinstance(evaluation_config, GeneratedCommonEvaluationConfig):
        raise TypeError("evaluation_config must be a closed checkpoint-evaluation profile")
    return MetricProfile(
        dataset,
        image_size,
        selected,
        evaluation_config.classifier_asset_id,
        evaluation_config.fid_asset_id,
        evaluation_config.fid_backend,
    )


def generated_metric_names(metric_names: Iterable[str]) -> tuple[str, ...]:
    selected = validate_metric_names(metric_names) or ()
    return tuple(name for name in selected if name in GENERATED_METRIC_NAMES)
