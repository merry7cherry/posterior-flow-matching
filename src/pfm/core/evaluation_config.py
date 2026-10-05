from __future__ import annotations

from collections.abc import Iterable, Mapping
from dataclasses import dataclass
from pathlib import Path
import struct
from typing import Any

from pfm.core.sampling_config import SamplingConfig
from pfm.core.inference_config import InferenceExecutionConfig
from pfm.core.layout import validate_path_component
from pfm.core.metric_names import (
    GENERATED_METRIC_NAMES,
    POSTERIOR_METRIC_NAMES,
    validate_metric_names,
)
from pfm.core.validation import require_finite_number
from pfm.core.yaml_config import load_yaml_mapping


GENERATED_COMMON_PROFILE = "generated_common"
POSTERIOR_DIAGNOSTICS_PROFILE = "posterior_diagnostics"


def validate_posterior_times(values: Iterable[object]) -> tuple[float, ...]:
    """Preserve authored order and values; require distinct interior FP32 times."""
    times = tuple(
        require_finite_number(value, "posterior_t_values", context="posterior diagnostics")
        for value in values
    )
    if not times:
        raise ValueError("posterior_t_values must be non-empty")
    seen = set()
    for value in times:
        if not 0.0 < value < 1.0:
            raise ValueError("posterior_t_values must stay strictly between zero and one")
        executed = struct.unpack("f", struct.pack("f", value))[0]
        if not 0.0 < executed < 1.0:
            raise ValueError("posterior_t_values must stay strictly between zero and one in FP32")
        if executed in seen:
            raise ValueError("posterior_t_values must be unique, including after FP32 conversion")
        seen.add(executed)
    return times


def posterior_time_key(value: float) -> str:
    """Round-trip identity for a validated authored time, also used to seed noise."""
    return repr(float(value))


class MetricEvaluationConfig:
    """Marker base for the two closed checkpoint-evaluation profiles."""


@dataclass(frozen=True)
class GeneratedCommonEvaluationConfig(MetricEvaluationConfig):
    profile: str
    metric_names: tuple[str, ...]
    generated_population: int
    sampling: SamplingConfig
    batch_size: int
    fid_batch_size: int
    fid_backend: str
    seed: int
    classifier_asset_id: str | None
    fid_asset_id: str | None

    def __post_init__(self) -> None:
        if self.profile != GENERATED_COMMON_PROFILE:
            raise ValueError(f"profile must equal {GENERATED_COMMON_PROFILE}")
        names = _metric_tuple(self.metric_names)
        if not names or not set(names).issubset(GENERATED_METRIC_NAMES):
            raise ValueError("generated_common may contain only generated quality metrics")
        _require_positive_int(self.generated_population, "generated_population")
        if not isinstance(self.sampling, SamplingConfig):
            raise TypeError("sampling must be a SamplingConfig")
        _require_positive_int(self.batch_size, "batch_size")
        _require_positive_int(self.fid_batch_size, "fid_batch_size")
        _validate_common_identity(self.seed)
        if self.fid_backend not in {"inception", "jit_imagenet"}:
            raise ValueError("fid_backend must be one of: inception, jit_imagenet")
        _validate_asset_id(self.classifier_asset_id, "classifier_asset_id")
        _validate_asset_id(self.fid_asset_id, "fid_asset_id")
        needs_classifier = "conditional_accuracy" in names
        needs_fid = bool({"fid", "inception_score"}.intersection(names))
        if needs_classifier != (self.classifier_asset_id is not None):
            raise ValueError("classifier asset must be bound exactly for conditional_accuracy")
        if needs_fid != (self.fid_asset_id is not None):
            raise ValueError("FID asset must be bound exactly for fid/inception_score")


@dataclass(frozen=True)
class PosteriorDiagnosticsEvaluationConfig(MetricEvaluationConfig):
    profile: str
    metric_names: tuple[str, ...]
    posterior_population: int
    batch_size: int
    num_bins: int
    posterior_t_values: tuple[float, ...]
    seed: int
    precision: str = "fp32"
    compile: bool = False
    num_workers: int = 0
    allow_tf32: bool = False
    cudnn_benchmark: bool = True

    def __post_init__(self) -> None:
        if self.profile != POSTERIOR_DIAGNOSTICS_PROFILE:
            raise ValueError(f"profile must equal {POSTERIOR_DIAGNOSTICS_PROFILE}")
        if _metric_tuple(self.metric_names) != POSTERIOR_METRIC_NAMES:
            raise ValueError(
                "posterior_diagnostics must request controlled and random-time suites together"
            )
        _require_positive_int(self.posterior_population, "posterior_population")
        _require_positive_int(self.batch_size, "batch_size")
        _require_positive_int(self.num_bins, "num_bins")
        _ = self.execution  # validate the shared execution choices
        if type(self.num_workers) is not int or self.num_workers < 0:
            raise ValueError("num_workers must be a nonnegative integer")
        if not isinstance(self.posterior_t_values, tuple) or not self.posterior_t_values:
            raise ValueError("posterior_t_values must be a non-empty tuple")
        object.__setattr__(self, "posterior_t_values", validate_posterior_times(self.posterior_t_values))
        _validate_common_identity(self.seed)

    @property
    def execution(self) -> InferenceExecutionConfig:
        return InferenceExecutionConfig(self.precision, self.compile, self.allow_tf32, self.cudnn_benchmark)


def load_metric_evaluation_config(path: str | Path) -> MetricEvaluationConfig:
    return metric_evaluation_config_from_mapping(load_yaml_mapping(Path(path)))


def metric_evaluation_config_from_mapping(
    raw: Mapping[str, Any],
) -> MetricEvaluationConfig:
    if not isinstance(raw, Mapping):
        raise ValueError("evaluation config must contain a mapping")
    profile = raw.get("profile")
    if profile == GENERATED_COMMON_PROFILE:
        data = _strict_fields(
            raw,
            required={
                "profile",
                "metric_names",
                "generated_population",
                "sampling",
                "batch_size",
                "fid_batch_size",
                "fid_backend",
                "seed",
                "fid_asset_id",
                "classifier_asset_id",
            },
            context=GENERATED_COMMON_PROFILE,
        )
        data["metric_names"] = _metric_tuple(data["metric_names"])
        data["sampling"] = SamplingConfig.from_mapping(data["sampling"])
        return GeneratedCommonEvaluationConfig(**data)
    if profile == POSTERIOR_DIAGNOSTICS_PROFILE:
        data = _strict_fields(
            raw,
            required={
                "profile",
                "metric_names",
                "posterior_population",
                "batch_size",
                "num_bins",
                "posterior_t_values",
                "seed",
                "precision",
                "compile",
                "num_workers",
                "allow_tf32",
                "cudnn_benchmark",
            },
            context=POSTERIOR_DIAGNOSTICS_PROFILE,
        )
        data["metric_names"] = _metric_tuple(data["metric_names"])
        values = data["posterior_t_values"]
        if not isinstance(values, (list, tuple)):
            raise ValueError("posterior_t_values must be a sequence")
        data["posterior_t_values"] = tuple(values)
        return PosteriorDiagnosticsEvaluationConfig(**data)
    raise ValueError(
        "profile is required and must be one of: generated_common, posterior_diagnostics"
    )


def _strict_fields(
    raw: Mapping[str, Any],
    *,
    required: set[str],
    optional: set[str] | None = None,
    context: str,
) -> dict[str, Any]:
    keys = set(raw)
    allowed_optional = optional or set()
    missing = sorted(required - keys)
    unknown = sorted(keys - required - allowed_optional)
    if missing:
        raise ValueError(f"Missing required authored {context} config keys: {missing}")
    if unknown:
        raise ValueError(f"Unknown {context} config keys: {unknown}")
    return dict(raw)


def _metric_tuple(value: object) -> tuple[str, ...]:
    if not isinstance(value, (list, tuple)):
        raise ValueError("metric_names must be a sequence")
    return validate_metric_names(value) or ()


def _require_positive_int(value: object, name: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value < 1:
        raise ValueError(f"{name} must be a positive integer")
    return value


def _validate_common_identity(seed: object) -> None:
    if type(seed) is not int or not 0 <= seed < 2**32:
        raise ValueError("seed must be an integer in [0, 2**32)")


def _validate_asset_id(value: object, name: str) -> None:
    if value is None:
        return
    try:
        validate_path_component(value, name)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"{name} must be null or one path component") from exc
