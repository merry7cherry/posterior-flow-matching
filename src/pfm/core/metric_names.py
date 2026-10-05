from __future__ import annotations

from collections.abc import Iterable


FID_METRIC_NAME = "fid"
INCEPTION_SCORE_METRIC_NAME = "inception_score"
CONDITIONAL_ACCURACY_METRIC_NAME = "conditional_accuracy"
POSTERIOR_CONTROLLED_METRIC_NAME = "posterior_controlled"
POSTERIOR_CALIBRATION_METRIC_NAME = "posterior_calibration"
GENERATED_METRIC_NAMES = (
    FID_METRIC_NAME,
    INCEPTION_SCORE_METRIC_NAME,
    CONDITIONAL_ACCURACY_METRIC_NAME,
)
POSTERIOR_METRIC_NAMES = (
    POSTERIOR_CONTROLLED_METRIC_NAME,
    POSTERIOR_CALIBRATION_METRIC_NAME,
)
CANONICAL_METRIC_NAMES = GENERATED_METRIC_NAMES + POSTERIOR_METRIC_NAMES


def validate_metric_names(metric_names: str | Iterable[str] | None) -> tuple[str, ...] | None:
    if metric_names is None:
        return None
    values = (metric_names,) if isinstance(metric_names, str) else tuple(metric_names)
    allowed = frozenset(CANONICAL_METRIC_NAMES)
    seen: set[str] = set()
    for value in values:
        if not isinstance(value, str):
            raise TypeError("metric request values must be strings")
        if value not in allowed:
            raise ValueError(
                f"metric request value {value!r} must be one of {CANONICAL_METRIC_NAMES}"
            )
        if value in seen:
            raise ValueError(f"duplicate metric request value: {value!r}")
        seen.add(value)
    return values
