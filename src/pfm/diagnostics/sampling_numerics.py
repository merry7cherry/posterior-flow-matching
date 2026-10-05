from __future__ import annotations

from collections.abc import Mapping

from pfm.core.validation import (
    require_finite_number,
    require_non_negative_integer,
    require_positive_integer,
)


_MEAN_SQUARE_KEYS = (
    "score_mean_square",
    "divergence_mean_square",
    "correction_mean_square",
    "expected_brownian_increment_mean_square",
    "derivative_update_mse_estimate",
)


class SamplingNumericalAccumulator:
    """Combine chunk means for one fixed sampling protocol without storing chunks."""

    def __init__(self) -> None:
        self._sample_count = 0
        self._nonfinite_count = 0
        self._weighted_sums = dict.fromkeys(_MEAN_SQUARE_KEYS, 0.0)
        self._error_available = True
        self._has_summary: bool | None = None

    def update(self, summary: Mapping[str, object], sample_count: int) -> None:
        context = "sampling numerical summary"
        weight = require_positive_integer(sample_count, "sample_count", context=context)
        if not isinstance(summary, Mapping):
            raise ValueError(f"{context} must be a mapping")
        has_summary = bool(summary)
        if self._has_summary is not None and has_summary != self._has_summary:
            raise ValueError(f"{context} availability must be consistent across chunks")
        if not has_summary:
            self._has_summary = False
            self._sample_count += weight
            return
        if set(summary) != {*_MEAN_SQUARE_KEYS, "nonfinite_count"}:
            raise ValueError(f"{context} must contain exactly the current fields")
        values = {}
        for key in _MEAN_SQUARE_KEYS:
            if key == "derivative_update_mse_estimate" and summary[key] is None:
                values[key] = None
                continue
            value = require_finite_number(summary[key], key, context=context)
            if value < 0.0:
                raise ValueError(f"{context} {key} must be non-negative")
            values[key] = value
        nonfinite_count = require_non_negative_integer(
            summary["nonfinite_count"], "nonfinite_count", context=context
        )
        self._has_summary = True
        self._sample_count += weight
        self._nonfinite_count += nonfinite_count
        for key, value in values.items():
            if value is None:
                self._error_available = False
            else:
                self._weighted_sums[key] += weight * value

    def finalize(self) -> dict[str, float | int | None]:
        if not self._has_summary:
            return {}
        result: dict[str, float | int | None] = {
            key: value / self._sample_count for key, value in self._weighted_sums.items()
        }
        result["nonfinite_count"] = self._nonfinite_count
        if not self._error_available:
            result["derivative_update_mse_estimate"] = None
        return result
