from __future__ import annotations

import math
from typing import Iterable, Mapping

import torch

from pfm.core.config import ExperimentConfig
from pfm.core.evaluation_config import posterior_time_key, validate_posterior_times
from pfm.core.validation import (
    require_finite_number,
    require_mapping_field,
    require_positive_integer,
)
from pfm.core.variant_metadata import requires_learned_variance
from pfm.core.variant_metadata import method_profile_name
from pfm.diagnostics.posterior_residuals import (
    collect_posterior_residual_statistics,
    posterior_generator,
    posterior_histogram_edges,
)
from pfm.diagnostics.standardized_residuals import (
    GAUSSIAN_COVERAGE_THRESHOLDS,
    STANDARDIZED_RESIDUAL_DEFINITION,
)
from pfm.diagnostics.posterior_schema import (
    CONTROLLED_POSTERIOR_ESTIMAND,
    CONTROLLED_POSTERIOR_SAMPLING,
    NLL_TENSOR_KEYS,
    POSTERIOR_SPACE_NAMES,
)


_VARIANCE_SUMMARY_KEYS = frozenset(
    {
        "count",
        "component_count",
        "requested_num_bins",
        "num_bins",
        "calibration_binning",
        "histogram_binning",
        "histogram_edges",
        "mean_pred_var",
        "mean_squared_error",
        "ece_abs",
        "ece_rel",
        "error_to_var_ratio",
        "spearman_var_error",
        "spearman_var_error_reason",
        "bins",
        "histogram_bins",
    }
)


@torch.no_grad()
def evaluate_controlled_interpolant_posterior(
    model: torch.nn.Module,
    variant: str,
    config: ExperimentConfig,
    data_loader,
    device: torch.device,
    num_samples: int,
    t_values: Iterable[float] = (0.05, 0.1, 0.25, 0.5, 0.75, 0.9),
    num_bins: int = 15,
    seed: int = 321,
) -> dict[str, object]:
    requires_learned_variance(variant, feature="posterior_controlled")
    method_profile = method_profile_name(variant)
    requested_samples = require_positive_integer(
        num_samples,
        "num_samples",
        context="controlled posterior diagnostics",
    )

    checked_t_values = validate_posterior_times(t_values)
    time_keys = tuple(posterior_time_key(value) for value in checked_t_values)
    model.eval()
    by_t: dict[str, object] = {}
    for t_value, time_key in zip(checked_t_values, time_keys, strict=True):
        def fixed_time_sampler(
            sampler_config: ExperimentConfig,
            batch_size: int,
            sampler_device: torch.device,
            sampler_variant: str,
            dtype: torch.dtype,
        ) -> torch.Tensor:
            del sampler_config, sampler_variant
            return torch.full((batch_size, 1, 1, 1), float(t_value), device=sampler_device, dtype=dtype)

        residual_stats = collect_posterior_residual_statistics(
            model=model,
            variant=variant,
            config=config,
            data_loader=data_loader,
            device=device,
            num_samples=requested_samples,
            num_bins=int(num_bins),
            t_sampler=fixed_time_sampler,
            include_time_calibration=False,
            noise_generator=posterior_generator(
                device,
                seed,
                "controlled",
                method_profile,
                time_key,
                "noise",
            ),
            histogram_edges=posterior_histogram_edges(
                config,
                num_bins=num_bins,
                endpoint_time=t_value,
            ),
        )
        validated = _validate_controlled_residual_stats(
            residual_stats,
            config=config,
            expected_samples=requested_samples,
        )
        variance_calibration = validated["variance_calibration"]
        by_t[time_key] = {
            "t": float(t_value),
            "num_batches": validated["num_batches"],
            "standardized_residual": validated["standardized_residual"],
            "actual_nfe_per_batch": validated["actual_nfe_per_batch"],
            "velocity": variance_calibration["velocity"],
            "endpoint": variance_calibration["endpoint"],
            "variance_nll": validated["variance_nll"],
            "raw_logvar_saturation": validated["raw_logvar_saturation"],
        }
    if tuple(by_t) != time_keys:
        raise ValueError("controlled posterior result must cover every requested time exactly once")
    actual_nfe_per_batch = require_positive_integer(
        _consistent_by_t_value(by_t, "actual_nfe_per_batch"),
        "actual_nfe_per_batch",
        context="controlled posterior diagnostics",
    )

    return {
        "available": True,
        "estimand": CONTROLLED_POSTERIOR_ESTIMAND,
        "interpolant_sampling": CONTROLLED_POSTERIOR_SAMPLING,
        "pair_reuse_across_t": False,
        "variant": variant,
        "method_profile": method_profile,
        "t_values": [float(value) for value in checked_t_values],
        "num_samples": requested_samples,
        "num_bins": int(num_bins),
        "rng": {
            "base_seed": int(seed),
            "suite": "controlled",
            "profile": method_profile,
        },
        "actual_nfe_per_batch": actual_nfe_per_batch,
        "by_t": by_t,
    }


def _consistent_by_t_value(by_t: Mapping[str, object], key: str) -> object:
    sentinel = object()
    expected: object = sentinel
    for t_key, payload in by_t.items():
        if not isinstance(payload, Mapping) or key not in payload:
            raise ValueError(f"posterior diagnostics missing {key} metadata for t={t_key}")
        value = payload[key]
        if expected is sentinel:
            expected = value
        elif value != expected:
            raise ValueError(f"posterior diagnostics found inconsistent {key} metadata across t values")
    if expected is sentinel:
        raise ValueError(f"posterior diagnostics missing {key} metadata")
    return expected


def _validate_controlled_residual_stats(
    value: object,
    *,
    config: ExperimentConfig,
    expected_samples: int,
) -> dict[str, object]:
    context = "controlled posterior producer"
    if not isinstance(value, Mapping):
        raise ValueError(f"{context} result must be a mapping")

    num_samples = require_positive_integer(
        require_mapping_field(value, "num_samples", context=context),
        "num_samples",
        context=context,
    )
    if num_samples != expected_samples:
        raise ValueError(
            f"controlled posterior observed {num_samples} samples, expected {expected_samples}"
        )
    num_batches = require_positive_integer(
        require_mapping_field(value, "num_batches", context=context),
        "num_batches",
        context=context,
    )
    if num_batches > num_samples:
        raise ValueError(f"{context} num_batches cannot exceed num_samples")
    actual_nfe = require_positive_integer(
        require_mapping_field(value, "actual_nfe_per_batch", context=context),
        "actual_nfe_per_batch",
        context=context,
    )
    if actual_nfe != 1:
        raise ValueError(f"{context} actual_nfe_per_batch does not match current forward policy")

    calibration = _require_non_empty_mapping(
        require_mapping_field(value, "variance_calibration", context=context),
        "variance_calibration",
        context=context,
    )
    if set(calibration) != set(POSTERIOR_SPACE_NAMES):
        raise ValueError(
            f"{context} variance_calibration must contain exactly velocity and endpoint"
        )
    validated_calibration: dict[str, Mapping[str, object]] = {}
    expected_components = (
        expected_samples * int(config.image_channels) * int(config.image_size) ** 2
    )
    for space in POSTERIOR_SPACE_NAMES:
        summary = _require_non_empty_mapping(
            calibration[space],
            space,
            context=f"{context} variance_calibration",
        )
        _validate_variance_summary(
            summary,
            expected_samples,
            expected_components,
            context=f"{context} {space}",
        )
        validated_calibration[space] = summary

    variance_nll = _require_non_empty_mapping(
        require_mapping_field(value, "variance_nll", context=context),
        "variance_nll",
        context=context,
    )
    if set(variance_nll) != set(NLL_TENSOR_KEYS):
        raise ValueError(f"{context} variance_nll must contain exactly the current NLL fields")
    validated_nll = {
        key: require_finite_number(variance_nll[key], key, context=f"{context} variance_nll")
        for key in NLL_TENSOR_KEYS
    }
    raw_logvar_saturation = _validate_raw_logvar_saturation(
        require_mapping_field(value, "raw_logvar_saturation", context=context),
        expected_components=expected_components,
        context=context,
    )
    return {
        "num_samples": num_samples,
        "num_batches": num_batches,
        "actual_nfe_per_batch": actual_nfe,
        "standardized_residual": _validate_standardized_residual(
            require_mapping_field(value, "standardized_residual", context=context),
            expected_components=expected_components,
            context=context,
        ),
        "variance_calibration": validated_calibration,
        "variance_nll": validated_nll,
        "raw_logvar_saturation": raw_logvar_saturation,
    }


def _validate_standardized_residual(
    value: object,
    *,
    expected_components: int,
    context: str,
) -> dict[str, object]:
    summary = _require_non_empty_mapping(value, "standardized_residual", context=context)
    expected_keys = {
        "definition", "shared_spaces", "component_count", "mean", "second_moment",
        "gaussian_coverage_90", "gaussian_coverage_95", "coverage_count_90",
        "coverage_count_95", "gaussian_threshold_90", "gaussian_threshold_95",
    }
    if set(summary) != expected_keys:
        raise ValueError(f"{context} standardized_residual must contain exactly the current fields")
    if summary["definition"] != STANDARDIZED_RESIDUAL_DEFINITION:
        raise ValueError(f"{context} standardized_residual definition is invalid")
    if summary["shared_spaces"] != list(POSTERIOR_SPACE_NAMES):
        raise ValueError(f"{context} standardized_residual shared_spaces is invalid")
    count = require_positive_integer(
        summary["component_count"], "component_count", context=f"{context} standardized_residual"
    )
    if count != expected_components:
        raise ValueError(f"{context} standardized_residual component_count does not match current image shape")
    mean = require_finite_number(summary["mean"], "mean", context=context)
    second_moment = require_finite_number(summary["second_moment"], "second_moment", context=context)
    if second_moment < 0 or second_moment + 1e-6 * max(1.0, second_moment) < mean * mean:
        raise ValueError(f"{context} standardized_residual second_moment is inconsistent with mean")
    for level, threshold in GAUSSIAN_COVERAGE_THRESHOLDS.items():
        threshold_value = require_finite_number(
            summary[f"gaussian_threshold_{level}"], f"gaussian_threshold_{level}", context=context
        )
        if threshold_value != threshold:
            raise ValueError(f"{context} standardized_residual Gaussian threshold is invalid")
        covered = summary[f"coverage_count_{level}"]
        if isinstance(covered, bool) or not isinstance(covered, int) or not 0 <= covered <= count:
            raise ValueError(f"{context} standardized_residual coverage count is invalid")
        coverage = require_finite_number(
            summary[f"gaussian_coverage_{level}"], f"gaussian_coverage_{level}", context=context
        )
        if not math.isclose(coverage, covered / count, rel_tol=1e-12, abs_tol=1e-12):
            raise ValueError(f"{context} standardized_residual coverage does not match its count")
    if summary["coverage_count_90"] > summary["coverage_count_95"]:
        raise ValueError(f"{context} standardized_residual Gaussian coverage is not nested")
    return dict(summary)


def _validate_raw_logvar_saturation(
    value: object,
    *,
    expected_components: int,
    context: str,
) -> dict[str, object]:
    summary = _require_non_empty_mapping(
        value,
        "raw_logvar_saturation",
        context=context,
    )
    expected_keys = {
        "definition",
        "tail_probability",
        "lower_raw_threshold",
        "upper_raw_threshold",
        "component_count",
        "lower_count",
        "upper_count",
        "lower_rate",
        "upper_rate",
    }
    if set(summary) != expected_keys:
        raise ValueError(
            f"{context} raw_logvar_saturation must contain exactly the current fields"
        )
    if summary["definition"] != "sigmoid_raw_tail_probability":
        raise ValueError(f"{context} raw_logvar_saturation definition is invalid")
    tail_probability = require_finite_number(
        summary["tail_probability"],
        "tail_probability",
        context=f"{context} raw_logvar_saturation",
    )
    if not 0.0 < tail_probability < 0.5:
        raise ValueError(f"{context} raw_logvar_saturation tail_probability is invalid")
    component_count = require_positive_integer(
        summary["component_count"],
        "component_count",
        context=f"{context} raw_logvar_saturation",
    )
    if component_count != expected_components:
        raise ValueError(
            f"{context} raw_logvar_saturation component_count does not match current image shape"
        )
    validated = dict(summary)
    for side in ("lower", "upper"):
        count = summary[f"{side}_count"]
        if isinstance(count, bool) or not isinstance(count, int) or not 0 <= count <= component_count:
            raise ValueError(f"{context} raw_logvar_saturation {side}_count is invalid")
        rate = require_finite_number(
            summary[f"{side}_rate"],
            f"{side}_rate",
            context=f"{context} raw_logvar_saturation",
        )
        if not 0.0 <= rate <= 1.0:
            raise ValueError(f"{context} raw_logvar_saturation {side}_rate is invalid")
        if not math.isclose(rate, count / component_count, rel_tol=1e-12, abs_tol=1e-12):
            raise ValueError(
                f"{context} raw_logvar_saturation {side}_rate does not match its count"
            )
    lower_threshold = require_finite_number(
        summary["lower_raw_threshold"],
        "lower_raw_threshold",
        context=f"{context} raw_logvar_saturation",
    )
    upper_threshold = require_finite_number(
        summary["upper_raw_threshold"],
        "upper_raw_threshold",
        context=f"{context} raw_logvar_saturation",
    )
    if lower_threshold >= upper_threshold:
        raise ValueError(f"{context} raw_logvar_saturation thresholds are invalid")
    return validated


def _validate_variance_summary(
    summary: Mapping[str, object],
    expected_samples: int,
    expected_components: int,
    *,
    context: str,
) -> None:
    if set(summary) != _VARIANCE_SUMMARY_KEYS:
        raise ValueError(
            f"{context} variance summary must contain exactly the current fields"
        )
    count = require_positive_integer(
        require_mapping_field(summary, "count", context=context),
        "count",
        context=context,
    )
    if count != expected_samples:
        raise ValueError(f"{context} count does not match num_samples")
    component_count = require_positive_integer(
        require_mapping_field(summary, "component_count", context=context),
        "component_count",
        context=context,
    )
    if component_count != expected_components:
        raise ValueError(
            f"{context} component_count does not match current image shape"
        )
    mean_variance = require_finite_number(
        require_mapping_field(summary, "mean_pred_var", context=context),
        "mean_pred_var",
        context=context,
    )
    mean_squared_error = require_finite_number(
        require_mapping_field(summary, "mean_squared_error", context=context),
        "mean_squared_error",
        context=context,
    )
    if mean_variance <= 0.0:
        raise ValueError(f"{context} mean_pred_var must be positive")
    if mean_squared_error < 0.0:
        raise ValueError(f"{context} mean_squared_error must be non-negative")


def _require_non_empty_mapping(
    value: object,
    name: str,
    *,
    context: str,
) -> Mapping[str, object]:
    if not isinstance(value, Mapping) or not value:
        raise ValueError(f"{context} {name} must be a non-empty mapping")
    return value
