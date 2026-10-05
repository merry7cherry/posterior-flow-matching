from __future__ import annotations

import hashlib
import math
from typing import Callable, Mapping

import torch

from pfm.core.config import ExperimentConfig
from pfm.core.runtime import autocast_context
from pfm.core.variant_metadata import (
    DELTA_VARIANTS,
    requires_learned_variance,
    validate_base_model_config,
)
from pfm.diagnostics.posterior_schema import NLL_TENSOR_KEYS
from pfm.probability import targets as probability_targets
from pfm.probability.diagonal_gaussian import (
    constant_free_nll_per_component,
    squared_residual,
)
from pfm.probability.posterior import posterior_from_outputs
from pfm.probability.targets import (
    direct_velocity_target,
    velocity_target,
)
from pfm.diagnostics.standardized_residuals import StandardizedResidualAccumulator


HALF_LOG_2PI = 0.5 * math.log(2.0 * math.pi)
RAW_LOGVAR_SATURATION_PROBABILITY = 0.01
_RAW_LOGVAR_LOWER_THRESHOLD = math.log(
    RAW_LOGVAR_SATURATION_PROBABILITY / (1.0 - RAW_LOGVAR_SATURATION_PROBABILITY)
)
_RAW_LOGVAR_UPPER_THRESHOLD = -_RAW_LOGVAR_LOWER_THRESHOLD
PosteriorTimeSampler = Callable[[ExperimentConfig, int, torch.device, str, torch.dtype], torch.Tensor]


def posterior_generator(
    device: torch.device,
    base_seed: int,
    *identity: object,
) -> torch.Generator:
    if isinstance(base_seed, bool) or not isinstance(base_seed, int) or base_seed < 0:
        raise ValueError("posterior base seed must be a non-negative integer")
    payload = "\0".join((str(base_seed), *(str(item) for item in identity))).encode("utf-8")
    derived_seed = int.from_bytes(hashlib.sha256(payload).digest()[:8], "big") % (2**63 - 1)
    generator = torch.Generator(device=device)
    generator.manual_seed(derived_seed)
    return generator


def posterior_histogram_edges(
    config: ExperimentConfig,
    *,
    num_bins: int,
    endpoint_time: float | None = None,
    random_time_endpoint: bool = False,
) -> dict[str, torch.Tensor]:
    bin_count = max(1, int(num_bins))
    velocity_log_edges = torch.linspace(
        float(config.logvar_min),
        float(config.logvar_max),
        bin_count + 1,
        dtype=torch.float64,
    )
    if endpoint_time is not None:
        endpoint_shift_low = endpoint_shift_high = 2.0 * math.log1p(-float(endpoint_time))
    elif random_time_endpoint:
        endpoint_shift_low = 2.0 * math.log(float(config.eps_t))
        endpoint_shift_high = 2.0 * math.log1p(-float(config.eps_t))
    else:
        endpoint_shift_low = endpoint_shift_high = 0.0
    endpoint_log_edges = torch.linspace(
        float(config.logvar_min) + endpoint_shift_low,
        float(config.logvar_max) + endpoint_shift_high,
        bin_count + 1,
        dtype=torch.float64,
    )
    return {
        "velocity": velocity_log_edges.exp(),
        "endpoint": endpoint_log_edges.exp(),
    }


class _PosteriorSpaceAccumulator:
    """Stream component totals while retaining only per-image calibration scalars."""

    def __init__(self, *, include_time: bool) -> None:
        self._image_variance: list[torch.Tensor] = []
        self._image_error: list[torch.Tensor] = []
        self._image_time: list[torch.Tensor] = []
        self._include_time = bool(include_time)
        self._component_count = 0
        self._variance_sum = 0.0
        self._error_sum = 0.0

    def update(
        self,
        predicted_variance: torch.Tensor,
        squared_error: torch.Tensor,
        validated_time: torch.Tensor,
    ) -> None:
        if predicted_variance.ndim < 2:
            raise ValueError("posterior component tensors must include batch and component dimensions")

        variance = predicted_variance.detach()
        error = squared_error.detach()
        _validate_variance_error_population(
            variance,
            error,
            context="posterior accumulator",
        )
        self._component_count += int(variance.numel())
        self._variance_sum += float(variance.sum(dtype=torch.float64).cpu())
        self._error_sum += float(error.sum(dtype=torch.float64).cpu())
        self._image_variance.append(_mean_per_sample(variance, dtype=torch.float64).cpu())
        self._image_error.append(_mean_per_sample(error, dtype=torch.float64).cpu())
        if self._include_time:
            image_time = _mean_per_sample(
                validated_time.detach(),
                dtype=torch.float64,
            )
            self._image_time.append(image_time.cpu())

    def variance_calibration(
        self,
        *,
        num_bins: int,
        histogram_edges: torch.Tensor,
    ) -> dict[str, object]:
        image_variance, image_error = self._image_pairs()
        summary = summarize_variance_calibration(
            image_variance,
            image_error,
            num_bins=num_bins,
            histogram_edges=histogram_edges,
        )
        component_count = self._require_components()
        mean_variance = self._variance_sum / component_count
        mean_error = self._error_sum / component_count
        summary.update(
            {
                "component_count": component_count,
                "mean_pred_var": mean_variance,
                "mean_squared_error": mean_error,
                "error_to_var_ratio": mean_error / max(mean_variance, 1e-12),
            }
        )
        return summary

    def time_calibration(
        self,
        *,
        num_bins: int,
        t_min: float,
        t_max: float,
    ) -> dict[str, object]:
        if not self._include_time:
            raise ValueError("time calibration was not enabled for this accumulator")
        image_variance, image_error = self._image_pairs()
        if not self._image_time:
            raise ValueError("posterior accumulator has no image-level times")
        return summarize_time_calibration(
            torch.cat(self._image_time),
            image_variance,
            image_error,
            num_bins=num_bins,
            t_min=t_min,
            t_max=t_max,
        )

    def _image_pairs(self) -> tuple[torch.Tensor, torch.Tensor]:
        if not self._image_variance or not self._image_error:
            raise ValueError("posterior accumulator has no image-level values")
        return torch.cat(self._image_variance), torch.cat(self._image_error)

    def _require_components(self) -> int:
        if self._component_count <= 0:
            raise ValueError("posterior accumulator has no component values")
        return self._component_count


class _RawLogvarSaturationAccumulator:
    """Stream elementwise sigmoid-tail counts without retaining model outputs."""

    def __init__(self, *, include_time: bool) -> None:
        self._include_time = bool(include_time)
        self._component_count = 0
        self._lower_count = 0
        self._upper_count = 0
        self._image_time: list[torch.Tensor] = []
        self._image_component_count: list[torch.Tensor] = []
        self._image_lower_count: list[torch.Tensor] = []
        self._image_upper_count: list[torch.Tensor] = []

    def update(self, raw_logvar: torch.Tensor, validated_time: torch.Tensor) -> None:
        raw = raw_logvar.detach().float()
        if raw.ndim < 2:
            raise ValueError("raw logvar saturation requires batch and component dimensions")
        _require_finite_tensor(raw, "posterior raw logvar")
        flattened = raw.flatten(1)
        lower_per_image = flattened.lt(_RAW_LOGVAR_LOWER_THRESHOLD).sum(dim=1)
        upper_per_image = flattened.gt(_RAW_LOGVAR_UPPER_THRESHOLD).sum(dim=1)
        components_per_image = torch.full_like(
            lower_per_image,
            int(flattened.shape[1]),
        )
        self._component_count += int(raw.numel())
        self._lower_count += int(lower_per_image.sum().cpu())
        self._upper_count += int(upper_per_image.sum().cpu())
        if self._include_time:
            self._image_time.append(
                _mean_per_sample(validated_time.detach(), dtype=torch.float64).cpu()
            )
            self._image_component_count.append(components_per_image.cpu())
            self._image_lower_count.append(lower_per_image.cpu())
            self._image_upper_count.append(upper_per_image.cpu())

    def summary(
        self,
        *,
        num_bins: int,
        t_min: float,
        t_max: float,
    ) -> dict[str, object]:
        if self._component_count <= 0:
            raise ValueError("raw logvar saturation accumulator has no component values")
        result: dict[str, object] = {
            "definition": "sigmoid_raw_tail_probability",
            "tail_probability": RAW_LOGVAR_SATURATION_PROBABILITY,
            "lower_raw_threshold": _RAW_LOGVAR_LOWER_THRESHOLD,
            "upper_raw_threshold": _RAW_LOGVAR_UPPER_THRESHOLD,
            "component_count": self._component_count,
            "lower_count": self._lower_count,
            "upper_count": self._upper_count,
            "lower_rate": self._lower_count / self._component_count,
            "upper_rate": self._upper_count / self._component_count,
        }
        if not self._include_time:
            return result
        if not self._image_time:
            raise ValueError("raw logvar saturation accumulator has no image-level times")
        time = torch.cat(self._image_time)
        component_count = torch.cat(self._image_component_count)
        lower_count = torch.cat(self._image_lower_count)
        upper_count = torch.cat(self._image_upper_count)
        time_summary = _summarize_time_binned_saturation(
            time,
            component_count,
            lower_count,
            upper_count,
            num_bins=num_bins,
            t_min=t_min,
            t_max=t_max,
        )
        time_bins = time_summary["time_bins"]
        if not isinstance(time_bins, list):
            raise ValueError("raw logvar saturation time bins must be a list")
        reconciled = {
            "component_count": sum(int(item["component_count"]) for item in time_bins),
            "lower_count": sum(int(item["lower_count"]) for item in time_bins),
            "upper_count": sum(int(item["upper_count"]) for item in time_bins),
        }
        expected = {
            "component_count": self._component_count,
            "lower_count": self._lower_count,
            "upper_count": self._upper_count,
        }
        if reconciled != expected:
            raise ValueError("raw logvar saturation time-bin counts do not reconcile")
        result.update(time_summary)
        return result


def pfm_residual_tensors(
    variant: str,
    outputs: Mapping[str, torch.Tensor | None],
    x0: torch.Tensor,
    x1: torch.Tensor,
    xt: torch.Tensor,
    t: torch.Tensor,
    config: ExperimentConfig,
) -> dict[str, torch.Tensor]:
    requires_learned_variance(variant, feature="posterior calibration")
    mean_key = "delta" if variant in DELTA_VARIANTS else "mu"
    posterior = posterior_from_outputs(
        {mean_key: _require(outputs, mean_key), "logvar": _require(outputs, "logvar")},
        variant, xt, t, config,
    )
    logvar = posterior.effective_logvar
    velocity_variance = posterior.velocity_var
    endpoint_variance = posterior.endpoint_var
    if logvar is None or velocity_variance is None or endpoint_variance is None:
        raise ValueError("posterior calibration requires learned variance")
    time = probability_targets.time_like(t, x1).float()
    velocity_mean = posterior.velocity
    endpoint_mean = posterior.endpoint_mean
    if variant in DELTA_VARIANTS:
        velocity_target_tensor = direct_velocity_target(x0, x1)
    else:
        velocity_target_tensor = velocity_target(x1, xt, t)
    velocity_residual = velocity_target_tensor - velocity_mean
    endpoint_residual = x1 - endpoint_mean
    velocity_squared_error = squared_residual(velocity_residual)
    endpoint_squared_error = squared_residual(endpoint_residual)
    num_components = int(velocity_residual.flatten(1).shape[1])
    velocity_constant_free_nll = constant_free_nll_per_component(
        velocity_residual,
        logvar,
    )
    endpoint_constant_free_nll = velocity_constant_free_nll + torch.log(
        (1.0 - time).clamp_min(1e-12)
    )
    velocity_constant_free_nll_per_dim = _mean_per_sample(velocity_constant_free_nll)
    endpoint_constant_free_nll_per_dim = _mean_per_sample(endpoint_constant_free_nll)

    velocity_complete_nll_per_dim = velocity_constant_free_nll_per_dim + HALF_LOG_2PI
    endpoint_complete_nll_per_dim = endpoint_constant_free_nll_per_dim + HALF_LOG_2PI
    velocity_complete_nll_total = velocity_complete_nll_per_dim * num_components
    endpoint_complete_nll_total = endpoint_complete_nll_per_dim * num_components
    return {
        "standardized_residual": velocity_residual.float() * torch.exp(-0.5 * logvar),
        "velocity_variance": velocity_variance,
        "velocity_squared_error": velocity_squared_error,
        "velocity_constant_free_nll_per_dim": velocity_constant_free_nll_per_dim,
        "velocity_complete_nll_per_dim": velocity_complete_nll_per_dim,
        "velocity_complete_nll_total": velocity_complete_nll_total,
        "endpoint_variance": endpoint_variance,
        "endpoint_squared_error": endpoint_squared_error,
        "endpoint_constant_free_nll_per_dim": endpoint_constant_free_nll_per_dim,
        "endpoint_complete_nll_per_dim": endpoint_complete_nll_per_dim,
        "endpoint_complete_nll_total": endpoint_complete_nll_total,
    }


def summarize_variance_calibration(
    predicted_variance: torch.Tensor,
    squared_error: torch.Tensor,
    num_bins: int = 15,
    *,
    histogram_edges: torch.Tensor | None = None,
) -> dict[str, object]:
    _validate_variance_error_population(
        predicted_variance,
        squared_error,
        context="variance calibration",
    )
    variance = predicted_variance.detach().flatten().cpu().to(torch.float64)
    error = squared_error.detach().flatten().cpu().to(torch.float64)

    order = torch.argsort(variance)
    variance = variance[order]
    error = error[order]
    total = int(variance.numel())
    requested_num_bins = max(1, min(int(num_bins), total))
    bins = []
    ece_abs = torch.tensor(0.0, dtype=torch.float64)
    for index, chunk_indices in enumerate(
        _tie_preserving_equal_count_indices(variance, requested_num_bins)
    ):
        chunk_var = variance[chunk_indices]
        chunk_error = error[chunk_indices]
        mean_var = chunk_var.mean()
        mean_error = chunk_error.mean()
        weight = chunk_indices.numel() / total
        ece_abs = ece_abs + weight * torch.abs(mean_var - mean_error)
        bins.append(
            {
                "bin": index,
                "count": int(chunk_indices.numel()),
                "min_pred_var": float(chunk_var.min()),
                "max_pred_var": float(chunk_var.max()),
                "mean_pred_var": float(mean_var),
                "mean_squared_error": float(mean_error),
                "error_to_var_ratio": float(mean_error / mean_var.clamp_min(1e-12)),
            }
        )
    if sum(int(item["count"]) for item in bins) != total:
        raise ValueError("variance calibration bins do not cover the complete population")

    mean_var = variance.mean()
    mean_error = error.mean()
    ece_rel = ece_abs / mean_error.clamp_min(1e-12)
    spearman, spearman_reason = _spearman(variance, error)
    histogram, resolved_histogram_edges = _variance_histogram(
        variance,
        num_bins=num_bins,
        histogram_edges=histogram_edges,
    )
    if sum(int(item["count"]) for item in histogram) != total:
        raise ValueError("variance histogram bins do not cover the complete population")
    return {
        "count": total,
        "requested_num_bins": requested_num_bins,
        "num_bins": len(bins),
        "calibration_binning": "tie_preserving_equal_count",
        "histogram_binning": "fixed_shared_edges",
        "histogram_edges": resolved_histogram_edges,
        "mean_pred_var": float(mean_var),
        "mean_squared_error": float(mean_error),
        "ece_abs": float(ece_abs),
        "ece_rel": float(ece_rel),
        "error_to_var_ratio": float(mean_error / mean_var.clamp_min(1e-12)),
        "spearman_var_error": spearman,
        "spearman_var_error_reason": spearman_reason,
        "bins": bins,
        "histogram_bins": histogram,
    }


def summarize_time_calibration(
    times: torch.Tensor,
    predicted_variance: torch.Tensor,
    squared_error: torch.Tensor,
    num_bins: int = 15,
    *,
    t_min: float = 0.0,
    t_max: float = 1.0,
) -> dict[str, object]:
    lower = float(t_min)
    upper = float(t_max)
    if not math.isfinite(lower) or not math.isfinite(upper) or not 0.0 <= lower < upper <= 1.0:
        raise ValueError("time calibration bounds must satisfy 0 <= t_min < t_max <= 1")
    _validate_variance_error_population(
        predicted_variance,
        squared_error,
        context="time calibration",
    )
    time = _validated_broadcast_time(
        times,
        predicted_variance,
        t_min=lower,
        t_max=upper,
        context="time calibration",
    ).detach().flatten().cpu().to(torch.float64)
    variance = predicted_variance.detach().flatten().cpu().to(torch.float64)
    error = squared_error.detach().flatten().cpu().to(torch.float64)

    bin_count = max(1, int(num_bins))
    normalized_time = (time - lower) / (upper - lower)
    bin_indices = torch.clamp(
        (normalized_time * bin_count).floor().to(torch.int64),
        min=0,
        max=bin_count - 1,
    )
    bins = []
    for index in range(bin_count):
        selected = bin_indices == index
        if not bool(selected.any()):
            continue
        bin_time = time[selected]
        bin_var = variance[selected]
        bin_error = error[selected]
        mean_var = bin_var.mean()
        mean_error = bin_error.mean()
        bins.append(
            {
                "bin": index,
                "t_min": float(bin_time.min()),
                "t_max": float(bin_time.max()),
                "t_mean": float(bin_time.mean()),
                "count": int(selected.sum()),
                "mean_pred_var": float(mean_var),
                "mean_squared_error": float(mean_error),
                "error_to_var_ratio": float(mean_error / mean_var.clamp_min(1e-12)),
            }
        )
    if sum(int(item["count"]) for item in bins) != int(time.numel()):
        raise ValueError("time calibration bins do not cover the complete population")

    return {
        "count": int(time.numel()),
        "num_bins": bin_count,
        "time_binning": "equal_width_bounded_interval",
        "time_bounds": [lower, upper],
        "bins": bins,
    }


def _summarize_time_binned_saturation(
    times: torch.Tensor,
    component_count: torch.Tensor,
    lower_count: torch.Tensor,
    upper_count: torch.Tensor,
    *,
    num_bins: int,
    t_min: float,
    t_max: float,
) -> dict[str, object]:
    lower = float(t_min)
    upper = float(t_max)
    if not math.isfinite(lower) or not math.isfinite(upper) or not 0.0 <= lower < upper <= 1.0:
        raise ValueError("raw logvar saturation time bounds must satisfy 0 <= t_min < t_max <= 1")
    time = times.detach().flatten().cpu().to(torch.float64)
    components = component_count.detach().flatten().cpu().to(torch.int64)
    lower_counts = lower_count.detach().flatten().cpu().to(torch.int64)
    upper_counts = upper_count.detach().flatten().cpu().to(torch.int64)
    if not (time.shape == components.shape == lower_counts.shape == upper_counts.shape):
        raise ValueError("raw logvar saturation image-level populations must have identical shapes")
    if time.numel() <= 0:
        raise ValueError("raw logvar saturation time population must not be empty")
    _require_finite_tensor(time, "raw logvar saturation time")
    tolerance = 1e-12
    if bool((time < lower - tolerance).any()) or bool((time > upper + tolerance).any()):
        raise ValueError("raw logvar saturation times fall outside the authored bounded interval")
    if bool((components <= 0).any()):
        raise ValueError("raw logvar saturation component counts must be positive")
    if bool((lower_counts < 0).any()) or bool((upper_counts < 0).any()):
        raise ValueError("raw logvar saturation counts must be non-negative")
    if bool((lower_counts > components).any()) or bool((upper_counts > components).any()):
        raise ValueError("raw logvar saturation counts cannot exceed component counts")

    bin_count = max(1, int(num_bins))
    normalized_time = (time - lower) / (upper - lower)
    bin_indices = torch.clamp(
        (normalized_time * bin_count).floor().to(torch.int64),
        min=0,
        max=bin_count - 1,
    )
    bins = []
    for index in range(bin_count):
        selected = bin_indices == index
        if not bool(selected.any()):
            continue
        bin_components = int(components[selected].sum())
        bin_lower = int(lower_counts[selected].sum())
        bin_upper = int(upper_counts[selected].sum())
        bin_time = time[selected]
        bins.append(
            {
                "bin": index,
                "t_min": float(bin_time.min()),
                "t_max": float(bin_time.max()),
                "t_mean": float(bin_time.mean()),
                "sample_count": int(selected.sum()),
                "component_count": bin_components,
                "lower_count": bin_lower,
                "upper_count": bin_upper,
                "lower_rate": bin_lower / bin_components,
                "upper_rate": bin_upper / bin_components,
            }
        )
    if sum(int(item["sample_count"]) for item in bins) != int(time.numel()):
        raise ValueError("raw logvar saturation time bins do not cover the complete population")
    return {
        "num_time_bins": bin_count,
        "time_binning": "equal_width_bounded_interval",
        "time_bounds": [lower, upper],
        "time_bins": bins,
    }


@torch.no_grad()
def collect_posterior_residual_statistics(
    model: torch.nn.Module,
    variant: str,
    config: ExperimentConfig,
    data_loader,
    device: torch.device,
    num_samples: int,
    num_bins: int,
    t_sampler: PosteriorTimeSampler,
    include_time_calibration: bool,
    noise_generator: torch.Generator | None = None,
    histogram_edges: Mapping[str, torch.Tensor] | None = None,
) -> dict[str, object]:
    validate_base_model_config(variant, config)
    resolved_histogram_edges = histogram_edges or posterior_histogram_edges(
        config,
        num_bins=num_bins,
        random_time_endpoint=True,
    )
    if set(resolved_histogram_edges) != {"velocity", "endpoint"}:
        raise ValueError("posterior histogram edges must define velocity and endpoint")
    velocity_accumulator = _PosteriorSpaceAccumulator(include_time=include_time_calibration)
    endpoint_accumulator = _PosteriorSpaceAccumulator(include_time=include_time_calibration)
    standardized_accumulator = StandardizedResidualAccumulator(include_time=include_time_calibration)
    saturation_accumulator = _RawLogvarSaturationAccumulator(
        include_time=include_time_calibration
    )
    nll_sums = {key: 0.0 for key in NLL_TENSOR_KEYS}
    nll_counts = {key: 0 for key in NLL_TENSOR_KEYS}
    if isinstance(num_samples, bool) or not isinstance(num_samples, int) or num_samples < 1:
        raise ValueError("posterior num_samples must be a positive integer")
    sampled_time_min = float(config.eps_t) if include_time_calibration else 0.0
    sampled_time_max = 1.0 - float(config.eps_t) if include_time_calibration else 1.0
    batches_seen = 0
    samples_seen = 0
    model.eval()
    for batch_idx, (images, labels) in enumerate(data_loader, start=1):
        remaining = int(num_samples) - samples_seen
        if remaining <= 0:
            break
        if int(images.shape[0]) > remaining:
            images = images[:remaining]
            labels = labels[:remaining]
        batches_seen = batch_idx
        samples_seen += int(images.shape[0])
        x1 = images.to(device)
        labels = labels.to(device)
        x0 = float(config.noise_scale) * _randn_like(x1, generator=noise_generator)
        t = t_sampler(config, x1.shape[0], device, variant, x1.dtype).to(device=device, dtype=x1.dtype)
        t = probability_targets.time_like(t, x1)
        validated_time = _validated_broadcast_time(
            t,
            x1,
            t_min=sampled_time_min,
            t_max=sampled_time_max,
            context="posterior sampled time",
        )
        xt = (1.0 - t) * x0 + t * x1
        with autocast_context(config, device):
            outputs = model(xt, t.view(-1), labels)
        saturation_accumulator.update(
            _require(outputs, "logvar"),
            validated_time,
        )
        tensors = pfm_residual_tensors(variant, outputs, x0, x1, xt, t, config)
        velocity_accumulator.update(
            tensors["velocity_variance"],
            tensors["velocity_squared_error"],
            validated_time,
        )
        endpoint_accumulator.update(
            tensors["endpoint_variance"],
            tensors["endpoint_squared_error"],
            validated_time,
        )
        standardized_accumulator.update(tensors["standardized_residual"], validated_time)
        for key in NLL_TENSOR_KEYS:
            values = tensors[key].detach()
            _require_finite_tensor(values, f"posterior {key}")
            nll_sums[key] += float(values.sum(dtype=torch.float64).cpu())
            nll_counts[key] += int(values.numel())
        if samples_seen >= int(num_samples):
            break

    if batches_seen <= 0:
        raise ValueError("posterior residual data loader did not yield any batches")
    if samples_seen != int(num_samples):
        raise ValueError(
            f"posterior residual data loader yielded {samples_seen} samples, "
            f"expected {num_samples}"
        )

    velocity_summary = velocity_accumulator.variance_calibration(
        num_bins=num_bins,
        histogram_edges=resolved_histogram_edges["velocity"],
    )
    endpoint_summary = endpoint_accumulator.variance_calibration(
        num_bins=num_bins,
        histogram_edges=resolved_histogram_edges["endpoint"],
    )
    _require_summary_population(
        velocity_summary,
        samples_seen,
        context="velocity variance calibration",
    )
    _require_summary_population(
        endpoint_summary,
        samples_seen,
        context="endpoint variance calibration",
    )
    for key, count in nll_counts.items():
        if count != samples_seen:
            raise ValueError(
                f"posterior {key} count {count} does not match population {samples_seen}"
            )
    result: dict[str, object] = {
        "num_batches": int(batches_seen),
        "num_samples": int(samples_seen),
        "actual_nfe_per_batch": 1,
        "standardized_residual": standardized_accumulator.summary(
            num_bins=num_bins, t_min=sampled_time_min, t_max=sampled_time_max,
        ),
        "variance_calibration": {
            "velocity": velocity_summary,
            "endpoint": endpoint_summary,
        },
        "variance_nll": {
            key: nll_sums[key] / nll_counts[key]
            for key in NLL_TENSOR_KEYS
            if nll_counts[key] > 0
        },
        "raw_logvar_saturation": saturation_accumulator.summary(
            num_bins=num_bins,
            t_min=sampled_time_min,
            t_max=sampled_time_max,
        ),
    }
    if include_time_calibration:
        velocity_time_summary = velocity_accumulator.time_calibration(
            num_bins=num_bins,
            t_min=float(config.eps_t),
            t_max=1.0 - float(config.eps_t),
        )
        endpoint_time_summary = endpoint_accumulator.time_calibration(
            num_bins=num_bins,
            t_min=float(config.eps_t),
            t_max=1.0 - float(config.eps_t),
        )
        _require_summary_population(
            velocity_time_summary,
            samples_seen,
            context="velocity time calibration",
        )
        _require_summary_population(
            endpoint_time_summary,
            samples_seen,
            context="endpoint time calibration",
        )
        result["time_calibration"] = {
            "velocity": velocity_time_summary,
            "endpoint": endpoint_time_summary,
        }
    return result


def _require(outputs: Mapping[str, torch.Tensor | None], key: str) -> torch.Tensor:
    value = outputs.get(key)
    if value is None:
        raise KeyError(f"Expected output {key!r}")
    return value


def _validate_variance_error_population(
    predicted_variance: torch.Tensor,
    squared_error: torch.Tensor,
    *,
    context: str,
) -> None:
    if predicted_variance.shape != squared_error.shape:
        raise ValueError(f"{context} variance and squared error must have identical shapes")
    if predicted_variance.numel() <= 0:
        raise ValueError(f"{context} population must not be empty")
    _require_finite_tensor(predicted_variance, f"{context} variance")
    _require_finite_tensor(squared_error, f"{context} squared error")
    if not bool(predicted_variance.gt(0.0).all()):
        raise ValueError(f"{context} variance must be strictly positive")
    if not bool(squared_error.ge(0.0).all()):
        raise ValueError(f"{context} squared error must be non-negative")


def _validated_broadcast_time(
    times: torch.Tensor,
    reference: torch.Tensor,
    *,
    t_min: float,
    t_max: float,
    context: str,
) -> torch.Tensor:
    try:
        normalized_time = probability_targets.time_like(times, reference)
        broadcast = normalized_time.expand_as(reference)
    except RuntimeError as exc:
        raise ValueError(
            f"{context} time shape {tuple(times.shape)} is incompatible with "
            f"population shape {tuple(reference.shape)}"
        ) from exc
    _require_finite_tensor(normalized_time, f"{context} time")
    tolerance = 1e-12
    if bool((normalized_time < float(t_min) - tolerance).any()) or bool(
        (normalized_time > float(t_max) + tolerance).any()
    ):
        raise ValueError(
            f"{context} samples fall outside the authored bounded interval "
            f"[{float(t_min)}, {float(t_max)}]"
        )
    return broadcast


def _require_finite_tensor(values: torch.Tensor, name: str) -> None:
    if not bool(torch.isfinite(values).all()):
        raise ValueError(f"{name} must contain only finite values")


def _require_summary_population(
    summary: Mapping[str, object],
    expected: int,
    *,
    context: str,
) -> None:
    if summary.get("count") != int(expected):
        raise ValueError(f"{context} count does not match population {expected}")


def _mean_per_sample(value: torch.Tensor, *, dtype: torch.dtype | None = None) -> torch.Tensor:
    return value.flatten(1).mean(dim=1, dtype=dtype)


def _randn_like(
    reference: torch.Tensor,
    *,
    generator: torch.Generator | None,
) -> torch.Tensor:
    if generator is None:
        return torch.randn_like(reference)
    return torch.randn(
        reference.shape,
        dtype=reference.dtype,
        device=reference.device,
        generator=generator,
    )


def _tie_preserving_equal_count_indices(
    sorted_variance: torch.Tensor,
    num_bins: int,
) -> list[torch.Tensor]:
    total = int(sorted_variance.numel())
    if total <= 0:
        return []
    requested = max(1, min(int(num_bins), total))
    if requested == 1:
        return [torch.arange(total)]
    internal_boundaries = (
        torch.nonzero(sorted_variance[1:] != sorted_variance[:-1], as_tuple=False).flatten() + 1
    ).tolist()
    selected_boundaries = (
        sorted(
            {
                min(
                    internal_boundaries,
                    key=lambda boundary: (abs(boundary - target), boundary),
                )
                for target in (total * index / requested for index in range(1, requested))
            }
        )
        if internal_boundaries
        else []
    )
    boundaries = [0, *selected_boundaries, total]
    return [torch.arange(left, right) for left, right in zip(boundaries, boundaries[1:])]


def _spearman(left: torch.Tensor, right: torch.Tensor) -> tuple[float | None, str | None]:
    if left.numel() < 2:
        return None, "insufficient_samples"
    left_rank = _rank(left)
    right_rank = _rank(right)
    left_centered = left_rank - left_rank.mean()
    right_centered = right_rank - right_rank.mean()
    denom = left_centered.norm() * right_centered.norm()
    if float(denom) == 0.0:
        return None, "constant_rank"
    return float((left_centered * right_centered).sum() / denom), None


def _rank(values: torch.Tensor) -> torch.Tensor:
    order = torch.argsort(values)
    sorted_values = values[order]
    ranks = torch.empty_like(values, dtype=torch.float64)
    start = 0
    while start < int(values.numel()):
        end = start + 1
        while end < int(values.numel()) and bool(sorted_values[end] == sorted_values[start]):
            end += 1
        ranks[order[start:end]] = 0.5 * (start + end - 1)
        start = end
    return ranks


def _variance_histogram(
    variance: torch.Tensor,
    num_bins: int,
    *,
    histogram_edges: torch.Tensor | None,
) -> tuple[list[dict[str, object]], list[float]]:
    total = int(variance.numel())
    if total == 0:
        return [], []
    bin_count = max(1, min(int(num_bins), total))
    if histogram_edges is None:
        var_min = float(variance.min())
        var_max = float(variance.max())
        if var_max <= var_min:
            var_min *= 0.5
            var_max *= 1.5
        edges = torch.linspace(var_min, var_max, bin_count + 1, dtype=torch.float64)
    else:
        edges = histogram_edges.detach().flatten().cpu().to(torch.float64)
        if edges.numel() < 2 or not bool(torch.isfinite(edges).all()):
            raise ValueError("posterior histogram edges must contain finite boundaries")
        if not bool((edges[1:] > edges[:-1]).all()) or float(edges[0]) <= 0.0:
            raise ValueError("posterior histogram edges must be positive and strictly increasing")
        bin_count = int(edges.numel()) - 1
    tolerance = max(abs(float(edges[0])), abs(float(edges[-1])), 1.0) * 1e-6
    if (
        float(variance.min()) < float(edges[0]) - tolerance
        or float(variance.max()) > float(edges[-1]) + tolerance
    ):
        raise ValueError("posterior variance falls outside the shared histogram edges")
    counts = torch.histogram(
        variance.clamp(float(edges[0]), float(edges[-1])),
        bins=edges,
    ).hist.to(torch.int64)
    histogram_bins = []
    for index in range(bin_count):
        left = float(edges[index])
        right = float(edges[index + 1])
        count = int(counts[index])
        histogram_bins.append(
            {
                "bin": index,
                "count": count,
                "min_pred_var": left,
                "max_pred_var": right,
                "center_pred_var": math.sqrt(left * right),
                "fraction": count / total,
            }
        )
    return histogram_bins, [float(value) for value in edges]
