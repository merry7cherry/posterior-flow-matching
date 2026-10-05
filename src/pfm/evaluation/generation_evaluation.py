from __future__ import annotations

import hashlib
from dataclasses import dataclass, field
import math
from collections.abc import Callable
from typing import Mapping

import torch

from pfm.core.config import ExperimentConfig
from pfm.core.sampling_config import SamplingConfig
from pfm.diagnostics.sampling_numerics import SamplingNumericalAccumulator
from pfm.core.validation import require_positive_integer
from pfm.evaluation.fid_metrics import (
    LTH14OnlineMetrics,
)
from pfm.evaluation.generation_metadata import initial_noise_policy, sampling_randomness_policy
from pfm.evaluation.generation_service import (
    GeneratedChunk,
    GenerationRequest,
    iter_loaded_generation,
)
from pfm.core.metric_names import (
    CONDITIONAL_ACCURACY_METRIC_NAME,
    FID_METRIC_NAME,
    INCEPTION_SCORE_METRIC_NAME,
)
from pfm.core.metric_profiles import generated_metric_names


CONDITIONAL_CLASSIFIER_INPUT_PROTOCOL = "raw_generated_sample_float_clamp_-1_1"


@dataclass
class _GenerationConsumers:
    sampling: SamplingConfig
    expected_count: int
    conditional_count: int
    fid_count: int
    next_start_index: int = 0
    actual_nfe: int | None = None
    actual_vjp: int | None = None
    total_forward_calls: int = 0
    total_vjp_calls: int = 0
    sampling_elapsed_seconds: float = 0.0
    peak_memory_allocated_bytes: int = 0
    numerics: SamplingNumericalAccumulator = field(default_factory=SamplingNumericalAccumulator)
    conditional: "_StreamingConditionalAccuracy | None" = None
    fid_seen: int = 0
    fid_metrics: LTH14OnlineMetrics | None = None


class _StreamingConditionalAccuracy:
    def __init__(self, num_classes: int) -> None:
        self.num_classes = int(num_classes)
        self.count = 0
        self.correct = 0
        self.confidence_sum = 0.0
        self.per_class_count = [0] * self.num_classes
        self.per_class_correct = [0] * self.num_classes

    def update(self, logits: torch.Tensor, labels: torch.Tensor) -> None:
        logits_cpu = logits.detach().cpu()
        labels_cpu = labels.detach().cpu().long()
        if logits_cpu.ndim != 2 or logits_cpu.shape[1] != self.num_classes:
            raise ValueError("conditional logits have an invalid class dimension")
        if labels_cpu.ndim != 1 or logits_cpu.shape[0] != labels_cpu.shape[0]:
            raise ValueError("conditional labels/logits have incompatible shapes")
        if not bool(torch.isfinite(logits_cpu).all()):
            raise ValueError("conditional logits must be finite")
        if bool((labels_cpu < 0).any()) or bool((labels_cpu >= self.num_classes).any()):
            raise ValueError("conditional labels must be valid class indices")
        predictions = logits_cpu.argmax(dim=1)
        correct = predictions.eq(labels_cpu)
        confidence = torch.softmax(logits_cpu, dim=1).max(dim=1).values
        self.count += int(labels_cpu.numel())
        self.correct += int(correct.sum())
        self.confidence_sum += float(confidence.sum(dtype=torch.float64))
        for class_id in range(self.num_classes):
            mask = labels_cpu.eq(class_id)
            self.per_class_count[class_id] += int(mask.sum())
            self.per_class_correct[class_id] += int(correct[mask].sum())

    def finalize(self) -> dict[str, object]:
        if self.count < 1 or any(count < 1 for count in self.per_class_count):
            raise ValueError("conditional accuracy requires samples from every class")
        accuracy = self.correct / self.count
        mean_confidence = self.confidence_sum / self.count
        if not math.isfinite(accuracy) or not math.isfinite(mean_confidence):
            raise ValueError("conditional accuracy results must be finite")
        return {
            "accuracy": accuracy,
            "mean_confidence": mean_confidence,
            "per_class_accuracy": [
                correct / count
                for correct, count in zip(self.per_class_correct, self.per_class_count)
            ],
            "per_class_count": list(self.per_class_count),
        }


@torch.no_grad()
def evaluate_generation(
    *,
    load_model: Callable[[], torch.nn.Module],
    variant: str,
    config: ExperimentConfig,
    metric_names: tuple[str, ...],
    classifier: torch.nn.Module | None,
    fid_asset_identity: Mapping[str, object] | None,
    sampling: SamplingConfig,
    generated_population: int,
    batch_size: int,
    fid_batch_size: int,
    device: torch.device,
    seed: int,
) -> dict[str, object]:
    selected = generated_metric_names(metric_names)
    if not selected:
        raise ValueError("generated evaluation requires at least one generated metric")
    if not isinstance(sampling, SamplingConfig):
        raise TypeError("sampling must be a SamplingConfig")
    resolved_batch_size = require_positive_integer(
        batch_size,
        "batch_size",
        context="generated evaluation",
    )
    resolved_fid_batch_size = require_positive_integer(
        fid_batch_size,
        "fid_batch_size",
        context="generated evaluation",
    )
    population = require_positive_integer(
        generated_population,
        "generated_population",
        context="generated evaluation",
    )
    num_classes = require_positive_integer(
        int(config.num_classes),
        "num_classes",
        context="generated evaluation",
    )
    if population < num_classes or population % num_classes != 0:
        raise ValueError(
            "generated_population must be at least num_classes and divisible by num_classes"
        )
    uses_conditional = CONDITIONAL_ACCURACY_METRIC_NAME in selected
    uses_fid_stream = FID_METRIC_NAME in selected or INCEPTION_SCORE_METRIC_NAME in selected
    conditional_count = population if uses_conditional else 0
    fid_count = population if uses_fid_stream else 0
    if uses_fid_stream:
        if fid_asset_identity is None:
            raise ValueError("FID asset identity is required for fid or inception_score")
    if uses_conditional:
        if classifier is None:
            raise ValueError("classifier and classifier asset identity are required")
    generation_count = population
    if generation_count < 1:
        raise ValueError("generated evaluation resolved an empty stream")

    labels = torch.arange(generation_count, dtype=torch.long).remainder(num_classes)
    request = GenerationRequest(
        labels=labels,
        sampling=sampling,
        batch_size=resolved_batch_size,
        seed=int(seed),
    )
    state = _GenerationConsumers(
        sampling=sampling,
        expected_count=generation_count,
        conditional_count=conditional_count,
        fid_count=fid_count,
        conditional=(
            _StreamingConditionalAccuracy(num_classes)
            if uses_conditional
            else None
        ),
        fid_metrics=(
            _build_online_fid_metrics(
                fid_asset_identity,
                total_samples=fid_count,
                batch_size=resolved_fid_batch_size,
                image_size=int(config.image_size),
                include_inception_score=(
                    INCEPTION_SCORE_METRIC_NAME in selected
                ),
                device=device,
            )
            if uses_fid_stream
            else None
        ),
    )
    model = load_model()
    for chunk in iter_loaded_generation(
        model,
        variant,
        config,
        request,
        device=device,
    ):
        _consume_generated_chunk(
            state,
            chunk,
            classifier=classifier,
            uses_conditional=uses_conditional,
            uses_fid_stream=uses_fid_stream,
            device=device,
        )

    if state.next_start_index != generation_count:
        raise ValueError(
            f"generated stream produced {state.next_start_index} items, "
            f"expected {generation_count}"
        )
    metrics, execution = _finalize_generation(
        state,
        metric_names=selected,
        seed=int(seed),
    )

    label_digest = hashlib.sha256(labels.numpy().tobytes()).hexdigest()
    generated_protocol: dict[str, object] = {
        "generation_count": generation_count,
        "fid_count": fid_count,
        "num_classes": num_classes,
        "batch_size": resolved_batch_size,
        "fid_batch_size": resolved_fid_batch_size,
        "seed": int(seed),
        "sampling": sampling.to_dict(),
        "actual_nfe": execution["actual_nfe"],
        "actual_vjp": execution["actual_vjp"],
        "total_forward_calls": state.total_forward_calls,
        "total_vjp_calls": state.total_vjp_calls,
        "sampling_elapsed_seconds": state.sampling_elapsed_seconds,
        "peak_memory_allocated_bytes": state.peak_memory_allocated_bytes,
        "timing_scope": "sampler_excludes_checkpoint_loading_metric_computation_and_initialization_service",
        "peak_memory_scope": "device_allocated_memory_during_sampling_including_resident_models",
        "randomness": sampling_randomness_policy(seed),
        "numerical_summary": state.numerics.finalize(),
        "label_sampling": "class_balanced_cyclic",
        "label_schedule_sha256": label_digest,
        "shared_generated_stream": True,
        "initial_noise_policy": initial_noise_policy(seed),
    }
    if uses_conditional:
        generated_protocol["conditional_count"] = conditional_count
        generated_protocol["conditional_classifier_input"] = (
            CONDITIONAL_CLASSIFIER_INPUT_PROTOCOL
        )
    return {"metrics": metrics, "generated_protocol": generated_protocol}


def _consume_generated_chunk(
    state: _GenerationConsumers,
    chunk: GeneratedChunk,
    *,
    classifier: torch.nn.Module | None,
    uses_conditional: bool,
    uses_fid_stream: bool,
    device: torch.device,
) -> None:
    if chunk.start_index != state.next_start_index:
        raise ValueError("generated chunks must be contiguous and start at zero")
    chunk_count = int(chunk.labels.numel())
    if chunk_count < 1 or state.next_start_index + chunk_count > state.expected_count:
        raise ValueError("generated chunk exceeds the requested shared stream")
    if int(chunk.sample_result.samples.shape[0]) != chunk_count:
        raise ValueError("generated chunk labels and samples must have matching counts")
    if chunk.sample_result.sampling != state.sampling:
        raise ValueError("generated chunk sampling metadata does not match its stream")
    actual_nfe = require_positive_integer(
        chunk.sample_result.actual_nfe,
        "actual_nfe",
        context="generated evaluation",
    )
    if state.actual_nfe is None:
        state.actual_nfe = actual_nfe
    elif state.actual_nfe != actual_nfe:
        raise ValueError("generated chunks have inconsistent actual_nfe")
    actual_vjp = chunk.sample_result.actual_vjp
    if type(actual_vjp) is not int or actual_vjp < 0:
        raise ValueError("generated chunk actual_vjp must be a non-negative integer")
    if state.actual_vjp is None:
        state.actual_vjp = actual_vjp
    elif state.actual_vjp != actual_vjp:
        raise ValueError("generated chunks have inconsistent actual_vjp")
    state.numerics.update(chunk.sample_result.numerical_summary, chunk_count)
    state.total_forward_calls += actual_nfe
    state.total_vjp_calls += actual_vjp
    state.sampling_elapsed_seconds += float(chunk.sample_result.elapsed_seconds)
    state.peak_memory_allocated_bytes = max(
        state.peak_memory_allocated_bytes, chunk.peak_memory_allocated_bytes
    )

    if uses_conditional and state.next_start_index < state.conditional_count:
        assert classifier is not None
        remaining = state.conditional_count - state.next_start_index
        count = min(chunk_count, remaining)
        logits = _classifier_logits_for_chunk(classifier, chunk, count=count, device=device)
        if state.conditional is None:
            raise ValueError("conditional accuracy accumulator is missing")
        state.conditional.update(logits, chunk.labels[:count])
    if uses_fid_stream and state.fid_seen < state.fid_count:
        count = min(chunk_count, state.fid_count - state.fid_seen)
        if state.fid_metrics is None:
            raise ValueError("online FID accumulator is missing")
        state.fid_metrics.update(chunk.sample_result.samples[:count])
        state.fid_seen += count
    state.next_start_index += chunk_count


def _finalize_generation(
    state: _GenerationConsumers,
    *,
    metric_names: tuple[str, ...],
    seed: int,
) -> tuple[dict[str, object], dict[str, object]]:
    metrics: dict[str, object] = {}
    details: dict[str, object] = {
        "sampling": state.sampling.to_dict(),
        "actual_nfe": state.actual_nfe,
        "num_generated": state.next_start_index,
        "generation_seed": seed,
        "label_sampling": "class_balanced_cyclic",
        "actual_vjp": state.actual_vjp,
    }
    if CONDITIONAL_ACCURACY_METRIC_NAME in metric_names:
        if state.conditional is None:
            raise ValueError("conditional accuracy did not consume its complete generated prefix")
        metrics[CONDITIONAL_ACCURACY_METRIC_NAME] = state.conditional.finalize()
    if FID_METRIC_NAME in metric_names or INCEPTION_SCORE_METRIC_NAME in metric_names:
        if state.fid_seen != state.fid_count or state.fid_metrics is None:
            raise ValueError("FID/IS did not consume its complete generated prefix")
        online = state.fid_metrics.finalize()
        if FID_METRIC_NAME in metric_names:
            metrics[FID_METRIC_NAME] = float(online[FID_METRIC_NAME])
        if INCEPTION_SCORE_METRIC_NAME in metric_names:
            metrics[INCEPTION_SCORE_METRIC_NAME] = online[INCEPTION_SCORE_METRIC_NAME]
    return metrics, details


def _classifier_logits_for_chunk(
    classifier: torch.nn.Module,
    chunk: GeneratedChunk,
    *,
    count: int,
    device: torch.device,
) -> torch.Tensor:
    raw_samples = chunk.sample_result.samples[:count]
    if not raw_samples.is_floating_point():
        raise ValueError("conditional classifier input must be floating point")
    classifier_input = raw_samples.to(device).clamp(-1.0, 1.0)
    return classifier(classifier_input)


def _build_online_fid_metrics(
    identity: Mapping[str, object] | None,
    *,
    total_samples: int,
    batch_size: int,
    image_size: int,
    include_inception_score: bool,
    device: torch.device,
) -> LTH14OnlineMetrics:
    if not isinstance(identity, Mapping):
        raise ValueError("FID asset identity is required for online metrics")
    stats_path = identity.get("stats_path")
    preprocessing = identity.get("preprocessing_identity")
    weights = (
        preprocessing.get("feature_extractor_weights")
        if isinstance(preprocessing, Mapping)
        else None
    )
    weights_sha256 = weights.get("sha256") if isinstance(weights, Mapping) else None
    if not isinstance(stats_path, str) or not isinstance(weights_sha256, str):
        raise ValueError("online FID requires bound stats and complete weights digest")
    return LTH14OnlineMetrics(
        fid_stats_path=stats_path,
        total_samples=total_samples,
        batch_size=batch_size,
        image_size=image_size,
        include_inception_score=include_inception_score,
        expected_weights_sha256=weights_sha256,
        device=device,
    )
