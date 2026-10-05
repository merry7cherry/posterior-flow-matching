from __future__ import annotations

from dataclasses import replace
from pathlib import Path
from typing import Mapping

import torch

from pfm.core.config import (
    ExperimentConfig,
    checkpoint_config_from_mapping,
)
from pfm.core.evaluation_config import (
    GeneratedCommonEvaluationConfig,
    MetricEvaluationConfig,
    PosteriorDiagnosticsEvaluationConfig,
)
from pfm.core.rng_context import preserve_rng_state
from pfm.core.runtime import set_seed
from pfm.core.execution import (
    RuntimePaths,
)
from pfm.data.datasets import make_dataloaders
from pfm.diagnostics.calibration_metrics import evaluate_pfm_calibration
from pfm.diagnostics.posterior_diagnostics import evaluate_controlled_interpolant_posterior
from pfm.evaluation.assets import load_classifier_asset, load_fid_asset
from pfm.evaluation.checkpoint_loading import (
    BoundEpochCheckpoint,
    load_bound_training_checkpoint,
)
from pfm.evaluation.generation_evaluation import evaluate_generation
from pfm.core.metric_names import (
    CONDITIONAL_ACCURACY_METRIC_NAME,
    FID_METRIC_NAME,
    GENERATED_METRIC_NAMES,
    INCEPTION_SCORE_METRIC_NAME,
    POSTERIOR_CALIBRATION_METRIC_NAME,
    POSTERIOR_CONTROLLED_METRIC_NAME,
)
from pfm.core.metric_profiles import (
    validate_evaluation_asset_bindings,
    validate_metric_request,
)
from pfm.core.variant_metadata import method_profile_name, requires_learned_variance
from pfm.core.sampling_config import SamplingConfig
from pfm.core.inference_config import InferenceExecutionConfig
from pfm.evaluation.checkpoint_loading import build_checkpoint_inference_model
from pfm.visualization.posterior_figures import write_posterior_figures


@torch.no_grad()
def evaluate_checkpoint_metrics(
    bound_checkpoint: BoundEpochCheckpoint,
    output_dir: str | Path,
    runtime: RuntimePaths,
    evaluation_config: MetricEvaluationConfig,
    device: torch.device,
) -> dict[str, object]:
    if not isinstance(bound_checkpoint, BoundEpochCheckpoint):
        raise TypeError("bound_checkpoint must be a BoundEpochCheckpoint")
    if not isinstance(evaluation_config, MetricEvaluationConfig):
        raise TypeError("evaluation_config must be a MetricEvaluationConfig")
    if not isinstance(device, torch.device):
        raise TypeError("device must be a torch.device")
    checkpoint = load_bound_training_checkpoint(bound_checkpoint)
    checkpoint_config = checkpoint_config_from_mapping(checkpoint["training_config"])
    variant = str(checkpoint["variant"])
    selected_metrics = validate_metric_request(
        evaluation_config.metric_names,
        checkpoint_config,
        variant=variant,
    )
    generated_names = {
        FID_METRIC_NAME,
        INCEPTION_SCORE_METRIC_NAME,
        CONDITIONAL_ACCURACY_METRIC_NAME,
    }
    if isinstance(evaluation_config, GeneratedCommonEvaluationConfig):
        if evaluation_config.sampling.solver == "sde":
            requires_learned_variance(variant, feature="SDE sampling")
        if not set(selected_metrics).issubset(generated_names):
            raise ValueError("generated_common cannot request posterior metrics")
    elif isinstance(evaluation_config, PosteriorDiagnosticsEvaluationConfig):
        if not set(selected_metrics).isdisjoint(generated_names):
            raise ValueError("posterior_diagnostics cannot request generated-common metrics")
    else:  # pragma: no cover - guarded by the public config loader
        raise TypeError("evaluation_config must be a closed checkpoint-evaluation profile")
    profile = validate_evaluation_asset_bindings(checkpoint_config, evaluation_config)
    if (
        isinstance(evaluation_config, GeneratedCommonEvaluationConfig)
        and (
            evaluation_config.generated_population < checkpoint_config.num_classes
            or evaluation_config.generated_population % checkpoint_config.num_classes != 0
        )
    ):
        raise ValueError(
            "generated_population must be at least num_classes and divisible by num_classes"
        )

    classifier = None
    if isinstance(evaluation_config, GeneratedCommonEvaluationConfig) and CONDITIONAL_ACCURACY_METRIC_NAME in selected_metrics:
        assert evaluation_config.classifier_asset_id is not None
        classifier, classifier_manifest = load_classifier_asset(
            runtime,
            profile.dataset,
            evaluation_config.classifier_asset_id,
            device,
        )
        _validate_classifier_matches_checkpoint(classifier_manifest, checkpoint_config)

    fid_identity = None
    if isinstance(evaluation_config, GeneratedCommonEvaluationConfig) and (
        FID_METRIC_NAME in selected_metrics or INCEPTION_SCORE_METRIC_NAME in selected_metrics
    ):
        assert evaluation_config.fid_asset_id is not None
        stats_path, fid_manifest = load_fid_asset(
            runtime,
            evaluation_config.fid_asset_id,
            dataset=profile.dataset,
            image_size=profile.image_size,
            backend=profile.fid_backend,
            split="train",
        )
        fid_identity = {
            "stats_path": str(stats_path),
            "preprocessing_identity": fid_manifest.get("preprocessing_identity"),
        }

    output_root = Path(output_dir)
    eval_config = replace(
        checkpoint_config,
        batch_size=evaluation_config.batch_size,
        drop_last=False,
    )
    if isinstance(evaluation_config, PosteriorDiagnosticsEvaluationConfig):
        eval_config = replace(evaluation_config.execution.apply(eval_config),
                              num_workers=evaluation_config.num_workers)
    metrics: dict[str, object] = {}
    generated_protocol: dict[str, object] | None = None
    generated_selected = tuple(
        metric for metric in selected_metrics if metric in GENERATED_METRIC_NAMES
    )
    if generated_selected:
        generated = evaluate_generation(
            load_model=lambda: _build_model_from_metric_checkpoint(
                checkpoint,
                variant,
                checkpoint_config,
                device,
                sampling=evaluation_config.sampling,
            ),
            variant=variant,
            config=eval_config,
            metric_names=generated_selected,
            classifier=classifier,
            fid_asset_identity=fid_identity,
            sampling=evaluation_config.sampling,
            generated_population=evaluation_config.generated_population,
            batch_size=evaluation_config.batch_size,
            fid_batch_size=evaluation_config.fid_batch_size,
            device=device,
            seed=evaluation_config.seed,
        )
        generated_protocol = dict(generated["generated_protocol"])
        generated_metrics = generated["metrics"]
        if not isinstance(generated_metrics, Mapping):
            raise ValueError("generated evaluation result must contain a metrics mapping")
        for metric_name in generated_selected:
            metrics[metric_name] = generated_metrics[metric_name]

    if any(
        metric in selected_metrics
        for metric in (POSTERIOR_CONTROLLED_METRIC_NAME, POSTERIOR_CALIBRATION_METRIC_NAME)
    ):
        with preserve_rng_state():
            model = _build_model_from_metric_checkpoint(
                checkpoint,
                variant,
                checkpoint_config,
                device,
                execution=evaluation_config.execution,
            )
            set_seed(evaluation_config.seed)
            _, test_loader = make_dataloaders(
                eval_config,
                runtime.dataset_root(profile.dataset),
            )
            if POSTERIOR_CONTROLLED_METRIC_NAME in selected_metrics:
                controlled = evaluate_controlled_interpolant_posterior(
                    model,
                    variant,
                    eval_config,
                    test_loader,
                    device,
                    t_values=evaluation_config.posterior_t_values,
                    num_samples=evaluation_config.posterior_population,
                    num_bins=evaluation_config.num_bins,
                    seed=evaluation_config.seed,
                )
                metrics[POSTERIOR_CONTROLLED_METRIC_NAME] = controlled
            if POSTERIOR_CALIBRATION_METRIC_NAME in selected_metrics:
                calibration = evaluate_pfm_calibration(
                    model,
                    variant,
                    eval_config,
                    test_loader,
                    device,
                    num_samples=evaluation_config.posterior_population,
                    num_bins=evaluation_config.num_bins,
                    seed=evaluation_config.seed,
                )
                metrics[POSTERIOR_CALIBRATION_METRIC_NAME] = calibration

    method = method_profile_name(variant)
    if isinstance(evaluation_config, GeneratedCommonEvaluationConfig):
        return {
            "profile": "generated_common",
            "method": method,
            "aggregate": {
                "metrics": metrics,
                "generated_protocol": generated_protocol,
            },
        }
    controlled = metrics[POSTERIOR_CONTROLLED_METRIC_NAME]
    random_time = metrics[POSTERIOR_CALIBRATION_METRIC_NAME]
    if not isinstance(controlled, Mapping) or not isinstance(random_time, Mapping):
        raise ValueError("posterior suites must be mappings")
    _validate_posterior_population(
        controlled,
        random_time,
        expected=int(evaluation_config.posterior_population),
    )
    presentation = write_posterior_figures(output_root, controlled, random_time)
    return {
        "profile": "posterior_diagnostics",
        "method": method,
        "controlled": controlled,
        "random_time": random_time,
        "presentation": presentation,
        "aggregate": _posterior_aggregate_summary(controlled, random_time),
    }


def _build_model_from_metric_checkpoint(
    checkpoint: dict[str, object],
    variant: str,
    config: ExperimentConfig,
    device: torch.device,
    *,
    sampling: SamplingConfig | None = None,
    execution: InferenceExecutionConfig | None = None,
) -> torch.nn.Module:
    return build_checkpoint_inference_model(
        checkpoint, variant, config, device, sampling=sampling, execution=execution
    )


def _validate_classifier_matches_checkpoint(
    manifest: Mapping[str, object],
    checkpoint_config: ExperimentConfig,
) -> None:
    expected = {
        "dataset": str(checkpoint_config.dataset),
        "image_size": int(checkpoint_config.image_size),
        "num_classes": int(checkpoint_config.num_classes),
    }
    for name, expected_value in expected.items():
        if manifest.get(name) != expected_value:
            raise ValueError(
                f"classifier asset {name} does not match bound checkpoint configuration"
            )
    classifier_config = manifest.get("config")
    if not isinstance(classifier_config, Mapping):
        raise ValueError("classifier asset config is missing from validated manifest")
    if classifier_config.get("image_channels") != int(checkpoint_config.image_channels):
        raise ValueError(
            "classifier asset image_channels does not match bound checkpoint configuration"
        )


def _posterior_aggregate_summary(
    controlled: object,
    random_time: object,
) -> dict[str, object]:
    if not isinstance(controlled, Mapping) or not isinstance(random_time, Mapping):
        raise ValueError("posterior numerical suites must be mappings")
    by_t = controlled.get("by_t")
    if not isinstance(by_t, Mapping):
        raise ValueError("controlled posterior result must contain by_t")
    controlled_rows = []
    for _, payload in sorted(
        by_t.items(),
        key=lambda item: float(item[1].get("t", 0.0)) if isinstance(item[1], Mapping) else 0.0,
    ):
        if not isinstance(payload, Mapping):
            raise ValueError("controlled posterior by_t entries must be mappings")
        controlled_rows.append(
            {
                "t": payload.get("t"),
                "velocity": _without_large_bins(payload.get("velocity")),
                "endpoint": _without_large_bins(payload.get("endpoint")),
                "variance_nll": payload.get("variance_nll"),
                "raw_logvar_saturation": payload.get("raw_logvar_saturation"),
            }
        )
    calibration = random_time.get("variance_calibration")
    time_calibration = random_time.get("time_calibration")
    if not isinstance(calibration, Mapping) or not isinstance(time_calibration, Mapping):
        raise ValueError("random-time posterior result is missing calibration summaries")
    return {
        "controlled": controlled_rows,
        "random_time": {
            "velocity": _without_large_bins(calibration.get("velocity")),
            "endpoint": _without_large_bins(calibration.get("endpoint")),
            "variance_nll": random_time.get("variance_nll"),
            "raw_logvar_saturation": random_time.get("raw_logvar_saturation"),
            "sample_count": random_time.get("num_samples"),
        },
    }


def _validate_posterior_population(
    controlled: Mapping[str, object],
    random_time: Mapping[str, object],
    *,
    expected: int,
) -> None:
    if (
        type(controlled.get("num_samples")) is not int
        or controlled["num_samples"] != expected
    ):
        raise ValueError("controlled posterior population does not match evaluation config")
    if (
        type(random_time.get("num_samples")) is not int
        or random_time["num_samples"] != expected
    ):
        raise ValueError("random-time posterior population does not match evaluation config")


def _without_large_bins(value: object) -> dict[str, object]:
    if not isinstance(value, Mapping):
        raise ValueError("posterior summary space must be a mapping")
    return {
        str(key): item
        for key, item in value.items()
        if key not in {"bins", "histogram_bins", "histogram_edges"}
    }
