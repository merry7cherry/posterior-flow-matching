from __future__ import annotations

import os
from pathlib import Path

import torch

from pfm.artifacts.consumer_artifact import publish_consumer_artifact
from pfm.core.evaluation_config import (
    GeneratedCommonEvaluationConfig,
    MetricEvaluationConfig,
    PosteriorDiagnosticsEvaluationConfig,
)
from pfm.core.layout import epoch_id, validate_path_component
from pfm.core.execution import ExecutionContext, classifier_asset_dir, fid_asset_dir
from pfm.core.variant_metadata import (
    CANONICAL_METHOD_PROFILES,
    parse_method_profile,
    requires_learned_variance,
)
from pfm.evaluation.checkpoint_loading import (
    build_checkpoint_consumer_binding,
    require_canonical_variant_root,
    resolve_bound_epoch_checkpoint,
)
from pfm.evaluation.metric_evaluate import evaluate_checkpoint_metrics


def evaluate_bound_checkpoint(
    variant_root: str | Path,
    epoch: int,
    evaluation_id: str,
    evaluation_config: MetricEvaluationConfig,
    *,
    evaluation_config_path: str | Path,
    job_context: ExecutionContext,
    device: torch.device,
) -> dict[str, object]:
    if not isinstance(evaluation_config, MetricEvaluationConfig):
        raise TypeError("evaluation_config must be a closed MetricEvaluationConfig")
    if not isinstance(device, torch.device):
        raise TypeError("device must be a torch.device")
    root = require_canonical_variant_root(
        variant_root,
        job_context.runtime.experiments_root,
    )
    selected_epoch_id = epoch_id(epoch)
    clean_evaluation_id = validate_path_component(evaluation_id, "evaluation_id")
    evaluations_parent = root / "epochs" / selected_epoch_id / "evaluations"
    target = evaluations_parent / clean_evaluation_id
    if os.path.lexists(target):
        raise FileExistsError(f"immutable target already exists: {target}")

    config_path = job_context.runtime.config_file(
        evaluation_config_path,
        name="evaluation config",
    )
    bound = resolve_bound_epoch_checkpoint(root, epoch)
    config_identity = bound.config_identity
    method = str(bound.manifest["variant_id"])
    if method not in CANONICAL_METHOD_PROFILES:
        raise ValueError(
            "checkpoint variant_id must be the canonical method identity"
        )
    if isinstance(evaluation_config, PosteriorDiagnosticsEvaluationConfig):
        requires_learned_variance(
            parse_method_profile(method).variant,
            feature="posterior_diagnostics",
        )
    if isinstance(evaluation_config, GeneratedCommonEvaluationConfig) and evaluation_config.sampling.solver == "sde":
        requires_learned_variance(method, feature="SDE sampling")
    dataset = str(config_identity["dataset"])
    binding, _ = build_checkpoint_consumer_binding(
        bound,
        profile=str(evaluation_config.profile),
        consumer_config_path=config_path,
        job_context=job_context,
        asset_paths=_asset_paths(
            evaluation_config,
            dataset=dataset,
            job_context=job_context,
        ),
    )

    def produce(staging: Path) -> dict[str, object]:
        result = evaluate_checkpoint_metrics(
            bound_checkpoint=bound,
            output_dir=staging,
            runtime=job_context.runtime,
            evaluation_config=evaluation_config,
            device=device,
        )
        if result.get("profile") != evaluation_config.profile:
            raise ValueError("evaluation numerical result profile does not match binding")
        if result.get("method") != method:
            raise ValueError("evaluation numerical result method does not match binding")
        return result

    published = publish_consumer_artifact(
        parent=evaluations_parent,
        artifact_id=clean_evaluation_id,
        binding=binding,
        numerical_result=produce,
    )
    return {
        "variant_root": str(root),
        "epoch_id": selected_epoch_id,
        "evaluation_id": clean_evaluation_id,
        "artifact_dir": str(published),
        "manifest": str(published / "consumer_manifest.json"),
        "result": str(published / "result.json"),
    }


def _asset_paths(
    config: MetricEvaluationConfig,
    *,
    dataset: str,
    job_context: ExecutionContext,
) -> dict[str, Path]:
    if not isinstance(config, GeneratedCommonEvaluationConfig):
        return {}
    result: dict[str, Path] = {}
    if config.classifier_asset_id is not None:
        manifest_path = (
            classifier_asset_dir(
                job_context.runtime,
                dataset,
                config.classifier_asset_id,
            )
            / "classifier_manifest.json"
        )
        result["classifier"] = manifest_path
    if config.fid_asset_id is not None:
        manifest_path = (
            fid_asset_dir(job_context.runtime, config.fid_asset_id)
            / "fid_manifest.json"
        )
        result["fid"] = manifest_path
    return result


def evaluate_checkpoint_file(checkpoint_path, evaluation_config, *, output_dir, job_context, device):
    """Evaluate a standalone checkpoint with the same scientific metric service."""
    from dataclasses import asdict
    import shutil
    from pfm.core.atomic_io import write_json_atomic
    from pfm.core.immutable_artifact import prepare_immutable_directory, publish_immutable_directory
    from pfm.evaluation.checkpoint_loading import bind_checkpoint_file, validate_bound_checkpoint_dataset
    bound = bind_checkpoint_file(checkpoint_path)
    validate_bound_checkpoint_dataset(bound, job_context.runtime)
    target = Path(output_dir).expanduser().absolute()
    staging, target = prepare_immutable_directory(target.parent, target.name)
    try:
        result = evaluate_checkpoint_metrics(bound, staging, job_context.runtime, evaluation_config, device)
        write_json_atomic(staging / "result.json", result)
        write_json_atomic(staging / "config.json", asdict(evaluation_config))
        write_json_atomic(staging / "checkpoint.json", {
            "sha256": bound.checkpoint_sha256,
            "producer": bound.checkpoint_runtime_identity,
            "consumer_revision": job_context.git_commit,
            "epoch": bound.manifest["epoch"], "step": bound.manifest["step"],
        })
        publish_immutable_directory(staging, target)
    except BaseException:
        shutil.rmtree(staging)
        raise
    return {"artifact_dir": str(target), "result": str(target / "result.json")}
