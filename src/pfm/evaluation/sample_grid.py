from __future__ import annotations

import os
from pathlib import Path

import torch

from pfm.artifacts.consumer_artifact import publish_consumer_artifact
from pfm.artifacts.sample_visualization import SampleVisualizationWriter
from pfm.core.layout import epoch_id, validate_path_component
from pfm.core.sample_config import SampleVisualizationConfig
from pfm.core.execution import ExecutionContext
from pfm.core.variant_metadata import (
    CANONICAL_METHOD_PROFILES,
    method_profile_name,
    requires_learned_variance,
)
from pfm.evaluation.checkpoint_loading import (
    build_checkpoint_consumer_binding,
    load_model_from_bound_checkpoint,
    require_canonical_variant_root,
    resolve_bound_epoch_checkpoint,
)
from pfm.evaluation.generation_metadata import sampling_randomness_policy
from pfm.evaluation.generation_service import (
    GenerationRequest,
    iter_loaded_generation,
)


def generate_sample_artifact(
    variant_root: str | Path,
    epoch: int,
    sample_id: str,
    sample_config: SampleVisualizationConfig,
    *,
    sample_config_path: str | Path,
    job_context: ExecutionContext,
    device: torch.device,
) -> Path:
    if not isinstance(sample_config, SampleVisualizationConfig):
        raise TypeError("sample_config must be a SampleVisualizationConfig")
    if not isinstance(device, torch.device):
        raise TypeError("device must be a torch.device")
    root = require_canonical_variant_root(
        variant_root,
        job_context.runtime.experiments_root,
    )
    selected_epoch_id = epoch_id(epoch)
    clean_sample_id = validate_path_component(sample_id, "sample_id")
    samples_parent = root / "epochs" / selected_epoch_id / "samples"
    target = samples_parent / clean_sample_id
    if os.path.lexists(target):
        raise FileExistsError(f"immutable target already exists: {target}")

    config_path = job_context.runtime.config_file(
        sample_config_path,
        name="sample visualization config",
    )
    bound = resolve_bound_epoch_checkpoint(root, epoch)
    config_identity = bound.config_identity
    method = str(bound.manifest["variant_id"])
    _validate_manifest_method(
        str(config_identity["dataset"]),
        method,
        sample_config,
    )
    binding, dataset_manifest = build_checkpoint_consumer_binding(
        bound,
        profile="sample_visualization",
        consumer_config_path=config_path,
        job_context=job_context,
    )
    _validate_manifest_profile(
        str(config_identity["dataset"]),
        method,
        sample_config,
        dataset_manifest,
    )
    model, variant, training_config = load_model_from_bound_checkpoint(
        bound,
        device,
        sampling=sample_config.sampling,
    )
    _validate_loaded_profile(training_config, variant, method, sample_config)
    labels = torch.arange(
        int(sample_config.class_start),
        int(sample_config.class_start + sample_config.class_count),
        dtype=torch.long,
    ).repeat_interleave(int(sample_config.num_per_class))
    request = GenerationRequest(
        labels=labels,
        sampling=sample_config.sampling,
        batch_size=sample_config.batch_size,
        seed=sample_config.seed,
        uncertainty_capture_t=sample_config.uncertainty_capture_t,
    )

    def produce(staging: Path) -> dict[str, object]:
        writer = SampleVisualizationWriter(
            artifact_root=staging,
            config=training_config,
            variant=variant,
            num_per_class=int(sample_config.num_per_class),
            class_start=int(sample_config.class_start),
            class_count=int(sample_config.class_count),
            expected_count=int(labels.numel()),
            sampling=sample_config.sampling,
            capture_t=sample_config.uncertainty_capture_t,
        )
        expected_start = 0
        peak_memory = 0
        for chunk in iter_loaded_generation(
            model,
            variant,
            training_config,
            request,
            device=device,
        ):
            if chunk.start_index != expected_start:
                raise ValueError("sample visualization chunks must be contiguous")
            writer.write_chunk(chunk.labels, chunk.sample_result)
            peak_memory = max(peak_memory, chunk.peak_memory_allocated_bytes)
            expected_start += int(chunk.labels.numel())
        result = writer.finalize()
        sampling = result["sampling"]
        assert isinstance(sampling, dict)
        sampling["seed"] = sample_config.seed
        sampling["randomness"] = sampling_randomness_policy(sample_config.seed)
        sampling["peak_memory_allocated_bytes"] = peak_memory
        return result

    return publish_consumer_artifact(
        parent=samples_parent,
        artifact_id=clean_sample_id,
        binding=binding,
        numerical_result=produce,
    )


def _validate_manifest_profile(
    dataset: str,
    method: str,
    sample_config: SampleVisualizationConfig,
    dataset_identity: dict[str, object],
) -> None:
    _validate_manifest_method(dataset, method, sample_config)
    scientific_identity = dataset_identity.get("scientific_identity")
    if not isinstance(scientific_identity, dict):
        raise ValueError("prepared dataset scientific identity must be a mapping")
    class_to_idx = scientific_identity.get("class_to_idx")
    if not isinstance(class_to_idx, dict) or not class_to_idx:
        raise ValueError("prepared dataset class mapping must be nonempty")
    prepared_class_count = len(class_to_idx)
    if sample_config.class_start + sample_config.class_count > prepared_class_count:
        raise ValueError("sample_visualization class subset exceeds prepared dataset classes")
    if sample_config.class_count * sample_config.num_per_class > 1000:
        raise ValueError("sample_visualization is limited to 1000 generated images")


def _validate_manifest_method(
    dataset: str,
    method: str,
    sample_config: SampleVisualizationConfig,
) -> None:
    if dataset not in {"mnist", "cifar10", "imagenet"}:
        raise ValueError("sample_visualization has no canonical dataset/image profile")
    if method not in CANONICAL_METHOD_PROFILES:
        raise ValueError("sample_visualization method is not supported")
    if sample_config.sampling.solver == "sde":
        requires_learned_variance(method, feature="SDE sampling")


def _validate_loaded_profile(
    training_config,
    variant: str,
    method: str,
    sample_config: SampleVisualizationConfig,
) -> None:
    if method_profile_name(variant) != method:
        raise ValueError("loaded checkpoint method identity does not match manifest")
    if sample_config.class_start + sample_config.class_count > int(training_config.num_classes):
        raise ValueError("sample_visualization class subset exceeds model classes")


def sample_checkpoint_file(
    checkpoint_path, sample_config, *, output_dir, device, consumer_revision: str | None = None,
) -> Path:
    """Generate bounded samples from a portable file; no training dataset is needed."""
    from dataclasses import asdict
    import shutil
    from pfm.core.atomic_io import write_json_atomic
    from pfm.core.immutable_artifact import prepare_immutable_directory, publish_immutable_directory
    from pfm.evaluation.checkpoint_loading import bind_checkpoint_file
    bound = bind_checkpoint_file(checkpoint_path)
    model, variant, config = load_model_from_bound_checkpoint(bound, device, sampling=sample_config.sampling)
    _validate_manifest_method(config.dataset, variant, sample_config)
    _validate_loaded_profile(config, variant, variant, sample_config)
    count = sample_config.class_count * sample_config.num_per_class
    if count > 1000:
        raise ValueError("sample_visualization is limited to 1000 generated images")
    labels = torch.arange(sample_config.class_start, sample_config.class_start + sample_config.class_count).repeat_interleave(sample_config.num_per_class)
    request = GenerationRequest(labels, sample_config.sampling, sample_config.batch_size,
                                sample_config.seed, sample_config.uncertainty_capture_t)
    target = Path(output_dir).expanduser().absolute()
    staging, target = prepare_immutable_directory(target.parent, target.name)
    try:
        writer = SampleVisualizationWriter(
            artifact_root=staging, config=config, variant=variant,
            num_per_class=sample_config.num_per_class, class_start=sample_config.class_start,
            class_count=sample_config.class_count, expected_count=count,
            sampling=sample_config.sampling, capture_t=sample_config.uncertainty_capture_t,
        )
        for chunk in iter_loaded_generation(model, variant, config, request, device=device):
            writer.write_chunk(chunk.labels, chunk.sample_result)
        result = writer.finalize()
        write_json_atomic(staging / "result.json", result)
        write_json_atomic(staging / "config.json", asdict(sample_config))
        write_json_atomic(staging / "checkpoint.json", {
            "sha256": bound.checkpoint_sha256,
            "producer": bound.checkpoint_runtime_identity,
            "consumer_revision": consumer_revision,
            "epoch": bound.manifest["epoch"], "step": bound.manifest["step"],
        })
        return publish_immutable_directory(staging, target)
    except BaseException:
        shutil.rmtree(staging)
        raise
