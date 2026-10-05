from __future__ import annotations

import json
import math
import time
from contextlib import contextmanager
from dataclasses import replace
from datetime import datetime, timezone
from pathlib import Path
from typing import Optional

import torch
import torch.distributed as dist
from tqdm import tqdm

from pfm.core.atomic_io import write_json_atomic, write_json_exclusive
from pfm.core.checkpointing import (
    validate_checkpoint_dataset_binding,
    validate_runtime_identity_payload,
)
from pfm.core.config import (
    ExperimentConfig,
    checkpoint_config_from_mapping,
    experiment_config_to_mapping,
    shared_training_protocol_sha256,
)
from pfm.core.identities import validate_architecture_identity, validate_dataset_identity
from pfm.core.distributed import barrier as distributed_barrier
from pfm.core.distributed import DistributedContext, unwrap_model, wrap_model_for_distributed
from pfm.core.layout import training_run_layout
from pfm.core.runtime import apply_runtime_settings, autocast_context, execution_device, set_seed
from pfm.core.rng_state import gather_rng_state_for_checkpoint, restore_rng_state, rng_state_for_rank
from pfm.core.serialization import to_jsonable
from pfm.core.execution import ExecutionContext, require_canonical_path, runtime_identity
from pfm.core.variant_metadata import SUPPORTED_VARIANTS, validate_base_model_config
from pfm.core.variant_lifecycle import variant_writer_lock
from pfm.data.datasets import (
    make_dataloaders_from_datasets,
    make_datasets,
    set_training_dataset_epoch,
)
from pfm.data.datasets import validate_dataset_manifest
from pfm.models.factory import build_model
from pfm.models.config_validation import validate_model_configuration
from pfm.sampling.time_sampling import sample_train_timesteps
from pfm.training.objectives import compute_loss
from pfm.training.checkpointing import (
    build_training_checkpoint,
    load_canonical_resume_checkpoint,
    publish_recovery_checkpoint,
    publish_recovery_pointer_to_immutable_checkpoint,
    publish_training_checkpoint,
    validate_resume_runtime_identity,
)
from pfm.training.rng import seed_rank_training_rng
from pfm.training.step import (
    build_ema_model,
    update_ema_model,
)


SINGLE_CARD_DATASETS = frozenset({"mnist", "cifar10", "imagenet"})
DISTRIBUTED_WORKLOADS = frozenset({("imagenet", "jit_transformer"), ("cifar10", "edm_ddpmpp")})
PERFORMANCE_RECORD_VERSION = 1
_SETUP_PERFORMANCE_FIELDS = (
    "total",
    "dataset_validation",
    "resume_validation",
    "data_loader",
    "model_ddp_ema",
    "optimizer",
    "state_and_manifests",
)
_EPOCH_PERFORMANCE_FIELDS = (
    "train_seconds",
    "data_wait_total_seconds",
    "data_wait_p50_ms",
    "data_wait_p90_ms",
    "data_wait_max_ms",
    "cuda_step_first_ms",
    "cuda_step_p50_ms",
    "cuda_step_p90_ms",
    "cuda_step_max_ms",
    "peak_cuda_allocated_bytes",
    "peak_cuda_reserved_bytes",
    "steps",
    "samples",
    "gradient_clip_triggered_steps",
    "checkpoint_seconds",
)


def validate_single_card_workload(config: ExperimentConfig, *, world_size: int) -> None:
    dataset = validate_dataset_identity(config.dataset)
    if dataset not in SINGLE_CARD_DATASETS:
        raise ValueError("single-card training supports MNIST, CIFAR10 and ImageNet workloads")
    if int(world_size) != 1:
        raise ValueError("single-card training requires world_size == 1")


def validate_distributed_workload(config: ExperimentConfig, world_size: int) -> None:
    dataset = validate_dataset_identity(config.dataset)
    architecture = validate_architecture_identity(config.architecture)
    if (dataset, architecture) not in DISTRIBUTED_WORKLOADS:
        raise ValueError("distributed training supports only ImageNet/JiT and CIFAR10/EDM-DDPM++ workloads")
    if int(world_size) <= 1:
        raise ValueError("distributed training requires world_size > 1")
    validate_model_configuration(config)


def validate_training_workload_contract(config: ExperimentConfig, *, world_size: int) -> None:
    dataset = validate_dataset_identity(config.dataset)
    architecture = validate_architecture_identity(config.architecture)
    resolved_world_size = int(world_size)
    if resolved_world_size < 1:
        raise ValueError("world_size must be >= 1")
    if (dataset, architecture) in DISTRIBUTED_WORKLOADS and (
        resolved_world_size > 1
    ):
        validate_distributed_workload(config, resolved_world_size)
        return
    if dataset in SINGLE_CARD_DATASETS:
        validate_single_card_workload(config, world_size=resolved_world_size)
        validate_model_configuration(config)
        return
    raise ValueError(
        "training supports MNIST/CIFAR10 single-card, CIFAR10/EDM-DDPM++ distributed, "
        "and ImageNet/JiT distributed workloads"
    )


def train_variant(
    variant: str,
    config: ExperimentConfig,
    *,
    job_context: ExecutionContext,
    distributed_context: DistributedContext,
    task_id: str,
    variant_id: str,
    epochs: Optional[int] = None,
    show_progress: bool = True,
    resume_from: Optional[str | Path] = None,
) -> dict[str, Path]:
    if variant not in SUPPORTED_VARIANTS:
        raise ValueError(f"Unknown variant {variant!r}; expected {sorted(SUPPORTED_VARIANTS)}")
    total_epochs = int(epochs if epochs is not None else config.epochs)
    if total_epochs < 1:
        raise ValueError("epochs must be >= 1")
    run_config = replace(config, epochs=total_epochs)
    _validate_training_contexts(job_context, distributed_context)
    current_runtime_identity = runtime_identity(job_context)
    validate_training_workload_contract(run_config, world_size=job_context.world_size)
    validate_base_model_config(variant, run_config)
    validate_model_configuration(run_config, variant=variant)
    layout = training_run_layout(
        job_context.runtime,
        task_id=task_id,
        variant_id=variant_id,
    )
    with _variant_training_lifecycle(layout, distributed_context):
        return _train_variant_owned(
            variant=variant,
            config=config,
            run_config=run_config,
            job_context=job_context,
            distributed_context=distributed_context,
            layout=layout,
            current_runtime_identity=current_runtime_identity,
            total_epochs=total_epochs,
            show_progress=show_progress,
            resume_from=resume_from,
        )


def _train_variant_owned(
    *,
    variant: str,
    config: ExperimentConfig,
    run_config: ExperimentConfig,
    job_context: ExecutionContext,
    distributed_context: DistributedContext,
    layout,
    current_runtime_identity: dict[str, object],
    total_epochs: int,
    show_progress: bool,
    resume_from: Optional[str | Path],
) -> dict[str, Path]:
    setup_started_at_utc = _utc_now()
    setup_started = time.perf_counter()
    resolved_learning_rate = resolve_effective_learning_rate(
        run_config,
        world_size=distributed_context.world_size,
    )
    resume_checkpoint = None
    if resume_from is None:
        validate_new_variant_root(layout)
    data_root = job_context.runtime.dataset_root(run_config.dataset)
    dataset_validation_started = time.perf_counter()
    dataset_manifest = validate_dataset_manifest(data_root, run_config.dataset)
    dataset_validation_seconds = time.perf_counter() - dataset_validation_started
    dataset_scientific_identity_sha256 = str(
        dataset_manifest["scientific_identity_sha256"]
    )
    resume_validation_started = time.perf_counter()
    if resume_from is not None:
        _, resume_checkpoint = load_canonical_resume_checkpoint(layout, resume_from)
        _validate_resume_checkpoint(
            resume_checkpoint,
            variant,
            run_config,
            distributed_context,
            current_runtime_identity=current_runtime_identity,
            current_dataset_scientific_identity_sha256=(
                dataset_scientific_identity_sha256
            ),
        )
        _validate_non_destructive_resume(
            layout,
            completed_epoch=int(resume_checkpoint["epoch"]),
        )
    resume_validation_seconds = time.perf_counter() - resume_validation_started
    set_seed(config.seed)
    device = distributed_context.device
    apply_runtime_settings(run_config, device)
    data_loader_started = time.perf_counter()
    train_dataset, test_dataset = make_datasets(run_config, data_root)
    train_sampler = _make_train_sampler(train_dataset, distributed_context, run_config)
    train_loader, _ = make_dataloaders_from_datasets(
        train_dataset,
        test_dataset,
        run_config,
        train_sampler=train_sampler,
        shuffle_train=train_sampler is None,
    )
    data_loader_seconds = time.perf_counter() - data_loader_started
    _synchronize_device(device)
    model_started = time.perf_counter()
    base_model = build_model(variant, run_config, device)
    model = wrap_model_for_distributed(base_model, distributed_context)
    ema_model = build_ema_model(base_model)
    _synchronize_device(device)
    model_seconds = time.perf_counter() - model_started
    optimizer_started = time.perf_counter()
    optimizer = _build_training_optimizer(
        model,
        lr=resolved_learning_rate,
        weight_decay=run_config.weight_decay,
        betas=(run_config.adamw_beta1, run_config.adamw_beta2),
        device=device,
    )
    optimizer_seconds = time.perf_counter() - optimizer_started
    seed_rank_training_rng(config.seed, distributed_context.rank)

    state_setup_started = time.perf_counter()
    if resume_from is None and distributed_context.is_main_process:
        layout.variant_root.mkdir(parents=True, exist_ok=True)
        _write_training_manifests(layout, variant, run_config, current_runtime_identity)
    metrics_path = layout.train_metrics_jsonl
    global_step = 0
    start_epoch = 1

    if resume_checkpoint is not None:
        unwrap_model(model).load_state_dict(resume_checkpoint["model_state"], strict=True)
        optimizer.load_state_dict(resume_checkpoint["optimizer_state"])
        if device.type == "cuda":
            _validate_optimizer_execution_contract(optimizer, device)
        ema_model.load_state_dict(resume_checkpoint["ema_model_state"], strict=True)
        global_step = int(resume_checkpoint["step"])
        start_epoch = int(resume_checkpoint["epoch"]) + 1
        if start_epoch > total_epochs:
            raise ValueError(
                f"resume checkpoint already completed epoch {start_epoch - 1}, total epochs is {total_epochs}"
            )
        restore_rng_state(
            rng_state_for_rank(
                resume_checkpoint,
                rank=distributed_context.rank,
                world_size=distributed_context.world_size,
            ),
            require_cuda=device.type == "cuda",
        )
        if distributed_context.is_main_process:
            layout.variant_root.mkdir(parents=True, exist_ok=True)
            _write_training_manifests(layout, variant, run_config, current_runtime_identity)
        distributed_barrier(distributed_context)
    _synchronize_device(device)
    state_setup_seconds = time.perf_counter() - state_setup_started
    setup_seconds = time.perf_counter() - setup_started
    setup_performance = _aggregate_setup_performance(
        {
            "total": setup_seconds,
            "dataset_validation": dataset_validation_seconds,
            "resume_validation": resume_validation_seconds,
            "data_loader": data_loader_seconds,
            "model_ddp_ema": model_seconds,
            "optimizer": optimizer_seconds,
            "state_and_manifests": state_setup_seconds,
        },
        distributed_context,
        device,
    )
    if distributed_context.is_main_process:
        _write_performance_record(
            {
                "record_type": "training_setup_performance",
                "schema_version": PERFORMANCE_RECORD_VERSION,
                "started_at_utc": setup_started_at_utc,
                "completed_at_utc": _utc_now(),
                "world_size": distributed_context.world_size,
                "slowest_total_rank": setup_performance["slowest_total_rank"],
                "slowest_total_rank_seconds": setup_performance["seconds"],
            }
        )

    first_job_step = True
    with _metrics_writer(metrics_path, distributed_context.is_main_process, append=resume_from is not None) as metrics_file:
        for epoch in range(start_epoch, total_epochs + 1):
            set_training_dataset_epoch(train_dataset, epoch)
            if train_sampler is not None:
                train_sampler.set_epoch(epoch)
            model.train()
            progress = tqdm(
                train_loader,
                desc=f"{variant} epoch {epoch}/{total_epochs}",
                leave=False,
                disable=not show_progress or not distributed_context.is_main_process,
            )
            distributed_barrier(distributed_context)
            _synchronize_device(device)
            _reset_peak_cuda_memory(device)
            epoch_started_at_utc = _utc_now()
            epoch_started = time.perf_counter()
            epoch_loss_totals = torch.zeros(2, dtype=torch.float64, device=device)
            epoch_sample_count = 0
            completed_steps = 0
            gradient_clip_triggered_steps = torch.zeros(
                (),
                dtype=torch.int64,
                device=device,
            )
            epoch_metrics = (
                _EpochMetricsBuffer()
                if distributed_context.is_main_process
                else None
            )
            data_wait_started = time.perf_counter()
            data_wait_seconds: list[float] = []
            cuda_step_events: list[tuple[torch.cuda.Event, torch.cuda.Event]] = []
            for batch_idx, (images, labels) in enumerate(progress, start=1):
                data_wait_seconds.append(time.perf_counter() - data_wait_started)
                learning_rate = _adjust_training_learning_rate(
                    optimizer,
                    base_learning_rate=resolved_learning_rate,
                    warmup_epochs=run_config.warmup_epochs,
                    lr_schedule=run_config.lr_schedule,
                    epoch=epoch,
                    batch_index=batch_idx,
                    batches_per_epoch=len(train_loader),
                )
                step_start_event, step_end_event = _new_cuda_event_pair(device)
                if step_start_event is not None:
                    step_start_event.record()
                optimizer.zero_grad(set_to_none=True)
                completed_steps = batch_idx
                x1 = images.to(device, non_blocking=True)
                labels = labels.to(device, non_blocking=True)
                x0 = float(run_config.noise_scale) * torch.randn_like(x1)
                t = sample_train_timesteps(run_config, x1.shape[0], device=device)
                xt = (1.0 - t) * x0 + t * x1
                if first_job_step:
                    _synchronize_device(device)
                    _write_first_step_phase(distributed_context, "forward_start")
                with autocast_context(run_config, device):
                    outputs = model(xt, t.view(-1), labels)
                    if first_job_step:
                        _synchronize_device(device)
                        _write_first_step_phase(distributed_context, "forward_complete")
                    loss, stats = compute_loss(variant, outputs, x0, x1, xt, t, run_config)

                loss.backward()
                if first_job_step:
                    _synchronize_device(device)
                    _write_first_step_phase(distributed_context, "backward_complete")
                grad_norm = _clip_training_gradients(model, max_norm=run_config.max_grad_norm)
                stats["grad_norm_pre_clip"] = grad_norm.detach()
                grad_clip_applied = grad_norm > float(run_config.max_grad_norm)
                stats["grad_clip_applied"] = grad_clip_applied.to(dtype=torch.float32)
                stats["learning_rate"] = loss.detach().new_tensor(learning_rate)
                gradient_clip_triggered_steps.add_(
                    grad_clip_applied.to(dtype=torch.int64)
                )
                optimizer.step()
                if first_job_step:
                    _synchronize_device(device)
                    _write_first_step_phase(distributed_context, "optimizer_complete")
                    first_job_step = False
                update_ema_model(ema_model, unwrap_model(model), run_config.ema_decay)
                if step_end_event is not None:
                    step_end_event.record()
                    cuda_step_events.append((step_start_event, step_end_event))

                global_step += 1
                batch_size = int(x1.shape[0])
                epoch_loss_totals[0].add_(
                    loss.detach().to(dtype=torch.float64),
                    alpha=batch_size,
                )
                epoch_loss_totals[1].add_(batch_size)
                epoch_sample_count += batch_size
                metric_scope = "rank0_local_batch" if distributed_context.enabled else "single_process_batch"
                if epoch_metrics is not None:
                    epoch_metrics.append(
                        epoch=epoch,
                        step=global_step,
                        batch=batch_idx,
                        metric_scope=metric_scope,
                        stats=stats,
                    )

                if run_config.limit_train_batches is not None and batch_idx >= run_config.limit_train_batches:
                    break
                data_wait_started = time.perf_counter()

            mean_loss = _mean_epoch_loss(
                epoch_loss_totals,
                distributed_context,
                device,
            )
            if not math.isfinite(mean_loss):
                raise ValueError("epoch mean loss must be finite")
            if epoch_metrics is not None:
                epoch_metrics.flush(metrics_file)
            _synchronize_device(device)
            train_seconds = time.perf_counter() - epoch_started
            cuda_step_ms = _cuda_event_durations_ms(cuda_step_events, device)
            peak_cuda_allocated_bytes, peak_cuda_reserved_bytes = (
                _peak_cuda_memory_bytes(device)
            )
            should_publish_immutable = _should_publish_immutable_checkpoint(
                epoch=epoch,
                checkpoint_every_epochs=run_config.checkpoint_every_epochs,
            )
            checkpoint_started = time.perf_counter()
            rng_state_by_rank = gather_rng_state_for_checkpoint(distributed_context)
            if distributed_context.is_main_process:
                checkpoint_payload = build_training_checkpoint(
                    model=unwrap_model(model),
                    optimizer=optimizer,
                    variant=variant,
                    config=run_config,
                    epoch=epoch,
                    step=global_step,
                    loss=mean_loss,
                    ema_model=ema_model,
                    distributed_world_size=distributed_context.world_size,
                    rng_state_by_rank=rng_state_by_rank,
                    runtime_identity=current_runtime_identity,
                    dataset_scientific_identity_sha256=(
                        dataset_scientific_identity_sha256
                    ),
                )
                config_identity = _checkpoint_config_identity(
                    run_config,
                    variant=variant,
                    producer_runtime_identity=current_runtime_identity,
                    dataset_scientific_identity_sha256=(
                        dataset_scientific_identity_sha256
                    ),
                )
                publication_arguments = {
                    "layout": layout,
                    "checkpoint_payload": checkpoint_payload,
                    "variant": variant,
                    "epoch": epoch,
                    "step": global_step,
                    "loss": mean_loss,
                    "config_identity": config_identity,
                }
                if should_publish_immutable:
                    publish_training_checkpoint(**publication_arguments)
                    publish_recovery_pointer_to_immutable_checkpoint(
                        layout=layout,
                        epoch=epoch,
                    )
                else:
                    publish_recovery_checkpoint(**publication_arguments)
            distributed_barrier(distributed_context)
            _synchronize_device(device)
            checkpoint_seconds = time.perf_counter() - checkpoint_started

            local_performance = _local_epoch_performance(
                train_seconds=train_seconds,
                data_wait_seconds=data_wait_seconds,
                cuda_step_ms=cuda_step_ms,
                peak_cuda_allocated_bytes=peak_cuda_allocated_bytes,
                peak_cuda_reserved_bytes=peak_cuda_reserved_bytes,
                steps=completed_steps,
                samples=epoch_sample_count,
                gradient_clip_triggered_steps=int(
                    gradient_clip_triggered_steps.item()
                ),
                checkpoint_seconds=checkpoint_seconds,
            )
            performance = _aggregate_epoch_performance(
                local_performance,
                distributed_context,
                device,
            )
            if distributed_context.is_main_process:
                _write_performance_record(
                    {
                        "record_type": "training_epoch_performance",
                        "schema_version": PERFORMANCE_RECORD_VERSION,
                        "epoch": epoch,
                        "global_step": global_step,
                        "started_at_utc": epoch_started_at_utc,
                        "recorded_at_utc": _utc_now(),
                        "world_size": distributed_context.world_size,
                        "slowest_train_rank": performance[
                            "slowest_train_rank"
                        ],
                        "steps_per_rank": performance["steps_per_rank"],
                        "global_samples": performance["global_samples"],
                        "global_images_per_second": performance[
                            "global_images_per_second"
                        ],
                        "max_rank_train_seconds": performance[
                            "max_rank_train_seconds"
                        ],
                        "data_wait": {
                            "max_rank_total_seconds": performance[
                                "max_rank_data_wait_total_seconds"
                            ],
                            "max_rank_share_of_train_time": performance[
                                "max_rank_data_wait_share_of_train_time"
                            ],
                            "max_rank_p50_ms": performance[
                                "max_rank_data_wait_p50_ms"
                            ],
                            "max_rank_p90_ms": performance[
                                "max_rank_data_wait_p90_ms"
                            ],
                            "max_rank_max_ms": performance[
                                "max_rank_data_wait_max_ms"
                            ],
                        },
                        "cuda_step_ms": {
                            "available": device.type == "cuda",
                            "max_rank_first": performance[
                                "max_rank_cuda_step_first_ms"
                            ],
                            "max_rank_p50": performance[
                                "max_rank_cuda_step_p50_ms"
                            ],
                            "max_rank_p90": performance[
                                "max_rank_cuda_step_p90_ms"
                            ],
                            "max_rank_max": performance[
                                "max_rank_cuda_step_max_ms"
                            ],
                        },
                        "peak_cuda_memory_bytes": {
                            "max_rank_allocated": performance[
                                "peak_cuda_allocated_bytes"
                            ],
                            "max_rank_reserved": performance[
                                "peak_cuda_reserved_bytes"
                            ],
                        },
                        "gradient_clipping": {
                            "max_norm": float(run_config.max_grad_norm),
                            "triggered_steps": performance[
                                "gradient_clip_triggered_steps"
                            ],
                            "total_steps": performance["steps_per_rank"],
                            "trigger_rate": (
                                float(
                                    performance["gradient_clip_triggered_steps"]
                                )
                                / float(performance["steps_per_rank"])
                            ),
                        },
                        "checkpoint": {
                            "recovery_published": True,
                            "immutable_published": should_publish_immutable,
                            "max_rank_seconds": performance[
                                "checkpoint_seconds"
                            ],
                        },
                    }
                )

    return {
        "task_root": layout.task_root,
        "variant_root": layout.variant_root,
        "metrics": metrics_path,
    }


def _utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def _build_training_optimizer(
    model: torch.nn.Module,
    *,
    lr: float,
    weight_decay: float,
    betas: tuple[float, float],
    device: torch.device,
) -> torch.optim.AdamW:
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=float(lr),
        weight_decay=float(weight_decay),
        betas=(float(betas[0]), float(betas[1])),
        fused=device.type == "cuda",
    )
    _validate_optimizer_execution_contract(optimizer, device)
    return optimizer


def _clip_training_gradients(
    model: torch.nn.Module,
    *,
    max_norm: float = 1.0,
) -> torch.Tensor:
    """Clip gradients and fail before the optimizer can apply non-finite values."""
    return torch.nn.utils.clip_grad_norm_(
        model.parameters(),
        max_norm=float(max_norm),
        error_if_nonfinite=True,
    )


def _adjust_training_learning_rate(
    optimizer: torch.optim.Optimizer,
    *,
    base_learning_rate: float,
    warmup_epochs: int,
    lr_schedule: str,
    epoch: int,
    batch_index: int,
    batches_per_epoch: int,
) -> float:
    if lr_schedule != "constant":
        raise ValueError("lr_schedule must be exactly 'constant'")
    if batches_per_epoch < 1:
        raise ValueError("batches_per_epoch must be positive")
    if batch_index < 1 or batch_index > batches_per_epoch:
        raise ValueError("batch_index must be within the epoch")
    epoch_progress = float(epoch - 1) + float(batch_index - 1) / float(
        batches_per_epoch
    )
    if warmup_epochs > 0 and epoch_progress < float(warmup_epochs):
        learning_rate = (
            float(base_learning_rate) * epoch_progress / float(warmup_epochs)
        )
    else:
        learning_rate = float(base_learning_rate)
    for parameter_group in optimizer.param_groups:
        parameter_group["lr"] = learning_rate
    return learning_rate


def _validate_optimizer_execution_contract(
    optimizer: torch.optim.Optimizer,
    device: torch.device,
) -> None:
    expected_fused = device.type == "cuda"
    for group_index, parameter_group in enumerate(optimizer.param_groups):
        if parameter_group.get("fused") is not expected_fused:
            raise RuntimeError(
                "AdamW execution contract mismatch for parameter group "
                f"{group_index}: expected fused={expected_fused} on {device.type}"
            )


def _synchronize_device(device: torch.device) -> None:
    if device.type == "cuda":
        torch.cuda.synchronize(device)


def _reset_peak_cuda_memory(device: torch.device) -> None:
    if device.type == "cuda":
        torch.cuda.reset_peak_memory_stats(device)


def _peak_cuda_memory_bytes(device: torch.device) -> tuple[int, int]:
    if device.type != "cuda":
        return 0, 0
    return (
        int(torch.cuda.max_memory_allocated(device)),
        int(torch.cuda.max_memory_reserved(device)),
    )


def _new_cuda_event_pair(
    device: torch.device,
) -> tuple[torch.cuda.Event | None, torch.cuda.Event | None]:
    if device.type != "cuda":
        return None, None
    return (
        torch.cuda.Event(enable_timing=True),
        torch.cuda.Event(enable_timing=True),
    )


def _cuda_event_durations_ms(
    event_pairs: list[tuple[torch.cuda.Event, torch.cuda.Event]],
    device: torch.device,
) -> list[float]:
    if device.type != "cuda":
        return []
    _synchronize_device(device)
    return [float(start.elapsed_time(end)) for start, end in event_pairs]


def _percentile(values: list[float], quantile: float) -> float:
    if not values:
        return 0.0
    ordered = sorted(float(value) for value in values)
    if len(ordered) == 1:
        return ordered[0]
    position = float(quantile) * float(len(ordered) - 1)
    lower_index = int(math.floor(position))
    upper_index = int(math.ceil(position))
    if lower_index == upper_index:
        return ordered[lower_index]
    weight = position - float(lower_index)
    return ordered[lower_index] * (1.0 - weight) + ordered[upper_index] * weight


def _aggregate_setup_performance(
    local_performance: dict[str, float],
    distributed_context: DistributedContext,
    device: torch.device,
) -> dict[str, object]:
    local_values = torch.tensor(
        [local_performance[field] for field in _SETUP_PERFORMANCE_FIELDS],
        dtype=torch.float64,
        device=device,
    )
    if distributed_context.enabled:
        gathered = [
            torch.empty_like(local_values)
            for _ in range(distributed_context.world_size)
        ]
        dist.all_gather(gathered, local_values)
        all_values = torch.stack(gathered)
    else:
        all_values = local_values.unsqueeze(0)
    all_rows = all_values.cpu().tolist()
    total_index = _SETUP_PERFORMANCE_FIELDS.index("total")
    critical_rank = max(
        range(len(all_rows)),
        key=lambda rank: all_rows[rank][total_index],
    )
    return {
        "slowest_total_rank": critical_rank,
        "seconds": dict(
            zip(_SETUP_PERFORMANCE_FIELDS, all_rows[critical_rank], strict=True)
        ),
    }


def _local_epoch_performance(
    *,
    train_seconds: float,
    data_wait_seconds: list[float],
    cuda_step_ms: list[float],
    peak_cuda_allocated_bytes: int,
    peak_cuda_reserved_bytes: int,
    steps: int,
    samples: int,
    gradient_clip_triggered_steps: int,
    checkpoint_seconds: float,
) -> dict[str, float]:
    data_wait_ms = [seconds * 1000.0 for seconds in data_wait_seconds]
    return {
        "train_seconds": float(train_seconds),
        "data_wait_total_seconds": float(sum(data_wait_seconds)),
        "data_wait_p50_ms": _percentile(data_wait_ms, 0.5),
        "data_wait_p90_ms": _percentile(data_wait_ms, 0.9),
        "data_wait_max_ms": max(data_wait_ms, default=0.0),
        "cuda_step_first_ms": cuda_step_ms[0] if cuda_step_ms else 0.0,
        "cuda_step_p50_ms": _percentile(cuda_step_ms, 0.5),
        "cuda_step_p90_ms": _percentile(cuda_step_ms, 0.9),
        "cuda_step_max_ms": max(cuda_step_ms, default=0.0),
        "peak_cuda_allocated_bytes": float(peak_cuda_allocated_bytes),
        "peak_cuda_reserved_bytes": float(peak_cuda_reserved_bytes),
        "steps": float(steps),
        "samples": float(samples),
        "gradient_clip_triggered_steps": float(gradient_clip_triggered_steps),
        "checkpoint_seconds": float(checkpoint_seconds),
    }


def _aggregate_epoch_performance(
    local_performance: dict[str, float],
    distributed_context: DistributedContext,
    device: torch.device,
) -> dict[str, float | int]:
    local_values = torch.tensor(
        [local_performance[field] for field in _EPOCH_PERFORMANCE_FIELDS],
        dtype=torch.float64,
        device=device,
    )
    if distributed_context.enabled:
        gathered = [
            torch.empty_like(local_values)
            for _ in range(distributed_context.world_size)
        ]
        dist.all_gather(gathered, local_values)
        all_values = torch.stack(gathered)
    else:
        all_values = local_values.unsqueeze(0)
    all_rows = all_values.cpu().tolist()
    train_seconds_index = _EPOCH_PERFORMANCE_FIELDS.index("train_seconds")
    data_wait_total_index = _EPOCH_PERFORMANCE_FIELDS.index(
        "data_wait_total_seconds"
    )
    samples_index = _EPOCH_PERFORMANCE_FIELDS.index("samples")
    steps_index = _EPOCH_PERFORMANCE_FIELDS.index("steps")
    clip_triggered_index = _EPOCH_PERFORMANCE_FIELDS.index(
        "gradient_clip_triggered_steps"
    )
    checkpoint_seconds_index = _EPOCH_PERFORMANCE_FIELDS.index("checkpoint_seconds")
    allocated_index = _EPOCH_PERFORMANCE_FIELDS.index("peak_cuda_allocated_bytes")
    reserved_index = _EPOCH_PERFORMANCE_FIELDS.index("peak_cuda_reserved_bytes")
    slowest_train_rank = max(
        range(len(all_rows)),
        key=lambda rank: all_rows[rank][train_seconds_index],
    )
    global_samples = int(sum(row[samples_index] for row in all_rows))
    train_seconds_by_rank = [float(row[train_seconds_index]) for row in all_rows]
    if any(seconds <= 0.0 for seconds in train_seconds_by_rank):
        raise ValueError("epoch train time must be positive")
    steps_by_rank = {int(row[steps_index]) for row in all_rows}
    if len(steps_by_rank) != 1:
        raise ValueError("distributed ranks completed different numbers of training steps")
    max_rank_train_seconds = max(train_seconds_by_rank)
    clip_triggered_by_rank = {
        int(row[clip_triggered_index]) for row in all_rows
    }
    if len(clip_triggered_by_rank) != 1:
        raise ValueError("distributed ranks observed different gradient clipping counts")

    def field_max(field: str) -> float:
        index = _EPOCH_PERFORMANCE_FIELDS.index(field)
        return max(float(row[index]) for row in all_rows)

    result: dict[str, float | int] = {
        "slowest_train_rank": slowest_train_rank,
        "steps_per_rank": steps_by_rank.pop(),
        "global_samples": global_samples,
        "global_images_per_second": (
            float(global_samples) / max_rank_train_seconds
        ),
        "max_rank_train_seconds": max_rank_train_seconds,
        "gradient_clip_triggered_steps": clip_triggered_by_rank.pop(),
        "max_rank_data_wait_total_seconds": field_max(
            "data_wait_total_seconds"
        ),
        "max_rank_data_wait_share_of_train_time": max(
            float(row[data_wait_total_index]) / float(row[train_seconds_index])
            for row in all_rows
        ),
        "max_rank_data_wait_p50_ms": field_max("data_wait_p50_ms"),
        "max_rank_data_wait_p90_ms": field_max("data_wait_p90_ms"),
        "max_rank_data_wait_max_ms": field_max("data_wait_max_ms"),
        "max_rank_cuda_step_first_ms": field_max("cuda_step_first_ms"),
        "max_rank_cuda_step_p50_ms": field_max("cuda_step_p50_ms"),
        "max_rank_cuda_step_p90_ms": field_max("cuda_step_p90_ms"),
        "max_rank_cuda_step_max_ms": field_max("cuda_step_max_ms"),
        "checkpoint_seconds": max(
            row[checkpoint_seconds_index] for row in all_rows
        ),
        "peak_cuda_allocated_bytes": int(
            max(row[allocated_index] for row in all_rows)
        ),
        "peak_cuda_reserved_bytes": int(
            max(row[reserved_index] for row in all_rows)
        ),
    }
    return result


def _write_performance_record(record: dict[str, object]) -> None:
    print(_json_line(record).rstrip("\n"), flush=True)


def _write_first_step_phase(
    distributed_context: DistributedContext,
    phase: str,
) -> None:
    _write_performance_record(
        {
            "record_type": "training_first_step_phase",
            "rank": distributed_context.rank,
            "phase": phase,
        }
    )


def _validate_training_contexts(
    job_context: ExecutionContext,
    distributed_context: DistributedContext,
) -> None:
    required_device = execution_device(job_context)
    expected = (
        job_context.rank,
        job_context.local_rank,
        job_context.world_size,
    )
    actual = (
        distributed_context.rank,
        distributed_context.local_rank,
        distributed_context.world_size,
    )
    if actual != expected:
        raise ValueError("distributed_context rank/local_rank/world_size must match job_context")
    if distributed_context.device != required_device:
        raise RuntimeError(
            f"training requires {required_device}, "
            f"got {distributed_context.device}"
        )
    if job_context.world_size == 1:
        if distributed_context.enabled or distributed_context.backend is not None:
            raise ValueError("single-process training requires enabled=False and backend=None")
        if dist.is_available() and dist.is_initialized():
            raise RuntimeError("single-process training requires no initialized process group")
        return
    if not distributed_context.enabled:
        raise ValueError("distributed training requires enabled=True")
    if distributed_context.backend != "nccl":
        raise RuntimeError("distributed training requires NCCL backend")
    if not (dist.is_available() and dist.is_initialized()):
        raise RuntimeError("distributed training requires an initialized process group")
    process_group_backend = dist.get_backend()
    if process_group_backend != "nccl":
        raise RuntimeError(f"initialized process group must use NCCL, got {process_group_backend!r}")
    process_group_rank = int(dist.get_rank())
    if process_group_rank != job_context.rank:
        raise RuntimeError(
            f"initialized process-group rank {process_group_rank} does not match execution context rank {job_context.rank}"
        )
    process_group_world_size = int(dist.get_world_size())
    if process_group_world_size != job_context.world_size:
        raise RuntimeError(
            "initialized process-group world_size "
            f"{process_group_world_size} does not match execution context world_size {job_context.world_size}"
        )


def _mean_epoch_loss(
    totals: torch.Tensor,
    distributed_context,
    device: torch.device,
) -> float:
    if totals.shape != (2,) or totals.dtype != torch.float64:
        raise ValueError("epoch loss totals must be a two-element float64 tensor")
    if totals.device != device:
        raise ValueError("epoch loss totals must reside on the training device")
    if distributed_context.enabled:
        if device.type != "cuda" or distributed_context.device.type != "cuda":
            raise RuntimeError("Distributed epoch-loss aggregation requires a CUDA device")
        if distributed_context.backend != "nccl":
            raise RuntimeError("Distributed epoch-loss aggregation requires NCCL backend")
        if not (dist.is_available() and dist.is_initialized()):
            raise RuntimeError(
                "Distributed epoch-loss aggregation requires an initialized process group"
            )
        dist.all_reduce(totals, op=dist.ReduceOp.SUM)
    global_loss_sum, global_count = totals.cpu().tolist()
    if global_count <= 0.0:
        raise ValueError("epoch produced no training samples")
    return float(global_loss_sum) / float(global_count)


def _should_publish_immutable_checkpoint(
    *,
    epoch: int,
    checkpoint_every_epochs: int,
) -> bool:
    interval = int(checkpoint_every_epochs)
    return interval > 0 and int(epoch) % interval == 0


def resolve_effective_learning_rate(config: ExperimentConfig, world_size: int) -> float:
    world_size = int(world_size)
    if world_size < 1:
        raise ValueError("world_size must be >= 1")
    if config.lr is not None:
        return float(config.lr)
    if not bool(config.scale_lr_by_effective_batch):
        raise ValueError("lr is required unless scale_lr_by_effective_batch=True")
    if config.base_lr is None:
        raise ValueError("base_lr is required when scale_lr_by_effective_batch=True")
    reference_batch = int(config.lr_reference_batch_size)
    if reference_batch < 1:
        raise ValueError("lr_reference_batch_size must be >= 1")
    effective_batch = int(config.batch_size) * world_size
    if effective_batch < 1:
        raise ValueError("effective batch size must be >= 1")
    resolved_lr = float(config.base_lr) * float(effective_batch) / float(reference_batch)
    return resolved_lr


def _make_train_sampler(dataset, distributed_context, config: ExperimentConfig):
    if not distributed_context.enabled:
        return None
    from torch.utils.data.distributed import DistributedSampler

    return DistributedSampler(
        dataset,
        num_replicas=distributed_context.world_size,
        rank=distributed_context.rank,
        shuffle=True,
        drop_last=bool(config.drop_last),
    )


@contextmanager
def _variant_training_lifecycle(layout, distributed_context: DistributedContext):
    if distributed_context.is_main_process:
        output_root = layout.output_root
        for name, path in (
            ("task_root", layout.task_root),
            ("variant_root", layout.variant_root),
            ("epochs_root", layout.epochs_root),
            ("recovery_root", layout.recovery_root),
            ("recovery_generations_root", layout.recovery_generations_root),
            ("recovery latest", layout.recovery_latest_path),
            ("training metrics", layout.train_metrics_jsonl),
            ("variant manifest", layout.variant_manifest_path),
            ("task manifest", layout.task_manifest_path),
        ):
            require_canonical_path(path, output_root, name=name)
        with variant_writer_lock(layout.variant_root):
            distributed_barrier(distributed_context)
            yield
    else:
        distributed_barrier(distributed_context)
        yield


def _metrics_writer(path: Path, enabled: bool, append: bool = False):
    if enabled:
        path.parent.mkdir(parents=True, exist_ok=True)
        return path.open("a" if append else "w", encoding="utf-8")
    return _NullMetricsWriter()


class _EpochMetricsBuffer:
    def __init__(self) -> None:
        self._records: list[
            tuple[
                dict[str, object],
                tuple[str, ...],
                tuple[torch.Tensor, ...],
            ]
        ] = []

    def append(
        self,
        *,
        epoch: int,
        step: int,
        batch: int,
        metric_scope: str,
        stats: dict[str, torch.Tensor],
    ) -> None:
        if not stats:
            raise ValueError("batch metrics stats must not be empty")
        stat_names = tuple(stats)
        stat_tensors = tuple(stats.values())
        for name, value in zip(stat_names, stat_tensors, strict=True):
            if not isinstance(value, torch.Tensor) or value.ndim != 0:
                raise TypeError(f"batch metric {name!r} must be a scalar tensor")
        self._records.append(
            (
                {
                    "epoch": int(epoch),
                    "step": int(step),
                    "batch": int(batch),
                    "metric_scope": str(metric_scope),
                },
                stat_names,
                tuple(value.detach().to(dtype=torch.float32) for value in stat_tensors),
            )
        )

    def flush(self, metrics_file) -> None:
        if not self._records:
            metrics_file.flush()
            return
        flattened = [
            value
            for _, _, stat_tensors in self._records
            for value in stat_tensors
        ]
        cpu_values = torch.stack(flattened).cpu().tolist()
        cursor = 0
        for metadata, stat_names, stat_tensors in self._records:
            next_cursor = cursor + len(stat_tensors)
            record = dict(metadata)
            record.update(
                zip(stat_names, cpu_values[cursor:next_cursor], strict=True)
            )
            _write_metrics_record(metrics_file, record)
            cursor = next_cursor
        metrics_file.flush()
        self._records.clear()


class _NullMetricsWriter:
    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc, traceback):
        return False

    def write(self, text: str) -> int:
        return len(text)


def _json_line(record: dict[str, object]) -> str:
    return json.dumps(to_jsonable(record), allow_nan=False) + "\n"


def _write_metrics_record(metrics_file, record: dict[str, object]) -> None:
    line = _json_line(record)
    metrics_file.write(line)


def _validate_non_destructive_resume(layout, *, completed_epoch: int) -> None:
    output_root = layout.output_root
    epochs_root = require_canonical_path(
        layout.epochs_root,
        output_root,
        name="epochs_root",
    )
    if epochs_root.exists():
        for candidate in epochs_root.glob("epoch_*"):
            suffix = candidate.name.removeprefix("epoch_")
            if not suffix.isdigit():
                raise ValueError(f"malformed epoch directory: {candidate}")
            canonical_epoch = require_canonical_path(
                candidate,
                output_root,
                name="resume epoch directory",
            )
            if not canonical_epoch.is_dir():
                raise ValueError(f"epoch path must be a directory: {canonical_epoch}")
            if int(suffix) > int(completed_epoch):
                raise ValueError(
                    f"resume checkpoint is older than later epoch directory: {canonical_epoch}"
                )


def _write_training_manifests(
    layout,
    variant: str,
    config: ExperimentConfig,
    producer_runtime_identity: dict[str, object],
) -> None:
    validated_identity = validate_runtime_identity_payload(
        producer_runtime_identity,
        context="training manifest",
    )
    _create_or_validate_task_manifest(layout, validated_identity)
    _atomic_json_dump(
        layout.variant_manifest_path,
        {
            "task_id": layout.task_id,
            "variant_id": layout.variant_id,
            "variant": str(variant),
            "training_config": experiment_config_to_mapping(config),
            "runtime_identity": validated_identity,
        },
    )


def _create_or_validate_task_manifest(layout, producer_runtime_identity: dict[str, object]) -> None:
    expected = {
        "task_id": layout.task_id,
        "output_root": str(layout.output_root),
        "producer_runtime_identity": producer_runtime_identity,
    }
    if write_json_exclusive(layout.task_manifest_path, expected):
        return
    try:
        existing = json.loads(layout.task_manifest_path.read_text(encoding="utf-8"))
    except json.JSONDecodeError as error:
        raise ValueError(f"task manifest contains invalid JSON: {layout.task_manifest_path}") from error
    if not isinstance(existing, dict):
        raise ValueError("task manifest must contain a JSON object")
    expected_keys = {"task_id", "output_root", "producer_runtime_identity"}
    if set(existing) != expected_keys:
        raise ValueError(f"task manifest keys must be exactly {sorted(expected_keys)}")
    existing_producer = validate_runtime_identity_payload(
        existing["producer_runtime_identity"],
        context="task manifest producer",
    )
    for field_name, expected_value in (
        ("task_id", expected["task_id"]),
        ("output_root", expected["output_root"]),
    ):
        if existing[field_name] != expected_value:
            raise ValueError(f"task manifest {field_name} does not match requested task")
    for field_name in ("project_name", "git_commit"):
        if existing_producer[field_name] != producer_runtime_identity[field_name]:
            raise ValueError(f"task manifest producer {field_name} does not match requested task")


def _checkpoint_config_identity(
    config: ExperimentConfig,
    *,
    variant: str,
    producer_runtime_identity: dict[str, object],
    dataset_scientific_identity_sha256: str,
) -> dict[str, object]:
    return {
        "dataset": str(config.dataset),
        "architecture": str(config.architecture),
        "variant": str(variant),
        "runtime_identity": producer_runtime_identity,
        "dataset_scientific_identity_sha256": str(
            dataset_scientific_identity_sha256
        ),
        "shared_training_protocol_sha256": shared_training_protocol_sha256(config),
    }


def validate_new_variant_root(layout) -> None:
    variant_root = layout.variant_root
    if not variant_root.exists():
        return
    if not variant_root.is_dir():
        raise ValueError(f"variant root already exists and is not a directory: {variant_root}")
    if any(variant_root.iterdir()):
        raise ValueError(f"variant root already exists: {variant_root}; use --resume to continue training")


def _atomic_json_dump(path: str | Path, payload: dict[str, object]) -> None:
    write_json_atomic(path, payload)


def _validate_resume_checkpoint(
    checkpoint: dict[str, object],
    variant: str,
    config: ExperimentConfig,
    distributed_context,
    *,
    current_runtime_identity: dict[str, object],
    current_dataset_scientific_identity_sha256: str,
) -> None:
    validate_resume_runtime_identity(checkpoint, current_runtime_identity)
    validate_continuation_state(
        checkpoint, variant, config, distributed_context,
        current_dataset_scientific_identity_sha256=current_dataset_scientific_identity_sha256,
    )


def validate_continuation_state(
    checkpoint, variant, config, distributed_context, *,
    current_dataset_scientific_identity_sha256: str,
) -> None:
    validate_checkpoint_dataset_binding(
        checkpoint,
        current_dataset_scientific_identity_sha256,
    )
    if str(checkpoint.get("variant")) != str(variant):
        raise ValueError(f"resume checkpoint variant {checkpoint.get('variant')!r} does not match requested {variant!r}")
    checkpoint_world_size = int(checkpoint["distributed_world_size"])
    if checkpoint_world_size != int(distributed_context.world_size):
        raise ValueError(
            f"resume checkpoint world size {checkpoint_world_size} does not match current world size "
            f"{int(distributed_context.world_size)}"
        )
    checkpoint_config = checkpoint_config_from_mapping(checkpoint["training_config"])
    checkpoint_mapping = experiment_config_to_mapping(checkpoint_config)
    current_mapping = experiment_config_to_mapping(config)
    for field_name in sorted(set(current_mapping) - {"epochs", "checkpoint_every_epochs"}):
        if checkpoint_mapping[field_name] != current_mapping[field_name]:
            raise ValueError(
                f"resume config field {field_name} differs from checkpoint; "
                "only epochs and checkpoint_every_epochs may change during continuation"
            )
