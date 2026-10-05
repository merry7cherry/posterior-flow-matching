from __future__ import annotations

import io
import json
from contextlib import contextmanager
from dataclasses import replace
from pathlib import Path

import pytest
import torch
import torch.distributed as dist
from torch.utils.data import Sampler, TensorDataset

import pfm.core.distributed as distributed_module
from pfm.core.config import ExperimentConfig
from pfm.data.datasets import make_dataloaders_from_datasets
from pfm.core.distributed import DistributedContext, barrier, distributed_context_from_job, unwrap_model
from pfm.training import runner as train_module
from pfm.training.runner import train_variant


PROJECT_ROOT = Path(__file__).resolve().parents[1]


class ReverseSampler(Sampler[int]):
    def __init__(self, size: int) -> None:
        self.size = int(size)

    def __iter__(self):
        return iter(reversed(range(self.size)))

    def __len__(self) -> int:
        return self.size


def test_distributed_module_exposes_no_env_or_current_factories():
    assert not hasattr(distributed_module, "distributed_context_from_env")
    assert not hasattr(distributed_module, "current_distributed_context")


def test_distributed_job_uses_nccl(monkeypatch, execution_context):
    context = replace(execution_context, rank=0, local_rank=0, world_size=2)
    monkeypatch.setattr(torch.cuda, "is_available", lambda: True)
    monkeypatch.setattr(torch.cuda, "device_count", lambda: 2)
    monkeypatch.setattr(torch.cuda, "set_device", lambda device: None)
    monkeypatch.setattr(dist, "is_initialized", lambda: False)
    calls = []
    monkeypatch.setattr(dist, "init_process_group", lambda **kwargs: calls.append(kwargs))

    result = distributed_context_from_job(context)

    assert result.backend == "nccl"
    assert calls == [
        {
            "backend": "nccl",
            "init_method": "env://",
            "device_id": torch.device("cuda", 0),
        }
    ]


def test_distributed_job_accepts_existing_nccl_backend(monkeypatch, execution_context):
    context = replace(execution_context, rank=0, local_rank=0, world_size=2)
    monkeypatch.setattr(torch.cuda, "is_available", lambda: True)
    monkeypatch.setattr(torch.cuda, "device_count", lambda: 2)
    monkeypatch.setattr(torch.cuda, "set_device", lambda device: None)
    monkeypatch.setattr(dist, "is_initialized", lambda: True)
    backend_calls = []
    monkeypatch.setattr(dist, "get_backend", lambda: backend_calls.append("get_backend") or "nccl")
    monkeypatch.setattr(dist, "get_rank", lambda: 0)
    monkeypatch.setattr(dist, "get_world_size", lambda: 2)
    monkeypatch.setattr(
        dist,
        "init_process_group",
        lambda **kwargs: pytest.fail("existing process group must not be reinitialized"),
    )

    result = distributed_context_from_job(context)

    assert result.backend == "nccl"
    assert backend_calls == ["get_backend"]


def test_distributed_job_rejects_existing_gloo_backend(monkeypatch, execution_context):
    context = replace(execution_context, rank=0, local_rank=0, world_size=2)
    monkeypatch.setattr(torch.cuda, "is_available", lambda: True)
    monkeypatch.setattr(torch.cuda, "device_count", lambda: 2)
    monkeypatch.setattr(torch.cuda, "set_device", lambda device: None)
    monkeypatch.setattr(dist, "is_initialized", lambda: True)
    monkeypatch.setattr(dist, "get_backend", lambda: "gloo")

    with pytest.raises(RuntimeError, match="NCCL.*gloo"):
        distributed_context_from_job(context)


@pytest.mark.parametrize(
    ("group_rank", "group_world_size", "match"),
    [
        (1, 2, "rank"),
        (0, 3, "world_size"),
    ],
)
def test_distributed_job_rejects_existing_process_group_identity_mismatch(
    monkeypatch,
    execution_context,
    group_rank,
    group_world_size,
    match,
):
    context = replace(execution_context, rank=0, local_rank=0, world_size=2)
    monkeypatch.setattr(torch.cuda, "is_available", lambda: True)
    monkeypatch.setattr(torch.cuda, "device_count", lambda: 2)
    monkeypatch.setattr(torch.cuda, "set_device", lambda device: None)
    monkeypatch.setattr(dist, "is_initialized", lambda: True)
    monkeypatch.setattr(dist, "get_backend", lambda: "nccl")
    monkeypatch.setattr(dist, "get_rank", lambda: group_rank)
    monkeypatch.setattr(dist, "get_world_size", lambda: group_world_size)

    with pytest.raises(RuntimeError, match=match):
        distributed_context_from_job(context)


def test_distributed_job_rejects_initialized_group_for_single_process_context(monkeypatch, execution_context):
    monkeypatch.setattr(torch.cuda, "is_available", lambda: True)
    monkeypatch.setattr(torch.cuda, "device_count", lambda: 1)
    monkeypatch.setattr(torch.cuda, "set_device", lambda device: None)
    monkeypatch.setattr(dist, "is_initialized", lambda: True)
    monkeypatch.setattr(dist, "get_backend", lambda: "nccl")
    monkeypatch.setattr(dist, "get_rank", lambda: 0)
    monkeypatch.setattr(dist, "get_world_size", lambda: 1)

    with pytest.raises(RuntimeError, match="world_size==1"):
        distributed_context_from_job(execution_context)


def test_distributed_job_initialize_false_requires_existing_process_group(monkeypatch, execution_context):
    context = replace(execution_context, rank=0, local_rank=0, world_size=2)
    monkeypatch.setattr(torch.cuda, "is_available", lambda: True)
    monkeypatch.setattr(torch.cuda, "device_count", lambda: 2)
    monkeypatch.setattr(torch.cuda, "set_device", lambda device: None)
    monkeypatch.setattr(dist, "is_initialized", lambda: False)
    monkeypatch.setattr(
        dist,
        "init_process_group",
        lambda **kwargs: pytest.fail("initialize=False must not initialize a process group"),
    )

    with pytest.raises(RuntimeError, match="existing initialized NCCL process group"):
        distributed_context_from_job(context, initialize=False)


def test_barrier_requires_explicit_distributed_context():
    with pytest.raises(TypeError):
        barrier()


def test_barrier_uses_explicit_distributed_context(monkeypatch):
    context = DistributedContext(True, 0, 0, 2, torch.device("cuda", 0), "nccl")
    calls = []

    monkeypatch.setattr(dist, "is_available", lambda: True)
    monkeypatch.setattr(dist, "is_initialized", lambda: True)
    monkeypatch.setattr(dist, "barrier", lambda **kwargs: calls.append(kwargs))

    barrier(context)

    assert calls == [{"device_ids": [0]}]


def test_barrier_rejects_enabled_context_without_process_group(monkeypatch):
    context = DistributedContext(True, 0, 0, 2, torch.device("cuda", 0), "nccl")
    monkeypatch.setattr(dist, "is_available", lambda: True)
    monkeypatch.setattr(dist, "is_initialized", lambda: False)

    with pytest.raises(RuntimeError, match="enabled distributed context requires an initialized process group"):
        barrier(context)


def test_rank0_training_lifecycle_acquires_exclusive_before_entry_barrier_and_fault_exit(
    monkeypatch,
    execution_context,
) -> None:
    layout = train_module.training_run_layout(
        execution_context.runtime,
        task_id="task_ab12",
        variant_id="pfm_delta",
    )
    context = DistributedContext(True, 0, 0, 2, torch.device("cpu"), "nccl")
    events: list[str] = []

    @contextmanager
    def traced_lock(root):
        assert root == layout.variant_root
        events.append("lock-enter")
        try:
            yield
        finally:
            events.append("lock-exit")

    monkeypatch.setattr(train_module, "variant_writer_lock", traced_lock)
    monkeypatch.setattr(
        train_module,
        "distributed_barrier",
        lambda distributed_context: events.append("entry-barrier"),
    )

    with pytest.raises(RuntimeError, match="injected rank failure"):
        with train_module._variant_training_lifecycle(layout, context):
            events.append("body")
            raise RuntimeError("injected rank failure")

    assert events == ["lock-enter", "entry-barrier", "body", "lock-exit"]


def test_nonzero_rank_training_lifecycle_waits_at_entry_without_taking_lock(
    monkeypatch,
    execution_context,
) -> None:
    layout = train_module.training_run_layout(
        execution_context.runtime,
        task_id="task_ab12",
        variant_id="pfm_delta",
    )
    context = DistributedContext(True, 1, 1, 2, torch.device("cpu"), "nccl")
    events: list[str] = []
    monkeypatch.setattr(
        train_module,
        "variant_writer_lock",
        lambda *args, **kwargs: pytest.fail("nonzero rank must not take the lifecycle lock"),
        raising=False,
    )
    monkeypatch.setattr(
        train_module,
        "distributed_barrier",
        lambda distributed_context: events.append("entry-barrier"),
    )

    with train_module._variant_training_lifecycle(layout, context):
        events.append("body")

    assert events == ["entry-barrier", "body"]


def test_make_dataloaders_from_datasets_respects_train_sampler_order():
    images = torch.arange(4, dtype=torch.float32).view(4, 1, 1, 1)
    labels = torch.arange(4)
    dataset = TensorDataset(images, labels)
    config = ExperimentConfig(batch_size=2, num_workers=0)

    train_loader, _ = make_dataloaders_from_datasets(
        dataset,
        dataset,
        config,
        train_sampler=ReverseSampler(len(dataset)),
    )

    _, first_labels = next(iter(train_loader))
    assert first_labels.tolist() == [3, 2]


def test_distributed_imagenet_training_requests_datasets_without_download(tmp_path, monkeypatch, execution_context):
    import pfm.training.runner as train_module

    captured = {}
    job_context = replace(execution_context, rank=0, local_rank=0, world_size=2)

    def fake_make_datasets(config, data_root):
        captured["dataset"] = config.dataset
        captured["data_root"] = data_root
        raise RuntimeError("stop after dataset contract")

    monkeypatch.setattr(train_module, "make_datasets", fake_make_datasets)
    monkeypatch.setattr(
        train_module,
        "validate_dataset_manifest",
        lambda data_root, dataset_name: {
            "scientific_identity": {"dataset": dataset_name},
            "scientific_identity_sha256": "d" * 64,
        },
    )
    monkeypatch.setattr(train_module, "_validate_training_contexts", lambda *args: None)
    monkeypatch.setattr(train_module, "distributed_barrier", lambda context: None)
    distributed_context = DistributedContext(True, 0, 0, 2, torch.device("cpu"), "nccl")

    config = ExperimentConfig(
        dataset="imagenet",
        architecture="jit_transformer",
        image_channels=3,
        image_size=256,
        num_classes=1000,
        batch_size=2,
        num_workers=0,
        time_sampling="logit_normal",
    )

    with pytest.raises(RuntimeError, match="stop after dataset contract"):
        train_variant(
            "pfm_delta",
            replace(config, limit_train_batches=1),
            job_context=job_context,
            distributed_context=distributed_context,
            task_id="task_ab12",
            variant_id="pfm_delta",
            show_progress=False,
        )

    assert captured == {
        "dataset": "imagenet",
        "data_root": job_context.runtime.dataset_root("imagenet"),
    }


def test_unwrap_model_returns_wrapped_module():
    module = torch.nn.Linear(2, 2)

    class Wrapper(torch.nn.Module):
        def __init__(self, wrapped):
            super().__init__()
            self.module = wrapped

    assert unwrap_model(Wrapper(module)) is module
    assert unwrap_model(module) is module


def test_mean_epoch_loss_uses_sample_weighted_local_average_without_distributed():
    import pfm.training.runner as train_module

    context = DistributedContext(False, 0, 0, 1, torch.device("cpu"), None)
    totals = torch.tensor([12.0, 3.0], dtype=torch.float64)

    assert train_module._mean_epoch_loss(totals, context, torch.device("cpu")) == 4.0
    with pytest.raises(ValueError, match="epoch produced no training samples"):
        train_module._mean_epoch_loss(
            torch.tensor([12.0, 0.0], dtype=torch.float64),
            context,
            torch.device("cpu"),
        )


def test_single_process_batch_metrics_record_scope(
    tmp_path,
    monkeypatch,
    execution_context,
    capsys,
):
    images = torch.zeros(4, 1, 28, 28)
    labels = torch.arange(4) % 10
    dataset = TensorDataset(images, labels)

    monkeypatch.setattr("pfm.training.runner.make_datasets", lambda config, data_root: (dataset, dataset))
    monkeypatch.setattr(
        train_module,
        "validate_dataset_manifest",
        lambda data_root, dataset_name: {
            "scientific_identity": {"dataset": dataset_name},
            "scientific_identity_sha256": "d" * 64,
        },
    )
    monkeypatch.setattr(train_module, "_validate_training_contexts", lambda *args: None)
    distributed_context = DistributedContext(False, 0, 0, 1, torch.device("cpu"), None)
    config = ExperimentConfig(
        epochs=1,
        batch_size=4,
        num_workers=0,
        dataset="mnist",
    )
    paths = train_variant(
        "pfm_full",
        replace(config, limit_train_batches=1),
        job_context=execution_context,
        distributed_context=distributed_context,
        task_id="metric_scope",
        variant_id="pfm_full",
        show_progress=False,
    )
    metric_path = paths["metrics"]
    record = json.loads(metric_path.read_text(encoding="utf-8").splitlines()[0])
    assert record["metric_scope"] == "single_process_batch"
    assert record["grad_norm_pre_clip"] >= 0.0
    assert record["grad_clip_applied"] in (0.0, 1.0)
    assert record["learning_rate"] == pytest.approx(0.0002)
    performance_records = [
        json.loads(line) for line in capsys.readouterr().out.splitlines()
    ]
    assert [record["record_type"] for record in performance_records] == [
        "training_setup_performance",
        "training_first_step_phase",
        "training_first_step_phase",
        "training_first_step_phase",
        "training_first_step_phase",
        "training_epoch_performance",
    ]
    assert [record["phase"] for record in performance_records[1:5]] == [
        "forward_start",
        "forward_complete",
        "backward_complete",
        "optimizer_complete",
    ]
    assert {record["rank"] for record in performance_records[1:5]} == {0}
    epoch_performance = performance_records[5]
    assert epoch_performance["epoch"] == 1
    assert epoch_performance["steps_per_rank"] == 1
    assert epoch_performance["global_samples"] == 4
    assert epoch_performance["cuda_step_ms"]["available"] is False
    assert epoch_performance["checkpoint"]["recovery_published"] is True
    assert epoch_performance["checkpoint"]["immutable_published"] is False
    assert epoch_performance["gradient_clipping"]["total_steps"] == 1
    assert 0.0 <= epoch_performance["gradient_clipping"]["trigger_rate"] <= 1.0


def test_epoch_performance_uses_slowest_rank_for_timing_and_global_peak_memory(
    monkeypatch,
):
    context = DistributedContext(True, 0, 0, 2, torch.device("cuda", 0), "nccl")
    local = {
        field: 0.0 for field in train_module._EPOCH_PERFORMANCE_FIELDS
    }
    local.update(
        {
            "train_seconds": 4.0,
            "data_wait_total_seconds": 1.0,
            "steps": 10.0,
            "samples": 20.0,
            "checkpoint_seconds": 2.0,
            "peak_cuda_allocated_bytes": 100.0,
            "peak_cuda_reserved_bytes": 200.0,
        }
    )
    remote = dict(local)
    remote.update(
        {
            "train_seconds": 5.0,
            "data_wait_total_seconds": 2.0,
            "data_wait_p50_ms": 3.0,
            "cuda_step_first_ms": 7.0,
            "samples": 18.0,
            "checkpoint_seconds": 1.0,
            "peak_cuda_allocated_bytes": 300.0,
            "peak_cuda_reserved_bytes": 400.0,
        }
    )
    real_tensor = torch.tensor

    def fake_tensor(values, *, dtype, device):
        assert device == torch.device("cuda", 0)
        return real_tensor(values, dtype=dtype)

    def fake_all_gather(gathered, local_values):
        gathered[0].copy_(local_values)
        gathered[1].copy_(
            real_tensor(
                [remote[field] for field in train_module._EPOCH_PERFORMANCE_FIELDS],
                dtype=local_values.dtype,
            )
        )

    monkeypatch.setattr(train_module.torch, "tensor", fake_tensor)
    monkeypatch.setattr(train_module.dist, "all_gather", fake_all_gather)

    summary = train_module._aggregate_epoch_performance(
        local,
        context,
        torch.device("cuda", 0),
    )

    assert summary["slowest_train_rank"] == 1
    assert summary["steps_per_rank"] == 10
    assert summary["max_rank_train_seconds"] == 5.0
    assert summary["global_samples"] == 38
    assert summary["global_images_per_second"] == pytest.approx(7.6)
    assert summary["max_rank_data_wait_share_of_train_time"] == pytest.approx(0.4)
    assert summary["max_rank_data_wait_p50_ms"] == 3.0
    assert summary["max_rank_cuda_step_first_ms"] == 7.0
    assert summary["checkpoint_seconds"] == 2.0
    assert summary["peak_cuda_allocated_bytes"] == 300
    assert summary["peak_cuda_reserved_bytes"] == 400


def test_performance_percentile_interpolates_small_samples():
    assert train_module._percentile([], 0.5) == 0.0
    assert train_module._percentile([3.0], 0.9) == 3.0
    assert train_module._percentile([1.0, 3.0], 0.5) == 2.0


def test_batch_metrics_record_rejects_nonfinite_values_before_writing():
    import pfm.training.runner as train_module

    metrics_file = io.StringIO()

    with pytest.raises(ValueError, match="finite|Out of range"):
        train_module._write_metrics_record(
            metrics_file,
            {
                "epoch": 1,
                "step": 1,
                "batch": 1,
                "metric_scope": "single_process_batch",
                "loss": float("nan"),
            },
        )

    assert metrics_file.getvalue() == ""


def test_epoch_metrics_buffer_writes_ordered_batch_schema_with_one_stack_and_flush(
    monkeypatch,
):
    buffer = train_module._EpochMetricsBuffer()
    buffer.append(
        epoch=2,
        step=5,
        batch=1,
        metric_scope="rank0_local_batch",
        stats={
            "loss": torch.tensor(1.25),
            "residual_mse": torch.tensor(2.5),
        },
    )
    buffer.append(
        epoch=2,
        step=6,
        batch=2,
        metric_scope="rank0_local_batch",
        stats={
            "loss": torch.tensor(0.75),
            "residual_mse": torch.tensor(1.5),
        },
    )
    real_stack = torch.stack
    stack_calls = 0

    def traced_stack(tensors, *args, **kwargs):
        nonlocal stack_calls
        stack_calls += 1
        return real_stack(tensors, *args, **kwargs)

    class RecordingWriter(io.StringIO):
        def __init__(self):
            super().__init__()
            self.flush_calls = 0

        def flush(self):
            self.flush_calls += 1
            return super().flush()

    writer = RecordingWriter()
    monkeypatch.setattr(train_module.torch, "stack", traced_stack)

    buffer.flush(writer)

    assert stack_calls == 1
    assert writer.flush_calls == 1
    assert [json.loads(line) for line in writer.getvalue().splitlines()] == [
        {
            "epoch": 2,
            "step": 5,
            "batch": 1,
            "metric_scope": "rank0_local_batch",
            "loss": 1.25,
            "residual_mse": 2.5,
        },
        {
            "epoch": 2,
            "step": 6,
            "batch": 2,
            "metric_scope": "rank0_local_batch",
            "loss": 0.75,
            "residual_mse": 1.5,
        },
    ]


def test_mean_epoch_loss_reduces_sample_weighted_totals_across_distributed_ranks(monkeypatch):
    import pfm.training.runner as train_module

    context = DistributedContext(True, 0, 0, 2, torch.device("cuda", 0), "nccl")
    calls = {}

    class FakeCudaTotals:
        shape = (2,)
        dtype = torch.float64
        device = torch.device("cuda", 0)

        def __init__(self):
            self.values = [10.0, 5.0]
            self.cpu_calls = 0

        def cpu(self):
            self.cpu_calls += 1
            return torch.tensor(self.values, dtype=self.dtype)

    totals = FakeCudaTotals()

    class FakeDist:
        class ReduceOp:
            SUM = "sum"

        @staticmethod
        def is_available():
            return True

        @staticmethod
        def is_initialized():
            return True

        @staticmethod
        def all_reduce(tensor, op=None):
            calls["op"] = op
            assert tensor.values == [10.0, 5.0]
            tensor.values = [40.0, 10.0]

    monkeypatch.setattr(train_module, "dist", FakeDist, raising=False)

    mean_loss = train_module._mean_epoch_loss(
        totals,
        context,
        torch.device("cuda", 0),
    )

    assert mean_loss == 4.0
    assert calls["op"] == FakeDist.ReduceOp.SUM
    assert totals.cpu_calls == 1


def test_mean_epoch_loss_reduces_even_when_local_rank_has_no_batches(monkeypatch):
    import pfm.training.runner as train_module

    context = DistributedContext(True, 0, 0, 2, torch.device("cuda", 0), "nccl")

    class FakeCudaTotals:
        shape = (2,)
        dtype = torch.float64
        device = torch.device("cuda", 0)

        def __init__(self):
            self.values = [0.0, 0.0]

        def cpu(self):
            return torch.tensor(self.values, dtype=self.dtype)

    totals = FakeCudaTotals()

    class FakeDist:
        class ReduceOp:
            SUM = "sum"

        @staticmethod
        def is_available():
            return True

        @staticmethod
        def is_initialized():
            return True

        @staticmethod
        def all_reduce(tensor, op=None):
            assert tensor.values == [0.0, 0.0]
            tensor.values = [12.0, 3.0]

    monkeypatch.setattr(train_module, "dist", FakeDist, raising=False)

    assert train_module._mean_epoch_loss(
        totals,
        context,
        torch.device("cuda", 0),
    ) == 4.0


@pytest.mark.parametrize(
    ("context", "device", "match"),
    [
        (DistributedContext(True, 0, 0, 2, torch.device("cpu"), "nccl"), torch.device("cpu"), "CUDA"),
        (DistributedContext(True, 0, 0, 2, torch.device("cuda", 0), "gloo"), torch.device("cuda", 0), "NCCL"),
    ],
)
def test_mean_epoch_loss_rejects_synthetic_distributed_contexts_before_collective(
    monkeypatch,
    context,
    device,
    match,
):
    import pfm.training.runner as train_module

    class FakeDist:
        class ReduceOp:
            SUM = "sum"

        @staticmethod
        def is_available():
            return True

        @staticmethod
        def is_initialized():
            return True

        @staticmethod
        def all_reduce(tensor, op=None):
            raise AssertionError("synthetic distributed contexts must fail before all_reduce")

    monkeypatch.setattr(train_module, "dist", FakeDist, raising=False)

    if device.type == "cuda":
        class FakeCudaTotals:
            shape = (2,)
            dtype = torch.float64
            device = torch.device("cuda", 0)

        totals = FakeCudaTotals()
    else:
        totals = torch.tensor([10.0, 5.0], dtype=torch.float64)

    with pytest.raises(RuntimeError, match=match):
        train_module._mean_epoch_loss(totals, context, device)


def test_training_adamw_fusion_matches_device_and_cuda_resume_cannot_disable_it():
    model = torch.nn.Linear(2, 2)

    cpu_optimizer = train_module._build_training_optimizer(
        model,
        lr=1e-3,
        weight_decay=0.01,
        betas=(0.9, 0.95),
        device=torch.device("cpu"),
    )
    cuda_optimizer = train_module._build_training_optimizer(
        model,
        lr=1e-3,
        weight_decay=0.01,
        betas=(0.9, 0.95),
        device=torch.device("cuda", 0),
    )

    assert all(group["fused"] is False for group in cpu_optimizer.param_groups)
    assert all(group["fused"] is True for group in cuda_optimizer.param_groups)
    assert cpu_optimizer.param_groups[0]["betas"] == pytest.approx((0.9, 0.95))
    cuda_optimizer.param_groups[0]["fused"] = False
    with pytest.raises(RuntimeError, match="expected fused=True"):
        train_module._validate_optimizer_execution_contract(
            cuda_optimizer,
            torch.device("cuda", 0),
        )


def test_training_gradient_clipping_rejects_nonfinite_gradients_before_optimizer_step():
    model = torch.nn.Linear(2, 2)
    model(torch.ones(1, 2)).sum().backward()
    model.weight.grad[0, 0] = float("nan")

    with pytest.raises(RuntimeError, match="non-finite"):
        train_module._clip_training_gradients(model)


def test_training_gradient_clipping_reports_pre_clip_norm():
    model = torch.nn.Linear(2, 1, bias=False)
    model.weight.grad = torch.tensor([[3.0, 4.0]])

    grad_norm = train_module._clip_training_gradients(model)

    assert float(grad_norm) == pytest.approx(5.0)
    assert float(torch.linalg.vector_norm(model.weight.grad)) == pytest.approx(1.0)


def test_training_learning_rate_matches_upstream_linear_warmup_then_constant():
    model = torch.nn.Linear(2, 2)
    optimizer = torch.optim.AdamW(model.parameters(), lr=1.0)

    first = train_module._adjust_training_learning_rate(
        optimizer,
        base_learning_rate=2e-4,
        warmup_epochs=5,
        lr_schedule="constant",
        epoch=1,
        batch_index=1,
        batches_per_epoch=4,
    )
    middle = train_module._adjust_training_learning_rate(
        optimizer,
        base_learning_rate=2e-4,
        warmup_epochs=5,
        lr_schedule="constant",
        epoch=3,
        batch_index=3,
        batches_per_epoch=4,
    )
    settled = train_module._adjust_training_learning_rate(
        optimizer,
        base_learning_rate=2e-4,
        warmup_epochs=5,
        lr_schedule="constant",
        epoch=6,
        batch_index=1,
        batches_per_epoch=4,
    )

    assert first == pytest.approx(0.0)
    assert middle == pytest.approx(1e-4)
    assert settled == pytest.approx(2e-4)
    assert optimizer.param_groups[0]["lr"] == pytest.approx(2e-4)


def test_jit_style_lr_scaling_uses_effective_batch_size():
    import pytest
    import pfm.training.runner as train_module

    config = ExperimentConfig(
        architecture="jit_transformer",
        batch_size=128,
        lr=None,
        base_lr=5e-5,
        lr_reference_batch_size=256,
        scale_lr_by_effective_batch=True,
    )

    eight_gpu = train_module.resolve_effective_learning_rate(config, world_size=8)
    four_gpu = train_module.resolve_effective_learning_rate(config, world_size=4)

    assert eight_gpu == pytest.approx(2e-4)
    assert four_gpu == pytest.approx(1e-4)


def test_explicit_lr_is_used_without_effective_batch_scaling():
    import pytest
    import pfm.training.runner as train_module

    config = ExperimentConfig(
        architecture="jit_transformer",
        batch_size=128,
        lr=3e-4,
        base_lr=None,
        lr_reference_batch_size=256,
        scale_lr_by_effective_batch=False,
    )

    resolved = train_module.resolve_effective_learning_rate(config, world_size=16)

    assert resolved == pytest.approx(3e-4)


def test_unified_train_cli_rejects_scientific_jit_overrides():
    from pfm.cli.train import parse_args

    with pytest.raises(SystemExit):
        parse_args(
            [
            "--config",
            "configs/imagenet256_jit_b16.yaml",
            "--variant",
            "pfm_delta",
            "--task-id",
            "task_ab12",
            "--architecture",
            "jit_transformer",
            "--jit-model",
            "JiT-B/16",
            "--noise-scale",
            "1.0",
            "--base-lr",
            "0.00005",
            "--lr-reference-batch-size",
            "256",
            "--scale-lr-by-effective-batch",
            ]
        )
