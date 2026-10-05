from __future__ import annotations

import argparse
from concurrent.futures import ThreadPoolExecutor
from contextlib import contextmanager
from dataclasses import replace
import inspect
import json
from threading import Barrier

import pytest
import torch

from pfm.core.config import ExperimentConfig
from pfm.training.cli import add_training_arguments
from pfm.training import runner as train_module
from pfm.core.distributed import DistributedContext
from pfm.core.execution import runtime_identity
from pfm.training.runner import validate_distributed_workload, validate_single_card_workload


def test_training_run_layout_uses_runtime_experiments_root(execution_context):
    layout = train_module.training_run_layout(
        execution_context.runtime,
        task_id="task_ab12",
        variant_id="pfm_delta",
    )

    assert layout.output_root == execution_context.runtime.experiments_root
    assert layout.variant_root == execution_context.runtime.experiments_root / "task_ab12" / "pfm_delta"


def test_training_runner_requires_explicit_execution_contexts() -> None:
    parameters = inspect.signature(train_module.train_variant).parameters

    assert "job_context" in parameters
    assert "distributed_context" in parameters
    assert "output_root" not in parameters
    assert not hasattr(train_module, "run_training_variant")


def test_train_variant_runs_complete_owned_body_under_exclusive_lifecycle(
    monkeypatch,
    execution_context,
) -> None:
    active = False
    events: list[str] = []

    @contextmanager
    def traced_lifecycle(layout, distributed_context):
        nonlocal active
        assert layout.variant_root.name == "velocity_fm"
        assert distributed_context.world_size == 1
        active = True
        events.append("enter")
        try:
            yield
        finally:
            events.append("exit")
            active = False

    def owned_body(**kwargs):
        assert active
        events.append("owned-body")
        return {"variant_root": kwargs["layout"].variant_root}

    context = DistributedContext(False, 0, 0, 1, torch.device("cpu"), None)
    monkeypatch.setattr(train_module, "_validate_training_contexts", lambda *args: None)
    monkeypatch.setattr(
        train_module,
        "_variant_training_lifecycle",
        traced_lifecycle,
        raising=False,
    )
    monkeypatch.setattr(train_module, "_train_variant_owned", owned_body)

    result = train_module.train_variant(
        "velocity_fm",
        ExperimentConfig(dataset="mnist", epochs=1),
        job_context=execution_context,
        distributed_context=context,
        task_id="task_ab12",
        variant_id="velocity_fm",
        show_progress=False,
    )

    assert result["variant_root"].name == "velocity_fm"
    assert events == ["enter", "owned-body", "exit"]


def test_training_workload_validation_requires_explicit_world_size() -> None:
    parameter = inspect.signature(train_module.validate_training_workload_contract).parameters["world_size"]

    assert parameter.default is inspect.Parameter.empty
    with pytest.raises(ValueError, match="world_size must be >= 1"):
        train_module.validate_training_workload_contract(ExperimentConfig(dataset="mnist"), world_size=0)


def test_training_arguments_require_config_and_exclude_runtime_fallbacks() -> None:
    parser = argparse.ArgumentParser()
    add_training_arguments(parser)

    with pytest.raises(SystemExit):
        parser.parse_args(["--variant", "pfm_delta", "--task-id", "task_ab12"])

    option_strings = {option for action in parser._actions for option in action.option_strings}
    assert "--device" not in option_strings
    assert "--data-root" not in option_strings
    assert "--output-root" not in option_strings
    assert set(inspect.signature(add_training_arguments).parameters) == {"parser"}


def test_train_variant_rejects_device_mismatch_before_data_work(monkeypatch, execution_context) -> None:
    data_work_attempted = False

    def fail_make_datasets(*args, **kwargs):
        nonlocal data_work_attempted
        data_work_attempted = True
        raise AssertionError("training guard must run before data work")

    monkeypatch.setattr(train_module, "make_datasets", fail_make_datasets)
    monkeypatch.setattr(
        train_module,
        "execution_device",
        lambda job_context: torch.device("cuda", job_context.local_rank),
    )
    context = DistributedContext(False, 0, 0, 1, torch.device("cpu"), None)

    with pytest.raises(RuntimeError, match="training requires cuda:0, got cpu"):
        train_module.train_variant(
            "velocity_fm",
            replace(ExperimentConfig(), limit_train_batches=1),
            job_context=execution_context,
            distributed_context=context,
            task_id="task_ab12",
            variant_id="velocity_fm",
            show_progress=False,
        )

    assert data_work_attempted is False


def test_training_context_guard_accepts_explicit_cpu_without_cuda(monkeypatch, execution_context) -> None:
    monkeypatch.setattr(torch.cuda, "is_available", lambda: False)
    monkeypatch.setattr(train_module.dist, "is_initialized", lambda: False)
    job_context = replace(execution_context, device="cpu")
    context = DistributedContext(False, 0, 0, 1, torch.device("cpu"), None)

    train_module._validate_training_contexts(job_context, context)


def test_train_variant_rejects_unavailable_cuda_before_data_work(monkeypatch, execution_context) -> None:
    monkeypatch.setattr(torch.cuda, "is_available", lambda: False)
    monkeypatch.setattr(
        train_module,
        "make_datasets",
        lambda *args, **kwargs: pytest.fail("CUDA guard must fail before data work"),
    )
    context = DistributedContext(False, 0, 0, 1, torch.device("cuda", 0), None)

    with pytest.raises(RuntimeError, match="CUDA requested but unavailable"):
        train_module.train_variant(
            "velocity_fm",
            replace(ExperimentConfig(), limit_train_batches=1),
            job_context=execution_context,
            distributed_context=context,
            task_id="task_ab12",
            variant_id="velocity_fm",
            show_progress=False,
        )


def test_train_variant_rejects_local_rank_beyond_visible_cuda_before_data_work(
    monkeypatch,
    execution_context,
) -> None:
    job_context = replace(execution_context, local_rank=1)
    monkeypatch.setattr(torch.cuda, "is_available", lambda: True)
    monkeypatch.setattr(torch.cuda, "device_count", lambda: 1)
    monkeypatch.setattr(
        train_module,
        "make_datasets",
        lambda *args, **kwargs: pytest.fail("CUDA guard must fail before data work"),
    )
    context = DistributedContext(False, 0, 1, 1, torch.device("cuda", 1), None)

    with pytest.raises(RuntimeError, match="LOCAL_RANK exceeds"):
        train_module.train_variant(
            "velocity_fm",
            replace(ExperimentConfig(), limit_train_batches=1),
            job_context=job_context,
            distributed_context=context,
            task_id="task_ab12",
            variant_id="velocity_fm",
            show_progress=False,
        )


@pytest.mark.parametrize(
    ("context", "match"),
    [
        (DistributedContext(False, 0, 0, 1, torch.device("cuda", 1), None), "training requires cuda:0, got cuda:1"),
        (DistributedContext(True, 0, 0, 1, torch.device("cuda", 0), "nccl"), "single-process"),
        (DistributedContext(False, 0, 0, 2, torch.device("cuda", 0), None), "distributed"),
        (DistributedContext(True, 0, 0, 2, torch.device("cuda", 0), "gloo"), "NCCL"),
    ],
)
def test_training_context_guard_requires_exact_mode_contract(
    context,
    match,
    monkeypatch,
    execution_context,
) -> None:
    job_context = replace(execution_context, world_size=context.world_size)
    monkeypatch.setattr(
        train_module,
        "execution_device",
        lambda context: torch.device("cuda", context.local_rank),
    )
    monkeypatch.setattr(train_module.dist, "is_available", lambda: True)
    monkeypatch.setattr(train_module.dist, "is_initialized", lambda: context.world_size > 1)

    with pytest.raises((ValueError, RuntimeError), match=match):
        train_module._validate_training_contexts(job_context, context)


@pytest.mark.parametrize(
    ("backend", "rank", "world_size", "match"),
    [
        ("gloo", 0, 2, "NCCL"),
        ("nccl", 1, 2, "rank"),
        ("nccl", 0, 3, "world_size"),
    ],
)
def test_training_context_guard_validates_initialized_process_group_identity(
    backend,
    rank,
    world_size,
    match,
    monkeypatch,
    execution_context,
) -> None:
    job_context = replace(execution_context, world_size=2)
    context = DistributedContext(True, 0, 0, 2, torch.device("cuda", 0), "nccl")
    monkeypatch.setattr(
        train_module,
        "execution_device",
        lambda context: torch.device("cuda", context.local_rank),
    )
    monkeypatch.setattr(train_module.dist, "is_available", lambda: True)
    monkeypatch.setattr(train_module.dist, "is_initialized", lambda: True)
    monkeypatch.setattr(train_module.dist, "get_backend", lambda: backend)
    monkeypatch.setattr(train_module.dist, "get_rank", lambda: rank)
    monkeypatch.setattr(train_module.dist, "get_world_size", lambda: world_size)

    with pytest.raises(RuntimeError, match=match):
        train_module._validate_training_contexts(job_context, context)


def test_training_context_guard_requires_initialized_group_for_distributed_job(
    monkeypatch,
    execution_context,
) -> None:
    job_context = replace(execution_context, world_size=2)
    context = DistributedContext(True, 0, 0, 2, torch.device("cuda", 0), "nccl")
    monkeypatch.setattr(
        train_module,
        "execution_device",
        lambda context: torch.device("cuda", context.local_rank),
    )
    monkeypatch.setattr(train_module.dist, "is_available", lambda: True)
    monkeypatch.setattr(train_module.dist, "is_initialized", lambda: False)

    with pytest.raises(RuntimeError, match="initialized process group"):
        train_module._validate_training_contexts(job_context, context)


def test_training_context_guard_rejects_process_group_for_single_process_job(
    monkeypatch,
    execution_context,
) -> None:
    context = DistributedContext(False, 0, 0, 1, torch.device("cuda", 0), None)
    monkeypatch.setattr(
        train_module,
        "execution_device",
        lambda context: torch.device("cuda", context.local_rank),
    )
    monkeypatch.setattr(train_module.dist, "is_available", lambda: True)
    monkeypatch.setattr(train_module.dist, "is_initialized", lambda: True)

    with pytest.raises(RuntimeError, match="no initialized process group"):
        train_module._validate_training_contexts(execution_context, context)


def test_task_manifest_is_created_once_and_validated_across_variant_jobs(execution_context) -> None:
    first_context = execution_context
    second_context = replace(execution_context, world_size=2)
    first_layout = train_module.training_run_layout(
        execution_context.runtime,
        task_id="shared_task",
        variant_id="velocity_fm",
    )
    second_layout = train_module.training_run_layout(
        execution_context.runtime,
        task_id="shared_task",
        variant_id="pfm_delta",
    )
    config = ExperimentConfig()

    train_module._write_training_manifests(
        first_layout,
        "velocity_fm",
        config,
        runtime_identity(first_context),
    )
    original_task_manifest = first_layout.task_manifest_path.read_bytes()
    train_module._write_training_manifests(
        second_layout,
        "pfm_delta",
        config,
        runtime_identity(second_context),
    )

    task_manifest = json.loads(first_layout.task_manifest_path.read_text(encoding="utf-8"))
    second_variant_manifest = json.loads(second_layout.variant_manifest_path.read_text(encoding="utf-8"))
    assert first_layout.task_manifest_path.read_bytes() == original_task_manifest
    assert set(task_manifest) == {
        "task_id",
        "output_root",
        "producer_runtime_identity",
    }
    assert task_manifest["producer_runtime_identity"] == runtime_identity(first_context)
    assert second_variant_manifest["runtime_identity"] == runtime_identity(second_context)


def test_task_manifest_rejects_project_or_commit_mismatch_without_replacement(execution_context) -> None:
    first_layout = train_module.training_run_layout(
        execution_context.runtime,
        task_id="shared_task",
        variant_id="velocity_fm",
    )
    config = ExperimentConfig()
    train_module._write_training_manifests(
        first_layout,
        "velocity_fm",
        config,
        runtime_identity(execution_context),
    )
    original = first_layout.task_manifest_path.read_bytes()
    mismatched_contexts = [
        ("git_commit", replace(execution_context, git_commit="b" * 40)),
        (
            "project_name",
            replace(
                execution_context,
                runtime=replace(execution_context.runtime, project_name="other_project"),
            ),
        ),
    ]

    for field_name, mismatched in mismatched_contexts:
        second_layout = train_module.training_run_layout(
            execution_context.runtime,
            task_id="shared_task",
            variant_id=f"pfm_delta_{field_name}",
        )
        with pytest.raises(ValueError, match=rf"task manifest.*{field_name}"):
            train_module._write_training_manifests(
                second_layout,
                "pfm_delta",
                config,
                runtime_identity(mismatched),
            )
        assert not second_layout.variant_manifest_path.exists()

    assert first_layout.task_manifest_path.read_bytes() == original


def test_racing_task_manifest_creators_with_different_commits_cannot_both_win(execution_context) -> None:
    contexts = [
        execution_context,
        replace(execution_context, git_commit="b" * 40),
    ]
    layouts = [
        train_module.training_run_layout(
            execution_context.runtime,
            task_id="racing_task",
            variant_id=variant_id,
        )
        for variant_id in ("velocity_fm", "pfm_delta")
    ]

    start = Barrier(2)

    def write(index):
        start.wait()
        train_module._write_training_manifests(
            layouts[index],
            layouts[index].variant_id,
            ExperimentConfig(),
            runtime_identity(contexts[index]),
        )
        return "created"

    with ThreadPoolExecutor(max_workers=2) as pool:
        futures = [pool.submit(write, index) for index in range(2)]
    outcomes = []
    for future in futures:
        try:
            outcomes.append(future.result())
        except ValueError as error:
            outcomes.append(str(error))

    assert outcomes.count("created") == 1
    assert sum("task manifest" in outcome for outcome in outcomes) == 1


def test_single_card_entrypoint_accepts_mnist_and_cifar10() -> None:
    validate_single_card_workload(ExperimentConfig(dataset="mnist"), world_size=1)
    validate_single_card_workload(ExperimentConfig(dataset="cifar10", image_channels=3, image_size=32), world_size=1)


def test_single_card_entrypoint_accepts_imagenet() -> None:
    validate_single_card_workload(
        ExperimentConfig(dataset="imagenet", architecture="jit_transformer", image_size=256),
        world_size=1,
    )


@pytest.mark.parametrize("world_size", [0, -1, 2])
def test_single_card_entrypoint_requires_world_size_one(world_size: int) -> None:
    with pytest.raises(ValueError, match="world_size"):
        validate_single_card_workload(ExperimentConfig(dataset="mnist"), world_size=world_size)


def test_distributed_entrypoint_accepts_imagenet_jit_and_cifar10_edm() -> None:
    config = ExperimentConfig(dataset="imagenet", architecture="jit_transformer", image_channels=3, num_classes=1000, image_size=256)
    validate_distributed_workload(config, world_size=2)
    cifar10 = ExperimentConfig(dataset="cifar10", architecture="edm_ddpmpp", image_channels=3, image_size=32)
    validate_distributed_workload(cifar10, world_size=2)
    train_module.validate_training_workload_contract(cifar10, world_size=2)
    train_module.validate_training_workload_contract(cifar10, world_size=1)


def test_distributed_entrypoint_rejects_single_card_and_unsupported_models() -> None:
    imagenet = ExperimentConfig(dataset="imagenet", architecture="jit_transformer")
    cifar10 = ExperimentConfig(dataset="cifar10", architecture="conditional_unet")
    with pytest.raises(ValueError, match="world_size"):
        validate_distributed_workload(imagenet, world_size=1)
    with pytest.raises(ValueError, match="ImageNet/JiT"):
        validate_distributed_workload(cifar10, world_size=2)


def test_direct_runner_accepts_single_card_imagenet() -> None:
    config = ExperimentConfig(
        dataset="imagenet",
        architecture="jit_transformer",
        image_channels=3,
        image_size=256,
        num_classes=1000,
        time_sampling="logit_normal",
    )
    train_module.validate_training_workload_contract(config, world_size=1)


def test_direct_runner_rejects_single_card_dataset_in_distributed_world() -> None:
    config = ExperimentConfig(dataset="mnist")
    with pytest.raises(ValueError, match="world_size"):
        train_module.validate_training_workload_contract(config, world_size=2)
