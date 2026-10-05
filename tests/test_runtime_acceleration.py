from __future__ import annotations

import contextlib
from dataclasses import replace
from pathlib import Path

import pytest
import torch
import yaml
from torch.utils.data import TensorDataset

from pfm.core.config import ExperimentConfig, load_config
from pfm.data.datasets import make_dataloaders_from_datasets
from pfm.models.factory import build_model


PROJECT_ROOT = Path(__file__).resolve().parents[1]


def _write_runtime_config(path: Path, updates: dict[str, object]) -> None:
    source = PROJECT_ROOT / "configs" / "imagenet256_jit_b16.yaml"
    payload = yaml.safe_load(source.read_text(encoding="utf-8"))
    payload.update(updates)
    path.write_text(yaml.safe_dump(payload, sort_keys=False), encoding="utf-8")


def test_config_loads_runtime_acceleration_options(tmp_path):
    config_path = tmp_path / "config.yaml"
    _write_runtime_config(
        config_path,
        {
            "mixed_precision": "bf16",
            "allow_tf32": True,
            "cudnn_benchmark": True,
            "jit_compile": True,
            "drop_last": True,
        },
    )

    config = load_config(config_path)

    assert config.mixed_precision == "bf16"
    assert config.allow_tf32 is True
    assert config.cudnn_benchmark is True
    assert config.jit_compile is True
    assert config.drop_last is True
    assert ExperimentConfig().mixed_precision == "bf16"
    assert ExperimentConfig().allow_tf32 is False
    assert ExperimentConfig().cudnn_benchmark is True
    assert ExperimentConfig().jit_compile is True
    assert ExperimentConfig().drop_last is True


def test_runtime_boolean_options_reject_string_values(tmp_path):
    config_path = tmp_path / "config.yaml"
    _write_runtime_config(
        config_path,
        {
            "allow_tf32": "false",
            "cudnn_benchmark": "false",
            "jit_compile": "false",
            "drop_last": "false",
        },
    )

    with pytest.raises(ValueError, match="boolean"):
        load_config(config_path)


def test_imagenet_jit_configs_enable_upstream_runtime_defaults():
    config = load_config(PROJECT_ROOT / "configs" / "imagenet256_jit_b16.yaml")
    paper_config = load_config(PROJECT_ROOT / "configs" / "imagenet256_jit_b16_batch256.yaml")

    assert config.mixed_precision == "bf16"
    assert config.allow_tf32 is False
    assert config.cudnn_benchmark is True
    assert config.jit_compile is True
    assert config.drop_last is True
    assert config.lr is None
    assert config.base_lr == 5e-5
    assert config.lr_reference_batch_size == 256
    assert config.scale_lr_by_effective_batch is True
    assert config.jit_attention_implementation == "upstream_fp32_qk"
    assert (config.adamw_beta1, config.adamw_beta2) == pytest.approx((0.9, 0.95))
    assert config.warmup_epochs == 5
    assert config.lr_schedule == "constant"
    assert config.max_grad_norm == pytest.approx(1.0)
    assert paper_config.image_size == 256
    assert paper_config.jit_model == "JiT-B/16"
    assert paper_config.mixed_precision == "bf16"
    assert paper_config.jit_compile is True
    assert paper_config.lr is None
    assert paper_config.scale_lr_by_effective_batch is True
    assert paper_config.batch_size == 256
    assert paper_config.max_grad_norm == pytest.approx(5.0)


def test_dataloader_respects_drop_last_training_batches():
    images = torch.arange(3, dtype=torch.float32).view(3, 1, 1, 1)
    labels = torch.arange(3)
    dataset = TensorDataset(images, labels)
    config = ExperimentConfig(batch_size=2, num_workers=0, drop_last=True)

    train_loader, test_loader = make_dataloaders_from_datasets(dataset, dataset, config)

    assert len(train_loader) == 1
    assert len(test_loader) == 2


def test_dataloaders_persist_worker_processes_when_workers_are_enabled():
    dataset = TensorDataset(torch.zeros(2, 1, 1, 1), torch.zeros(2, dtype=torch.long))
    config = ExperimentConfig(batch_size=1, num_workers=1)

    train_loader, test_loader = make_dataloaders_from_datasets(dataset, dataset, config)

    assert train_loader.persistent_workers is True
    assert test_loader.persistent_workers is True


def test_dataloaders_disable_persistent_workers_without_worker_processes():
    dataset = TensorDataset(torch.zeros(2, 1, 1, 1), torch.zeros(2, dtype=torch.long))
    config = ExperimentConfig(batch_size=1, num_workers=0)

    train_loader, test_loader = make_dataloaders_from_datasets(dataset, dataset, config)

    assert train_loader.persistent_workers is False
    assert test_loader.persistent_workers is False


def test_execution_device_fails_without_cuda(monkeypatch, execution_context):
    from pfm.core.runtime import execution_device

    monkeypatch.setattr(torch.cuda, "is_available", lambda: False)

    with pytest.raises(RuntimeError, match="CUDA requested but unavailable"):
        execution_device(execution_context)


def test_execution_device_sets_local_rank_device(monkeypatch, execution_context):
    from pfm.core.runtime import execution_device

    context = replace(execution_context, local_rank=1)
    calls = []
    monkeypatch.setattr(torch.cuda, "is_available", lambda: True)
    monkeypatch.setattr(torch.cuda, "device_count", lambda: 2)
    monkeypatch.setattr(torch.cuda, "set_device", lambda device: calls.append(device))

    device = execution_device(context)

    assert device == torch.device("cuda", 1)
    assert calls == [torch.device("cuda", 1)]


def test_autocast_context_uses_bf16_only_for_cuda(monkeypatch):
    calls = []

    @contextlib.contextmanager
    def fake_autocast(device_type, dtype):
        calls.append((device_type, dtype))
        yield

    from pfm.core import runtime

    monkeypatch.setattr(runtime.torch.amp, "autocast", fake_autocast)

    with runtime.autocast_context(ExperimentConfig(mixed_precision="bf16"), torch.device("cuda")):
        pass
    with runtime.autocast_context(ExperimentConfig(mixed_precision="bf16"), torch.device("cpu")):
        pass

    assert calls == [("cuda", torch.bfloat16)]


@pytest.mark.parametrize(
    "value",
    [None, False, "", "off", "false", "fp32", "float32", "bfloat16", "BF16", " bf16 "],
)
def test_autocast_context_rejects_noncanonical_mixed_precision(value):
    from pfm.core import runtime

    config = ExperimentConfig(mixed_precision=value)

    with pytest.raises(ValueError, match="mixed_precision must be one of"):
        runtime.autocast_context(config, torch.device("cuda"))


def test_runtime_settings_reset_cuda_backend_flags():
    from pfm.core import runtime

    original_matmul_tf32 = torch.backends.cuda.matmul.allow_tf32
    original_cudnn_tf32 = torch.backends.cudnn.allow_tf32
    original_cudnn_benchmark = torch.backends.cudnn.benchmark
    try:
        runtime.apply_runtime_settings(
            ExperimentConfig(allow_tf32=True, cudnn_benchmark=True),
            torch.device("cuda"),
        )
        assert torch.backends.cuda.matmul.allow_tf32 is True
        assert torch.backends.cudnn.allow_tf32 is True
        assert torch.backends.cudnn.benchmark is True

        runtime.apply_runtime_settings(
            ExperimentConfig(allow_tf32=False, cudnn_benchmark=False),
            torch.device("cuda"),
        )
        assert torch.backends.cuda.matmul.allow_tf32 is False
        assert torch.backends.cudnn.allow_tf32 is False
        assert torch.backends.cudnn.benchmark is False
    finally:
        torch.backends.cuda.matmul.allow_tf32 = original_matmul_tf32
        torch.backends.cudnn.allow_tf32 = original_cudnn_tf32
        torch.backends.cudnn.benchmark = original_cudnn_benchmark


def test_jit_compile_wraps_blocks_and_heads_when_enabled(monkeypatch, tiny_jit_spec):
    calls = []

    def fake_compile(module):
        calls.append(module.__class__.__name__)
        return module

    import pfm.models.jit_backbone as jit_backbone

    monkeypatch.setattr(jit_backbone.torch, "compile", fake_compile, raising=False)
    config = ExperimentConfig(
        architecture="jit_transformer",
        jit_model="JiT-Tiny/4",
        image_channels=3,
        image_size=8,
        num_classes=4,
        jit_compile=True,
    )

    build_model("pfm_delta", config)

    assert calls.count("JiTBlock") == 2
    assert calls.count("JiTImageHead") == 2


def test_jit_compile_true_rejects_missing_torch_compile(monkeypatch):
    import pfm.models.jit_backbone as jit_backbone

    monkeypatch.delattr(jit_backbone.torch, "compile", raising=False)
    with pytest.raises(RuntimeError, match="jit_compile=True"):
        jit_backbone.compile_module(torch.nn.Identity())


def test_train_cli_rejects_runtime_acceleration_overrides():
    from pfm.cli.train import parse_args

    with pytest.raises(SystemExit):
        parse_args(
            [
            "--config",
            "configs/mnist_unet.yaml",
            "--variant",
            "pfm_delta",
            "--task-id",
            "task_ab12",
            "--mixed-precision",
            "bf16",
            "--allow-tf32",
            "--allow-matmul-tf32",
            "--allow-cudnn-tf32",
            "--cudnn-benchmark",
            "--jit-compile",
            "--drop-last",
            ]
        )
