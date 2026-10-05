from dataclasses import replace
import json
from pathlib import Path

import pytest
import torch
from torch.utils.data import TensorDataset

from pfm.evaluation import sample_grid as sample_grid_module
from pfm.training import runner as train_module
from pfm.core.config import ExperimentConfig
from pfm.core.distributed import DistributedContext
from pfm.core.execution import runtime_identity


def _tiny_config(tmp_path, **overrides):
    del tmp_path
    values = {
        "batch_size": 2,
        "num_workers": 0,
        "epochs": 1,
        "time_proj_dim": 16,
        "class_proj_dim": 16,
        "unet_base_channels": 4,
        "unet_channel_mult": (1, 2),
        "unet_num_res_blocks": 1,
        "unet_attention_resolutions": (),
        "unet_norm_groups": 4,
    }
    values.update(overrides)
    return ExperimentConfig(**values)


def _single_process_context(job_context):
    return DistributedContext(
        enabled=False,
        rank=job_context.rank,
        local_rank=job_context.local_rank,
        world_size=job_context.world_size,
        device=torch.device("cpu"),
    )


def _patch_datasets(monkeypatch, dataset):
    monkeypatch.setattr(train_module, "_validate_training_contexts", lambda *args: None)
    monkeypatch.setattr(
        train_module,
        "validate_dataset_manifest",
        lambda data_root, dataset_name: {
            "scientific_identity": {"dataset": dataset_name},
            "scientific_identity_sha256": "d" * 64,
        },
        raising=False,
    )
    monkeypatch.setattr(train_module, "make_datasets", lambda config, data_root: (dataset, dataset))


def _load_recovery_checkpoint(variant_root):
    latest = json.loads((variant_root / "recovery" / "latest.json").read_text())
    manifest_path = variant_root / latest["manifest"]
    manifest = json.loads(manifest_path.read_text())
    return torch.load(manifest_path.parent / manifest["checkpoint"], map_location="cpu")


def test_train_variant_never_runs_evaluation_hooks(tmp_path, monkeypatch, execution_context):
    dataset = TensorDataset(torch.zeros(2, 1, 28, 28), torch.tensor([0, 1]))

    def fail_visualization(*args, **kwargs):
        raise AssertionError("training must not generate visualizations")

    _patch_datasets(monkeypatch, dataset)
    monkeypatch.setattr(sample_grid_module, "generate_sample_artifact", fail_visualization)

    config = _tiny_config(tmp_path)

    train_module.train_variant(
        "velocity_fm",
        replace(config, limit_train_batches=1),
        job_context=execution_context,
        distributed_context=_single_process_context(execution_context),
        task_id="task_ab12",
        variant_id="velocity_fm",
        show_progress=False,
    )


def test_train_variant_writes_immutable_milestones_and_latest_recovery(
    tmp_path, monkeypatch, execution_context
):
    dataset = TensorDataset(torch.zeros(2, 1, 28, 28), torch.tensor([0, 1]))
    _patch_datasets(monkeypatch, dataset)

    config = _tiny_config(tmp_path, epochs=3, checkpoint_every_epochs=2)

    paths = train_module.train_variant(
        "velocity_fm",
        replace(config, limit_train_batches=1),
        job_context=execution_context,
        distributed_context=_single_process_context(execution_context),
        task_id="task_ab12",
        variant_id="velocity_fm",
        show_progress=False,
    )

    variant_root = execution_context.runtime.experiments_root / "task_ab12" / "velocity_fm"
    assert not (variant_root / "epochs" / "epoch_0001" / "checkpoint.pt").exists()
    assert (variant_root / "epochs" / "epoch_0002" / "checkpoint.pt").exists()
    assert not (variant_root / "epochs" / "epoch_0003").exists()
    assert _load_recovery_checkpoint(variant_root)["epoch"] == 3
    assert not (variant_root / "best.pt").exists()
    assert not (variant_root / "final.pt").exists()
    assert paths["variant_root"] == variant_root


def test_train_variant_gathers_rng_for_every_epoch_recovery_checkpoint(
    tmp_path, monkeypatch, execution_context
):
    dataset = TensorDataset(torch.zeros(2, 1, 28, 28), torch.tensor([0, 1]))
    _patch_datasets(monkeypatch, dataset)
    gather_epochs: list[int] = []
    original_gather = train_module.gather_rng_state_for_checkpoint

    def traced_gather(distributed_context):
        gather_epochs.append(len(gather_epochs) + 1)
        return original_gather(distributed_context)

    monkeypatch.setattr(train_module, "gather_rng_state_for_checkpoint", traced_gather)

    train_module.train_variant(
        "velocity_fm",
        replace(_tiny_config(tmp_path, epochs=5, checkpoint_every_epochs=2), limit_train_batches=1),
        job_context=execution_context,
        distributed_context=_single_process_context(execution_context),
        task_id="checkpoint_schedule",
        variant_id="velocity_fm",
        show_progress=False,
    )

    variant_root = execution_context.runtime.experiments_root / "checkpoint_schedule" / "velocity_fm"
    assert gather_epochs == [1, 2, 3, 4, 5]
    assert sorted(path.name for path in (variant_root / "epochs").iterdir()) == [
        "epoch_0002",
        "epoch_0004",
    ]
    assert _load_recovery_checkpoint(variant_root)["epoch"] == 5


def test_train_variant_binds_dataset_identity_before_model_work(
    tmp_path, monkeypatch, execution_context
):
    dataset = TensorDataset(torch.zeros(2, 1, 28, 28), torch.tensor([0, 1]))
    digest = "e" * 64
    events: list[str] = []
    monkeypatch.setattr(train_module, "_validate_training_contexts", lambda *args: None)

    def validate_manifest(data_root, dataset_name):
        del data_root
        events.append("dataset-binding")
        return {
            "scientific_identity": {"dataset": dataset_name},
            "scientific_identity_sha256": digest,
        }

    original_build_model = train_module.build_model

    def build_model_after_binding(*args, **kwargs):
        assert events == ["dataset-binding"]
        events.append("model")
        return original_build_model(*args, **kwargs)

    monkeypatch.setattr(train_module, "validate_dataset_manifest", validate_manifest, raising=False)
    monkeypatch.setattr(train_module, "make_datasets", lambda config, data_root: (dataset, dataset))
    monkeypatch.setattr(train_module, "build_model", build_model_after_binding)

    train_module.train_variant(
        "velocity_fm",
        replace(_tiny_config(tmp_path), limit_train_batches=1),
        job_context=execution_context,
        distributed_context=_single_process_context(execution_context),
        task_id="dataset_binding",
        variant_id="velocity_fm",
        show_progress=False,
    )

    variant_root = (
        execution_context.runtime.experiments_root / "dataset_binding" / "velocity_fm"
    )
    checkpoint = _load_recovery_checkpoint(variant_root)
    assert events == ["dataset-binding", "model"]
    assert checkpoint["dataset_scientific_identity_sha256"] == digest


def test_train_variant_checkpoints_record_effective_runtime_overrides(tmp_path, monkeypatch, execution_context):
    dataset = TensorDataset(torch.zeros(2, 1, 28, 28), torch.tensor([0, 1]))
    _patch_datasets(monkeypatch, dataset)

    config = _tiny_config(tmp_path, epochs=9)

    paths = train_module.train_variant(
        "velocity_fm",
        replace(config, limit_train_batches=1),
        job_context=execution_context,
        distributed_context=_single_process_context(execution_context),
        epochs=2,
        task_id="task_ab12",
        variant_id="velocity_fm",
        show_progress=False,
    )
    variant_root = execution_context.runtime.experiments_root / "task_ab12" / "velocity_fm"
    checkpoint = _load_recovery_checkpoint(variant_root)

    assert checkpoint["training_config"]["epochs"] == 2
    assert "metric_names" not in checkpoint["training_config"]
    assert "sample_every_epochs" not in checkpoint["training_config"]
    assert paths["metrics"] == variant_root / "train_metrics.jsonl"


def test_train_variant_keeps_authored_lr_scaling_and_optimizer_records_resolved_lr(
    tmp_path, monkeypatch, execution_context
):
    dataset = TensorDataset(torch.zeros(2, 1, 28, 28), torch.tensor([0, 1]))
    _patch_datasets(monkeypatch, dataset)

    config = _tiny_config(tmp_path, epochs=1)
    config.lr = None
    config.base_lr = 1e-3
    config.lr_reference_batch_size = 4
    config.scale_lr_by_effective_batch = True

    train_module.train_variant(
        "velocity_fm",
        replace(config, limit_train_batches=1),
        job_context=execution_context,
        distributed_context=_single_process_context(execution_context),
        task_id="task_ab12",
        variant_id="velocity_fm",
        show_progress=False,
    )

    variant_root = execution_context.runtime.experiments_root / "task_ab12" / "velocity_fm"
    checkpoint = _load_recovery_checkpoint(variant_root)
    variant_manifest = json.loads((variant_root / "variant_manifest.json").read_text())
    task_manifest = json.loads((variant_root.parent / "task_manifest.json").read_text())
    latest = json.loads((variant_root / "recovery" / "latest.json").read_text())
    checkpoint_manifest = json.loads((variant_root / latest["manifest"]).read_text())
    producer_identity = runtime_identity(execution_context)

    assert checkpoint["training_config"]["lr"] is None
    assert variant_manifest["training_config"]["lr"] is None
    assert checkpoint["training_config"]["base_lr"] == pytest.approx(1e-3)
    assert checkpoint["training_config"]["scale_lr_by_effective_batch"] is True
    assert checkpoint["optimizer_state"]["param_groups"][0]["lr"] == pytest.approx(5e-4)
    assert task_manifest["producer_runtime_identity"] == producer_identity
    assert variant_manifest["runtime_identity"] == producer_identity
    assert checkpoint["runtime_identity"] == producer_identity
    assert checkpoint_manifest["config_identity"]["runtime_identity"] == producer_identity


def test_train_variant_variant_id_controls_written_layout(tmp_path, monkeypatch, execution_context):
    dataset = TensorDataset(torch.zeros(2, 1, 28, 28), torch.tensor([0, 1]))
    _patch_datasets(monkeypatch, dataset)

    config = _tiny_config(tmp_path)

    paths = train_module.train_variant(
        "velocity_fm",
        replace(config, limit_train_batches=1),
        job_context=execution_context,
        distributed_context=_single_process_context(execution_context),
        task_id="task_ab12",
        variant_id="manual_remote_run",
        show_progress=False,
    )

    variant_root = execution_context.runtime.experiments_root / "task_ab12" / "manual_remote_run"
    assert paths["variant_root"] == variant_root
    assert paths["metrics"] == variant_root / "train_metrics.jsonl"
    assert not (variant_root / "best.pt").exists()
    assert not (execution_context.runtime.experiments_root / "velocity_fm_sc_mean").exists()


def test_train_variant_rejects_existing_variant_root_without_resume(tmp_path, monkeypatch, execution_context):
    dataset = TensorDataset(torch.zeros(2, 1, 28, 28), torch.tensor([0, 1]))
    _patch_datasets(monkeypatch, dataset)
    config = _tiny_config(tmp_path)

    train_module.train_variant(
        "velocity_fm",
        replace(config, limit_train_batches=1),
        job_context=execution_context,
        distributed_context=_single_process_context(execution_context),
        task_id="task_ab12",
        variant_id="velocity_fm",
        show_progress=False,
    )

    with pytest.raises(ValueError, match="already exists"):
        train_module.train_variant(
            "velocity_fm",
            replace(config, limit_train_batches=1),
            job_context=execution_context,
            distributed_context=_single_process_context(execution_context),
            task_id="task_ab12",
            variant_id="velocity_fm",
            show_progress=False,
        )


def test_train_variant_rejects_nonempty_variant_root_without_markers(tmp_path, monkeypatch, execution_context):
    monkeypatch.setattr(train_module, "_validate_training_contexts", lambda *args: None)
    config = _tiny_config(tmp_path)
    stale_root = execution_context.runtime.experiments_root / "task_ab12" / "velocity_fm"
    (stale_root / "epochs" / "epoch_0001").mkdir(parents=True)
    (stale_root / "epochs" / "epoch_0001" / "orphan.txt").write_text("stale\n", encoding="utf-8")

    with pytest.raises(ValueError, match="variant root already exists"):
        train_module.train_variant(
            "velocity_fm",
            replace(config, limit_train_batches=1),
            job_context=execution_context,
            distributed_context=_single_process_context(execution_context),
            task_id="task_ab12",
            variant_id="velocity_fm",
            show_progress=False,
        )


def test_train_variant_scales_training_noise_by_config_noise_scale(tmp_path, monkeypatch, execution_context):
    dataset = TensorDataset(torch.zeros(2, 1, 28, 28), torch.tensor([0, 1]))

    def fake_timesteps(config, batch_size, device):
        del config
        return torch.zeros(batch_size, 1, 1, 1, device=device)

    def fake_randn_like(x):
        return torch.ones_like(x)

    class RecordingVelocityModel(torch.nn.Module):
        def __init__(self) -> None:
            super().__init__()
            self.weight = torch.nn.Parameter(torch.tensor(0.0))
            self.seen_xt = None

        def forward(self, xt, t, labels):
            del t, labels
            self.seen_xt = xt.detach().clone()
            return {"velocity": self.weight * torch.ones_like(xt)}

    model = RecordingVelocityModel()

    def fake_build_model(variant, config, device):
        del variant, config
        return model.to(device)

    _patch_datasets(monkeypatch, dataset)
    monkeypatch.setattr(train_module, "sample_train_timesteps", fake_timesteps)
    monkeypatch.setattr(train_module.torch, "randn_like", fake_randn_like)
    monkeypatch.setattr(train_module, "build_model", fake_build_model)

    config = _tiny_config(
        tmp_path,
        image_size=28,
        unet_channel_mult=(1,),
        noise_scale=3.0,
        ema_decay=0.5,
    )

    train_module.train_variant(
        "velocity_fm",
        replace(config, limit_train_batches=1),
        job_context=execution_context,
        distributed_context=_single_process_context(execution_context),
        task_id="task_ab12",
        variant_id="velocity_fm",
        show_progress=False,
    )

    assert model.seen_xt is not None
    assert torch.allclose(model.seen_xt, torch.full_like(model.seen_xt, 3.0))
