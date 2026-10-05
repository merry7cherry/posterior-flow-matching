import hashlib
import json
from dataclasses import replace
from pathlib import Path

import pytest
import torch
from torch.utils.data import TensorDataset

from pfm.training import runner as train_module
from pfm.core.config import ExperimentConfig
from pfm.core.distributed import DistributedContext
from pfm.core.layout import training_run_layout
from pfm.core.rng_state import capture_rng_state
from pfm.evaluation.checkpoint_loading import (
    load_bound_training_checkpoint,
    resolve_bound_epoch_checkpoint,
)
from pfm.training.checkpointing import load_canonical_resume_checkpoint


def _tiny_config(output_root, epochs, *, num_workers=0, limit_train_batches=None):
    del output_root
    return ExperimentConfig(
        batch_size=2,
        epochs=epochs,
        num_workers=num_workers,
        limit_train_batches=limit_train_batches,
        time_proj_dim=16,
        class_proj_dim=16,
        unet_base_channels=4,
        unet_channel_mult=(1, 2),
        unet_num_res_blocks=1,
        unet_attention_resolutions=(),
        unet_norm_groups=4,
        checkpoint_every_epochs=1,
        ema_decay=0.9,
        seed=7,
    )


def _patch_loader(monkeypatch, *, scientific_identity_sha256: str = "d" * 64):
    images = torch.arange(8 * 28 * 28, dtype=torch.float32).view(8, 1, 28, 28) / 1000.0
    labels = torch.arange(8) % 10
    dataset = TensorDataset(images, labels)

    monkeypatch.setattr(train_module, "_validate_training_contexts", lambda *args: None)
    monkeypatch.setattr(
        train_module,
        "validate_dataset_manifest",
        lambda data_root, dataset_name: {
            "scientific_identity": {"dataset": dataset_name},
            "scientific_identity_sha256": scientific_identity_sha256,
        },
    )
    monkeypatch.setattr(train_module, "make_datasets", lambda config, data_root: (dataset, dataset))


def _distributed_context(job_context):
    return DistributedContext(
        enabled=False,
        rank=job_context.rank,
        local_rank=job_context.local_rank,
        world_size=job_context.world_size,
        device=torch.device("cpu"),
    )


def _train(job_context, variant, config, **kwargs):
    return train_module.train_variant(
        variant,
        config,
        job_context=job_context,
        distributed_context=_distributed_context(job_context),
        **kwargs,
    )


def _model_state(path):
    checkpoint = torch.load(path, map_location="cpu")
    return checkpoint["model_state"], checkpoint


def _manifest_reference(checkpoint_path: Path) -> Path:
    return checkpoint_path.with_name("checkpoint_manifest.json")


def _refresh_manifest_checkpoint_sha256(checkpoint_path: Path) -> None:
    manifest_path = _manifest_reference(checkpoint_path)
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    manifest["checkpoint_sha256"] = hashlib.sha256(checkpoint_path.read_bytes()).hexdigest()
    manifest_path.write_text(json.dumps(manifest) + "\n", encoding="utf-8")


def _write_resume_checkpoint(tmp_path, monkeypatch, execution_context):
    del tmp_path
    _patch_loader(monkeypatch)
    _train(
        execution_context,
        "velocity_fm",
        _tiny_config(None, epochs=1),
        task_id="resume_payload",
        variant_id="velocity_fm",
        show_progress=False,
    )
    return (
        execution_context.runtime.experiments_root
        / "resume_payload"
        / "velocity_fm"
        / "epochs"
        / "epoch_0001"
        / "checkpoint.pt"
    )


def test_resume_rejects_direct_checkpoint_reference(tmp_path, monkeypatch, execution_context):
    checkpoint_path = _write_resume_checkpoint(tmp_path, monkeypatch, execution_context)

    with pytest.raises(ValueError, match="checkpoint_manifest.json"):
        _train(
            execution_context,
            "velocity_fm",
            _tiny_config(None, epochs=2),
            task_id="resume_payload",
            variant_id="velocity_fm",
            resume_from=checkpoint_path,
            show_progress=False,
        )


def test_canonical_resume_resolves_output_root_alias(
    tmp_path,
    monkeypatch,
    execution_context,
):
    checkpoint_path = _write_resume_checkpoint(tmp_path, monkeypatch, execution_context)
    physical_runs_root = execution_context.runtime.runs_root
    lexical_runs_root = tmp_path / "lexical-runs"
    lexical_runs_root.symlink_to(physical_runs_root, target_is_directory=True)
    lexical_runtime = replace(execution_context.runtime, runs_root=lexical_runs_root)
    layout = training_run_layout(
        lexical_runtime,
        task_id="resume_payload",
        variant_id="velocity_fm",
    )

    resumed_path, checkpoint = load_canonical_resume_checkpoint(
        layout,
        layout.checkpoint_manifest_path(1),
    )

    assert resumed_path == layout.checkpoint_path(1)
    assert resumed_path == checkpoint_path.resolve()
    assert checkpoint["epoch"] == 1


def test_train_resume_and_consumer_resolve_the_same_output_root_alias(
    tmp_path,
    monkeypatch,
    execution_context,
):
    _patch_loader(monkeypatch)
    physical_runs_root = execution_context.runtime.runs_root
    physical_runs_root.mkdir(parents=True)
    logical_runs_root = tmp_path / "logical-runs"
    logical_runs_root.symlink_to(physical_runs_root, target_is_directory=True)
    logical_runtime = replace(execution_context.runtime, runs_root=logical_runs_root)
    logical_job_context = replace(execution_context, runtime=logical_runtime)
    layout = training_run_layout(
        logical_runtime,
        task_id="resume_payload",
        variant_id="velocity_fm",
    )
    first_outputs = _train(
        logical_job_context,
        "velocity_fm",
        _tiny_config(None, epochs=1),
        task_id="resume_payload",
        variant_id="velocity_fm",
        show_progress=False,
    )
    epoch_one_bytes = {
        path.relative_to(layout.epoch_dir(1)): path.read_bytes()
        for path in layout.epoch_dir(1).rglob("*")
        if path.is_file()
    }
    metrics_prefix = layout.train_metrics_jsonl.read_bytes()

    second_outputs = _train(
        logical_job_context,
        "velocity_fm",
        _tiny_config(None, epochs=2),
        task_id="resume_payload",
        variant_id="velocity_fm",
        resume_from=layout.checkpoint_manifest_path(1),
        show_progress=False,
    )

    assert layout.output_root == physical_runs_root.resolve() / "experiments"
    for outputs in (first_outputs, second_outputs):
        assert outputs["task_root"] == layout.task_root
        assert outputs["variant_root"] == layout.variant_root
        assert outputs["metrics"] == layout.train_metrics_jsonl
    task_manifest = json.loads(layout.task_manifest_path.read_text(encoding="utf-8"))
    assert task_manifest["output_root"] == str(layout.output_root)
    assert layout.checkpoint_path(2).resolve().is_relative_to(
        physical_runs_root.resolve()
    )
    assert {
        path.relative_to(layout.epoch_dir(1)): path.read_bytes()
        for path in layout.epoch_dir(1).rglob("*")
        if path.is_file()
    } == epoch_one_bytes
    assert layout.train_metrics_jsonl.read_bytes().startswith(metrics_prefix)

    bound = resolve_bound_epoch_checkpoint(layout.variant_root, 2)
    checkpoint = load_bound_training_checkpoint(bound)

    assert bound.checkpoint_path == layout.checkpoint_path(2)
    assert bound.manifest_path == layout.checkpoint_manifest_path(2)
    assert checkpoint["epoch"] == 2


def test_resume_rejects_external_manifest_reference(tmp_path, monkeypatch, execution_context):
    checkpoint_path = _write_resume_checkpoint(tmp_path, monkeypatch, execution_context)
    external = tmp_path / "checkpoint_manifest.json"
    external.write_bytes(checkpoint_path.with_name("checkpoint_manifest.json").read_bytes())

    with pytest.raises(ValueError, match="same canonical variant root"):
        _train(
            execution_context,
            "velocity_fm",
            _tiny_config(None, epochs=2),
            task_id="resume_payload",
            variant_id="velocity_fm",
            resume_from=external,
            show_progress=False,
        )


def test_resume_rejects_manifest_identity_mismatch(tmp_path, monkeypatch, execution_context):
    checkpoint_path = _write_resume_checkpoint(tmp_path, monkeypatch, execution_context)
    manifest_path = checkpoint_path.with_name("checkpoint_manifest.json")
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    manifest["variant_id"] = "other_variant"
    manifest_path.write_text(json.dumps(manifest) + "\n", encoding="utf-8")

    with pytest.raises(ValueError, match="variant_id"):
        _train(
            execution_context,
            "velocity_fm",
            _tiny_config(None, epochs=2),
            task_id="resume_payload",
            variant_id="velocity_fm",
            resume_from=manifest_path,
            show_progress=False,
        )


def test_resume_rejects_rewind_when_later_epoch_exists(tmp_path, monkeypatch, execution_context):
    _patch_loader(monkeypatch)
    _train(
        execution_context,
        "velocity_fm",
        _tiny_config(None, epochs=2),
        task_id="rewind",
        variant_id="velocity_fm",
        show_progress=False,
    )
    variant_root = execution_context.runtime.experiments_root / "rewind" / "velocity_fm"
    epoch_one_manifest = variant_root / "epochs" / "epoch_0001" / "checkpoint_manifest.json"
    before = {path: path.read_bytes() for path in variant_root.rglob("*") if path.is_file()}

    with pytest.raises(ValueError, match="later epoch"):
        _train(
            execution_context,
            "velocity_fm",
            _tiny_config(None, epochs=3),
            task_id="rewind",
            variant_id="velocity_fm",
            resume_from=epoch_one_manifest,
            show_progress=False,
        )

    assert {path: path.read_bytes() for path in variant_root.rglob("*") if path.is_file()} == before


def test_resume_rejects_checkpoint_missing_distributed_world_size(tmp_path, monkeypatch, execution_context):
    checkpoint_path = _write_resume_checkpoint(tmp_path, monkeypatch, execution_context)
    checkpoint = torch.load(checkpoint_path, map_location="cpu")
    checkpoint.pop("distributed_world_size")
    torch.save(checkpoint, checkpoint_path)
    _refresh_manifest_checkpoint_sha256(checkpoint_path)

    with pytest.raises(ValueError, match="distributed_world_size|missing fields"):
        _train(
            execution_context,
            "velocity_fm",
            _tiny_config(tmp_path / "resume_payload", epochs=2),
            task_id="resume_payload",
            variant_id="velocity_fm",
            resume_from=_manifest_reference(checkpoint_path),
            show_progress=False,
        )


def test_resume_rejects_legacy_rng_state(tmp_path, monkeypatch, execution_context):
    checkpoint_path = _write_resume_checkpoint(tmp_path, monkeypatch, execution_context)
    checkpoint = torch.load(checkpoint_path, map_location="cpu")
    checkpoint["rng_state"] = checkpoint["rng_state_by_rank"][0]
    checkpoint.pop("rng_state_by_rank")
    torch.save(checkpoint, checkpoint_path)
    _refresh_manifest_checkpoint_sha256(checkpoint_path)

    with pytest.raises(ValueError, match="rng_state_by_rank|unknown fields|missing fields"):
        _train(
            execution_context,
            "velocity_fm",
            _tiny_config(tmp_path / "resume_payload", epochs=2),
            task_id="resume_payload",
            variant_id="velocity_fm",
            resume_from=_manifest_reference(checkpoint_path),
            show_progress=False,
        )


def test_cpu_resume_rejects_checkpoint_with_cuda_rng_topology(
    tmp_path,
    monkeypatch,
    execution_context,
):
    checkpoint_path = _write_resume_checkpoint(tmp_path, monkeypatch, execution_context)
    checkpoint = torch.load(checkpoint_path, map_location="cpu")
    checkpoint["rng_state_by_rank"][0]["torch_cuda"] = [
        capture_rng_state()["torch_cpu"]
    ]
    torch.save(checkpoint, checkpoint_path)
    _refresh_manifest_checkpoint_sha256(checkpoint_path)

    with pytest.raises(ValueError, match="torch_cuda cannot be restored for CPU training"):
        _train(
            execution_context,
            "velocity_fm",
            _tiny_config(tmp_path / "resume_payload", epochs=2),
            task_id="resume_payload",
            variant_id="velocity_fm",
            resume_from=_manifest_reference(checkpoint_path),
            show_progress=False,
        )


@pytest.mark.parametrize(
    ("identity_field", "changed_value", "match"),
    [
        ("project_name", "other_project", "project_name"),
        ("git_commit", "b" * 40, "git_commit"),
    ],
)
def test_resume_rejects_runtime_identity_mismatch_before_restore(
    tmp_path,
    monkeypatch,
    execution_context,
    identity_field,
    changed_value,
    match,
):
    checkpoint_path = _write_resume_checkpoint(tmp_path, monkeypatch, execution_context)
    checkpoint = torch.load(checkpoint_path, map_location="cpu")
    checkpoint["runtime_identity"][identity_field] = changed_value
    torch.save(checkpoint, checkpoint_path)
    _refresh_manifest_checkpoint_sha256(checkpoint_path)
    manifest_path = _manifest_reference(checkpoint_path)
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    manifest["config_identity"]["runtime_identity"][identity_field] = changed_value
    manifest_path.write_text(json.dumps(manifest) + "\n", encoding="utf-8")

    class ModelRestoreSentinel:
        def load_state_dict(self, state_dict, strict=True):
            raise AssertionError("model restore must not run after runtime_identity mismatch")

    def fail_optimizer_restore(self, state_dict):
        raise AssertionError("optimizer restore must not run after runtime_identity mismatch")

    monkeypatch.setattr(train_module, "unwrap_model", lambda model: ModelRestoreSentinel())
    monkeypatch.setattr(torch.optim.AdamW, "load_state_dict", fail_optimizer_restore)

    with pytest.raises(ValueError, match=match):
        _train(
            execution_context,
            "velocity_fm",
            _tiny_config(tmp_path / "resume_payload", epochs=2),
            task_id="resume_payload",
            variant_id="velocity_fm",
            resume_from=manifest_path,
            show_progress=False,
        )


def test_resume_rejects_dataset_identity_mismatch_before_model_work(
    tmp_path,
    monkeypatch,
    execution_context,
):
    checkpoint_path = _write_resume_checkpoint(tmp_path, monkeypatch, execution_context)
    _patch_loader(monkeypatch, scientific_identity_sha256="e" * 64)

    def fail_build_model(*args, **kwargs):
        raise AssertionError("model construction must not run after dataset mismatch")

    monkeypatch.setattr(train_module, "build_model", fail_build_model)

    with pytest.raises(ValueError, match="dataset scientific identity mismatch"):
        _train(
            execution_context,
            "velocity_fm",
            _tiny_config(tmp_path, epochs=2),
            task_id="resume_payload",
            variant_id="velocity_fm",
            resume_from=_manifest_reference(checkpoint_path),
            show_progress=False,
        )


@pytest.mark.parametrize(("num_workers", "limit_train_batches"), [(0, None), (1, None), (0, 2), (1, 2)])
def test_resume_matches_uninterrupted_epoch_boundary_training(
    tmp_path,
    monkeypatch,
    execution_context,
    num_workers,
    limit_train_batches,
):
    _patch_loader(monkeypatch)

    experiments_root = execution_context.runtime.experiments_root
    uninterrupted = experiments_root / "uninterrupted"
    resumed = experiments_root / "resumed"

    _train(
        execution_context,
        "velocity_fm",
        _tiny_config(uninterrupted, epochs=3, num_workers=num_workers, limit_train_batches=limit_train_batches),
        task_id="uninterrupted",
        variant_id="velocity_fm",
        show_progress=False,
    )

    _train(
        execution_context,
        "velocity_fm",
        _tiny_config(resumed, epochs=1, num_workers=num_workers, limit_train_batches=limit_train_batches),
        task_id="resumed",
        variant_id="velocity_fm",
        show_progress=False,
    )
    resume_path = resumed / "velocity_fm" / "epochs" / "epoch_0001" / "checkpoint_manifest.json"
    metrics_path = resumed / "velocity_fm" / "train_metrics.jsonl"
    source_metrics_prefix = metrics_path.read_bytes()
    _train(
        execution_context,
        "velocity_fm",
        _tiny_config(resumed, epochs=3, num_workers=num_workers, limit_train_batches=limit_train_batches),
        task_id="resumed",
        variant_id="velocity_fm",
        resume_from=resume_path,
        show_progress=False,
    )

    resumed_metrics = metrics_path.read_bytes()
    assert resumed_metrics.startswith(source_metrics_prefix)
    metric_records = [
        json.loads(line)
        for line in resumed_metrics.decode("utf-8").splitlines()
    ]
    batches_per_epoch = limit_train_batches or 4
    assert [record["step"] for record in metric_records] == list(range(1, 3 * batches_per_epoch + 1))
    assert [(record["epoch"], record["batch"]) for record in metric_records] == [
        (epoch, batch)
        for epoch in range(1, 4)
        for batch in range(1, batches_per_epoch + 1)
    ]

    state_a, ckpt_a = _model_state(
        uninterrupted / "velocity_fm" / "epochs" / "epoch_0003" / "checkpoint.pt"
    )
    state_b, ckpt_b = _model_state(
        resumed / "velocity_fm" / "epochs" / "epoch_0003" / "checkpoint.pt"
    )

    assert ckpt_a["epoch"] == 3
    assert ckpt_b["epoch"] == 3
    assert ckpt_a["training_config"]["limit_train_batches"] == limit_train_batches
    assert ckpt_b["training_config"]["limit_train_batches"] == limit_train_batches
    assert ckpt_a["step"] == ckpt_b["step"]
    for key in state_a:
        assert torch.equal(state_a[key], state_b[key]), key
    ema_state_a = ckpt_a["ema_model_state"]
    ema_state_b = ckpt_b["ema_model_state"]
    assert ema_state_a.keys() == ema_state_b.keys()
    for key in ema_state_a:
        assert torch.equal(ema_state_a[key], ema_state_b[key]), key

    def assert_equal(left, right):
        if isinstance(left, torch.Tensor):
            assert torch.equal(left, right)
        elif isinstance(left, dict):
            assert left.keys() == right.keys()
            for key in left:
                assert_equal(left[key], right[key])
        elif isinstance(left, (list, tuple)):
            assert len(left) == len(right)
            for a, b in zip(left, right, strict=True):
                assert_equal(a, b)
        else:
            assert left == right

    for field in ("optimizer_state", "rng_state_by_rank"):
        assert_equal(ckpt_a[field], ckpt_b[field])


@pytest.mark.parametrize(("original_limit", "changed_limit"), [(1, None), (None, 1), (1, 2)])
def test_resume_rejects_changed_batch_limit_before_model_work(
    tmp_path, monkeypatch, execution_context, original_limit, changed_limit,
):
    _patch_loader(monkeypatch)
    config = _tiny_config(tmp_path, epochs=1, limit_train_batches=original_limit)
    _train(execution_context, "velocity_fm", config,
           task_id="batch_limit", variant_id="velocity_fm", show_progress=False)
    variant_root = execution_context.runtime.experiments_root / "batch_limit/velocity_fm"
    checkpoint_path = variant_root / "epochs/epoch_0001/checkpoint.pt"
    assert torch.load(checkpoint_path, map_location="cpu")["training_config"]["limit_train_batches"] == original_limit
    before = {path: path.read_bytes() for path in variant_root.rglob("*") if path.is_file()}
    monkeypatch.setattr(train_module, "build_model", lambda *args, **kwargs:
                        pytest.fail("batch-limit mismatch must fail before model work"))

    with pytest.raises(ValueError, match="resume config field limit_train_batches differs"):
        _train(execution_context, "velocity_fm",
               replace(config, epochs=2, limit_train_batches=changed_limit),
               task_id="batch_limit", variant_id="velocity_fm", show_progress=False,
               resume_from=_manifest_reference(checkpoint_path))

    assert {path: path.read_bytes() for path in variant_root.rglob("*") if path.is_file()} == before


def test_resume_resolves_scaled_learning_rate_before_config_validation(
    tmp_path,
    monkeypatch,
    execution_context,
):
    _patch_loader(monkeypatch)
    config = _tiny_config(tmp_path, epochs=1)
    config.lr = None
    config.base_lr = 0.0002
    config.lr_reference_batch_size = 2
    config.scale_lr_by_effective_batch = True

    _train(
        execution_context,
        "velocity_fm",
        config,
        task_id="scaled_lr_resume",
        variant_id="velocity_fm",
        show_progress=False,
    )
    manifest = (
        execution_context.runtime.experiments_root
        / "scaled_lr_resume"
        / "velocity_fm"
        / "epochs"
        / "epoch_0001"
        / "checkpoint_manifest.json"
    )
    _train(
        execution_context,
        "velocity_fm",
        replace(config, epochs=2),
        task_id="scaled_lr_resume",
        variant_id="velocity_fm",
        show_progress=False,
        resume_from=manifest,
    )

    assert manifest.parent.parent.joinpath("epoch_0002", "checkpoint.pt").is_file()


def test_resume_rejects_changed_authored_lr_scaling_even_when_resolved_lr_matches(
    tmp_path,
    monkeypatch,
    execution_context,
):
    _patch_loader(monkeypatch)
    config = _tiny_config(tmp_path, epochs=1)
    config.lr = None
    config.base_lr = 0.0002
    config.lr_reference_batch_size = 2
    config.scale_lr_by_effective_batch = True
    _train(
        execution_context,
        "velocity_fm",
        config,
        task_id="scaled_lr_identity",
        variant_id="velocity_fm",
        show_progress=False,
    )
    manifest = (
        execution_context.runtime.experiments_root
        / "scaled_lr_identity"
        / "velocity_fm"
        / "epochs"
        / "epoch_0001"
        / "checkpoint_manifest.json"
    )
    changed = replace(
        config,
        epochs=2,
        base_lr=0.0004,
        lr_reference_batch_size=4,
    )

    with pytest.raises(ValueError, match="resume config field base_lr differs"):
        _train(
            execution_context,
            "velocity_fm",
            changed,
            task_id="scaled_lr_identity",
            variant_id="velocity_fm",
            show_progress=False,
            resume_from=manifest,
        )


@pytest.mark.parametrize(
    ("field_name", "changed_value"),
    [
        ("mixed_precision", "none"),
        ("allow_tf32", True),
        ("unet_dropout", 0.25),
        ("num_workers", 1),
    ],
)
def test_resume_rejects_changed_training_dynamics(tmp_path, monkeypatch, execution_context, field_name, changed_value):
    _patch_loader(monkeypatch)

    root = execution_context.runtime.experiments_root / "strict"
    _train(
        execution_context,
        "velocity_fm",
        _tiny_config(root, epochs=1),
        task_id="strict",
        variant_id="velocity_fm",
        show_progress=False,
    )
    manifest_path = root / "velocity_fm" / "variant_manifest.json"
    original_manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    changed = _tiny_config(root, epochs=2)
    setattr(changed, field_name, changed_value)

    with pytest.raises(ValueError, match=f"resume config field {field_name} differs"):
        _train(
            execution_context,
            "velocity_fm",
            changed,
            task_id="strict",
            variant_id="velocity_fm",
            resume_from=root / "velocity_fm" / "epochs" / "epoch_0001" / "checkpoint_manifest.json",
            show_progress=False,
        )
    assert json.loads(manifest_path.read_text(encoding="utf-8")) == original_manifest


@pytest.mark.parametrize(
    ("metrics_state", "payload"),
    [
        ("missing", None),
        ("truncated", b'{"epoch": 1'),
        ("schema_changed", b'{"telemetry_version": 99, "value": "changed"}\n'),
        ("step_discontinuous", b'{"epoch": 1, "step": 999, "batch": 42}\n'),
    ],
)
def test_resume_treats_existing_metrics_as_optional_telemetry(
    tmp_path,
    monkeypatch,
    execution_context,
    metrics_state,
    payload,
):
    _patch_loader(monkeypatch)

    root = execution_context.runtime.experiments_root / metrics_state
    _train(
        execution_context,
        "velocity_fm",
        _tiny_config(root, epochs=1),
        task_id=metrics_state,
        variant_id="velocity_fm",
        show_progress=False,
    )
    variant_root = root / "velocity_fm"
    metrics_path = variant_root / "train_metrics.jsonl"
    if payload is None:
        metrics_path.unlink()
        prefix = b""
    else:
        metrics_path.write_bytes(payload)
        prefix = payload

    _train(
        execution_context,
        "velocity_fm",
        _tiny_config(root, epochs=2),
        task_id=metrics_state,
        variant_id="velocity_fm",
        resume_from=(
            variant_root
            / "epochs"
            / "epoch_0001"
            / "checkpoint_manifest.json"
        ),
        show_progress=False,
    )

    assert metrics_path.read_bytes().startswith(prefix)
    assert (variant_root / "epochs" / "epoch_0002" / "checkpoint.pt").is_file()
