"""Operator-level equivalence, completeness, and frozen comparison behavior."""
import hashlib
from pathlib import Path

import pytest
import yaml

from pfm.cli.config_inputs import execution_config, resolved_config_bytes
from pfm.core.config import ExperimentConfig
from pfm.core.inference_config import InferenceExecutionConfig

ROOT = Path(__file__).resolve().parents[1]


@pytest.mark.parametrize("category,filename,field,value", [
    ("train", "cifar10_edm_ddpmpp.yaml", "logvar_max", "1.0"),
    ("evaluate", "evaluation/cifar10_edm_sde.yaml", "sampling.diffusion_scale", "0.1"),
    ("sample", "sampling/imagenet256_sde_visualization.yaml", "sampling.terminal_time", "0.99"),
    ("sde_acceptance", "diagnostics/imagenet256_sde_acceptance.yaml", "diffusion_scale", "0.1"),
    ("prepare_fid", "fid/cifar10_inception.yaml", "batch_size", "256"),
    ("train_classifier", "classifier/mnist.yaml", "weight_decay", "0.0001"),
])
def test_equal_settings_have_one_snapshot(tmp_path, category, filename, field, value):
    source = ROOT / "configs" / filename
    raw = yaml.safe_load(source.read_text())
    reordered = tmp_path / "formatted.yaml"
    reordered.write_text("# Formatting has no protocol meaning\n" + yaml.safe_dump(raw, sort_keys=True, indent=4))
    base = resolved_config_bytes(source, [], category=category)
    assert resolved_config_bytes(reordered, [], category=category) == base
    assert resolved_config_bytes(source, [f"{field}={value}"], category=category) == base
    reordered.write_bytes(base)
    assert resolved_config_bytes(reordered, [], category=category) == base


def test_yaml_is_complete_and_cli_only_changes_explicit_choices(tmp_path):
    source = ROOT / "configs/evaluation/cifar10_edm_sde.yaml"
    raw = yaml.safe_load(source.read_text())
    changed = yaml.safe_load(resolved_config_bytes(source, ["sampling.steps=250", "sampling.compile=false"], category="evaluate"))
    expected = {**raw, "sampling": {**raw["sampling"], "steps": 250, "compile": False}}
    assert changed == expected
    del raw["sampling"]["precision"]
    incomplete = tmp_path / "incomplete.yaml"
    incomplete.write_text(yaml.safe_dump(raw))
    with pytest.raises(ValueError, match="precision"):
        resolved_config_bytes(incomplete, [], category="evaluate")
    # A CLI value may explicitly complete a field, but Python never supplies it.
    assert yaml.safe_load(resolved_config_bytes(incomplete, ["sampling.precision=fp32"], category="evaluate"))["sampling"]["precision"] == "fp32"


def test_comparison_freezes_consumer_configs_and_does_not_follow_source(runtime_paths):
    project = runtime_paths.project_root
    project.mkdir(parents=True)
    source = project / "posterior.yaml"
    source.write_text((ROOT / "configs/evaluation/cifar10_edm_posterior.yaml").read_text())
    authored = project / "comparison.yaml"
    authored.write_text(yaml.safe_dump({
        "comparison_id": "pfm-check", "dataset": "cifar10",
        "required_methods": ["pfm_delta"],
        "required_consumers": {"pfm_delta": ["posterior_diagnostics"]},
        "consumer_protocols": {"posterior_diagnostics": {"owner": "project", "path": "posterior.yaml"}},
    }))
    comparison_path = execution_config(runtime_paths, str(authored), [], category="aggregate", name="comparison")
    spec = yaml.safe_load(comparison_path.read_text())
    reference = spec["consumer_protocols"]["posterior_diagnostics"]
    snapshot = runtime_paths.runs_root / reference["path"]
    original = snapshot.read_bytes()
    assert hashlib.sha256(original).hexdigest() == reference["sha256"]
    assert original == resolved_config_bytes(source, [], category="evaluate")
    raw = yaml.safe_load(source.read_text()); raw["seed"] += 1
    source.write_text(yaml.safe_dump(raw))
    assert execution_config(runtime_paths, str(comparison_path), [], category="aggregate", name="comparison") == comparison_path
    assert snapshot.read_bytes() == original
    snapshot.write_bytes(original + b"# modified frozen bytes\n")
    with pytest.raises(ValueError, match="digest mismatch"):
        execution_config(runtime_paths, str(comparison_path), [], category="aggregate", name="comparison")


@pytest.mark.parametrize("filename,overrides,solver,methods", [
    ("cifar10_sde.yaml", [], "sde", {"pfm_full", "pfm_delta"}),
    ("cifar10_primary.yaml",
     ["consumer_protocols.generated_common.path=configs/evaluation/cifar10_edm_ode.yaml"],
     "ode", {"velocity_fm", "endpoint_fm", "pfm_full", "pfm_delta"}),
])
def test_documented_comparisons_freeze_matched_protocols(runtime_paths, filename, overrides, solver, methods):
    dependencies = {}
    prepared = yaml.safe_load(resolved_config_bytes(
        ROOT / "configs/comparisons" / filename, overrides, category="aggregate",
        runtime=runtime_paths, project_root=ROOT, dependencies=dependencies,
    ))
    assert set(prepared["required_methods"]) == methods
    for method in methods:
        expected = ["generated_common", "posterior_diagnostics"] if method.startswith("pfm_") else ["generated_common"]
        assert prepared["required_consumers"][method] == expected
    protocols = {}
    for profile, reference in prepared["consumer_protocols"].items():
        assert reference["owner"] == "runs"
        payload = dependencies[runtime_paths.runs_root / reference["path"]]
        assert hashlib.sha256(payload).hexdigest() == reference["sha256"]
        protocols[profile] = yaml.safe_load(payload)
        assert protocols[profile]["profile"] == profile
    sampling = protocols["generated_common"]["sampling"]
    assert sampling["solver"] == solver
    assert (sampling["steps"], sampling["terminal_time"], sampling["terminal_readout"]) == (50, 0.99, "endpoint_mean")
    assert protocols["posterior_diagnostics"] == yaml.safe_load(resolved_config_bytes(
        ROOT / "configs/evaluation/cifar10_edm_posterior.yaml", [], category="evaluate",
    ))


def test_consumer_execution_does_not_inherit_training_backend_flags():
    training = ExperimentConfig(architecture="jit_transformer", allow_tf32=True, cudnn_benchmark=False)
    execution = InferenceExecutionConfig("fp32", False, False, True).apply(training)
    assert execution.mixed_precision == "none"
    assert (execution.jit_compile, execution.allow_tf32, execution.cudnn_benchmark) == (False, False, True)
    assert training.allow_tf32 is True and training.cudnn_benchmark is False
    assert execution.logvar_min == training.logvar_min
    assert execution.noise_scale == training.noise_scale


def test_direct_training_uses_the_complete_frozen_config(monkeypatch, execution_context):
    from types import SimpleNamespace
    from pfm.cli import train
    runtime = execution_context.runtime
    runtime.project_root.mkdir(parents=True)
    source = runtime.project_root / "train.yaml"
    source.write_text((ROOT / "configs/cifar10_edm_ddpmpp.yaml").read_text())
    monkeypatch.setattr(train, "context_from_args", lambda *args, **kwargs: execution_context)
    monkeypatch.setattr(train, "distributed_context_from_job", lambda *a, **kw: SimpleNamespace(is_main_process=True))
    monkeypatch.setattr(train, "cleanup_distributed", lambda: None)
    calls = []
    monkeypatch.setattr(train, "train_variant", lambda variant, config, **kw: calls.append((config, kw)) or {})
    train.main(["--config", str(source), "--variant", "pfm_delta", "--task-id", "config-contract",
                "--set", "epochs=12", "--set", "lr=0.001", "--quiet"])
    config, options = calls[0]
    assert config.epochs == 12 and config.lr == 0.001
    assert options["epochs"] is None
    expected = resolved_config_bytes(source, ["epochs=12", "lr=0.001"], category="train")
    snapshots = list((runtime.runs_root / "inputs/configs").glob("*.yaml"))
    assert len(snapshots) == 1 and snapshots[0].read_bytes() == expected
    with pytest.raises(ValueError, match="duplicate or overlapping"):
        train.main(["--config", str(source), "--variant", "pfm_delta", "--task-id", "config-contract",
                    "--set", "epochs=12", "--set", "epochs=13"])
    assert len(calls) == 1


def test_explicit_backend_flags_are_applied_and_validated(monkeypatch):
    import torch
    from pfm.core.runtime import apply_runtime_settings
    # Restore process-wide flags after the test; no GPU kernels are required.
    monkeypatch.setattr(torch.backends.cuda.matmul, "allow_tf32", torch.backends.cuda.matmul.allow_tf32)
    monkeypatch.setattr(torch.backends.cudnn, "allow_tf32", torch.backends.cudnn.allow_tf32)
    monkeypatch.setattr(torch.backends.cudnn, "benchmark", torch.backends.cudnn.benchmark)
    for enabled in (True, False):
        config = InferenceExecutionConfig("fp32", False, enabled, enabled).apply(ExperimentConfig())
        apply_runtime_settings(config, torch.device("cuda"))
        assert torch.backends.cuda.matmul.allow_tf32 is enabled
        assert torch.backends.cudnn.allow_tf32 is enabled
        assert torch.backends.cudnn.benchmark is enabled
    with pytest.raises(ValueError, match="allow_tf32"):
        InferenceExecutionConfig("bf16", False, "false", True)
