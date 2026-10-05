"""Portable user paths, immutable config inputs, and explicit data preparation."""
from __future__ import annotations

import json
from pathlib import Path
import shutil
import subprocess
from types import SimpleNamespace

import pytest
import yaml

from pfm.cli.common import context_from_args
from pfm.cli.config_inputs import execution_config
from pfm.core import execution
from pfm.core.execution import ExecutionContext, execution_context_from_env, runtime_identity, runtime_paths
from pfm.data import preparation
from pfm.data.identity import validate_dataset_manifest_payload


ROOT = Path(__file__).resolve().parents[1]


@pytest.fixture(autouse=True)
def restore_torch_hub_cache():
    import torch
    original, setter = torch.hub.get_dir(), torch.hub.set_dir
    yield
    setter(original)


def test_source_revision_distinguishes_clean_dirty_and_installed_code(tmp_path, monkeypatch):
    git = shutil.which("git")
    if git is None:
        pytest.skip("Git is optional; provenance is unknown without it")
    source_root = tmp_path / "source"
    source_file = source_root / "src/pfm/core/execution.py"
    source_file.parent.mkdir(parents=True)
    source_file.write_text("# test-owned source checkout\n")
    monkeypatch.setattr(execution, "__file__", str(source_file))
    assert execution.source_git_commit() is None
    subprocess.run([git, "init", "--quiet", str(source_root)], check=True, capture_output=True)
    subprocess.run([git, "-C", str(source_root), "add", "src"], check=True, capture_output=True)
    subprocess.run([
        git, "-C", str(source_root), "-c", "user.name=PFM test", "-c", "user.email=test@example.invalid",
        "-c", "commit.gpgsign=false", "-c", "core.hooksPath=/dev/null", "commit", "--quiet", "-m", "fixture",
    ], check=True, capture_output=True)
    revision = execution.source_git_commit()
    assert revision is not None and len(revision) == 40
    for output_name in ("custom-runs", "samples-ode", "evaluation-ode"):
        output = source_root / output_name
        output.mkdir()
        (output / "result.json").write_text("{}\n")
    assert execution.source_git_commit() == revision
    source_file.write_text("# modified source\n")
    assert execution.source_git_commit() is None
    source_file.write_text("# test-owned source checkout\n")
    assert execution.source_git_commit() == revision
    (source_root / "src/new_module.py").write_text("# untracked implementation\n")
    assert execution.source_git_commit() is None


def test_execution_requires_no_scheduler_environment(tmp_path, monkeypatch):
    monkeypatch.setattr(execution, "source_git_commit", lambda: None)
    context = execution_context_from_env({}, project_root=tmp_path,
                                         data_root="datasets", output_dir="runs", device="cpu")
    assert context.runtime.dataset_root("mnist") == tmp_path / "datasets/mnist"
    assert context.runtime.experiments_root == tmp_path / "runs/experiments"
    assert (context.rank, context.local_rank, context.world_size, context.device) == (0, 0, 1, "cpu")
    assert runtime_identity(context) == {"project_name": "pfm", "git_commit": None, "rank": 0, "world_size": 1}


def test_torchrun_identity_is_explicit_and_complete(tmp_path):
    context = execution_context_from_env(
        {"RANK": "1", "LOCAL_RANK": "1", "WORLD_SIZE": "2"}, project_root=tmp_path)
    assert (context.rank, context.local_rank, context.world_size) == (1, 1, 2)
    with pytest.raises(ValueError, match="provided together"):
        execution_context_from_env({"RANK": "0"}, project_root=tmp_path)
    with pytest.raises(ValueError, match="less than WORLD_SIZE"):
        execution_context_from_env({"RANK": "2", "LOCAL_RANK": "0", "WORLD_SIZE": "2"},
                                   project_root=tmp_path)


def test_consumers_reject_distributed_launch(monkeypatch, tmp_path):
    from pfm.cli import common
    monkeypatch.setattr(common, "execution_context_from_env", lambda **kwargs:
                        ExecutionContext(runtime_paths(project_root=tmp_path), None, 0, 0, 2))
    args = SimpleNamespace(data_root="data", output_dir="out", assets_root="assets", cache_root="cache")
    with pytest.raises(ValueError, match="single-process"):
        context_from_args(args)
    assert context_from_args(args, distributed=True).world_size == 2


def test_external_config_is_frozen_before_source_can_change(tmp_path):
    project = tmp_path / "project"
    project.mkdir()
    source = tmp_path / "custom-training.yaml"
    source.write_text((ROOT / "configs/mnist_unet.yaml").read_text())
    paths = runtime_paths(project_root=project, output_dir=tmp_path / "results")
    frozen = execution_config(paths, str(source), ["epochs=2", "limit_train_batches=3"], name="training", category="train")
    before = frozen.read_bytes()
    assert frozen.is_relative_to(tmp_path / "results/inputs/configs")
    assert yaml.safe_load(before)["epochs"] == 2
    assert yaml.safe_load(before)["limit_train_batches"] == 3
    source.write_text("not: a complete config\n")
    assert frozen.read_bytes() == before
    assert execution_config(paths, str(frozen), [], name="training", category="train") == frozen


@pytest.mark.parametrize("download", [False, True])
def test_prepare_data_download_is_explicit_and_manifest_is_valid(tmp_path, monkeypatch, download):
    calls = []
    class TinyDataset:
        classes = ["zero", "one"]
        targets = [0, 1]
        def __len__(self):
            return 2
    def constructor(**kwargs):
        calls.append(kwargs)
        return TinyDataset()
    monkeypatch.setattr(preparation.datasets, "MNIST", constructor)
    monkeypatch.setitem(preparation.OFFICIAL_SPLIT_COUNTS, "mnist", {"train": 2, "test": 2})
    monkeypatch.setitem(preparation.OFFICIAL_CLASS_COUNTS, "mnist", 2)
    path = preparation.prepare_dataset("mnist", tmp_path / "mnist", download=download)
    payload = json.loads(path.read_text())
    validate_dataset_manifest_payload(payload, expected_dataset="mnist")
    assert [call["download"] for call in calls] == [download, download]
    assert [call["train"] for call in calls] == [True, False]
    assert preparation.prepare_dataset("mnist", tmp_path / "mnist") == path
    before = path.read_bytes()
    TinyDataset.targets = [1, 0]
    with pytest.raises(ValueError, match="identity differs"):
        preparation.prepare_dataset("mnist", tmp_path / "mnist")
    assert path.read_bytes() == before


def test_prepare_imagenet_never_downloads_or_accepts_missing_splits(tmp_path):
    with pytest.raises(ValueError, match="user-provided"):
        preparation.prepare_dataset("imagenet", tmp_path, download=True)
    with pytest.raises(FileNotFoundError, match="train/ and val/"):
        preparation.prepare_dataset("imagenet", tmp_path)


def test_prepare_data_cli_uses_same_dataset_directory_as_training(tmp_path, monkeypatch):
    from pfm.cli import prepare_data
    calls = []
    monkeypatch.setattr(prepare_data, "prepare_dataset", lambda *args, **kwargs: calls.append((args, kwargs)))
    assert prepare_data.main(["--dataset", "cifar10", "--data-root", str(tmp_path), "--download"]) == 0
    assert calls == [(("cifar10", tmp_path / "cifar10"), {"download": True})]


@pytest.mark.parametrize("module_name", ["sample", "evaluate"])
def test_portable_consumer_selects_one_explicit_checkpoint(module_name):
    from importlib import import_module
    module = import_module(f"pfm.cli.{module_name}")
    args = module.parse_args(["--checkpoint", "model.pt", "--config", "protocol.yaml"])
    assert args.checkpoint == "model.pt" and args.variant_root is None
    for extra in (["--epoch", "1"], ["--variant-root", "experiment"]):
        with pytest.raises(SystemExit):
            module.parse_args(["--checkpoint", "model.pt", "--config", "protocol.yaml", *extra])


@pytest.mark.parametrize("module_name,config_path,function_name", [
    ("sample", "sampling/mnist_visualization.yaml", "sample_checkpoint_file"),
    ("evaluate", "evaluation/mnist_generation.yaml", "evaluate_checkpoint_file"),
])
def test_portable_consumer_freezes_config_without_precreating_output(
    tmp_path, monkeypatch, module_name, config_path, function_name,
):
    from importlib import import_module
    import torch
    module = import_module(f"pfm.cli.{module_name}")
    service = import_module("pfm.evaluation.sample_grid" if module_name == "sample"
                            else "pfm.evaluation.checkpoint_evaluation")
    target = tmp_path / "independent-output"
    context = ExecutionContext(runtime_paths(project_root=tmp_path, output_dir=target), None,
                               device="cpu")
    monkeypatch.setattr(module, "context_from_args", lambda args: context)
    monkeypatch.setattr(module, "execution_device", lambda context: torch.device("cpu"))
    calls = []
    def consume(path, config, **kwargs):
        assert not target.exists()
        calls.append((path, config, kwargs))
        return target if module_name == "sample" else {"output": str(target)}
    monkeypatch.setattr(service, function_name, consume, raising=False)
    assert module.main(["--checkpoint", "model.pt", "--config", str(ROOT / "configs" / config_path),
                        "--set", "seed=17", "--set", "sampling.precision=fp32",
                        "--output-dir", str(target), "--device", "cpu"]) == 0
    path, config, options = calls[0]
    assert path == "model.pt" and config.seed == 17
    assert options["output_dir"] == target and options["device"] == torch.device("cpu")
    assert not target.exists()


@pytest.mark.parametrize("module_name,config_path,selection,override", [
    ("train", "mnist_unet.yaml", ["--variant", "pfm_delta"], "mixed_precision=none"),
    ("sample", "sampling/mnist_visualization.yaml", ["--checkpoint", "missing.pt"], "sampling.precision=fp32"),
    ("evaluate", "evaluation/mnist_generation.yaml", ["--checkpoint", "missing.pt"], "sampling.precision=fp32"),
    ("evaluate", "evaluation/mnist_posterior.yaml", ["--checkpoint", "missing.pt"], "precision=fp32"),
    ("train_classifier", "classifier/mnist.yaml", ["--classifier-asset-id", "test-classifier"], "mixed_precision=none"),
])
def test_cpu_cli_rejects_misreported_bf16_before_model_work(
    tmp_path, monkeypatch, module_name, config_path, selection, override,
):
    from importlib import import_module
    module = import_module(f"pfm.cli.{module_name}")
    context = ExecutionContext(runtime_paths(project_root=tmp_path), None, device="cpu")
    monkeypatch.setattr(module, "context_from_args", lambda *args, **kwargs: context)
    with pytest.raises(ValueError, match=f"--set {override}"):
        module.main(["--config", str(ROOT / "configs" / config_path), *selection, "--device", "cpu"])


def test_all_contexts_select_the_requested_torch_weight_cache(tmp_path, monkeypatch):
    import torch
    from pfm.cli import common
    context = ExecutionContext(runtime_paths(project_root=tmp_path, cache_root="model-cache"), None)
    monkeypatch.setattr(common, "execution_context_from_env", lambda **kwargs: context)
    calls = []
    monkeypatch.setattr(torch.hub, "set_dir", calls.append)
    args = SimpleNamespace(data_root="data", output_dir="out", assets_root="assets", cache_root="model-cache")
    assert common.context_from_args(args) == context
    assert calls == [str(tmp_path / "model-cache/torch")]


def test_fid_weight_download_requires_the_explicit_cli_flag(tmp_path, monkeypatch):
    import torch
    from pfm.cli import prepare_fid
    from pfm.evaluation import fid_metrics
    calls = []
    monkeypatch.setattr(torch.hub, "set_dir", lambda value: calls.append("cache"))
    monkeypatch.setattr(fid_metrics, "prepare_inception_weights", lambda: calls.append("download"))
    monkeypatch.setattr(prepare_fid, "prepare_fid_asset", lambda *args, **kwargs: calls.append("statistics"))
    arguments = ["--config", str(ROOT / "configs/fid/cifar10_inception.yaml"),
                 "--fid-asset-id", "test-reference", "--output-dir", str(tmp_path / "outputs"),
                 "--cache-root", str(tmp_path / "cache"), "--device", "cpu"]
    assert prepare_fid.main(arguments) == 0
    assert calls == ["cache", "statistics"]
    calls.clear()
    assert prepare_fid.main([*arguments, "--download-weights"]) == 0
    assert calls == ["cache", "download", "statistics"]
