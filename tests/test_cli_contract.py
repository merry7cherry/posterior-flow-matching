from __future__ import annotations

import argparse
import inspect
from pathlib import Path
import tomllib
from types import SimpleNamespace

import pytest
import torch


PROJECT_ROOT = Path(__file__).resolve().parents[1]
EXPECTED_SCRIPTS = {
    "pfm-aggregate": "pfm.cli.aggregate:main",
    "pfm-evaluate": "pfm.cli.evaluate:main",
    "pfm-prepare-fid": "pfm.cli.prepare_fid:main",
    "pfm-prepare-data": "pfm.cli.prepare_data:main",
    "pfm-sample": "pfm.cli.sample:main",
    "pfm-train": "pfm.cli.train:main",
    "pfm-train-classifier": "pfm.cli.train_classifier:main",
}
OFFICIAL_OPERATOR_COMMANDS = {
    "pfm-aggregate",
    "pfm-evaluate",
    "pfm-prepare-fid",
    "pfm-prepare-data",
    "pfm-sample",
    "pfm-train",
    "pfm-train-classifier",
}
def _option_strings(module) -> set[str]:
    def collect(parser: argparse.ArgumentParser):
        for action in parser._actions:
            yield from action.option_strings
            if isinstance(action, argparse._SubParsersAction):
                for subparser in action.choices.values():
                    yield from collect(subparser)

    return set(collect(module.build_parser()))


def _fake_job(runtime, *, world_size: int = 1):
    return SimpleNamespace(
        runtime=runtime,
        git_commit="a" * 40,
        rank=0,
        local_rank=0,
        world_size=world_size,
        device="cuda",
    )


def test_option_strings_recurses_into_nested_subparsers():
    parser = argparse.ArgumentParser()
    commands = parser.add_subparsers()
    profile_parser = commands.add_parser("profile")
    modes = profile_parser.add_subparsers()
    training_parser = modes.add_parser("training")
    training_parser.add_argument("--device")
    module = SimpleNamespace(build_parser=lambda: parser)

    assert "--device" in _option_strings(module)


def test_pyproject_declares_complete_command_surface():
    data = tomllib.loads((PROJECT_ROOT / "pyproject.toml").read_text(encoding="utf-8"))

    assert data["project"]["scripts"] == EXPECTED_SCRIPTS


def test_installed_commands_separate_official_operators_from_utilities():
    assert OFFICIAL_OPERATOR_COMMANDS == set(EXPECTED_SCRIPTS)


def test_pyproject_declares_supported_python_and_does_not_inject_source_path():
    data = tomllib.loads((PROJECT_ROOT / "pyproject.toml").read_text(encoding="utf-8"))

    assert data["project"]["requires-python"] == ">=3.13,<3.14"
    assert "pythonpath" not in data.get("tool", {}).get("pytest", {}).get("ini_options", {})


def test_no_root_python_entrypoints_remain():
    assert list((PROJECT_ROOT / "scripts").glob("*.py")) == []


def test_every_command_exposes_argv_aware_parse_and_main():
    from pfm.cli import (
        aggregate,
        evaluate,
        prepare_fid,
        prepare_data,
        sample,
        train,
        train_classifier,
    )

    for module in (
        aggregate,
        evaluate,
        prepare_fid,
        prepare_data,
        sample,
        train,
        train_classifier,
    ):
        assert list(inspect.signature(module.parse_args).parameters) == ["argv"]
        assert list(inspect.signature(module.main).parameters) == ["argv"]
        assert inspect.signature(module.main).return_annotation in {int, "int"}


def test_every_installed_parser_disables_long_option_abbreviation():
    from pfm.cli import (
        aggregate,
        evaluate,
        prepare_fid,
        prepare_data,
        sample,
        train,
        train_classifier,
    )

    for module in (
        aggregate,
        evaluate,
        prepare_fid,
        prepare_data,
        sample,
        train,
        train_classifier,
    ):
        assert module.build_parser().allow_abbrev is False, module.__name__


def test_train_parser_requires_config_task_and_variant():
    from pfm.cli import train

    with pytest.raises(SystemExit):
        train.parse_args([])

    args = train.parse_args(
        [
            "--config",
            "configs/mnist_unet.yaml",
            "--variant",
            "pfm_full",
            "--task-id",
            "task_ab12",
            "--resume",
            "epochs/epoch_0003/checkpoint_manifest.json",
        ]
    )

    assert args.task_id == "task_ab12"
    assert not hasattr(args, "variant_id")
    assert args.resume == "epochs/epoch_0003/checkpoint_manifest.json"


def test_train_resume_help_is_manifest_only():
    from pfm.cli import train

    help_text = " ".join(train.build_parser().format_help().split())
    assert "explicit checkpoint_manifest.json" in help_text
    assert "epoch checkpoint" not in help_text


def test_train_parser_rejects_manual_variant_identity():
    from pfm.cli import train

    with pytest.raises(SystemExit):
        train.parse_args(
            [
                "--config",
                "configs/mnist_unet.yaml",
                "--variant",
                "pfm_full",
                "--task-id",
                "task_ab12",
                "--variant-id",
                "manual_variant",
            ]
        )


def test_training_variant_id_is_derived_from_method_and_rejects_retired_conditioning():
    from pfm.core.config import ExperimentConfig
    from pfm.training.cli import resolve_training_variant_id

    config = ExperimentConfig()

    assert resolve_training_variant_id("pfm_full", config) == "pfm_full"
    config.self_conditioning_mode = "mean"
    with pytest.raises(ValueError, match="self-conditioning runtime is retired"):
        resolve_training_variant_id("pfm_full", config)


def test_train_exposes_only_tracked_method_and_operational_controls():
    from pfm.cli import train

    assert _option_strings(train) == {
        "-h", "--help", "--set", "--config", "--variant", "--task-id",
        "--resume", "--quiet",
        "--device", "--data-root", "--output-dir", "--assets-root", "--cache-root",
    }


def test_train_rejects_retired_epochs_alias():
    from pfm.cli import train

    with pytest.raises(SystemExit):
        train.parse_args([
            "--config", "config.yaml", "--variant", "pfm_delta", "--task-id", "preview",
            "--epochs", "12",
        ])


def test_training_batch_limit_uses_only_the_config_override_interface():
    from pfm.cli import train

    args = train.parse_args(["--config", "config.yaml", "--variant", "pfm_delta",
                             "--set", "limit_train_batches=2"])
    assert args.set == ["limit_train_batches=2"]
    with pytest.raises(SystemExit):
        train.parse_args(["--config", "config.yaml", "--variant", "pfm_delta",
                          "--limit-train-batches", "2"])


@pytest.mark.parametrize(
    "removed",
    [
        ["--self-conditioning-mode", "mean"],
        ["--jit-attention-implementation", "sdpa"],
        ["--covariance-type", "diagonal"],
        ["--covariance-jitter", "1e-6"],
        ["--self-conditioning-sample-use-logvar"],
        ["--batch-size", "32"],
        ["--lr", "0.0003"],
        ["--mixed-precision", "bf16"],
    ],
)
def test_train_rejects_removed_covariance_and_self_conditioning_flags(removed):
    from pfm.cli import train

    argv = [
        "--config",
        "configs/mnist_unet.yaml",
        "--variant",
        "pfm_full",
        "--task-id",
        "task_ab12",
    ]
    with pytest.raises(SystemExit):
        train.parse_args([*argv, *removed])


def test_train_exposes_explicit_paths_and_device():
    from pfm.cli import train

    assert {"--device", "--data-root", "--output-dir", "--assets-root", "--cache-root"}.issubset(_option_strings(train))


def test_evaluate_parser_accepts_only_one_explicit_evaluation_identity():
    from pfm.cli import evaluate

    args = evaluate.parse_args(
        [
            "--variant-root",
            "/work/outputs/experiments/task/variant",
            "--epoch",
            "2",
            "--evaluation-id",
            "eval-cifar-final",
            "--eval-config",
            "configs/evaluation/cifar10_edm_generation.yaml",
        ]
    )

    assert args.epoch == 2
    assert args.evaluation_id == "eval-cifar-final"
    assert _option_strings(evaluate) == {
        "-h",
        "--help", "--set",
        "--variant-root",
        "--epoch",
        "--evaluation-id",
        "--eval-config", "--config", "--checkpoint",
        "--device", "--data-root", "--output-dir", "--assets-root", "--cache-root",
    }


@pytest.mark.parametrize(
    "epoch_args",
    [
        [],
        ["--epoch", "all"],
        ["--epoch", "latest"],
        ["--epochs", "1"],
        ["--epoch", "1", "2"],
        ["--epoch", "1", "--overwrite-evaluation"],
    ],
)
def test_evaluate_parser_rejects_implicit_plural_or_overwrite_epoch_surfaces(epoch_args):
    from pfm.cli import evaluate

    with pytest.raises(SystemExit):
        evaluate.parse_args(
            [
                "--variant-root",
                "/work/outputs/experiments/task/variant",
                *epoch_args,
                "--evaluation-id",
                "eval-fixed",
                "--eval-config",
                "configs/evaluation/cifar10_edm_generation.yaml",
            ]
        )


def test_evaluate_parser_rejects_repeated_epoch_options():
    from pfm.cli import evaluate

    with pytest.raises(SystemExit):
        evaluate.parse_args(
            [
                "--variant-root",
                "/work/outputs/experiments/task/variant",
                "--epoch",
                "1",
                "--epoch",
                "2",
                "--evaluation-id",
                "eval-fixed",
                "--eval-config",
                "configs/evaluation/cifar10_edm_generation.yaml",
            ]
        )


@pytest.mark.parametrize(
    "epoch",
    ["0", "01", "+1", "-1", " 1", "1 ", "1.0", "epoch_0001", "\u0661"],
)
@pytest.mark.parametrize("module_name", ["evaluate", "sample"])
def test_consumer_parsers_reject_noncanonical_operator_epochs(module_name, epoch):
    from pfm.cli import evaluate, sample

    if module_name == "evaluate":
        argv = [
            "--variant-root",
            "/work/outputs/experiments/task/variant",
            "--epoch",
            epoch,
            "--evaluation-id",
            "eval-fixed",
            "--eval-config",
            "configs/evaluation/cifar10_edm_generation.yaml",
        ]
        module = evaluate
    else:
        argv = [
            "--variant-root",
            "/work/outputs/experiments/task/variant",
            "--epoch",
            epoch,
            "--sample-id",
            "sample-fixed",
            "--sample-config",
            "configs/sampling/cifar10_visualization.yaml",
        ]
        module = sample

    with pytest.raises(SystemExit):
        module.parse_args(argv)


def test_sample_parser_parses_operator_epoch_once_to_integer():
    from pfm.cli import sample

    args = sample.parse_args(
        [
            "--variant-root",
            "/work/outputs/experiments/task/variant",
            "--epoch",
            "2",
            "--sample-id",
            "sample-fixed",
            "--sample-config",
            "configs/sampling/cifar10_visualization.yaml",
        ]
    )

    assert args.epoch == 2


def test_classifier_cli_accepts_only_config_and_explicit_asset_id():
    from pfm.cli import train_classifier

    with pytest.raises(SystemExit):
        train_classifier.parse_args([])

    with pytest.raises(SystemExit):
        train_classifier.parse_args(["--config", "configs/classifier/mnist.yaml"])
    with pytest.raises(SystemExit):
        train_classifier.parse_args(["--classifier-asset-id", "mnist-feature-seed123"])

    args = train_classifier.parse_args(
        [
            "--config",
            "configs/classifier/mnist.yaml",
            "--classifier-asset-id",
            "mnist-feature-seed123",
        ]
    )

    assert args.config == "configs/classifier/mnist.yaml"
    assert args.classifier_asset_id == "mnist-feature-seed123"
    assert _option_strings(train_classifier) == {
        "-h",
        "--help", "--set",
        "--config",
        "--classifier-asset-id",
        "--device", "--data-root", "--output-dir", "--assets-root", "--cache-root",
    }


def test_prepare_fid_cli_accepts_only_config_and_explicit_asset_id():
    from pfm.cli import prepare_fid

    with pytest.raises(SystemExit):
        prepare_fid.parse_args([])
    with pytest.raises(SystemExit):
        prepare_fid.parse_args(["--config", "configs/fid/cifar10_inception.yaml"])
    with pytest.raises(SystemExit):
        prepare_fid.parse_args(["--fid-asset-id", "cifar10-inception-train-pfm-v4"])

    args = prepare_fid.parse_args(
        [
            "--config",
            "configs/fid/cifar10_inception.yaml",
            "--fid-asset-id",
            "cifar10-inception-train-pfm-v4",
        ]
    )

    assert args.config == "configs/fid/cifar10_inception.yaml"
    assert args.fid_asset_id == "cifar10-inception-train-pfm-v4"
    assert _option_strings(prepare_fid) == {
        "-h",
        "--help", "--set",
        "--config",
        "--fid-asset-id", "--download-weights",
        "--device", "--data-root", "--output-dir", "--assets-root", "--cache-root",
    }


def test_train_main_forwards_tracked_method_and_operational_controls(monkeypatch):
    from pfm.cli import train

    runtime = SimpleNamespace(config_file=lambda value, **kwargs: Path("/project") / value)
    job = _fake_job(runtime)
    distributed = SimpleNamespace(world_size=1, is_main_process=True)
    config = SimpleNamespace(self_conditioning_mode="none")
    load_calls = []
    training_calls = []
    cleanup_calls = []
    monkeypatch.setattr(train, "context_from_args", lambda *args, **kwargs: job)
    monkeypatch.setattr(train, "execution_config", lambda runtime, value, items, **kwargs: runtime.config_file(value, name=kwargs["name"]))
    monkeypatch.setattr(
        train,
        "load_config",
        lambda *args, **kwargs: load_calls.append((args, kwargs)) or config,
    )
    monkeypatch.setattr(train, "distributed_context_from_job", lambda *args, **kwargs: distributed)
    monkeypatch.setattr(
        train,
        "train_variant",
        lambda *args, **kwargs: training_calls.append((args, kwargs)) or {},
    )
    monkeypatch.setattr(train, "cleanup_distributed", lambda: cleanup_calls.append(True))

    result = train.main(
        [
            "--config",
            "config.yaml",
            "--variant",
            "pfm_full",
            "--task-id",
            "task_ab12",
            "--set",
            "epochs=12",
            "--set",
            "limit_train_batches=3",
            "--resume",
            "epochs/epoch_0003/checkpoint_manifest.json",
            "--quiet",
        ]
    )

    assert result == 0
    assert load_calls[0][0] == (Path("/project/config.yaml"),)
    assert load_calls[0][1] == {}
    assert training_calls[0][0] == ("pfm_full", config)
    assert training_calls[0][1]["job_context"] is job
    assert training_calls[0][1]["distributed_context"] is distributed
    assert training_calls[0][1]["task_id"] == "task_ab12"
    assert training_calls[0][1]["variant_id"] == "pfm_full"
    assert training_calls[0][1]["epochs"] is None
    assert "limit_train_batches" not in training_calls[0][1]
    assert training_calls[0][1]["resume_from"] == (
        "epochs/epoch_0003/checkpoint_manifest.json"
    )
    assert training_calls[0][1]["show_progress"] is False
    assert cleanup_calls == [True]


def test_evaluate_main_loads_project_config_and_forwards_execution_context(monkeypatch):
    from pfm.cli import evaluate

    runtime = SimpleNamespace(config_file=lambda value, **kwargs: Path("/project") / value)
    job = _fake_job(runtime)
    base_config = SimpleNamespace(profile="generated_common")
    calls = []
    monkeypatch.setattr(evaluate, "context_from_args", lambda *args, **kwargs: job)
    monkeypatch.setattr(evaluate, "execution_config", lambda runtime, value, items, **kwargs: runtime.config_file(value, name=kwargs["name"]))
    monkeypatch.setattr(evaluate, "execution_device", lambda context: torch.device("cuda", 0))
    monkeypatch.setattr(evaluate, "load_metric_evaluation_config", lambda path: base_config)
    monkeypatch.setattr(
        evaluate,
        "evaluate_bound_checkpoint",
        lambda *args, **kwargs: calls.append((args, kwargs)) or {},
    )

    result = evaluate.main(
        [
            "--variant-root",
            "/work/outputs/experiments/task/variant",
            "--epoch",
            "1",
            "--evaluation-id",
            "eval-fixed",
            "--eval-config",
            "eval.yaml",
        ]
    )

    assert result == 0
    assert calls[0][0][0:3] == (
        "/work/outputs/experiments/task/variant",
        1,
        "eval-fixed",
    )
    assert calls[0][0][3] is base_config
    assert calls[0][1] == {
        "evaluation_config_path": Path("/project/eval.yaml"),
        "job_context": job,
        "device": torch.device("cuda", 0),
    }


def test_aggregate_main_has_exact_task_id_surface_and_derives_task_root(monkeypatch):
    from pfm.cli import aggregate

    option_strings = {
        option
        for action in aggregate.build_parser()._actions
        for option in action.option_strings
    }
    assert option_strings == {
        "-h",
        "--help", "--set",
        "--task-id",
        "--comparison-config",
        "--epoch",
        "--aggregate-id",
        "--input-manifest",
        "--data-root", "--output-dir", "--assets-root", "--cache-root",
    }


def test_aggregate_main_forwards_executor_revision(monkeypatch, tmp_path):
    from pfm.cli import aggregate

    project_root = tmp_path / "project"
    comparison_path = project_root / "configs" / "comparisons" / "primary.yaml"
    comparison_path.parent.mkdir(parents=True)
    comparison_path.write_text("comparison_id: primary\n", encoding="utf-8")
    experiments_root = tmp_path / "runs" / "experiments"
    input_manifest = (
        experiments_root
        / "task_ab12"
        / "velocity_fm"
        / "consumer_manifest.json"
    )
    input_manifest.parent.mkdir(parents=True)
    input_manifest.write_text("{}\n", encoding="utf-8")
    runtime = SimpleNamespace(
        experiments_root=experiments_root,
        config_file=lambda value, *, name: comparison_path,
    )
    job = _fake_job(runtime)
    comparison = {"comparison_id": "primary"}
    calls = []
    published = experiments_root / "task_ab12" / "aggregates" / "primary-v2"
    monkeypatch.setattr(aggregate, "context_from_args", lambda *args, **kwargs: job)
    monkeypatch.setattr(aggregate, "execution_config", lambda runtime, value, items, **kwargs: runtime.config_file(value, name=kwargs["name"]))
    monkeypatch.setattr(aggregate, "load_yaml_mapping", lambda path: comparison)
    monkeypatch.setattr(aggregate, "validate_comparison_spec", lambda value: comparison)
    monkeypatch.setattr(
        aggregate,
        "publish_aggregate_result",
        lambda **kwargs: calls.append(kwargs) or published,
    )

    assert aggregate.main(
        [
            "--task-id",
            "task_ab12",
            "--comparison-config",
            "configs/comparisons/primary.yaml",
            "--epoch",
            "1",
            "--aggregate-id",
            "primary-v2",
            "--input-manifest",
            "velocity_fm/consumer_manifest.json",
        ]
    ) == 0
    assert calls[0]["aggregate_revision"] == job.git_commit
    assert calls[0]["requested_epoch"] == 1


@pytest.mark.parametrize(
    "forbidden_args",
    [
        ["--allow-empty"],
        ["--variant-root", "/work/variant"],
        ["--root", "/work/task"],
        ["--overwrite"],
        ["--latest"],
        ["--all"],
        ["--lock"],
    ],
)
def test_aggregate_rejects_noncanonical_selection_and_mutation_options(forbidden_args):
    from pfm.cli import aggregate

    with pytest.raises(SystemExit) as error:
        aggregate.parse_args(["--task-id", "task_ab12", *forbidden_args])

    assert error.value.code == 2


def test_prepare_fid_main_loads_strict_config_and_forwards_explicit_asset_id(monkeypatch):
    from pfm.cli import prepare_fid

    config_path = Path("/project/configs/fid/cifar10_inception.yaml")
    runtime = SimpleNamespace(config_file=lambda value, *, name: config_path)
    job = _fake_job(runtime)
    config = SimpleNamespace(dataset="cifar10", image_size=32, backend="inception")
    calls = []
    monkeypatch.setattr(prepare_fid, "context_from_args", lambda *args, **kwargs: job)
    monkeypatch.setattr(prepare_fid, "execution_config", lambda runtime, value, items, **kwargs: runtime.config_file(value, name=kwargs["name"]))
    monkeypatch.setattr(prepare_fid, "execution_device", lambda context: torch.device("cuda", 0))
    monkeypatch.setattr(
        prepare_fid,
        "load_fid_config",
        lambda path: calls.append(("load", path)) or config,
    )
    monkeypatch.setattr(
        prepare_fid,
        "prepare_fid_asset",
        lambda *args, **kwargs: calls.append(("prepare", args, kwargs)) or Path("fid-asset"),
    )

    assert prepare_fid.main(
        [
            "--config",
            "configs/fid/cifar10_inception.yaml",
            "--fid-asset-id",
            "cifar10-inception-train-pfm-v4",
        ]
    ) == 0
    assert calls[0] == ("load", config_path)
    assert calls[1][1] == (config,)
    assert calls[1][2]["fid_asset_id"] == "cifar10-inception-train-pfm-v4"
    assert calls[1][2]["job_context"] is job
    assert calls[1][2]["device"] == torch.device("cuda", 0)


def test_train_classifier_main_loads_strict_config_and_forwards_explicit_asset_id(monkeypatch):
    from pfm.cli import train_classifier
    from pfm.models.classifier import ClassifierConfig

    config_path = Path("/project/configs/classifier/mnist.yaml")
    runtime = SimpleNamespace(
        config_file=lambda value, *, name: config_path,
    )
    job = _fake_job(runtime)
    config = ClassifierConfig(dataset="mnist", num_classes=10, image_channels=1, image_size=28)
    calls = []
    monkeypatch.setattr(train_classifier, "context_from_args", lambda *args, **kwargs: job)
    monkeypatch.setattr(train_classifier, "execution_config", lambda runtime, value, items, **kwargs: runtime.config_file(value, name=kwargs["name"]))
    monkeypatch.setattr(
        train_classifier,
        "execution_device",
        lambda context: torch.device("cuda", 0),
    )
    monkeypatch.setattr(
        train_classifier,
        "load_classifier_config",
        lambda path: calls.append(("load", path)) or config,
    )
    monkeypatch.setattr(
        train_classifier,
        "train_classifier",
        lambda *args, **kwargs: calls.append(("train", args, kwargs)) or {},
    )

    assert train_classifier.main(
        [
            "--config",
            "configs/classifier/mnist.yaml",
            "--classifier-asset-id",
            "mnist-feature-seed123",
        ]
    ) == 0
    assert calls[0] == ("load", config_path)
    assert calls[1][1] == (config,)
    assert calls[1][2]["classifier_asset_id"] == "mnist-feature-seed123"
    assert calls[1][2]["job_context"] is job
    assert calls[1][2]["device"] == torch.device("cuda", 0)
