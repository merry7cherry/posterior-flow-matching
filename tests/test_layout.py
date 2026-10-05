from pathlib import Path

import pytest

from pfm.core.layout import (
    epoch_id,
    parse_operator_epoch,
    training_run_layout,
    validate_path_component,
)


def test_epoch_id_is_zero_padded():
    assert epoch_id(1) == "epoch_0001"
    assert epoch_id(50) == "epoch_0050"
    assert epoch_id(1200) == "epoch_1200"


@pytest.mark.parametrize("value", [True, False, 1.0, "1", None])
def test_epoch_id_rejects_non_integer_values(value):
    with pytest.raises(TypeError, match="integer"):
        epoch_id(value)


@pytest.mark.parametrize(
    "value",
    ["", "0", "01", "+1", "-1", " 1", "1 ", "1.0", "epoch_0001", "\u0661"],
)
def test_operator_epoch_accepts_only_unpadded_positive_ascii_decimal(value):
    with pytest.raises(ValueError, match="positive ASCII decimal"):
        parse_operator_epoch(value)


def test_operator_epoch_parses_once_to_integer():
    assert parse_operator_epoch("1") == 1
    assert parse_operator_epoch("12000") == 12000


def test_training_layout_places_checkpoint_and_evaluation_under_epoch_dir(runtime_paths):
    layout = training_run_layout(
        runtime_paths,
        task_id="cifar10_edm_e800_ab12cd34",
        variant_id="pfm_full_uniform",
    )

    assert layout.task_root == runtime_paths.experiments_root / "cifar10_edm_e800_ab12cd34"
    assert layout.variant_root == layout.task_root / "pfm_full_uniform"
    assert layout.train_metrics_jsonl == layout.variant_root / "train_metrics.jsonl"
    assert layout.epoch_dir(50) == layout.variant_root / "epochs" / "epoch_0050"
    assert layout.checkpoint_path(50) == layout.variant_root / "epochs" / "epoch_0050" / "checkpoint.pt"
    assert layout.checkpoint_manifest_path(50) == layout.variant_root / "epochs" / "epoch_0050" / "checkpoint_manifest.json"
    assert layout.variant_manifest_path == layout.variant_root / "variant_manifest.json"
    assert layout.task_manifest_path == layout.task_root / "task_manifest.json"


def test_training_layout_exposes_immutable_evaluation_and_sample_artifact_dirs(runtime_paths):
    layout = training_run_layout(
        runtime_paths,
        task_id="cifar10_edm_e800_ab12cd34",
        variant_id="pfm_full_uniform",
    )

    epoch_dir = layout.variant_root / "epochs" / "epoch_0050"
    assert layout.evaluation_artifact_dir(50, "eval-seed123") == (
        epoch_dir / "evaluations" / "eval-seed123"
    )
    assert layout.sample_artifact_dir(50, "sample-seed456") == (
        epoch_dir / "samples" / "sample-seed456"
    )


@pytest.mark.parametrize("value", [True, 1.0, "1", "epoch_0001"])
def test_layout_epoch_inputs_are_internal_positive_integers(runtime_paths, value):
    layout = training_run_layout(runtime_paths, task_id="task_ab12", variant_id="pfm_full")

    with pytest.raises(TypeError, match="integer"):
        layout.checkpoint_path(value)


@pytest.mark.parametrize("bad", ["", ".", "..", "nested/id", r"nested\id", " id", "id ", 7])
def test_immutable_artifact_layout_helpers_reject_non_component_ids(runtime_paths, bad):
    layout = training_run_layout(runtime_paths, task_id="task_ab12", variant_id="pfm_full")

    with pytest.raises((TypeError, ValueError), match="component"):
        layout.evaluation_artifact_dir(1, bad)
    with pytest.raises((TypeError, ValueError), match="component"):
        layout.sample_artifact_dir(1, bad)


def test_training_layout_is_bound_to_selected_experiments_root(runtime_paths):
    layout = training_run_layout(runtime_paths, task_id="task_ab12", variant_id="pfm_delta")

    assert layout.output_root == runtime_paths.experiments_root
    assert layout.task_root == runtime_paths.experiments_root / "task_ab12"


@pytest.mark.parametrize("bad", ["", ".", "..", "a/b", "a\\b"])
def test_training_layout_rejects_non_component_ids(runtime_paths, bad):
    with pytest.raises(ValueError, match="single path component"):
        training_run_layout(runtime_paths, task_id=bad, variant_id="pfm_full")
    with pytest.raises(ValueError, match="single path component"):
        training_run_layout(runtime_paths, task_id="task_abcd", variant_id=bad)


@pytest.mark.parametrize("value", [" task", "task ", "\ttask", 7, None])
def test_path_components_reject_whitespace_and_non_strings(value):
    with pytest.raises((TypeError, ValueError), match="component"):
        validate_path_component(value, "task_id")


def test_path_component_returns_exact_original_value():
    assert validate_path_component("task_ab12", "task_id") == "task_ab12"


def test_training_layout_rejects_task_symlink_even_when_target_stays_inside_root(runtime_paths):
    experiments_root = runtime_paths.experiments_root
    real_task = experiments_root / "real_task"
    real_task.mkdir(parents=True)
    (experiments_root / "task_link").symlink_to(real_task, target_is_directory=True)

    with pytest.raises(ValueError, match="symlink"):
        training_run_layout(runtime_paths, task_id="task_link", variant_id="variant")


def test_training_layout_rejects_variant_symlink_even_when_target_stays_inside_root(runtime_paths):
    task_root = runtime_paths.experiments_root / "task"
    real_variant = task_root / "real_variant"
    real_variant.mkdir(parents=True)
    (task_root / "variant_link").symlink_to(real_variant, target_is_directory=True)

    with pytest.raises(ValueError, match="symlink"):
        training_run_layout(runtime_paths, task_id="task", variant_id="variant_link")


def test_training_layout_rejects_epoch_symlink_even_when_target_stays_inside_root(runtime_paths):
    layout = training_run_layout(runtime_paths, task_id="task", variant_id="variant")
    real_epoch = layout.variant_root / "epochs" / "epoch_0002"
    real_epoch.mkdir(parents=True)
    (layout.variant_root / "epochs" / "epoch_0001").symlink_to(
        real_epoch,
        target_is_directory=True,
    )

    with pytest.raises(ValueError, match="symlink"):
        layout.checkpoint_path(1)
