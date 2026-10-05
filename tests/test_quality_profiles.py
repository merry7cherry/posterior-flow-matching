from __future__ import annotations

import inspect
import hashlib
from pathlib import Path

import pytest
import torch

from pfm.core.sampling_config import SamplingConfig
import yaml

from pfm.core.evaluation_config import (
    load_metric_evaluation_config,
    metric_evaluation_config_from_mapping,
)
from pfm.evaluation import fid_metrics, generation_evaluation
from pfm.evaluation.evaluation_summary import validate_comparison_spec
from pfm.evaluation.assets import (
    canonical_fid_preprocessing_identity,
    load_fid_config,
    validate_fid_config_asset_binding,
)
from pfm.sampling.samplers import sample


PROJECT_ROOT = Path(__file__).resolve().parents[1]


@pytest.mark.parametrize(
    ("filename", "metric_names", "population", "steps"),
    [
        ("mnist_generation.yaml", ("conditional_accuracy",), 10_000, 50),
        (
            "cifar10_edm_generation.yaml",
            ("fid",),
            50_000,
            50,
        ),
        (
            "imagenet256_jit_generation.yaml",
            ("fid", "inception_score"),
            50_000,
            250,
        ),
    ],
)
def test_official_generated_common_profiles_are_metric_only_and_canonical(
    filename: str,
    metric_names: tuple[str, ...],
    population: int,
    steps: int,
) -> None:
    path = PROJECT_ROOT / "configs" / "evaluation" / filename
    raw = yaml.safe_load(path.read_text(encoding="utf-8"))

    assert raw["profile"] == "generated_common"
    assert not any(key.startswith("posterior_") for key in raw)
    assert not any(key.startswith("keep_") for key in raw)

    config = load_metric_evaluation_config(path)
    assert config.profile == "generated_common"
    assert config.metric_names == metric_names
    assert config.generated_population == population
    assert config.sampling.steps == steps
    assert config.seed == 321


@pytest.mark.parametrize(
    "filename",
    ["imagenet256_jit_generation.yaml", "imagenet256_jit_smoke.yaml"],
)
def test_imagenet_generated_profiles_use_fid_metrics_without_classifier(
    filename: str,
) -> None:
    path = PROJECT_ROOT / "configs" / "evaluation" / filename
    raw = yaml.safe_load(path.read_text(encoding="utf-8"))

    assert tuple(raw["metric_names"]) == ("fid", "inception_score")
    assert raw["classifier_asset_id"] is None
    assert load_metric_evaluation_config(path).classifier_asset_id is None


def test_imagenet_smoke_exercises_production_batch_sizes() -> None:
    smoke = load_metric_evaluation_config(
        PROJECT_ROOT / "configs" / "evaluation" / "imagenet256_jit_smoke.yaml"
    )
    production = load_metric_evaluation_config(
        PROJECT_ROOT / "configs" / "evaluation" / "imagenet256_jit_generation.yaml"
    )

    assert smoke.batch_size == production.batch_size == 64
    assert smoke.fid_batch_size == production.fid_batch_size == 256


@pytest.mark.parametrize(
    ("filename", "classifier_asset_id", "fid_asset_id"),
    [
        ("mnist_generation.yaml", "mnist-feature-seed123-bf16-pfm-v4", None),
        (
            "cifar10_edm_generation.yaml",
            None,
            "cifar10-inception-train-pfm-v4",
        ),
        (
            "imagenet256_jit_generation.yaml",
            None,
            "imagenet256-jit-train-pfm-v4",
        ),
    ],
)
def test_generated_common_profiles_bind_current_asset_namespace(
    filename: str,
    classifier_asset_id: str | None,
    fid_asset_id: str | None,
) -> None:
    config = load_metric_evaluation_config(
        PROJECT_ROOT / "configs" / "evaluation" / filename
    )

    assert config.classifier_asset_id == classifier_asset_id
    assert config.fid_asset_id == fid_asset_id


def test_evaluation_config_rejects_retired_ema_selector() -> None:
    path = PROJECT_ROOT / "configs" / "evaluation" / "mnist_generation.yaml"
    raw = yaml.safe_load(path.read_text(encoding="utf-8"))
    raw["prefer_ema"] = True

    with pytest.raises(ValueError, match="Unknown generated_common config keys"):
        metric_evaluation_config_from_mapping(raw)


@pytest.mark.parametrize(
    ("filename", "asset_id"),
    [
        ("cifar10_inception.yaml", "cifar10-inception-train-pfm-v4"),
        ("imagenet256_jit.yaml", "imagenet256-jit-train-pfm-v4"),
    ],
)
def test_fid_producer_configs_bind_current_asset_namespace(
    filename: str,
    asset_id: str,
) -> None:
    config = load_fid_config(PROJECT_ROOT / "configs" / "fid" / filename)

    assert validate_fid_config_asset_binding(config, asset_id) == asset_id


def test_generated_common_has_no_image_or_variance_retention_controls() -> None:
    parameters = inspect.signature(generation_evaluation.evaluate_generation).parameters
    assert not {
        "keep_generated_images",
        "keep_generation_variance_maps",
        "keep_generation_visualization_grids",
    }.intersection(parameters)


def test_cifar_fid_profile_is_full_lth14_protocol() -> None:
    config = load_fid_config(PROJECT_ROOT / "configs" / "fid" / "cifar10_inception.yaml")
    assert config.num_samples == 50_000

    identity = canonical_fid_preprocessing_identity(
        "cifar10",
        32,
        "train",
        "inception",
    )
    assert identity["backend_commit"] == "bfe72b995473e4fee9351705d7b005f17442a4ca"
    assert identity["feature_layer"] == 2048
    assert identity["real_population"] == "official_cifar10_train_all_50000"
    assert identity["input_geometry"] == "original_32x32_rgb_no_resize"


def test_fid_config_rejects_duplicate_yaml_keys(tmp_path: Path) -> None:
    path = tmp_path / "fid.yaml"
    path.write_text(
        "dataset: cifar10\n"
        "dataset: imagenet\n"
        "image_size: 32\n"
        "split: train\n"
        "backend: inception\n"
        "num_samples: 50000\n"
        "batch_size: 64\n",
        encoding="utf-8",
    )

    with pytest.raises(ValueError, match="duplicate YAML key"):
        load_fid_config(path)


@pytest.mark.parametrize(
    "filename",
    ["mnist_primary.yaml", "cifar10_primary.yaml", "imagenet256_primary.yaml"],
)
def test_primary_comparisons_bind_current_tracked_consumer_protocols(
    filename: str, runtime_paths,
) -> None:
    path = PROJECT_ROOT / "configs" / "comparisons" / filename
    from pfm.evaluation.evaluation_summary import prepare_comparison_config
    from pfm.cli.config_inputs import resolved_config_bytes
    authored = yaml.safe_load(path.read_text())
    spec, snapshots = prepare_comparison_config(authored, runtime_paths, PROJECT_ROOT)
    for profile, reference in spec["consumer_protocols"].items():
        target = runtime_paths.runs_root / reference["path"]
        assert hashlib.sha256(snapshots[target]).hexdigest() == reference["sha256"]
        assert snapshots[target] == resolved_config_bytes(
            PROJECT_ROOT / authored["consumer_protocols"][profile]["path"], [], category="evaluate")


@pytest.mark.parametrize("image_size", [32, 256])
def test_generated_quantization_exactly_matches_protocol(image_size: int) -> None:
    values = torch.zeros(1, 3, image_size, image_size, dtype=torch.float32)
    values[0, 0, 0, :8] = torch.tensor(
        [-2.0, -1.0, -0.5, 0.0, 0.5, 1.0, 2.0, float("nan")],
        dtype=torch.float32,
    )
    metrics = object.__new__(fid_metrics.LTH14OnlineMetrics)
    metrics.image_size = image_size
    metrics.total_samples = 2
    metrics.batch_size = 2
    metrics._buffer = []
    metrics._buffer_count = 0
    metrics._seen = 0

    with pytest.raises(ValueError, match="finite"):
        metrics.update(values)

    finite = values.clone()
    finite[0, 0, 0, 7] = 0.25
    expected = torch.round((finite.clamp(-1, 1) + 1) * 127.5).clamp(0, 255).to(torch.uint8)
    metrics.update(finite)
    actual = metrics._buffer[0]
    assert actual.dtype is torch.uint8
    assert torch.equal(actual, expected)
    assert tuple(actual.shape) == tuple(finite.shape)


def test_generated_quantization_requires_bound_geometry() -> None:
    metrics = object.__new__(fid_metrics.LTH14OnlineMetrics)
    metrics.image_size = 256
    metrics.total_samples = 2
    metrics.batch_size = 2
    metrics._buffer = []
    metrics._buffer_count = 0
    metrics._seen = 0
    with pytest.raises(ValueError, match="geometry"):
        metrics.update(torch.zeros(1, 3, 32, 32))


def test_sample_sampler_exposes_exact_pre_update_capture_request() -> None:
    parameters = inspect.signature(sample).parameters
    assert "uncertainty_capture_t" in parameters


def test_lth14_weights_identity_records_the_complete_digest(monkeypatch, tmp_path) -> None:
    weights = tmp_path / "weights-inception-2015-12-05-6726825d.pth"
    weights.write_bytes(b"resolved LTH14 inception weights")
    monkeypatch.setattr(
        fid_metrics,
        "validate_torch_fidelity_inception_weights",
        lambda: weights,
    )

    identity = fid_metrics.torch_fidelity_inception_weights_identity()

    assert identity["filename"] == weights.name
    assert identity["sha256"] == hashlib.sha256(weights.read_bytes()).hexdigest()
    assert len(identity["sha256"]) == 64
