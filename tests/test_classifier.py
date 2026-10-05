import contextlib
import hashlib
import json
from dataclasses import asdict, fields
from pathlib import Path
from types import SimpleNamespace

import pytest
import torch
import yaml
from torch import nn
from torch.utils.data import DataLoader, Dataset, TensorDataset

from pfm.core.atomic_io import write_json_atomic
from pfm.core.execution import classifier_asset_dir, runtime_identity
from pfm.data import datasets as dataset_module
from pfm.data.identity import DATASET_MANIFEST_NAME, dataset_manifest_payload
from pfm.evaluation import assets as classifier_assets
from pfm.models import classifier as classifier_model
from pfm.models.classifier import (
    ClassifierConfig,
    FeatureClassifier,
    evaluate_classifier_accuracy,
)
from pfm.training import classifier as classifier_training
from conftest import dataset_scientific_identity


_RUNTIME_IDENTITY = {
    "project_name": "pfm",
    "git_commit": "a" * 40,
    "rank": 0,
    "world_size": 1,
}
_CONFIG_DEFAULTS = {
    "seed": 123,
    "batch_size": 512,
    "num_workers": 0,
    "epochs": 20,
    "lr": 1e-3,
    "weight_decay": 1e-4,
    "feature_dim": 256,
    "pool_size": 7,
    "conv_channels": (64, 128, 128),
    "norm": "group",
}
_AUTHORED_CONFIGS = (
    ("mnist.yaml", "mnist", 10, 1, 28, 1024, 2),
)


class _MarkerTransform:
    def __init__(self, token):
        self.token = token

    def __call__(self, value):
        return value


class _TinyClassifierDataset(Dataset):
    def __init__(self, images, labels, *, transform, class_to_idx):
        self._images = images
        self._labels = labels
        self.transform = transform
        self.class_to_idx = class_to_idx

    def __len__(self):
        return len(self._labels)

    def __getitem__(self, index):
        return self.transform(self._images[index]), self._labels[index]


def test_classifier_defaults_and_feature_extractor_are_unchanged():
    config = ClassifierConfig()
    model = FeatureClassifier.from_config(config)

    for name, expected in _CONFIG_DEFAULTS.items():
        assert getattr(config, name) == expected
    assert config.mixed_precision == "none"
    assert model.feature_dim == 256
    assert model.conv_channels == (64, 128, 128)
    assert any(isinstance(module, nn.GroupNorm) for module in model.modules())


def test_classifier_config_excludes_runtime_owned_fields() -> None:
    field_names = {field.name for field in fields(ClassifierConfig)}

    assert "device" not in field_names
    assert "data_root" not in field_names
    assert "output_root" not in field_names
    assert "classifier_asset_id" not in field_names


@pytest.mark.parametrize(
    (
        "filename",
        "dataset",
        "num_classes",
        "image_channels",
        "image_size",
        "batch_size",
        "num_workers",
    ),
    _AUTHORED_CONFIGS,
)
def test_authored_classifier_configs_are_strict_and_use_exact_shapes(
    filename,
    dataset,
    num_classes,
    image_channels,
    image_size,
    batch_size,
    num_workers,
):
    project_root = Path(__file__).resolve().parents[1]

    config = classifier_model.load_classifier_config(
        project_root / "configs" / "classifier" / filename
    )

    assert (config.dataset, config.num_classes, config.image_channels, config.image_size) == (
        dataset,
        num_classes,
        image_channels,
        image_size,
    )
    assert config.mixed_precision == "bf16"
    assert config.batch_size == batch_size
    assert config.num_workers == num_workers
    for name, expected in _CONFIG_DEFAULTS.items():
        if name in {"batch_size", "num_workers"}:
            continue
        assert getattr(config, name) == expected


@pytest.mark.parametrize(
    ("mutation", "message"),
    [
        ("missing_mixed_precision", "missing fields.*mixed_precision"),
        ("unknown", "unknown fields.*pretrained"),
        ("wrong_shape", "shape"),
    ],
)
def test_classifier_config_parser_rejects_noncanonical_documents(tmp_path, mutation, message):
    payload = {
        **_CONFIG_DEFAULTS,
        "dataset": "mnist",
        "num_classes": 10,
        "image_channels": 1,
        "image_size": 28,
        "mixed_precision": "bf16",
    }
    payload["conv_channels"] = list(payload["conv_channels"])
    if mutation == "missing_mixed_precision":
        payload.pop("mixed_precision")
    elif mutation == "unknown":
        payload["pretrained"] = True
    else:
        payload["image_size"] = 32
    path = tmp_path / "classifier.yaml"
    path.write_text(yaml.safe_dump(payload), encoding="utf-8")

    with pytest.raises(ValueError, match=message):
        classifier_model.load_classifier_config(path)


@pytest.mark.parametrize("mixed_precision", ["none", "bf16"])
def test_classifier_config_parser_accepts_only_canonical_mixed_precision_values(
    tmp_path,
    mixed_precision,
):
    payload = {
        **_CONFIG_DEFAULTS,
        "dataset": "mnist",
        "num_classes": 10,
        "image_channels": 1,
        "image_size": 28,
        "mixed_precision": mixed_precision,
    }
    payload["conv_channels"] = list(payload["conv_channels"])
    path = tmp_path / "classifier.yaml"
    path.write_text(yaml.safe_dump(payload), encoding="utf-8")

    config = classifier_model.load_classifier_config(path)

    assert config.mixed_precision == mixed_precision


@pytest.mark.parametrize(
    "mixed_precision",
    [True, "fp16", "bfloat16", "BF16", " bf16 "],
)
def test_classifier_config_parser_rejects_noncanonical_mixed_precision_values(
    tmp_path,
    mixed_precision,
):
    payload = {
        **_CONFIG_DEFAULTS,
        "dataset": "mnist",
        "num_classes": 10,
        "image_channels": 1,
        "image_size": 28,
        "mixed_precision": mixed_precision,
    }
    payload["conv_channels"] = list(payload["conv_channels"])
    path = tmp_path / "classifier.yaml"
    path.write_text(yaml.safe_dump(payload), encoding="utf-8")

    with pytest.raises(ValueError, match="mixed_precision must be one of"):
        classifier_model.load_classifier_config(path)


def test_classifier_cuda_training_uses_bf16_autocast_and_nonblocking_transfers(
    monkeypatch,
    execution_context,
):
    config = _tiny_config("mnist", 10, 1, 28)
    config.mixed_precision = "bf16"
    _write_dataset_manifest(execution_context.runtime, config.dataset)
    state = {"inside_autocast": False}
    calls = []
    captured = {}

    class RecordingBatchValue:
        def __init__(self, name):
            self.name = name

        def to(self, device, *, non_blocking=False):
            calls.append((f"{self.name}.to", device.type, non_blocking))
            return self

    class RecordingLoader:
        def __init__(self, role):
            self.dataset = SimpleNamespace(
                class_to_idx=_class_to_idx(config.num_classes),
                transform=_MarkerTransform(f"{role}-transform"),
            )

        def __iter__(self):
            yield RecordingBatchValue("images"), RecordingBatchValue("labels")

    class RecordingModel(nn.Module):
        def __init__(self):
            super().__init__()
            self.weight = nn.Parameter(torch.zeros(config.num_classes, dtype=torch.float32))

        def to(self, device):
            calls.append(("model.to", device.type))
            return self

        def forward(self, images):
            assert state["inside_autocast"]
            calls.append(("forward",))
            return self.weight.unsqueeze(0)

    class RecordingLoss:
        def __call__(self, logits, labels):
            assert state["inside_autocast"]
            calls.append(("loss",))
            return logits.sum()

    class RecordingOptimizer:
        def zero_grad(self, *, set_to_none):
            assert not state["inside_autocast"]
            calls.append(("zero_grad", set_to_none))

        def step(self):
            assert not state["inside_autocast"]
            calls.append(("step",))

    @contextlib.contextmanager
    def recording_autocast(loader_config, device):
        assert loader_config.mixed_precision == "bf16"
        assert device.type == "cuda"
        assert not state["inside_autocast"]
        calls.append(("autocast.enter", loader_config.mixed_precision, device.type))
        state["inside_autocast"] = True
        try:
            yield
        finally:
            state["inside_autocast"] = False
            calls.append(("autocast.exit",))

    real_experiment_config = classifier_training.ExperimentConfig

    def recording_experiment_config(**kwargs):
        captured["loader_config_kwargs"] = kwargs
        return real_experiment_config(**kwargs)

    def recording_save(*args, **kwargs):
        model = args[3]
        captured["checkpoint_dtypes"] = {
            tensor.dtype for tensor in model.state_dict().values() if tensor.is_floating_point()
        }

    monkeypatch.setattr(classifier_training, "ExperimentConfig", recording_experiment_config)
    monkeypatch.setattr(
        classifier_training,
        "make_dataloaders",
        lambda loader_config, data_root: (RecordingLoader("training"), RecordingLoader("validation")),
    )
    monkeypatch.setattr(classifier_training.FeatureClassifier, "from_config", lambda config: RecordingModel())
    monkeypatch.setattr(classifier_training.torch.optim, "AdamW", lambda *args, **kwargs: RecordingOptimizer())
    monkeypatch.setattr(classifier_training.nn, "CrossEntropyLoss", lambda: RecordingLoss())
    monkeypatch.setattr(classifier_training, "autocast_context", recording_autocast, raising=False)
    monkeypatch.setattr(
        classifier_training.torch.amp,
        "GradScaler",
        lambda *args, **kwargs: pytest.fail("classifier BF16 must not use GradScaler"),
    )
    monkeypatch.setattr(
        classifier_training,
        "evaluate_classifier_accuracy",
        lambda model, loader, device: _complete_metrics(config.num_classes),
    )
    monkeypatch.setattr(classifier_training, "_save_classifier_asset_with_binding", recording_save)

    classifier_training.train_classifier(
        config,
        classifier_asset_id="mnist-feature-seed123-bf16",
        job_context=execution_context,
        device=torch.device("cuda"),
        show_progress=False,
    )

    assert captured["loader_config_kwargs"]["mixed_precision"] == "bf16"
    assert calls == [
        ("model.to", "cuda"),
        ("images.to", "cuda", True),
        ("labels.to", "cuda", True),
        ("autocast.enter", "bf16", "cuda"),
        ("forward",),
        ("loss",),
        ("autocast.exit",),
        ("zero_grad", True),
        ("step",),
    ]
    assert captured["checkpoint_dtypes"] == {torch.float32}


def test_classifier_cpu_training_stays_outside_torch_autocast(monkeypatch, execution_context):
    config = _tiny_config("mnist", 10, 1, 28)
    config.mixed_precision = "bf16"
    monkeypatch.setattr(
        classifier_training.torch.amp,
        "autocast",
        lambda *args, **kwargs: pytest.fail("CPU classifier training must not enter torch autocast"),
    )

    target, _ = _train_classifier_asset(
        monkeypatch,
        execution_context,
        config=config,
        classifier_asset_id="mnist-feature-seed123-bf16",
    )

    checkpoint = torch.load(target / "classifier.pt", map_location="cpu")
    floating_dtypes = {
        tensor.dtype
        for tensor in checkpoint["model_state"].values()
        if tensor.is_floating_point()
    }
    assert floating_dtypes == {torch.float32}


def test_feature_classifier_accepts_mnist_authored_shape():
    config = ClassifierConfig(
        feature_dim=8,
        pool_size=1,
        conv_channels=(4, 4, 4),
    )
    model = FeatureClassifier.from_config(config)

    logits = model(torch.zeros(1, 1, 28, 28))

    assert logits.shape == (1, 10)


def test_classifier_asset_publication_is_one_complete_directory(monkeypatch, execution_context):
    config = _tiny_config("mnist", 10, 1, 28)
    target, _ = _train_classifier_asset(
        monkeypatch,
        execution_context,
        config=config,
        classifier_asset_id="mnist-feature-seed123",
    )

    runtime = execution_context.runtime
    expected = classifier_asset_dir(runtime, "mnist", "mnist-feature-seed123")
    assert target == expected
    assert {path.name for path in target.iterdir()} == {
        "classifier.pt",
        "classifier_manifest.json",
    }
    assert not list(target.parent.glob(".mnist-feature-seed123.staging-*"))
    assert not (runtime.classifier_root / "mnist_feature_classifier.pt").exists()


def test_classifier_asset_manifest_binds_complete_identity(monkeypatch, execution_context):
    config = _tiny_config("mnist", 10, 1, 28)
    runtime = execution_context.runtime
    dataset_manifest_path = _write_dataset_manifest(runtime, "mnist")
    class_to_idx = _class_to_idx(10)

    target, metrics = _train_classifier_asset(
        monkeypatch,
        execution_context,
        config=config,
        classifier_asset_id="mnist-feature-seed123",
    )
    checkpoint_path = target / "classifier.pt"
    checkpoint = torch.load(checkpoint_path, map_location="cpu")
    manifest = json.loads((target / "classifier_manifest.json").read_text(encoding="utf-8"))

    assert set(checkpoint) == {"model_state"}
    assert set(manifest) == {
        "classifier_asset_id",
        "paths",
        "dataset",
        "image_size",
        "num_classes",
        "training_split",
        "validation_split",
        "config",
        "optimizer",
        "preprocessing_identity",
        "class_to_idx",
        "class_mapping_sha256",
        "dataset_scientific_identity_sha256",
        "checkpoint_sha256",
        "producer_runtime_identity",
        "validation_metrics",
    }
    assert manifest["classifier_asset_id"] == "mnist-feature-seed123"
    assert manifest["paths"] == {
        "checkpoint": "classifier.pt",
        "manifest": "classifier_manifest.json",
    }
    assert (manifest["dataset"], manifest["training_split"], manifest["validation_split"]) == (
        "mnist",
        "train",
        "test",
    )
    assert (manifest["num_classes"], manifest["image_size"]) == (10, 28)
    assert manifest["config"] == {**asdict(config), "conv_channels": [4, 4, 4]}
    assert manifest["optimizer"] == {
        "type": "AdamW",
        "lr": config.lr,
        "weight_decay": config.weight_decay,
    }
    assert manifest["preprocessing_identity"] == _expected_preprocessing_identity(
        "mnist",
        28,
    )
    serialized_preprocessing = json.dumps(manifest["preprocessing_identity"])
    assert "code_sha256" not in serialized_preprocessing
    assert "closure" not in serialized_preprocessing
    assert manifest["class_to_idx"] == class_to_idx
    assert manifest["class_mapping_sha256"] == _canonical_json_sha256(class_to_idx)
    dataset_manifest = json.loads(dataset_manifest_path.read_text(encoding="utf-8"))
    assert manifest["dataset_scientific_identity_sha256"] == dataset_manifest[
        "scientific_identity_sha256"
    ]
    assert manifest["checkpoint_sha256"] == _sha256(checkpoint_path)
    assert manifest["producer_runtime_identity"] == _RUNTIME_IDENTITY
    assert manifest["validation_metrics"] == metrics


@pytest.mark.parametrize(
    ("name", "mutation"),
    [
        ("type", "missing"),
        ("type", "mismatch"),
        ("lr", "missing"),
        ("lr", "mismatch"),
        ("weight_decay", "missing"),
        ("weight_decay", "mismatch"),
    ],
)
def test_classifier_asset_loader_rejects_tampered_optimizer_contract(
    monkeypatch,
    execution_context,
    name,
    mutation,
):
    config = _tiny_config("mnist", 10, 1, 28)
    target, _ = _train_classifier_asset(
        monkeypatch,
        execution_context,
        config=config,
        classifier_asset_id="mnist-feature-seed123",
    )
    manifest_path = target / "classifier_manifest.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    if mutation == "missing":
        manifest["optimizer"].pop(name)
    elif name == "type":
        manifest["optimizer"][name] = "SGD"
    else:
        manifest["optimizer"][name] = (
            config.lr * 2.0 if name == "lr" else config.weight_decay * 3.0
        )
    write_json_atomic(manifest_path, manifest)

    with pytest.raises(ValueError, match="optimizer"):
        classifier_assets.load_classifier_asset(
            execution_context.runtime,
            "mnist",
            "mnist-feature-seed123",
            torch.device("cpu"),
        )


def test_classifier_asset_target_exists_fails_without_staging(monkeypatch, execution_context):
    config = _tiny_config("mnist", 10, 1, 28)
    _train_classifier_asset(
        monkeypatch,
        execution_context,
        config=config,
        classifier_asset_id="mnist-feature-seed123",
    )

    with pytest.raises(FileExistsError, match="immutable target already exists"):
        classifier_training.train_classifier(
            config,
            classifier_asset_id="mnist-feature-seed123",
            job_context=execution_context,
            device=torch.device("cpu"),
            show_progress=False,
        )

    parent = classifier_asset_dir(
        execution_context.runtime,
        "mnist",
        "mnist-feature-seed123",
    ).parent
    assert not list(parent.glob(".mnist-feature-seed123.staging-*"))


@pytest.mark.parametrize(
    "class_to_idx",
    [
        {"zero": 0},
        {"zero": 0, "one": 2},
        {"zero": 0, "also_zero": 0},
    ],
)
def test_classifier_asset_rejects_incomplete_or_noncontiguous_class_mapping(
    class_to_idx,
):
    with pytest.raises(ValueError, match="class.*mapping"):
        classifier_model.validate_classifier_class_mapping(class_to_idx, 10)


@pytest.mark.parametrize(
    "metrics",
    [
        {"test_accuracy": 0.5, "test_loss": 1.0},
        {
            "test_accuracy": 0.5,
            "test_loss": float("nan"),
            "per_class_accuracy": [0.5, 0.5],
            "per_class_correct": [1, 1],
            "per_class_total": [2, 2],
        },
        {
            "test_accuracy": 0.5,
            "test_loss": 1.0,
            "per_class_accuracy": [0.5],
            "per_class_correct": [1, 1],
            "per_class_total": [2, 2],
        },
    ],
)
def test_classifier_asset_rejects_incomplete_or_nonfinite_validation_metrics(
    metrics,
):
    with pytest.raises(ValueError, match="validation metrics"):
        classifier_model.validate_classifier_validation_metrics(metrics, 10)


@pytest.mark.parametrize(
    ("mutation", "message"),
    [
        ("per_class", "class 0 counts do not match accuracy"),
        ("overall", "overall accuracy does not match per-class counts"),
    ],
)
def test_classifier_validation_metrics_reject_inconsistent_accuracy_and_counts(
    mutation,
    message,
):
    metrics = _complete_metrics(10)
    if mutation == "per_class":
        metrics["per_class_accuracy"][0] = 0.25
    else:
        metrics["test_accuracy"] = 0.6

    with pytest.raises(ValueError, match=message):
        classifier_model.validate_classifier_validation_metrics(metrics, 10)


def test_classifier_asset_loader_validates_identity_and_digest_before_weights(
    monkeypatch,
    execution_context,
):
    config = _tiny_config("mnist", 10, 1, 28)
    target, _ = _train_classifier_asset(
        monkeypatch,
        execution_context,
        config=config,
        classifier_asset_id="mnist-feature-seed123",
    )
    checkpoint_path = target / "classifier.pt"
    checkpoint_path.write_bytes(checkpoint_path.read_bytes() + b"tampered")
    weight_load_attempted = False

    def fail_torch_load(*args, **kwargs):
        nonlocal weight_load_attempted
        weight_load_attempted = True
        raise AssertionError("digest mismatch must fail before torch.load")

    monkeypatch.setattr(torch, "load", fail_torch_load)

    with pytest.raises(ValueError, match="SHA-256"):
        classifier_assets.load_classifier_asset(
            execution_context.runtime,
            "mnist",
            "mnist-feature-seed123",
            torch.device("cpu"),
        )

    assert weight_load_attempted is False


def test_classifier_asset_loader_rejects_class_mapping_digest_tampering(
    monkeypatch,
    execution_context,
):
    config = _tiny_config("mnist", 10, 1, 28)
    target, _ = _train_classifier_asset(
        monkeypatch,
        execution_context,
        config=config,
        classifier_asset_id="mnist-feature-seed123",
    )
    manifest_path = target / "classifier_manifest.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    manifest["class_mapping_sha256"] = "0" * 64
    write_json_atomic(manifest_path, manifest)

    with pytest.raises(ValueError, match="class mapping SHA-256"):
        classifier_assets.load_classifier_asset(
            execution_context.runtime,
            "mnist",
            "mnist-feature-seed123",
            torch.device("cpu"),
        )


def test_classifier_asset_loader_rejects_changed_dataset_manifest_binding(
    monkeypatch,
    execution_context,
):
    config = _tiny_config("mnist", 10, 1, 28)
    runtime = execution_context.runtime
    dataset_manifest_path = _write_dataset_manifest(runtime, "mnist")
    _train_classifier_asset(
        monkeypatch,
        execution_context,
        config=config,
        classifier_asset_id="mnist-feature-seed123",
    )
    write_json_atomic(
        dataset_manifest_path,
        dataset_manifest_payload(
            "mnist",
            dataset_scientific_identity("mnist", marker="b"),
        ),
    )

    with pytest.raises(ValueError, match="dataset scientific identity"):
        classifier_assets.load_classifier_asset(
            runtime,
            "mnist",
            "mnist-feature-seed123",
            torch.device("cpu"),
        )


def test_classifier_asset_loader_starts_from_dataset_and_explicit_asset_id(
    monkeypatch,
    execution_context,
):
    config = _tiny_config("mnist", 10, 1, 28)
    _, metrics = _train_classifier_asset(
        monkeypatch,
        execution_context,
        config=config,
        classifier_asset_id="mnist-feature-seed123",
    )

    loaded, manifest = classifier_assets.load_classifier_asset(
        execution_context.runtime,
        "mnist",
        "mnist-feature-seed123",
        torch.device("cpu"),
    )

    assert loaded.num_classes == 10
    assert manifest["classifier_asset_id"] == "mnist-feature-seed123"
    assert manifest["validation_metrics"] == metrics


def test_classifier_evaluation_reports_complete_per_class_metrics():
    class FixedLogits(nn.Module):
        num_classes = 3

        def forward(self, images):
            return images[:, :, 0, 0]

    images = torch.tensor(
        [
            [[[4.0]], [[0.0]], [[0.0]]],
            [[[0.0]], [[4.0]], [[0.0]]],
            [[[4.0]], [[0.0]], [[0.0]]],
            [[[0.0]], [[0.0]], [[4.0]]],
        ]
    )
    labels = torch.tensor([0, 1, 2, 2])
    loader = DataLoader(TensorDataset(images, labels), batch_size=2)

    metrics = evaluate_classifier_accuracy(FixedLogits(), loader, torch.device("cpu"))

    assert metrics["test_accuracy"] == pytest.approx(0.75)
    assert metrics["test_loss"] == pytest.approx(
        float(nn.CrossEntropyLoss()(images[:, :, 0, 0], labels))
    )
    assert metrics["per_class_accuracy"] == pytest.approx([1.0, 1.0, 0.5])
    assert metrics["per_class_correct"] == [1, 1, 1]
    assert metrics["per_class_total"] == [1, 1, 2]


def test_classifier_evaluation_rejects_missing_class_metrics():
    config = _tiny_config("mnist", 3, 1, 28)
    model = FeatureClassifier.from_config(config)
    loader = DataLoader(
        TensorDataset(torch.zeros(2, 1, 28, 28), torch.tensor([0, 1])),
        batch_size=2,
    )

    with pytest.raises(ValueError, match="per-class"):
        evaluate_classifier_accuracy(model, loader, torch.device("cpu"))


def test_mnist_classifier_training_publishes_explicit_immutable_asset(
    monkeypatch,
    execution_context,
):
    dataset_name = "mnist"
    asset_id = "mnist-feature-seed123"
    _write_dataset_manifest(execution_context.runtime, dataset_name)
    training_dataset = TensorDataset(
        torch.zeros(2, 1, 28, 28),
        torch.tensor([0, 1]),
    )
    validation_dataset = TensorDataset(
        torch.zeros(2, 1, 28, 28),
        torch.tensor([0, 1]),
    )
    mapping = _class_to_idx(10)
    training_dataset.classes = list(mapping)
    validation_dataset.classes = list(mapping)
    training_transform = dataset_module.mnist_transform()
    validation_transform = dataset_module.mnist_transform()
    training_dataset.transform = training_transform
    validation_dataset.transform = validation_transform
    captured = {}

    def fake_make_dataloaders(config, data_root):
        captured["data_root"] = data_root
        return (
            DataLoader(training_dataset, batch_size=2),
            DataLoader(validation_dataset, batch_size=2),
        )

    monkeypatch.setattr(classifier_training, "make_dataloaders", fake_make_dataloaders)
    monkeypatch.setattr(
        classifier_training,
        "evaluate_classifier_accuracy",
        lambda model, loader, device: _complete_metrics(10),
    )
    config = _tiny_config(dataset_name, 10, 1, 28)

    metrics = classifier_training.train_classifier(
        config,
        classifier_asset_id=asset_id,
        job_context=execution_context,
        device=torch.device("cpu"),
        show_progress=False,
    )

    target = classifier_asset_dir(execution_context.runtime, dataset_name, asset_id)
    assert target.is_dir()
    assert {path.name for path in target.iterdir()} == {
        "classifier.pt",
        "classifier_manifest.json",
    }
    assert metrics == _complete_metrics(10)
    assert captured["data_root"] == execution_context.runtime.dataset_root(dataset_name)
    assert not (execution_context.runtime.classifier_root / f"{dataset_name}_feature_classifier.pt").exists()
    manifest = json.loads((target / "classifier_manifest.json").read_text(encoding="utf-8"))
    assert (manifest["training_split"], manifest["validation_split"]) == ("train", "test")
    assert manifest["preprocessing_identity"] == _expected_preprocessing_identity(
        dataset_name,
        28,
    )


def test_classifier_manifest_uses_authored_protocol_not_runtime_object_provenance(
    monkeypatch,
    execution_context,
):
    dataset_name = "mnist"
    _write_dataset_manifest(execution_context.runtime, dataset_name)
    class_to_idx = _class_to_idx(10)
    training_dataset = TensorDataset(torch.zeros(2, 1, 28, 28), torch.tensor([0, 1]))
    training_dataset.class_to_idx = class_to_idx
    training_dataset.transform = _MarkerTransform("actual-training-transform")
    validation_dataset = TensorDataset(torch.zeros(2, 1, 28, 28), torch.tensor([0, 1]))
    validation_dataset.class_to_idx = class_to_idx
    validation_dataset.transform = _MarkerTransform("actual-validation-transform")
    monkeypatch.setattr(
        classifier_training,
        "make_dataloaders",
        lambda config, data_root: (
            DataLoader(training_dataset, batch_size=2),
            DataLoader(validation_dataset, batch_size=2),
        ),
    )
    monkeypatch.setattr(
        classifier_training,
        "evaluate_classifier_accuracy",
        lambda model, data_loader, device: _complete_metrics(10),
    )
    config = _tiny_config(dataset_name, 10, 1, 28)

    classifier_training.train_classifier(
        config,
        classifier_asset_id="mnist-feature-seed123",
        job_context=execution_context,
        device=torch.device("cpu"),
        show_progress=False,
    )

    asset_dir = classifier_asset_dir(
        execution_context.runtime,
        dataset_name,
        "mnist-feature-seed123",
    )
    manifest = json.loads((asset_dir / "classifier_manifest.json").read_text(encoding="utf-8"))
    assert (manifest["training_split"], manifest["validation_split"]) == ("train", "test")
    assert manifest["preprocessing_identity"] == _expected_preprocessing_identity(
        dataset_name,
        28,
    )
    assert "actual-training-transform" not in json.dumps(manifest)
    assert "actual-validation-transform" not in json.dumps(manifest)
    assert manifest["optimizer"] == {
        "type": "AdamW",
        "lr": config.lr,
        "weight_decay": config.weight_decay,
    }
    assert "defaults" not in manifest["optimizer"]
    assert "param_groups" not in manifest["optimizer"]
    assert not hasattr(classifier_training, "save_classifier_asset")


def test_classifier_asset_loader_rejects_tampered_preprocessing_protocol(
    monkeypatch,
    execution_context,
):
    config = _tiny_config("mnist", 10, 1, 28)
    target, _ = _train_classifier_asset(
        monkeypatch,
        execution_context,
        config=config,
        classifier_asset_id="mnist-feature-seed123",
    )
    manifest_path = target / "classifier_manifest.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    manifest["preprocessing_identity"]["training"][
        "random_horizontal_flip_probability"
    ] = 0.5
    write_json_atomic(manifest_path, manifest)

    with pytest.raises(ValueError, match="preprocessing"):
        classifier_assets.load_classifier_asset(
            execution_context.runtime,
            "mnist",
            "mnist-feature-seed123",
            torch.device("cpu"),
        )


def test_training_rejects_incomplete_class_mapping_before_model_or_optimizer_work(
    monkeypatch,
    execution_context,
):
    _write_dataset_manifest(execution_context.runtime, "mnist")
    dataset = TensorDataset(torch.zeros(2, 1, 28, 28), torch.tensor([0, 1]))
    dataset.class_to_idx = {"only_one_class": 0}
    loader = DataLoader(dataset, batch_size=2)
    monkeypatch.setattr(
        classifier_training,
        "make_dataloaders",
        lambda config, data_root: (loader, loader),
    )
    touched = []

    def record_work(*args, **kwargs):
        touched.append((args, kwargs))
        raise AssertionError("model, optimizer, evaluation, and publication must not start")

    monkeypatch.setattr(classifier_training.FeatureClassifier, "from_config", record_work)
    monkeypatch.setattr(classifier_training.torch.optim, "AdamW", record_work)
    monkeypatch.setattr(classifier_training, "evaluate_classifier_accuracy", record_work)

    with pytest.raises(ValueError, match="class mapping must contain every class"):
        classifier_training.train_classifier(
            _tiny_config("mnist", 10, 1, 28),
            classifier_asset_id="mnist-feature-seed123",
            job_context=execution_context,
            device=torch.device("cpu"),
            show_progress=False,
        )

    assert touched == []


def test_training_rejects_dataset_manifest_change_before_asset_publication(
    monkeypatch,
    execution_context,
):
    dataset_name = "mnist"
    asset_id = "mnist-feature-seed123"
    manifest_path = _write_dataset_manifest(execution_context.runtime, dataset_name)
    dataset = TensorDataset(torch.zeros(2, 1, 28, 28), torch.tensor([0, 1]))
    dataset.classes = list(_class_to_idx(10))
    dataset.transform = _MarkerTransform("manifest-mutation-transform")
    loader = DataLoader(dataset, batch_size=2)
    monkeypatch.setattr(
        classifier_training,
        "make_dataloaders",
        lambda config, data_root: (loader, loader),
    )

    def mutate_manifest_after_training(model, data_loader, device):
        write_json_atomic(
            manifest_path,
            dataset_manifest_payload(
                dataset_name,
                dataset_scientific_identity(dataset_name, marker="b"),
            ),
        )
        return _complete_metrics(10)

    monkeypatch.setattr(
        classifier_training,
        "evaluate_classifier_accuracy",
        mutate_manifest_after_training,
    )

    with pytest.raises(ValueError, match="dataset manifest changed during classifier training"):
        classifier_training.train_classifier(
            _tiny_config(dataset_name, 10, 1, 28),
            classifier_asset_id=asset_id,
            job_context=execution_context,
            device=torch.device("cpu"),
            show_progress=False,
        )

    assert not classifier_asset_dir(execution_context.runtime, dataset_name, asset_id).exists()


def test_training_verifies_dataset_manifest_binding_once_before_publication(
    monkeypatch,
    execution_context,
):
    calls = 0
    verify = classifier_training._verify_dataset_manifest_binding

    def recording_verify(binding):
        nonlocal calls
        calls += 1
        verify(binding)

    monkeypatch.setattr(
        classifier_training,
        "_verify_dataset_manifest_binding",
        recording_verify,
    )

    _train_classifier_asset(
        monkeypatch,
        execution_context,
        config=_tiny_config("mnist", 10, 1, 28),
        classifier_asset_id="mnist-feature-seed123",
    )

    assert calls == 1


@pytest.mark.parametrize("epochs", [0, -1, True, 1.5, "1"])
def test_train_classifier_rejects_invalid_epochs_before_any_work(
    epochs,
    monkeypatch,
    execution_context,
):
    touched = []

    def record_work(*args, **kwargs):
        touched.append((args, kwargs))
        raise AssertionError("classifier work must not start")

    monkeypatch.setattr(classifier_training, "set_seed", record_work)
    monkeypatch.setattr(classifier_training, "make_dataloaders", record_work)

    with pytest.raises(ValueError, match="classifier training epochs must be (?:a positive|an) integer"):
        classifier_training.train_classifier(
            ClassifierConfig(epochs=epochs),
            classifier_asset_id="mnist-feature-seed123",
            job_context=execution_context,
            device=torch.device("cpu"),
            show_progress=False,
        )

    assert touched == []


def _train_classifier_asset(
    monkeypatch,
    job_context,
    *,
    config,
    classifier_asset_id,
    training_transform=None,
    validation_transform=None,
):
    runtime = job_context.runtime
    dataset = config.dataset
    manifest_path = runtime.dataset_root(dataset) / DATASET_MANIFEST_NAME
    if not manifest_path.exists():
        _write_dataset_manifest(runtime, dataset)
    class_to_idx = _class_to_idx(config.num_classes)
    labels = torch.arange(config.num_classes)
    images = torch.zeros(
        config.num_classes,
        config.image_channels,
        config.image_size,
        config.image_size,
    )
    training_dataset = _TinyClassifierDataset(
        images,
        labels,
        transform=training_transform or _MarkerTransform(f"{dataset}-training-transform"),
        class_to_idx=class_to_idx,
    )
    validation_dataset = _TinyClassifierDataset(
        images.clone(),
        labels.clone(),
        transform=validation_transform or _MarkerTransform(f"{dataset}-validation-transform"),
        class_to_idx=class_to_idx,
    )
    monkeypatch.setattr(
        classifier_training,
        "make_dataloaders",
        lambda loader_config, data_root: (
            DataLoader(training_dataset, batch_size=config.batch_size),
            DataLoader(validation_dataset, batch_size=config.batch_size),
        ),
    )
    metrics = classifier_training.train_classifier(
        config,
        classifier_asset_id=classifier_asset_id,
        job_context=job_context,
        device=torch.device("cpu"),
        show_progress=False,
    )
    return classifier_asset_dir(runtime, dataset, classifier_asset_id), metrics


def test_cpu_classifier_cli_records_explicit_fp32_execution(monkeypatch, execution_context):
    from dataclasses import replace
    from pfm.cli import train_classifier as cli

    context = replace(execution_context, device="cpu")
    monkeypatch.setattr(cli, "context_from_args", lambda args: context)
    observed_dtypes = set()
    original_forward = FeatureClassifier.forward

    def forward(self, *args, **kwargs):
        logits = original_forward(self, *args, **kwargs)
        observed_dtypes.add(logits.dtype)
        return logits

    def train(config, *, classifier_asset_id, job_context, device):
        assert device == torch.device("cpu")
        _, metrics = _train_classifier_asset(
            monkeypatch, job_context, config=config, classifier_asset_id=classifier_asset_id,
        )
        return metrics

    monkeypatch.setattr(FeatureClassifier, "forward", forward)
    monkeypatch.setattr(cli, "train_classifier", train)
    config_path = Path(__file__).resolve().parents[1] / "configs/classifier/mnist.yaml"
    assert cli.main([
        "--config", str(config_path), "--classifier-asset-id", "cpu-fp32", "--device", "cpu",
        "--set", "mixed_precision=none", "--set", "epochs=1", "--set", "num_workers=0",
        "--set", "batch_size=10", "--set", "conv_channels=[4,8,8]", "--set", "feature_dim=8",
    ]) == 0
    target = classifier_asset_dir(context.runtime, "mnist", "cpu-fp32")
    manifest = json.loads((target / "classifier_manifest.json").read_text())
    assert manifest["config"]["mixed_precision"] == "none"
    assert observed_dtypes == {torch.float32}


def test_cuda_classifier_cli_preserves_requested_bf16(monkeypatch, execution_context):
    from pfm.cli import train_classifier as cli

    calls = []
    monkeypatch.setattr(cli, "context_from_args", lambda args: execution_context)
    monkeypatch.setattr(cli, "execution_device", lambda job: torch.device("cuda", 0))
    monkeypatch.setattr(cli, "train_classifier", lambda config, **kwargs:
                        calls.append((config.mixed_precision, kwargs["device"])) or {})
    config_path = Path(__file__).resolve().parents[1] / "configs/classifier/mnist.yaml"
    assert cli.main(["--config", str(config_path), "--classifier-asset-id", "cuda-bf16"]) == 0
    assert calls == [("bf16", torch.device("cuda", 0))]


def _expected_preprocessing_identity(dataset, image_size):
    return {
        "protocol": "pfm_classifier_preprocessing_v1",
        "dataset": dataset,
        "image_size": image_size,
        "tensor_conversion": "to_float_tensor",
        "normalization": {
            "source_range": [0.0, 1.0],
            "target_range": [-1.0, 1.0],
        },
        "training": {
            "center_crop": None,
            "random_horizontal_flip_probability": 0.0,
        },
        "validation": {
            "center_crop": None,
            "random_horizontal_flip_probability": 0.0,
        },
    }


def _tiny_config(dataset, num_classes, image_channels, image_size):
    return ClassifierConfig(
        dataset=dataset,
        epochs=1,
        batch_size=2,
        feature_dim=8,
        num_classes=num_classes,
        image_channels=image_channels,
        image_size=image_size,
        pool_size=1,
        conv_channels=(4, 4, 4),
    )


def _write_dataset_manifest(runtime, dataset):
    path = runtime.dataset_root(dataset) / DATASET_MANIFEST_NAME
    write_json_atomic(
        path,
        dataset_manifest_payload(dataset, dataset_scientific_identity(dataset)),
    )
    return path


def _class_to_idx(num_classes):
    return {f"class_{index:04d}": index for index in range(num_classes)}


def _complete_metrics(num_classes):
    return {
        "test_accuracy": 0.5,
        "test_loss": 1.0,
        "per_class_accuracy": [0.5] * num_classes,
        "per_class_correct": [1] * num_classes,
        "per_class_total": [2] * num_classes,
    }


def _sha256(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def _canonical_json_sha256(payload):
    encoded = json.dumps(
        payload,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=False,
    ).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()
