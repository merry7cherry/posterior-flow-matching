from __future__ import annotations

from dataclasses import fields
from pathlib import Path

import pytest
import torch

from pfm.core.sampling_config import SamplingConfig

from pfm.core.config import ExperimentConfig
from pfm.evaluation import generation_evaluation
from pfm.evaluation.generation_service import GeneratedChunk
from pfm.sampling.samplers import SampleResult


class _Classifier(torch.nn.Module):
    def forward(self, images):
        labels = images[:, 0, 0, 0].long()
        logits = torch.zeros(images.shape[0], 2)
        logits[torch.arange(images.shape[0]), labels] = 9.0
        return logits


def _sample_result(samples: torch.Tensor, sampling_steps: int) -> SampleResult:
    return SampleResult(
        samples=samples,
        endpoint_var=None,
        velocity_var=None,
        variance_available=False,
        sampling_steps=sampling_steps,
        actual_nfe=sampling_steps,
        sampling=SamplingConfig(steps=sampling_steps),
        actual_vjp=0,
        elapsed_seconds=0.1,
        numerical_summary={},
        uncertainty_source_step=None,
        uncertainty_source_t=None,
        uncertainty_source_state=None,


    )


def _fid_identity(dataset: str) -> dict[str, object]:
    return {
        "stats_path": "/scratch/data/fid_stats/stats.npz",
        "preprocessing_identity": {
            "feature_extractor_weights": {"sha256": "a" * 64}
        },
        "dataset": dataset,
    }


@pytest.mark.parametrize(
    ("dataset", "image_size", "metric_names"),
    [
        ("mnist", 28, ("conditional_accuracy",)),
        ("cifar10", 32, ("fid",)),
        ("imagenet", 256, ("fid", "inception_score")),
    ],
)
def test_generated_common_streams_requested_metrics(
    monkeypatch,
    dataset: str,
    image_size: int,
    metric_names: tuple[str, ...],
) -> None:
    assert "conditional_logits" not in {field.name for field in fields(generation_evaluation._GenerationConsumers)}
    assert "conditional_labels" not in {field.name for field in fields(generation_evaluation._GenerationConsumers)}
    config = ExperimentConfig(
        dataset=dataset,
        image_size=image_size,
        image_channels=1 if dataset == "mnist" else 3,
        num_classes=2,
    )
    requests = []
    online_builds = []

    events = []

    def fake_iter(_model, _variant, _config, request, *, device):
        del device
        events.append("iterate")
        requests.append(request)
        for start in range(0, request.labels.numel(), 2):
            labels = request.labels[start : start + 2]
            samples = labels.float().view(-1, 1, 1, 1).expand(
                -1, config.image_channels, image_size, image_size
            )
            yield GeneratedChunk(
                start,
                labels,
                _sample_result(samples, request.sampling.steps),
            )

    class FakeOnline:
        def __init__(self):
            self.count = 0

        def update(self, images):
            self.count += int(images.shape[0])

        def finalize(self):
            assert self.count == 4
            return {"fid": 1.25, "inception_score": {"mean": 3.0, "std": 0.1}}

    def fake_build(identity, **kwargs):
        events.append("fid")
        online_builds.append((identity, kwargs))
        return FakeOnline()

    monkeypatch.setattr(generation_evaluation, "iter_loaded_generation", fake_iter)
    monkeypatch.setattr(generation_evaluation, "_build_online_fid_metrics", fake_build)

    result = generation_evaluation.evaluate_generation(
        load_model=lambda: events.append("model") or torch.nn.Identity(),
        variant="velocity_fm",
        config=config,
        metric_names=metric_names,
        classifier=(
            _Classifier()
            if "conditional_accuracy" in metric_names
            else None
        ),
        fid_asset_identity=(None if dataset == "mnist" else _fid_identity(dataset)),
        sampling=SamplingConfig(steps=2),
        generated_population=4,
        batch_size=2,
        fid_batch_size=3,
        device=torch.device("cpu"),
        seed=321,
    )

    assert len(requests) == 1
    if "conditional_accuracy" in metric_names:
        assert result["metrics"]["conditional_accuracy"]["accuracy"] == pytest.approx(1.0)
        assert result["metrics"]["conditional_accuracy"]["per_class_count"] == [2, 2]
        assert result["generated_protocol"]["conditional_count"] == 4
    else:
        assert "conditional_accuracy" not in result["metrics"]
        assert "conditional_count" not in result["generated_protocol"]
    assert result["generated_protocol"]["actual_nfe"] == 2
    if online_builds:
        assert online_builds[0][1]["batch_size"] == 3
        assert online_builds[0][1]["image_size"] == image_size
        assert events[:3] == ["fid", "model", "iterate"]


def test_generated_stream_clamps_only_the_float_classifier_view_and_records_protocol(monkeypatch):
    config = ExperimentConfig(
        dataset="mnist",
        image_size=2,
        image_channels=1,
        num_classes=2,
    )
    raw_samples = torch.tensor(
        [
            [[[-2.0, -1.5], [-2.0, -1.5]]],
            [[[2.0, 1.5], [2.0, 1.5]]],
        ],
        dtype=torch.float32,
    )
    classifier_inputs = []

    class RecordingClassifier(torch.nn.Module):
        def forward(self, images):
            classifier_inputs.append(images.detach().clone())
            labels = images[:, 0, 0, 0].gt(0).long()
            logits = torch.zeros(images.shape[0], 2)
            logits[torch.arange(images.shape[0]), labels] = 9.0
            return logits

    def fake_iter(_model, _variant, _config, request, *, device):
        del device
        yield GeneratedChunk(
            0,
            request.labels,
            _sample_result(raw_samples, request.sampling.steps),
        )

    monkeypatch.setattr(generation_evaluation, "iter_loaded_generation", fake_iter)

    result = generation_evaluation.evaluate_generation(
        load_model=lambda: torch.nn.Identity(),
        variant="velocity_fm",
        config=config,
        metric_names=("conditional_accuracy",),
        classifier=RecordingClassifier(),
        fid_asset_identity=None,
        sampling=SamplingConfig(steps=1),
        generated_population=2,
        batch_size=2,
        fid_batch_size=2,
        device=torch.device("cpu"),
        seed=321,
    )

    assert len(classifier_inputs) == 1
    assert classifier_inputs[0].is_floating_point()
    assert torch.equal(classifier_inputs[0], raw_samples.clamp(-1.0, 1.0))
    assert torch.equal(raw_samples, torch.tensor(
        [
            [[[-2.0, -1.5], [-2.0, -1.5]]],
            [[[2.0, 1.5], [2.0, 1.5]]],
        ]
    ))
    assert result["generated_protocol"]["conditional_classifier_input"] == (
        "raw_generated_sample_float_clamp_-1_1"
    )


@pytest.mark.parametrize(
    ("generated_population", "num_classes"),
    [(1, 2), (3, 2), (9, 10)],
)
def test_generated_population_must_be_class_balanced(
    generated_population: int,
    num_classes: int,
) -> None:
    config = ExperimentConfig(dataset="cifar10", image_size=32, num_classes=num_classes)
    with pytest.raises(ValueError, match="generated_population"):
        generation_evaluation.evaluate_generation(
            load_model=lambda: torch.nn.Identity(),
            variant="velocity_fm",
            config=config,
            metric_names=("fid",),
            classifier=None,
            fid_asset_identity=_fid_identity("cifar10"),
            sampling=SamplingConfig(steps=1),
            generated_population=generated_population,
            batch_size=2,
            fid_batch_size=2,
            device=torch.device("cpu"),
            seed=321,
        )
