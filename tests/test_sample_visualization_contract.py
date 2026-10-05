from __future__ import annotations

import math
from pathlib import Path
from types import SimpleNamespace

import pytest
import torch

from pfm.core.sampling_config import SamplingConfig
from PIL import Image

from pfm.artifacts.sample_visualization import SampleVisualizationWriter
from pfm.core.config import ExperimentConfig
from pfm.core.sample_config import load_sample_visualization_config


PROJECT_ROOT = Path(__file__).resolve().parents[1]


@pytest.mark.parametrize(
    ("filename", "steps", "class_count", "per_class"),
    [
        ("mnist_visualization.yaml", 50, 10, 8),
        ("cifar10_visualization.yaml", 50, 10, 8),
        ("imagenet256_visualization.yaml", 250, 1000, 1),
    ],
)
def test_tracked_sample_profiles_are_closed(
    filename: str,
    steps: int,
    class_count: int,
    per_class: int,
) -> None:
    path = PROJECT_ROOT / "configs" / "sampling" / filename
    config = load_sample_visualization_config(path)
    assert config.sampling.steps == steps
    assert config.batch_size == 256
    assert config.class_start == 0
    assert config.class_count == class_count
    assert config.num_per_class == per_class
    assert config.seed == 321
    assert config.uncertainty_capture_t == 0.495


def test_sample_config_rejects_retired_ema_selector(tmp_path: Path) -> None:
    source = PROJECT_ROOT / "configs" / "sampling" / "mnist_visualization.yaml"
    path = tmp_path / "sample.yaml"
    path.write_text(
        source.read_text(encoding="utf-8") + "prefer_ema: true\n",
        encoding="utf-8",
    )

    with pytest.raises(ValueError, match="unknown=\\['prefer_ema'\\]"):
        load_sample_visualization_config(path)


@pytest.mark.parametrize("value", [0, -1, True, 2.5, "8"])
def test_sample_batch_size_is_a_positive_integer(tmp_path: Path, value) -> None:
    import yaml
    source = PROJECT_ROOT / "configs/sampling/mnist_visualization.yaml"
    raw = yaml.safe_load(source.read_text())
    raw["batch_size"] = value
    path = tmp_path / "sample.yaml"
    path.write_text(yaml.safe_dump(raw))
    with pytest.raises(ValueError, match="batch_size must be a positive integer"):
        load_sample_visualization_config(path)


def test_sample_requires_explicit_batch_size_without_checkpoint_fallback(tmp_path: Path) -> None:
    source = PROJECT_ROOT / "configs/sampling/mnist_visualization.yaml"
    path = tmp_path / "sample.yaml"
    path.write_text(source.read_text().replace("batch_size: 256\n", ""))
    with pytest.raises(ValueError, match="missing=\\['batch_size'\\]"):
        load_sample_visualization_config(path)


def test_pfm_sample_writer_publishes_fixed_scale_and_lower_bound_contrast_pages(
    tmp_path: Path,
) -> None:
    config = ExperimentConfig(num_classes=2, logvar_min=-1.0, logvar_max=1.0)
    labels = torch.tensor([0, 1])
    samples = torch.zeros(2, 1, 28, 28)
    velocity = torch.ones_like(samples)
    endpoint = 0.25 * velocity
    writer = SampleVisualizationWriter(
        artifact_root=tmp_path,
        config=config,
        variant="pfm_full",
        num_per_class=1,
        class_start=0,
        class_count=2,
        expected_count=2,
        sampling=SamplingConfig(steps=50),
        capture_t=0.5,

    )
    writer.write_chunk(
        labels,
        SimpleNamespace(
            samples=samples,
            actual_nfe=50,
            actual_vjp=0,
            elapsed_seconds=0.1,
            numerical_summary={},
            sampling=SamplingConfig(steps=50),
            variance_available=True,
            velocity_var=velocity,
            endpoint_var=endpoint,
            uncertainty_source_step=25,
            uncertainty_source_t=0.5,
            uncertainty_source_state="pre_update",


        ),
    )
    result = writer.finalize()

    scales = result["presentation"]["variance_scales"]
    assert scales["velocity"] == pytest.approx([math.exp(-1.0), math.exp(1.0)])
    assert scales["endpoint"] == pytest.approx(
        [0.25 * math.exp(-1.0), 0.25 * math.exp(1.0)]
    )
    assert scales["endpoint_scale_at_t_vis"] == 0.25
    assert result["presentation"]["variance_position_scale"] == {
        "definition": "(variance - lower_bound) / (upper_bound - lower_bound)",
        "display_transform": "log10(max(position, 1e-8))",
        "display_range": [-8.0, -2.0],
    }
    assert result["sampling"]["uncertainty_capture_observed_t"] == 0.5
    assert sorted(path.relative_to(tmp_path).as_posix() for path in tmp_path.rglob("*.png")) == [
        "presentation/endpoint_variance/classes_0000_0001.png",
        "presentation/endpoint_variance_position/classes_0000_0001.png",
        "presentation/samples/classes_0000_0001.png",
        "presentation/velocity_variance/classes_0000_0001.png",
        "presentation/velocity_variance_position/classes_0000_0001.png",
    ]
    assert not list(tmp_path.rglob("*.pt"))
    assert not list(tmp_path.rglob("*.pdf"))
    assert not list(tmp_path.rglob("manifest.json"))


def test_large_class_sample_page_uses_compact_rgb_layout(tmp_path: Path) -> None:
    class_count = 12
    config = ExperimentConfig(
        num_classes=class_count,
        image_channels=3,
        image_size=8,
    )
    writer = SampleVisualizationWriter(
        artifact_root=tmp_path,
        config=config,
        variant="velocity_fm",
        num_per_class=1,
        class_start=0,
        class_count=class_count,
        expected_count=class_count,
        sampling=SamplingConfig(steps=10),
        capture_t=0.5,

    )
    writer.write_chunk(
        torch.arange(class_count),
        SimpleNamespace(
            samples=torch.zeros(class_count, 3, 8, 8),
            actual_nfe=10,
            actual_vjp=0,
            elapsed_seconds=0.1,
            numerical_summary={},
            sampling=SamplingConfig(steps=10),
            variance_available=False,


        ),
    )
    writer.finalize()

    output = tmp_path / "presentation/samples/classes_0000_0011.png"
    with Image.open(output) as image:
        assert image.width < 3000
        assert image.height < 5000


def test_sde_writer_keeps_posterior_and_diffusion_map_semantics_separate(tmp_path):
    from pfm.sampling.samplers import sample

    class Posterior(torch.nn.Module):
        def forward(self, x, t, labels):
            return {"delta": 0.05 * x, "logvar": 0.2 * x}

    config = ExperimentConfig(image_channels=1, image_size=4, num_classes=2, logvar_min=-3.0, logvar_max=1.0)
    sampling = SamplingConfig(solver="sde", steps=4, terminal_time=0.8, diffusion_scale=0.1, derivative_probes=2)
    labels = torch.tensor([0, 1])
    result = sample(Posterior(), "pfm_delta", labels, config, sampling, torch.device("cpu"), uncertainty_capture_t=0.4)
    writer = SampleVisualizationWriter(
        artifact_root=tmp_path, config=config, variant="pfm_delta", num_per_class=1,
        class_start=0, class_count=2, expected_count=2, sampling=sampling, capture_t=0.4,
    )
    writer.write_chunk(labels, result)
    metadata = writer.finalize()
    torch.testing.assert_close(result.diffusion_rate, 0.1 * result.velocity_var)
    torch.testing.assert_close(result.noise_covariance, 0.2 * result.diffusion_rate)
    assert metadata["sampling"]["actual_vjp"] == 8
    assert metadata["sampling"]["protocol"] == sampling.to_dict()
    assert metadata["presentation"]["diffusion_scales"]["diffusion_rate"] == pytest.approx([0.1 * math.exp(-3), 0.1 * math.exp(1)])
    pages = metadata["presentation"]["pages"]
    for key in ("velocity_variance", "endpoint_variance", "diffusion_rate", "brownian_step_covariance"):
        assert (tmp_path / pages[0][key]).is_file()
