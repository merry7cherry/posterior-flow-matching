import torch

from pfm.core.config import ExperimentConfig
from pfm.models.factory import ConditionalUNet
from pfm.sampling.samplers import sample
from pfm.core.sampling_config import SamplingConfig


def test_sampling_layer_is_explicit_package():
    import pfm.sampling as sampling

    assert sampling.__name__ == "pfm.sampling"


def test_sampling_outputs_and_variance_shapes_follow_variant_contracts():
    cfg = ExperimentConfig(time_proj_dim=16, class_proj_dim=16, unet_base_channels=8, image_size=8)
    labels = torch.tensor([0, 1, 2])

    for variant in ["pfm_full", "pfm_delta", "endpoint_fm", "velocity_fm"]:
        torch.manual_seed(0)
        model = ConditionalUNet(variant=variant, config=cfg)
        result = sample(model, variant, labels, cfg, sampling=SamplingConfig(solver="ode", steps=2), device=torch.device("cpu"))

        expected_shape = (labels.shape[0], cfg.image_channels, cfg.image_size, cfg.image_size)
        assert result.samples.shape == expected_shape
        if variant in {"pfm_full", "pfm_delta"}:
            assert result.endpoint_var is not None
            assert result.velocity_var is not None
            assert result.endpoint_var.shape == expected_shape
            assert result.velocity_var.shape == expected_shape
        else:
            assert result.endpoint_var is None
            assert result.velocity_var is None


def test_sampling_can_reuse_fixed_initial_noise_without_mutating_it():
    cfg = ExperimentConfig(time_proj_dim=16, class_proj_dim=16, unet_base_channels=8, image_size=8)
    labels = torch.tensor([0, 1, 2])
    torch.manual_seed(0)
    model = ConditionalUNet(variant="velocity_fm", config=cfg)
    initial_noise = torch.randn(labels.shape[0], cfg.image_channels, cfg.image_size, cfg.image_size)
    saved_noise = initial_noise.clone()

    result = sample(
        model,
        "velocity_fm",
        labels,
        cfg,
        sampling=SamplingConfig(solver="ode", steps=2),
        device=torch.device("cpu"),
        initial_noise=initial_noise,
    )

    assert result.samples.shape == initial_noise.shape
    assert torch.allclose(initial_noise, saved_noise)


def test_pfm_delta_sampling_uses_delta_velocity_directly():
    cfg = ExperimentConfig(image_channels=1, image_size=2, logvar_min=-1.0, logvar_max=1.0)
    labels = torch.tensor([0])
    initial_noise = torch.zeros(1, cfg.image_channels, cfg.image_size, cfg.image_size)

    class DeltaOnlyModel(torch.nn.Module):
        def eval(self):
            return self

        def forward(self, x, t, labels):
            del t, labels
            return {
                "mu": torch.zeros_like(x),
                "delta": torch.ones_like(x),
                "logvar": torch.zeros_like(x),
            }

    result = sample(
        DeltaOnlyModel(),
        "pfm_delta",
        labels,
        cfg,
        sampling=SamplingConfig(solver="ode", steps=1),
        device=torch.device("cpu"),
        initial_noise=initial_noise,
    )

    assert torch.allclose(result.samples, torch.ones_like(initial_noise))


def test_pfm_delta_sampling_variance_uses_diagonal_logvar():
    cfg = ExperimentConfig(image_channels=1, image_size=2, logvar_min=-1.0, logvar_max=1.0)
    labels = torch.tensor([0])
    initial_noise = torch.zeros(1, cfg.image_channels, cfg.image_size, cfg.image_size)

    class DiagonalDeltaModel(torch.nn.Module):
        def eval(self):
            return self

        def forward(self, x, t, labels):
            del t, labels
            return {
                "mu": x,
                "delta": torch.zeros_like(x),
                "logvar": torch.zeros_like(x),
            }

    result = sample(
        DiagonalDeltaModel(),
        "pfm_delta",
        labels,
        cfg,
        sampling=SamplingConfig(solver="ode", steps=1),
        device=torch.device("cpu"),
        initial_noise=initial_noise,
    )

    assert torch.allclose(result.velocity_var, torch.ones_like(initial_noise))
    assert torch.allclose(result.endpoint_var, torch.ones_like(initial_noise))


def test_bfloat16_sampling_uncertainty_uses_fp32_probability_owner():
    cfg = ExperimentConfig(
        image_channels=1,
        image_size=2,
        logvar_min=-5.0,
        logvar_max=2.0,
    )
    labels = torch.tensor([0])
    initial_noise = torch.zeros(
        1,
        cfg.image_channels,
        cfg.image_size,
        cfg.image_size,
        dtype=torch.bfloat16,
    )

    class BFloat16DeltaModel(torch.nn.Module):
        def eval(self):
            return self

        def forward(self, x, t, labels):
            del t, labels
            return {
                "mu": x,
                "delta": torch.zeros_like(x),
                "logvar": torch.full_like(x, 7.0),
            }

    result = sample(
        BFloat16DeltaModel(),
        "pfm_delta",
        labels,
        cfg,
        sampling=SamplingConfig(solver="ode", steps=2),
        device=torch.device("cpu"),
        initial_noise=initial_noise,
        uncertainty_capture_t=0.5,
    )

    assert result.velocity_var is not None
    assert result.endpoint_var is not None
    assert result.velocity_var.dtype == torch.float32
    assert result.endpoint_var.dtype == torch.float32
    assert torch.all(result.velocity_var < torch.exp(torch.tensor(cfg.logvar_max)))


def test_sample_ode_records_euler_actual_nfe_from_model_calls():
    cfg = ExperimentConfig(image_channels=1, image_size=2)
    labels = torch.tensor([0])
    initial_noise = torch.zeros(1, cfg.image_channels, cfg.image_size, cfg.image_size)

    class ConstantVelocityModel(torch.nn.Module):
        def __init__(self) -> None:
            super().__init__()
            self.calls = 0

        def eval(self):
            return self

        def forward(self, x, t, labels):
            del t, labels
            self.calls += 1
            return {"velocity": torch.ones_like(x)}

    model = ConstantVelocityModel()
    result = sample(
        model,
        "velocity_fm",
        labels,
        cfg,
        sampling=SamplingConfig(solver="ode", steps=3),
        device=torch.device("cpu"),
        initial_noise=initial_noise,
    )

    assert result.sampling_steps == 3
    assert result.actual_nfe == 3
    assert model.calls == 3
    assert not hasattr(result, "sampling_" + "method")
    assert torch.allclose(result.samples, torch.ones_like(initial_noise))


def test_sampler_applies_noise_scale_when_initial_noise_is_not_supplied():
    cfg = ExperimentConfig(image_channels=1, image_size=2, noise_scale=2.0)
    labels = torch.tensor([0])

    class ZeroVelocityModel(torch.nn.Module):
        def eval(self):
            return self

        def forward(self, x, t, labels):
            del t, labels
            return {"velocity": torch.zeros_like(x)}

    generator = torch.Generator(device="cpu").manual_seed(123)
    result = sample(
        ZeroVelocityModel(),
        "velocity_fm",
        labels,
        cfg,
        sampling=SamplingConfig(solver="ode", steps=1),
        device=torch.device("cpu"),
        generator=generator,
    )

    generator = torch.Generator(device="cpu").manual_seed(123)
    expected = 2.0 * torch.randn(1, 1, 2, 2, generator=generator)
    assert torch.allclose(result.samples, expected)
