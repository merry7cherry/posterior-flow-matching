import pytest
import torch
from pathlib import Path

from pfm.core.config import ExperimentConfig, load_config
from pfm.probability.targets import velocity_target
from pfm.sampling.time_sampling import logit_normal_timestep_sample, sample_train_timesteps


def test_uniform_time_sampling_respects_eps_bounds_and_training_shape():
    cfg = ExperimentConfig(eps_t=0.1)

    t = sample_train_timesteps(cfg, num_samples=512, device=torch.device("cpu"))

    assert t.shape == (512, 1, 1, 1)
    assert torch.all(t >= 0.1)
    assert torch.all(t <= 0.9)


def test_uniform_time_sampling_is_method_neutral_and_uses_eps_bounds():
    cfg = ExperimentConfig(eps_t=0.2)

    torch.manual_seed(0)
    sampled = sample_train_timesteps(cfg, num_samples=4096, device=torch.device("cpu"))

    assert torch.all(sampled >= 0.2)
    assert torch.all(sampled <= 0.8)


def test_logit_normal_timestep_sample_matches_requested_formula():
    t = logit_normal_timestep_sample(
        P_mean=0.0,
        P_std=0.0,
        num_samples=4,
        device=torch.device("cpu"),
    )

    assert t.shape == (4,)
    assert torch.allclose(t, torch.full((4,), 0.5))


def test_logit_normal_train_timesteps_use_configured_distribution():
    cfg = ExperimentConfig(
        time_sampling="logit_normal",
        logit_normal_p_mean=0.0,
        logit_normal_p_std=0.0,
    )

    t = sample_train_timesteps(cfg, num_samples=3, device=torch.device("cpu"))

    assert t.shape == (3, 1, 1, 1)
    assert torch.allclose(t, torch.full((3, 1, 1, 1), 0.5))


def test_logit_normal_time_sampling_always_clamps_to_common_bounds():
    cfg = ExperimentConfig(
        eps_t=0.1,
        time_sampling="logit_normal",
        logit_normal_p_mean=100.0,
        logit_normal_p_std=0.0,
    )

    sampled = sample_train_timesteps(cfg, num_samples=2, device=torch.device("cpu"))

    assert torch.allclose(sampled, torch.full((2, 1, 1, 1), 0.9))


def test_unknown_time_sampling_mode_fails_explicitly():
    cfg = ExperimentConfig(time_sampling="not_a_sampler")

    with pytest.raises(ValueError, match="Unknown time_sampling"):
        sample_train_timesteps(cfg, num_samples=1, device=torch.device("cpu"))


def test_train_time_sampling_has_no_method_specific_input_and_is_always_bounded():
    cfg = ExperimentConfig(eps_t=0.2)

    torch.manual_seed(17)
    sampled = sample_train_timesteps(cfg, num_samples=4096, device=torch.device("cpu"))

    assert sampled.shape == (4096, 1, 1, 1)
    assert torch.all(sampled >= 0.2)
    assert torch.all(sampled <= 0.8)


@pytest.mark.parametrize("eps_t", [1e-8, 1e-100, 0.5 - 1e-9])
def test_config_rejects_unrepresentable_training_time_interval(eps_t):
    with pytest.raises(ValueError, match="FP32 endpoints"):
        load_config(Path(__file__).resolve().parents[1] / "configs/mnist_unet.yaml",
                    overrides={"eps_t": eps_t})


@pytest.mark.parametrize("mean", [-100.0, 100.0])
def test_small_representable_time_margin_keeps_extreme_samples_and_targets_finite(mean):
    config = load_config(Path(__file__).resolve().parents[1] / "configs/mnist_unet.yaml",
                         overrides={"eps_t": 2**-24, "time_sampling": "logit_normal",
                                    "logit_normal_p_mean": mean})
    t = sample_train_timesteps(config, 64, torch.device("cpu"))
    assert t.dtype == torch.float32
    assert torch.all((t > 0) & (t < 1))
    assert torch.isfinite(velocity_target(torch.ones_like(t), t, t)).all()


@pytest.mark.parametrize("mode", ["uniform", "logit_normal"])
def test_training_time_precision_does_not_follow_global_default(mode):
    previous = torch.get_default_dtype()
    try:
        torch.set_default_dtype(torch.float64)
        t = sample_train_timesteps(ExperimentConfig(time_sampling=mode), 8, torch.device("cpu"))
        assert t.dtype == torch.float32
    finally:
        torch.set_default_dtype(previous)
