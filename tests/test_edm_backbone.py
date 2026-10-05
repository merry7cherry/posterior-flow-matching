import pytest
import torch

import pfm.models.factory as models
from pfm.core.config import ExperimentConfig, load_config


PROJECT_ROOT = __import__("pathlib").Path(__file__).resolve().parents[1]


def _tiny_edm_config(tmp_path):
    del tmp_path
    return ExperimentConfig(
        dataset="cifar10",
        architecture="edm_ddpmpp",
        image_channels=3,
        image_size=8,
        num_classes=10,
        batch_size=2,
        num_workers=0,
        edm_model_channels=8,
        edm_channel_mult=(1, 2),
        edm_num_blocks=1,
        edm_norm_groups=8,
        edm_attn_resolutions=(),
        edm_dropout=0.0,
    )


def test_cifar10_edm_config_loads_ddpmpp_defaults():
    config = load_config(PROJECT_ROOT / "configs" / "cifar10_edm_ddpmpp.yaml")

    assert config.dataset == "cifar10"
    assert config.architecture == "edm_ddpmpp"
    assert config.image_channels == 3
    assert config.image_size == 32
    assert config.num_classes == 10
    assert config.edm_model_channels == 128
    assert config.edm_channel_mult == (1, 2, 2, 2)
    assert config.edm_num_blocks == 4
    assert config.edm_attn_resolutions == (16,)
    assert config.edm_label_dropout == 0.0


def test_build_model_dispatches_edm_ddpmpp_from_registry(tmp_path):
    config = _tiny_edm_config(tmp_path)

    model = models.build_model("velocity_fm", config, device=torch.device("cpu"))

    assert "edm_ddpmpp" in models.VALID_ARCHITECTURES
    assert model.__class__.__name__ == "EDMDDPMppModel"


def test_edm_rejects_norm_groups_that_do_not_divide_channels() -> None:
    config = ExperimentConfig(
        dataset="cifar10",
        architecture="edm_ddpmpp",
        image_channels=3,
        image_size=32,
        num_classes=10,
        edm_model_channels=30,
        edm_channel_mult=(1,),
        edm_num_blocks=1,
        edm_norm_groups=8,
    )

    with pytest.raises(ValueError, match="GroupNorm requested_groups=8 must divide num_channels=30"):
        models.build_model("velocity_fm", config)


@pytest.mark.parametrize("norm_groups", ["8", 8.5, True])
def test_edm_rejects_non_integer_norm_groups(norm_groups) -> None:
    config = ExperimentConfig(
        dataset="cifar10",
        architecture="edm_ddpmpp",
        image_channels=3,
        image_size=32,
        num_classes=10,
        edm_model_channels=16,
        edm_channel_mult=(1,),
        edm_num_blocks=1,
        edm_norm_groups=norm_groups,
        edm_attn_resolutions=(),
    )

    with pytest.raises(ValueError, match="GroupNorm requested_groups must be an integer"):
        models.build_model("velocity_fm", config)


def test_edm_ddpmpp_outputs_match_variant_contracts(tmp_path):
    config = _tiny_edm_config(tmp_path)
    removed_factor_key = "cov_" + "factor"
    xt = torch.randn(2, config.image_channels, config.image_size, config.image_size)
    t = torch.tensor([0.1, 0.9])
    labels = torch.tensor([1, 7])

    expected_keys = {
        "pfm_full": {"mu", "logvar"},
        "pfm_delta": {"mu", "delta", "logvar"},
        "endpoint_fm": {"x1_hat"},
        "velocity_fm": {"velocity"},
    }

    for variant, keys in expected_keys.items():
        model = models.build_model(variant, config)
        outputs = model(xt, t, labels)
        assert set(outputs) == keys
        assert removed_factor_key not in outputs
        if hasattr(model, "output_heads"):
            assert removed_factor_key not in model.output_heads
        for value in outputs.values():
            if value is not None:
                assert value.shape == xt.shape


def test_edm_ddpmpp_rejects_retired_self_conditioning(tmp_path):
    config = _tiny_edm_config(tmp_path)
    config.self_conditioning_mode = "mean"
    with pytest.raises(ValueError, match="self-conditioning runtime is retired"):
        models.build_model("pfm_full", config)

def test_edm_pfm_delta_zero_initialization_returns_current_state_as_endpoint_mean(tmp_path):
    config = _tiny_edm_config(tmp_path)
    model = models.build_model("pfm_delta", config).eval()
    xt = torch.randn(2, config.image_channels, config.image_size, config.image_size)
    t = torch.tensor([0.2, 0.8])
    labels = torch.tensor([2, 3])

    outputs = model(xt, t, labels)

    assert torch.allclose(outputs["mu"], xt)
