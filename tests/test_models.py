import pytest

from pfm.core.config import ExperimentConfig
from pfm.core.variant_metadata import (
    SUPPORTED_VARIANTS,
    requires_learned_variance,
    variant_capabilities,
)
from pfm.models.embeddings import DiffusionTimestepEmbedding
import pfm.models.factory as models
from pfm.models.factory import ConditionalUNet
from pfm.models.variants import VariantAdapter
from pfm.probability.logvar import bound_normalized_logvar


def test_diffusion_timestep_embedding_uses_scaled_time_signal():
    import torch

    embedding = DiffusionTimestepEmbedding(dim=128, scale=1000.0)
    values = embedding(torch.tensor([0.0, 1.0]))

    assert values.shape == (2, 128)
    assert torch.linalg.vector_norm(values[0] - values[1]) > 5.0


def test_conditional_unet_outputs_match_variant_contracts():
    cfg = ExperimentConfig(time_proj_dim=32, class_proj_dim=32, unet_base_channels=8, image_size=8)
    batch_size = 4
    removed_factor_key = "cov_" + "factor"

    import torch

    xt = torch.randn(batch_size, cfg.image_channels, cfg.image_size, cfg.image_size)
    t = torch.linspace(0.1, 0.9, batch_size)
    labels = torch.arange(batch_size) % cfg.num_classes

    expected_keys = {
        "pfm_full": {"mu", "logvar"},
        "pfm_delta": {"mu", "delta", "logvar"},
        "endpoint_fm": {"x1_hat"},
        "velocity_fm": {"velocity"},
    }

    for variant, keys in expected_keys.items():
        model = ConditionalUNet(variant=variant, config=cfg)
        outputs = model(xt, t, labels)
        assert set(outputs.keys()) == keys
        assert removed_factor_key not in outputs
        assert isinstance(model.variant_adapter, VariantAdapter)
        assert removed_factor_key not in model.variant_adapter.heads
        for value in outputs.values():
            if value is not None:
                assert value.shape == xt.shape


def test_unet_rejects_norm_groups_that_do_not_divide_channels() -> None:
    config = ExperimentConfig(
        dataset="mnist",
        architecture="conditional_unet",
        image_channels=1,
        image_size=28,
        num_classes=10,
        unet_base_channels=30,
        unet_channel_mult=(1,),
        unet_num_res_blocks=1,
        unet_norm_groups=8,
    )

    with pytest.raises(ValueError, match="GroupNorm requested_groups=8 must divide num_channels=30"):
        models.build_model("velocity_fm", config)


@pytest.mark.parametrize("norm_groups", ["8", 8.5, True])
def test_unet_rejects_non_integer_norm_groups(norm_groups) -> None:
    config = ExperimentConfig(
        dataset="mnist",
        architecture="conditional_unet",
        image_channels=1,
        image_size=28,
        num_classes=10,
        unet_base_channels=16,
        unet_channel_mult=(1,),
        unet_num_res_blocks=1,
        unet_norm_groups=norm_groups,
    )

    with pytest.raises(ValueError, match="GroupNorm requested_groups must be an integer"):
        models.build_model("velocity_fm", config)


def test_structure_preflight_checks_only_attention_enabled_widths():
    from pfm.models.config_validation import validate_model_configuration

    config = ExperimentConfig(unet_base_channels=8, unet_channel_mult=(1, 2, 3),
                              unet_num_heads=3, unet_attention_resolutions=())
    # Only the 24-channel middle block uses attention; non-attended widths
    # need not divide the attention head count.
    validate_model_configuration(config)
    config.unet_attention_resolutions = (1,)
    with pytest.raises(ValueError, match="channels=8 must be divisible by num_heads=3"):
        validate_model_configuration(config)


def test_variant_capabilities_mark_pfm_variants_as_variance_capable():
    for variant in ("pfm_full", "pfm_delta"):
        capabilities = variant_capabilities(variant)
        assert capabilities.predicts_logvar is True
        assert capabilities.supports_posterior_diagnostics is True
        assert capabilities.supports_generation_variance_maps is True


def test_variant_capabilities_mark_baselines_as_no_variance():
    for variant in ("endpoint_fm", "velocity_fm"):
        capabilities = variant_capabilities(variant)
        assert capabilities.predicts_logvar is False
        assert capabilities.supports_posterior_diagnostics is False
        assert capabilities.supports_generation_variance_maps is False


def test_requires_learned_variance_rejects_baselines():
    with pytest.raises(ValueError, match="requires learned variance"):
        requires_learned_variance("velocity_fm", feature="posterior_calibration")


def test_old_model_head_keys_are_invalid_checkpoint_contract():
    import torch

    cfg = ExperimentConfig(time_proj_dim=32, class_proj_dim=32, unet_base_channels=8, image_size=8)
    model = ConditionalUNet(variant="pfm_full", config=cfg)
    current_state = model.state_dict()
    old_style_state = {
        key.replace("variant_adapter.heads", "output_heads"): value
        for key, value in current_state.items()
    }

    with pytest.raises(RuntimeError):
        model.load_state_dict(old_style_state, strict=True)


@pytest.mark.parametrize("mode", ["mean", "mean_logvar"])
@pytest.mark.parametrize("variant", sorted(SUPPORTED_VARIANTS))
def test_conditional_unet_rejects_retired_self_conditioning(mode, variant):
    config = ExperimentConfig(self_conditioning_mode=mode)
    with pytest.raises(ValueError, match="self-conditioning runtime is retired"):
        ConditionalUNet(variant=variant, config=config)


def test_fixed_variance_variant_is_removed_from_runtime_variants():
    cfg = ExperimentConfig(time_proj_dim=32, class_proj_dim=32, unet_base_channels=8, image_size=8)

    assert "pfm_fixed_var" not in SUPPORTED_VARIANTS
    assert not hasattr(models, "VALID_VARIANTS")
    with pytest.raises(ValueError, match="Unknown variant"):
        ConditionalUNet(variant="pfm_fixed_var", config=cfg)


def test_conditional_unet_uses_convolutional_encoder_decoder_path():
    import torch

    cfg = ExperimentConfig(time_proj_dim=32, class_proj_dim=32, unet_base_channels=8)
    model = ConditionalUNet(variant="velocity_fm", config=cfg)
    conv_layers = [module for module in model.modules() if isinstance(module, torch.nn.Conv2d)]
    upsample_layers = [module for module in model.modules() if isinstance(module, getattr(models, "Upsample"))]

    assert len(conv_layers) >= 8
    assert len(upsample_layers) == 2


def test_conditional_unet_matches_enhanced_mnist_unet_structure():
    cfg = ExperimentConfig(time_proj_dim=32, class_proj_dim=32, unet_base_channels=32)
    model = ConditionalUNet(variant="pfm_full", config=cfg)

    residual_blocks = [module for module in model.modules() if isinstance(module, models.ResidualBlock)]
    attention_blocks = [module for module in model.modules() if isinstance(module, getattr(models, "AttentionBlock"))]
    prediction_heads = [module for module in model.modules() if isinstance(module, getattr(models, "PredictionHead"))]

    assert model.channel_mult == (1, 2, 4)
    assert model.num_res_blocks == 2
    assert model.attention_resolutions == ()
    assert model.model_channels == 32
    assert len(model.input_blocks) == 9
    assert len(model.output_blocks) == 9
    assert len(residual_blocks) == 21
    assert len(attention_blocks) == 1
    assert len(prediction_heads) == 2
    assert all(block.dropout.p == cfg.unet_dropout for block in residual_blocks)


def test_pfm_full_uses_independent_mean_and_logvar_heads():
    cfg = ExperimentConfig(time_proj_dim=32, class_proj_dim=32, unet_base_channels=16)
    model = ConditionalUNet(variant="pfm_full", config=cfg)

    heads = model.variant_adapter.heads
    assert set(heads.keys()) == {"mu", "logvar"}
    assert heads["mu"] is not heads["logvar"]
    assert isinstance(heads["mu"], models.PredictionHead)
    assert isinstance(heads["logvar"], models.PredictionHead)
    assert len(heads["mu"].private_blocks) == 2
    assert len(heads["logvar"].private_blocks) == 2
    assert heads["mu"].output.proj.out_channels == cfg.image_channels
    assert heads["logvar"].output.proj.out_channels == cfg.image_channels


def test_pfm_delta_zero_initialization_returns_current_state_as_endpoint_mean():
    import torch

    cfg = ExperimentConfig(time_proj_dim=32, class_proj_dim=32, unet_base_channels=16)
    model = ConditionalUNet(variant="pfm_delta", config=cfg).eval()
    batch_size = 3
    xt = torch.randn(batch_size, cfg.image_channels, cfg.image_size, cfg.image_size)
    t = torch.tensor([0.1, 0.5, 0.9])
    labels = torch.arange(batch_size) % cfg.num_classes

    outputs = model(xt, t, labels)

    assert torch.allclose(outputs["mu"], xt)
    assert torch.allclose(
        bound_normalized_logvar(outputs["logvar"], cfg),
        torch.full_like(outputs["logvar"], cfg.logvar_init),
        atol=1e-6,
    )
    assert set(outputs.keys()) == {"mu", "delta", "logvar"}
    assert set(model.variant_adapter.heads.keys()) == {"delta", "logvar"}


def test_non_pfm_variants_use_the_shared_prediction_head():
    cfg = ExperimentConfig(time_proj_dim=32, class_proj_dim=32, unet_base_channels=16)

    for variant in ["endpoint_fm", "velocity_fm"]:
        model = ConditionalUNet(variant=variant, config=cfg)
        output_name = "x1_hat" if variant == "endpoint_fm" else "velocity"
        head = model.variant_adapter.heads[output_name]
        assert isinstance(head, models.PredictionHead)
        assert head.output.proj.out_channels == cfg.image_channels


@pytest.mark.parametrize("architecture", ["conditional_unet", "edm_ddpmpp"])
def test_all_learned_main_heads_share_one_architectural_family_and_capacity(architecture):
    if architecture == "conditional_unet":
        cfg = ExperimentConfig(
            architecture=architecture,
            time_proj_dim=16,
            class_proj_dim=16,
            unet_base_channels=8,
            unet_channel_mult=(1,),
            unet_num_res_blocks=1,
            unet_norm_groups=8,
            image_size=8,
        )
    else:
        cfg = ExperimentConfig(
            dataset="cifar10",
            architecture=architecture,
            image_channels=3,
            image_size=8,
            edm_model_channels=8,
            edm_channel_mult=(1,),
            edm_num_blocks=1,
            edm_norm_groups=8,
            edm_attn_resolutions=(),
            edm_dropout=0.0,
        )

    main_head_names = {
        "velocity_fm": "velocity",
        "endpoint_fm": "x1_hat",
        "pfm_full": "mu",
        "pfm_delta": "delta",
    }
    main_heads = {
        variant: models.build_model(variant, cfg).variant_adapter.heads[name]
        for variant, name in main_head_names.items()
    }
    parameter_counts = {
        variant: sum(parameter.numel() for parameter in head.parameters())
        for variant, head in main_heads.items()
    }

    assert {type(head) for head in main_heads.values()} == {models.PredictionHead}
    assert len(set(parameter_counts.values())) == 1

    pfm_model = models.build_model("pfm_full", cfg)
    assert type(pfm_model.variant_adapter.heads["logvar"]) is type(
        pfm_model.variant_adapter.heads["mu"]
    )


def test_reference_unet_zero_initializes_residual_attention_and_output_branches():
    import torch

    cfg = ExperimentConfig(time_proj_dim=32, class_proj_dim=32, unet_base_channels=32)
    model = ConditionalUNet(variant="pfm_full", config=cfg)

    for module in model.modules():
        if isinstance(module, models.ResidualBlock):
            assert torch.count_nonzero(module.conv2.weight) == 0
            assert torch.count_nonzero(module.conv2.bias) == 0
        if isinstance(module, models.AttentionBlock):
            assert torch.count_nonzero(module.proj.weight) == 0
            assert torch.count_nonzero(module.proj.bias) == 0

    heads = model.variant_adapter.heads
    for head in heads.values():
        assert torch.count_nonzero(head.output.proj.weight) == 0
    assert torch.allclose(heads["mu"].output.proj.bias, torch.zeros_like(heads["mu"].output.proj.bias))
    assert torch.allclose(
        bound_normalized_logvar(heads["logvar"].output.proj.bias, cfg),
        torch.full_like(heads["logvar"].output.proj.bias, cfg.logvar_init),
    )


def test_conditional_unet_label_conditioning_reaches_condition_vector():
    import torch

    cfg = ExperimentConfig(time_proj_dim=32, class_proj_dim=32, unet_base_channels=8)
    model = ConditionalUNet(variant="velocity_fm", config=cfg)
    t = torch.full((2,), 0.5)

    time_features, class_features_a = model.embeddings(t, torch.tensor([1, 1]))
    _, class_features_b = model.embeddings(t, torch.tensor([1, 2]))
    condition_a = model.condition_projector(time_features + class_features_a)
    condition_b = model.condition_projector(time_features + class_features_b)

    assert not torch.allclose(condition_a[1], condition_b[1])


def test_conditional_unet_requires_matching_time_and_class_condition_dims():
    import pytest

    cfg = ExperimentConfig(time_proj_dim=32, class_proj_dim=16, unet_base_channels=8)

    with pytest.raises(ValueError, match="time_proj_dim and class_proj_dim"):
        ConditionalUNet(variant="velocity_fm", config=cfg)


def test_build_model_dispatches_conditional_unet_from_registry():
    import torch

    cfg = ExperimentConfig(time_proj_dim=16, class_proj_dim=16, unet_base_channels=8, image_size=8)

    model = models.build_model("velocity_fm", cfg, device=torch.device("cpu"))

    assert isinstance(model, ConditionalUNet)
    assert "conditional_unet" in models.VALID_ARCHITECTURES
