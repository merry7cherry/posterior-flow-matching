import pytest
import torch

import pfm.models.factory as models
from pfm.core.config import ExperimentConfig
from pfm.models.jit_backbone import (
    Attention,
    JIT_UPSTREAM_COMMIT,
    JiTBackbone,
    RMSNorm,
    VisionRotaryEmbeddingFast,
    _upstream_scaled_dot_product_attention,
)
from pfm.models.jit_config import JIT_MODEL_SPECS, JiTModelSpec


def _tiny_jit_config(**overrides):
    config = ExperimentConfig(
        dataset="imagenet",
        architecture="jit_transformer",
        jit_model="JiT-Tiny/4",
        image_channels=3,
        image_size=8,
        num_classes=1000,
        batch_size=2,
        num_workers=0,
        jit_compile=False,
    )
    for key, value in overrides.items():
        setattr(config, key, value)
    return config


def test_build_model_dispatches_jit_transformer_from_registry(tiny_jit_spec):
    config = _tiny_jit_config()

    model = models.build_model("velocity_fm", config, device=torch.device("cpu"))

    assert "jit_transformer" in models.VALID_ARCHITECTURES
    assert model.__class__.__name__ == "JiTPFMModel"


def test_jit_attention_uses_upstream_qk_norm_and_qkv_projection():
    attention = Attention(dim=32, num_heads=4, attn_dropout=0.0, proj_dropout=0.0)
    per_head = torch.randn(2, 4, 5, 8)

    normalized_query = attention.q_norm(per_head)
    normalized_key = attention.k_norm(per_head)

    assert JIT_UPSTREAM_COMMIT == "cbc743a2ada5e9762697da2c83f8c4f8379e8c17"
    assert isinstance(attention.q_norm, RMSNorm)
    assert isinstance(attention.k_norm, RMSNorm)
    assert attention.qkv.in_features == 32
    assert attention.qkv.out_features == 96
    assert attention.proj.in_features == attention.proj.out_features == 32
    expected_rms = torch.ones(2, 4, 5)
    assert torch.allclose(normalized_query.float().square().mean(dim=-1), expected_rms, atol=2e-5)
    assert torch.allclose(normalized_key.float().square().mean(dim=-1), expected_rms, atol=2e-5)


def test_jit_vision_rope_preserves_shape_and_pads_in_context_tokens():
    rope = VisionRotaryEmbeddingFast(dim=4, pt_seq_len=2, num_cls_token=3)
    tokens = torch.randn(2, 4, 7, 8)

    rotated = rope(tokens)

    assert rope.freqs_cos.shape == (7, 8)
    assert rope.freqs_sin.shape == (7, 8)
    assert rotated.shape == tokens.shape
    assert torch.equal(rotated[:, :, :3], tokens[:, :, :3])
    assert not torch.allclose(rotated[:, :, -1], tokens[:, :, -1])


def test_jit_attention_disables_attention_dropout_in_eval():
    attention = Attention(dim=32, num_heads=4, attn_dropout=0.75, proj_dropout=0.0).eval()
    rope = VisionRotaryEmbeddingFast(dim=4, pt_seq_len=2)
    tokens = torch.randn(2, 4, 32)

    first = attention(tokens, rope)
    second = attention(tokens, rope)

    assert torch.equal(first, second)


def _reference_scaled_dot_product_attention(
    query: torch.Tensor,
    key: torch.Tensor,
    value: torch.Tensor,
) -> torch.Tensor:
    scale = query.shape[-1] ** -0.5
    weights = torch.softmax(torch.matmul(query, key.transpose(-2, -1)) * scale, dim=-1)
    return torch.matmul(weights, value)




def test_jit_upstream_fp32_qk_matches_pinned_reference_forward_and_gradients():
    generator = torch.Generator().manual_seed(19)
    inputs = [
        torch.randn(2, 2, 3, 4, dtype=torch.float32, generator=generator, requires_grad=True)
        for _ in range(3)
    ]
    references = [value.detach().clone().requires_grad_() for value in inputs]

    actual = _upstream_scaled_dot_product_attention(*inputs, dropout_p=0.0)
    expected = _reference_scaled_dot_product_attention(*references)
    upstream_gradient = torch.randn(actual.shape, dtype=actual.dtype, generator=generator)
    actual_gradients = torch.autograd.grad(actual, inputs, upstream_gradient)
    expected_gradients = torch.autograd.grad(expected, references, upstream_gradient)

    assert torch.allclose(actual, expected, atol=1e-6, rtol=1e-5)
    for actual_gradient, expected_gradient in zip(actual_gradients, expected_gradients, strict=True):
        assert torch.allclose(actual_gradient, expected_gradient, atol=1e-6, rtol=1e-5)


def test_jit_attention_propagates_train_and_eval_dropout(monkeypatch):
    observed_dropout = []

    def fake_attention(query, key, value, dropout_p):
        observed_dropout.append(dropout_p)
        return value

    monkeypatch.setattr("pfm.models.jit_backbone._upstream_scaled_dot_product_attention", fake_attention)
    attention = Attention(dim=32, num_heads=4, attn_dropout=0.375, proj_dropout=0.0)
    rope = VisionRotaryEmbeddingFast(dim=4, pt_seq_len=2)
    tokens = torch.randn(2, 4, 32)

    attention.train()
    attention(tokens, rope)
    attention.eval()
    attention(tokens, rope)

    assert observed_dropout == [0.375, 0.0]


def test_jit_upstream_attention_jvp_vjp_agree():
    generator = torch.Generator().manual_seed(23)
    inputs = tuple(torch.randn(1, 2, 3, 4, generator=generator) for _ in range(3))
    tangents = tuple(torch.randn(value.shape, generator=generator) for value in inputs)
    fn = lambda *values: _upstream_scaled_dot_product_attention(*values, dropout_p=0.0)
    output, jvp = torch.func.jvp(fn, inputs, tangents)
    cotangent = torch.randn(output.shape, generator=generator)
    _, pullback = torch.func.vjp(fn, *inputs)
    vjps = pullback(cotangent)
    lhs = (jvp * cotangent).sum()
    rhs = sum((tangent * vjp).sum() for tangent, vjp in zip(tangents, vjps, strict=True))
    torch.testing.assert_close(lhs, rhs, atol=2e-6, rtol=2e-5)

def test_jit_upstream_attention_bfloat16_autocast_has_finite_backward():
    generator = torch.Generator().manual_seed(29)
    inputs = tuple(
        torch.randn(
            1,
            2,
            4,
            8,
            dtype=torch.bfloat16,
            generator=generator,
            requires_grad=True,
        )
        for _ in range(3)
    )

    with torch.amp.autocast("cpu", dtype=torch.bfloat16):
        output = _upstream_scaled_dot_product_attention(*inputs, dropout_p=0.0)
    output.float().square().mean().backward()

    assert output.dtype == torch.bfloat16
    assert torch.all(torch.isfinite(output))
    for value in inputs:
        assert value.grad is not None
        assert value.grad.dtype == torch.bfloat16
        assert torch.all(torch.isfinite(value.grad))


def test_jit_upstream_attention_is_captured_by_fullgraph_compile():
    generator = torch.Generator().manual_seed(31)
    inputs = tuple(
        torch.randn(1, 2, 4, 8, generator=generator, requires_grad=True)
        for _ in range(3)
    )
    compiled = torch.compile(
        lambda query, key, value: _upstream_scaled_dot_product_attention(
            query,
            key,
            value,
            dropout_p=0.0,
        ),
        backend="eager",
        fullgraph=True,
    )

    output = compiled(*inputs)
    output.square().mean().backward()

    assert output.shape == inputs[0].shape
    assert torch.all(torch.isfinite(output))
    assert all(value.grad is not None for value in inputs)
    assert all(torch.all(torch.isfinite(value.grad)) for value in inputs)


def test_jit_backbone_switches_to_in_context_rope_and_returns_image_tokens(monkeypatch):
    spec = JiTModelSpec("JiT-TinyContext/4", 4, 32, 2, 4, 16, 2, 1, (8,))
    monkeypatch.setitem(JIT_MODEL_SPECS, spec.name, spec)
    config = _tiny_jit_config(jit_model=spec.name, jit_compile=False)
    backbone = JiTBackbone(config)

    tokens, condition = backbone.forward_tokens(
        torch.randn(2, 3, 8, 8),
        torch.tensor([0.1, 0.9]),
        torch.tensor([1, 7]),
    )

    assert backbone.feat_rope.freqs_cos.shape == (4, 8)
    assert backbone.feat_rope_incontext.freqs_cos.shape == (6, 8)
    assert tokens.shape == (2, 4, 32)
    assert condition.shape == (2, 32)


def test_jit_transformer_outputs_match_variant_contracts(tiny_jit_spec):
    config = _tiny_jit_config()
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


def test_jit_transformer_rejects_retired_self_conditioning(tiny_jit_spec):
    config = _tiny_jit_config(self_conditioning_mode="mean")
    with pytest.raises(ValueError, match="self-conditioning runtime is retired"):
        models.build_model("pfm_full", config)


def test_jit_transformer_rejects_retired_sdpa(tiny_jit_spec):
    config = _tiny_jit_config(jit_attention_implementation="sdpa")
    with pytest.raises(ValueError, match="SDPA is retired"):
        models.build_model("pfm_full", config)

def test_compile_module_rejects_missing_torch_compile(monkeypatch):
    import pfm.models.jit_backbone as jit_backbone

    monkeypatch.delattr(jit_backbone.torch, "compile", raising=False)
    with pytest.raises(RuntimeError, match="jit_compile=True requires torch.compile"):
        jit_backbone.compile_module(torch.nn.Identity())


def test_jit_transformer_pfm_delta_zero_initialization_returns_current_state_as_endpoint_mean(
    tiny_jit_spec,
):
    config = _tiny_jit_config()
    model = models.build_model("pfm_delta", config).eval()
    xt = torch.randn(2, config.image_channels, config.image_size, config.image_size)
    t = torch.tensor([0.2, 0.8])

    outputs = model(xt, t, torch.tensor([2, 3]))

    assert torch.allclose(outputs["mu"], xt)


def test_jit_execution_switch_preserves_loaded_parameters_and_wrapper_layout(tiny_jit_spec):
    config = _tiny_jit_config(jit_compile=True)
    model = models.build_model("pfm_delta", config)
    original_state = dict(model.state_dict())
    parameter_ids = {id(value) for value in model.parameters()}
    assert any("._orig_mod." in key for key in original_state)

    models.configure_jit_execution(model, compile=False)
    assert {id(value) for value in model.parameters()} == parameter_ids
    assert not any("._orig_mod." in key for key in model.state_dict())
    models.configure_jit_execution(model, compile=True)
    assert {id(value) for value in model.parameters()} == parameter_ids
    restored_state = model.state_dict()
    assert original_state.keys() == restored_state.keys()
    for key in original_state:
        torch.testing.assert_close(restored_state[key], original_state[key], rtol=0, atol=0)
    wrapper_ids = [id(block) for block in model.backbone.blocks]
    models.configure_jit_execution(model, compile=True)
    assert [id(block) for block in model.backbone.blocks] == wrapper_ids
