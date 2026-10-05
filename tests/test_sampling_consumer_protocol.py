from dataclasses import asdict
from pathlib import Path

import pytest
import yaml

from pfm.core.config import ExperimentConfig
from pfm.core.evaluation_config import load_metric_evaluation_config
from pfm.core.sample_config import load_sample_visualization_config
from pfm.core.sampling_config import SamplingConfig
from pfm.core.yaml_config import merge_config, parse_overrides


ROOT = Path(__file__).resolve().parents[1]


def test_matched_cifar_consumers_share_one_sampling_protocol():
    protocols = {}
    for solver in ("ode", "sde"):
        metrics = load_metric_evaluation_config(ROOT / f"configs/evaluation/cifar10_edm_{solver}.yaml")
        samples = load_sample_visualization_config(ROOT / f"configs/sampling/cifar10_{solver}_visualization.yaml")
        assert metrics.sampling == samples.sampling
        assert metrics.seed == samples.seed
        protocols[solver] = metrics.sampling
    assert protocols["ode"].terminal_time == protocols["sde"].terminal_time
    assert protocols["ode"].terminal_readout == protocols["sde"].terminal_readout
    assert protocols["ode"].steps == protocols["sde"].steps


def test_nested_sampling_overrides_preserve_unmodified_protocol_and_source():
    path = ROOT / "configs/evaluation/cifar10_edm_sde.yaml"
    original = yaml.safe_load(path.read_text())
    snapshot = yaml.safe_dump(original)
    result = merge_config(original, parse_overrides(["sampling.derivative_probes=8", "sampling.steps=100"]))
    assert result["sampling"]["derivative_probes"] == 8
    assert result["sampling"]["steps"] == 100
    assert result["sampling"]["diffusion_scale"] == original["sampling"]["diffusion_scale"]
    assert yaml.safe_dump(original) == snapshot


@pytest.mark.parametrize("items", [
    ["sampling.steps=10", "sampling.steps=20"],
    ["sampling={}", "sampling.steps=20"],
    ["sampling.steps=20", "sampling={}"],
])
def test_nested_overrides_reject_order_dependent_assignments(items):
    with pytest.raises(ValueError, match="overlapping"):
        parse_overrides(items)


def test_inference_execution_overrides_preserve_training_identity():
    training = ExperimentConfig(architecture="jit_transformer", mixed_precision="bf16", jit_compile=True)
    before = asdict(training)
    execution = SamplingConfig(precision="fp32", compile=False).execution.apply(training)
    assert execution.mixed_precision == "none"
    assert execution.jit_compile is False
    assert asdict(training) == before
    changed = {key for key, value in asdict(execution).items() if value != before[key]}
    assert changed == {"mixed_precision", "jit_compile"}


def test_retired_sample_policy_is_rejected(tmp_path):
    path = ROOT / "configs/sampling/cifar10_visualization.yaml"
    raw = yaml.safe_load(path.read_text())
    raw["self_conditioning_policy"] = "carry"
    target = tmp_path / "sample.yaml"
    target.write_text(yaml.safe_dump(raw))
    with pytest.raises(ValueError, match="self_conditioning_policy"):
        load_sample_visualization_config(target)


def test_sde_generation_service_replays_indexed_inputs_across_chunks():
    import torch
    from pfm.evaluation.generation_service import GenerationRequest, iter_loaded_generation

    class Posterior(torch.nn.Module):
        def forward(self, x, t, labels):
            return {"delta": 0.05 * x, "logvar": 0.2 * x}

    config = ExperimentConfig(image_channels=1, image_size=2, num_classes=2, logvar_min=-3.0, logvar_max=1.0)
    protocol = SamplingConfig(solver="sde", steps=4, terminal_time=0.8, diffusion_scale=0.1, derivative_probes=2)
    labels = torch.tensor([0, 1, 0])
    before = torch.get_rng_state().clone()
    outputs = []
    for batch_size in (1, 3):
        chunks = list(iter_loaded_generation(
            Posterior(), "pfm_delta", config,
            GenerationRequest(labels, protocol, batch_size, 117, uncertainty_capture_t=0.4),
            device=torch.device("cpu"),
        ))
        outputs.append(torch.cat([chunk.sample_result.samples for chunk in chunks]))
        assert all(chunk.sample_result.sampling == protocol for chunk in chunks)
        assert all(chunk.sample_result.actual_nfe == 4 for chunk in chunks)
        assert all(chunk.sample_result.actual_vjp == 8 for chunk in chunks)
        assert all(chunk.sample_result.diffusion_rate is not None for chunk in chunks)
        assert all(chunk.sample_result.numerical_summary["nonfinite_count"] == 0 for chunk in chunks)
    torch.testing.assert_close(outputs[0], outputs[1], rtol=0, atol=0)
    assert torch.equal(torch.get_rng_state(), before)


@pytest.mark.parametrize("stored_compile", [False, True])
@pytest.mark.parametrize("inference_compile", [False, True])
def test_jit_inference_loads_recorded_wrappers_before_switching_execution(
    monkeypatch, tiny_jit_spec, stored_compile, inference_compile,
):
    import torch
    from pfm.models.factory import build_model
    from pfm.evaluation.checkpoint_loading import build_checkpoint_inference_model

    real_compile = torch.compile
    monkeypatch.setattr(torch, "compile", lambda module: real_compile(module, backend="eager"))
    config = ExperimentConfig(
        dataset="imagenet", architecture="jit_transformer", jit_model=tiny_jit_spec.name,
        image_channels=3, image_size=8, num_classes=1000, jit_compile=stored_compile,
        mixed_precision="none",
    )
    source = build_model("pfm_delta", config, torch.device("cpu")).eval()
    with torch.no_grad():
        for name, parameter in source.named_parameters():
            if "variant_adapter" in name and name.endswith("linear.weight"):
                parameter.normal_(std=0.02)
    checkpoint = {"ema_model_state": source.state_dict()}
    requested = SamplingConfig(compile=inference_compile)
    loaded = build_checkpoint_inference_model(checkpoint, "pfm_delta", config, torch.device("cpu"), sampling=requested)
    canonical = lambda state: {key.replace("._orig_mod.", "."): value for key, value in state.items()}
    original_state = canonical(source.state_dict())
    loaded_state = canonical(loaded.state_dict())
    assert original_state.keys() == loaded_state.keys()
    for key in original_state:
        torch.testing.assert_close(original_state[key], loaded_state[key], rtol=0, atol=0)
    assert any("._orig_mod." in key for key in loaded.state_dict()) is inference_compile
    x = torch.randn(2, 3, 8, 8)
    t = torch.tensor([0.2, 0.7])
    labels = torch.tensor([1, 2])
    with torch.no_grad():
        expected = source(x, t, labels)
        actual = loaded(x, t, labels)
    for key in expected:
        torch.testing.assert_close(actual[key], expected[key], rtol=0, atol=0)
    assert config.jit_compile is stored_compile
    assert loaded.config.jit_compile is inference_compile
    broken = dict(checkpoint["ema_model_state"])
    broken.pop(next(iter(broken)))
    with pytest.raises(RuntimeError, match="Missing key"):
        build_checkpoint_inference_model({"ema_model_state": broken}, "pfm_delta", config, torch.device("cpu"), sampling=requested)


@pytest.mark.parametrize('key,value', [('precision', 'fp16'), ('compile', 'yes'), ('num_workers', -1), ('num_workers', True)])
def test_posterior_execution_overrides_are_typed(tmp_path, key, value):
    raw = yaml.safe_load((ROOT / 'configs/evaluation/imagenet256_jit_posterior.yaml').read_text())
    raw[key] = value
    path = tmp_path / 'posterior.yaml'
    path.write_text(yaml.safe_dump(raw))
    with pytest.raises(ValueError):
        load_metric_evaluation_config(path)


def test_posterior_execution_fields_are_explicit_and_cli_configurable(tmp_path):
    raw = yaml.safe_load((ROOT / 'configs/evaluation/imagenet256_jit_posterior.yaml').read_text())
    updated = merge_config(raw, parse_overrides(['precision=bf16', 'compile=true', 'num_workers=2']))
    path = tmp_path / 'posterior.yaml'
    path.write_text(yaml.safe_dump(updated))
    config = load_metric_evaluation_config(path)
    training = ExperimentConfig(architecture='jit_transformer', noise_scale=1, logvar_min=-1)
    execution = config.execution.apply(training)
    assert execution.mixed_precision == 'bf16' and execution.jit_compile is True
    assert execution.noise_scale == training.noise_scale and execution.logvar_min == -1
    assert config.num_workers == 2
    del updated['precision']
    path.write_text(yaml.safe_dump(updated))
    with pytest.raises(ValueError, match='precision'):
        load_metric_evaluation_config(path)


def test_matched_imagenet_profiles_keep_full_visualization_population():
    profiles = [load_sample_visualization_config(ROOT / f'configs/sampling/imagenet256_{solver}_visualization.yaml') for solver in ('ode', 'sde')]
    for config in profiles:
        assert config.class_count == 1000 and config.num_per_class == 1
        assert config.sampling.steps == 250 and config.sampling.terminal_time == .99
        assert config.sampling.precision == 'bf16' and config.sampling.compile is True
        assert config.uncertainty_capture_t == .495
    assert profiles[1].sampling.derivative_probes == 2
