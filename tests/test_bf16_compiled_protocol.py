"""Execution defaults, old EDM identity, and compiled input-gradient contract."""
from copy import deepcopy
from dataclasses import replace
from pathlib import Path

import pytest
import torch

from pfm.core.config import (ExperimentConfig, load_config, experiment_config_to_mapping,
    checkpoint_config_from_mapping, experiment_config_from_mapping, shared_training_protocol_sha256)
from pfm.core.inference_config import InferenceExecutionConfig
from pfm.core.sample_config import load_sample_visualization_config
from pfm.models.factory import build_model
from pfm.probability.posterior import posterior_from_outputs
from pfm.sampling.derivatives import estimate_variance_diagonal

ROOT = Path(__file__).resolve().parents[1]


def test_old_edm_checkpoint_is_eager_but_new_yaml_requires_execution_field():
    config = load_config(ROOT / 'configs/cifar10_edm_ddpmpp.yaml')
    raw = experiment_config_to_mapping(config)
    del raw['edm_compile']
    with pytest.raises(ValueError, match='edm_compile'):
        experiment_config_from_mapping(raw)
    old = checkpoint_config_from_mapping(raw)
    assert old.edm_compile is False
    import hashlib, json
    expected = {k:v for k,v in raw.items() if k not in {'self_conditioning_mode','epochs','checkpoint_every_epochs'}}
    digest = hashlib.sha256(json.dumps(expected, sort_keys=True, separators=(',',':'), allow_nan=False).encode()).hexdigest()
    assert shared_training_protocol_sha256(old, legacy_edm=True) == digest
    assert InferenceExecutionConfig('bf16', True).apply(old).edm_compile is True
    assert old.edm_compile is False


@pytest.mark.parametrize('dataset,compiled', [('cifar10', True), ('imagenet256', True)])
@pytest.mark.parametrize('solver', ['ode','sde'])
def test_matched_recommended_execution_and_capture(dataset,compiled,solver):
    config = load_sample_visualization_config(ROOT / f'configs/sampling/{dataset}_{solver}_visualization.yaml')
    s = config.sampling
    assert (s.precision,s.compile,s.terminal_time,s.terminal_readout) == ('bf16',compiled,.99,'endpoint_mean')
    assert config.uncertainty_capture_t == .495
    assert config.uncertainty_capture_t / s.terminal_time * s.steps == pytest.approx(s.steps/2)
    if solver == 'sde': assert s.derivative_probes == 2


@pytest.mark.parametrize('variant',['pfm_delta','pfm_full'])
def test_edm_compile_preserves_checkpoint_ema_and_input_derivatives(monkeypatch,variant):
    real = torch.compile
    monkeypatch.setattr(torch,'compile',lambda fn,*a,**kw: real(fn, backend='aot_eager'))
    cfg = ExperimentConfig(dataset='cifar10', architecture='edm_ddpmpp', image_size=32, image_channels=3,
        edm_model_channels=8, edm_channel_mult=(1,), edm_num_blocks=1, edm_norm_groups=4,
        edm_attn_resolutions=(), edm_dropout=0, edm_compile=False, mixed_precision='none')
    reference = build_model(variant,cfg,torch.device('cpu')).eval()
    with torch.no_grad():
        for name,p in reference.named_parameters():
            if 'variant_adapter' in name and p.ndim>1: p.normal_(std=.02)
    compiled = build_model(variant,replace(cfg,edm_compile=True),torch.device('cpu')).eval()
    compiled.load_state_dict(reference.state_dict(),strict=True)
    assert compiled.state_dict().keys() == reference.state_dict().keys()
    ema = deepcopy(compiled)
    assert getattr(ema,'_compiled_call_impl',None) is None
    x = torch.randn(1,3,32,32);t=torch.tensor([.5]);labels=torch.tensor([1]);z=torch.ones_like(x)
    results=[]
    for model in (reference,compiled):
        current=x.clone().requires_grad_(True)
        out=model(current,t,labels)
        post=posterior_from_outputs(out,variant,current,t,cfg)
        diag=estimate_variance_diagonal(post.velocity_var,current,[z,-z],2)
        results.append((post.velocity.detach(),post.velocity_var.detach(),diag.value))
    for a,b in zip(*results): torch.testing.assert_close(a,b,rtol=1e-5,atol=1e-6)




def test_compiled_consumer_reuses_backward_across_sde_batches(monkeypatch):
    import torch._functorch.config as aot_config
    from pfm.core.sampling_config import SamplingConfig
    from pfm.evaluation.checkpoint_loading import build_checkpoint_inference_model
    from pfm.sampling.samplers import sample

    real = torch.compile
    monkeypatch.setattr(torch, 'compile', lambda fn, *a, **kw: real(fn, backend='aot_eager'))
    monkeypatch.setattr(aot_config, 'donated_buffer', True)
    cfg = ExperimentConfig(dataset='cifar10', architecture='edm_ddpmpp', image_size=32,
        image_channels=3, edm_model_channels=8, edm_channel_mult=(1,), edm_num_blocks=1,
        edm_norm_groups=4, edm_attn_resolutions=(), edm_dropout=0, edm_compile=False,
        mixed_precision='none')
    device = torch.device('cpu')
    eager = build_model('pfm_delta', cfg, device).eval()
    with torch.no_grad():
        for name, parameter in eager.named_parameters():
            if 'variant_adapter' in name and parameter.ndim > 1:
                parameter.normal_(std=.02)
    sampling = SamplingConfig(solver='sde', steps=2, terminal_time=.99,
        terminal_readout='endpoint_mean', precision='fp32', compile=True,
        derivative_probes=2, diffusion_scale=.1, diffusion_schedule='linear_decay')
    compiled = build_checkpoint_inference_model(
        {'ema_model_state': eager.state_dict()}, 'pfm_delta', cfg, device, sampling=sampling)
    assert aot_config.donated_buffer is False
    # The final partial batch and a repeated shape exercise cached backwards,
    # not just the first compiled VJP. Capture preserves an intermediate map.
    for batch in (2, 1, 2):
        noise = torch.randn(batch, 3, 32, 32)
        labels = torch.arange(batch)
        kwargs = dict(initial_noise=noise, uncertainty_capture_t=.495, seed=123)
        expected = sample(eager, 'pfm_delta', labels, cfg, replace(sampling, compile=False), device, **kwargs)
        actual = sample(compiled, 'pfm_delta', labels, cfg, sampling, device, **kwargs)
        assert actual.actual_vjp == 4
        assert actual.uncertainty_source_t == .495
        for field in ('samples', 'velocity_var', 'endpoint_var'):
            torch.testing.assert_close(getattr(actual, field), getattr(expected, field), rtol=1e-5, atol=1e-6)
