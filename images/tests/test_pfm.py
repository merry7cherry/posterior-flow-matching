from dataclasses import replace
import numpy as np
import pytest
import torch
from torch.utils.data import TensorDataset
from pfm.config import ImageConfig, SamplingConfig
from pfm.models.factory import build_model
from pfm.models.jit_config import JIT_MODEL_SPECS, JiTModelSpec
from pfm.probability import pfm_loss, posterior, score_from_velocity
from pfm.sampling.samplers import sample, variance_divergence
from pfm.generate import generate_batches
from pfm.train import train
from pfm.checkpoint import load_model
from pfm.diagnostics import posterior_diagnostics
from pfm.metrics import _OnlineFeatureStatistics, _OnlineInceptionScore


torch.set_num_threads(2)


def small_config(**changes):
    config = ImageConfig(image_size=16, num_classes=3, mixed_precision="none", batch_size=2, num_workers=0,
        epochs=1, limit_train_batches=2, checkpoint_every_epochs=1, unet_base_channels=8,
        unet_channel_mult=(1, 2), unet_num_res_blocks=1, unet_norm_groups=4, unet_dropout=.05,
        time_embed_dim=16, time_proj_dim=16, class_embed_dim=16, class_proj_dim=16,
        edm_model_channels=8, edm_channel_mult=(1, 2), edm_num_blocks=1, edm_norm_groups=4,
        edm_attn_resolutions=(8,), edm_dropout=0)
    return replace(config, **changes)


@pytest.mark.parametrize("parameterization", ["velocity", "endpoint"])
def test_gaussian_nll_and_endpoint_change_of_coordinates(parameterization):
    config = small_config(mean_parameterization=parameterization)
    x0 = torch.tensor([-.4, .6]).reshape(2, 1, 1, 1)
    x1 = torch.tensor([.8, -.2]).reshape(2, 1, 1, 1)
    t = torch.tensor([.3, .7]).reshape(2, 1, 1, 1)
    xt = (1 - t) * x0 + t * x1
    v = torch.tensor([.2, -.1]).reshape(2, 1, 1, 1)
    output = {"mean": v if parameterization == "velocity" else xt + (1-t)*v,
              "logvar": torch.zeros_like(xt, requires_grad=True)}
    result = posterior(output, xt, t, config)
    expected = .5 * ((x1-x0-v)**2).mean()
    torch.testing.assert_close(pfm_loss(output, x0, x1, xt, t, config), expected)
    torch.testing.assert_close(result.endpoint_var, (1-t)**2)
    torch.testing.assert_close(result.endpoint_mean, xt+(1-t)*v)


def test_gaussian_score_and_input_variance_derivative():
    # Exact scalar Gaussian conditional velocity implies the Gaussian marginal score.
    t, source_var, data_var, mean = .4, 2., 3., .7
    x = torch.tensor([-.8, .2, 1.1]).reshape(3, 1, 1, 1)
    marginal_var = (1-t)**2*source_var+t*t*data_var
    velocity = mean+(t*data_var-(1-t)*source_var)/marginal_var*(x-t*mean)
    torch.testing.assert_close(score_from_velocity(x, torch.full((3,),t), velocity, source_var**.5), -(x-t*mean)/marginal_var)
    state = x.clone().requires_grad_(True)
    gamma = (state+2).square()
    probes = [torch.ones_like(state), -torch.ones_like(state)]
    torch.testing.assert_close(variance_divergence(gamma,state,probes,2), 2*(state+2))


@pytest.mark.parametrize("architecture", ["conditional_unet", "edm_ddpmpp", "jit_transformer"])
@pytest.mark.parametrize("parameterization", ["velocity", "endpoint"])
def test_native_architecture_forward_backward(architecture, parameterization, monkeypatch):
    monkeypatch.setitem(JIT_MODEL_SPECS, "JiT-B/16", JiTModelSpec("JiT-B/16",4,32,2,4,16,4,1,(16,)))
    config=small_config(architecture=architecture, mean_parameterization=parameterization)
    model=build_model(config)
    x0=torch.randn(2,1,16,16);x1=torch.randn_like(x0);t=torch.tensor([.2,.7]).view(2,1,1,1)
    xt=(1-t)*x0+t*x1
    loss=pfm_loss(model(xt,t.flatten(),torch.tensor([0,1])),x0,x1,xt,t,config)
    loss.backward()
    assert torch.isfinite(loss)
    for head in [model.posterior_heads.mean,model.posterior_heads.logvar]:
        gradients=[p.grad for p in head.parameters() if p.grad is not None]
        assert gradients and all(torch.isfinite(g).all() for g in gradients)
        assert sum(g.abs().sum() for g in gradients)>0


def test_ode_equals_zero_sde_and_brownian_batching():
    torch.manual_seed(9)
    config=small_config(unet_dropout=0)
    model=build_model(config)
    labels=torch.tensor([0,1])
    ode=sample(model,labels,config,SamplingConfig(steps=2),'cpu',seed=8)
    zero=sample(model,labels,config,SamplingConfig(solver='sde',steps=2),'cpu',seed=8)
    torch.testing.assert_close(ode['samples'],zero['samples'],rtol=0,atol=0)
    protocol=SamplingConfig(solver='sde',steps=2,diffusion_scale=.1)
    whole=list(generate_batches(model,config,protocol,'cpu',count=4,batch_size=4))
    split=list(generate_batches(model,config,protocol,'cpu',count=4,batch_size=2))
    torch.testing.assert_close(whole[0]['samples'],torch.cat([c['samples'] for c in split]),rtol=1e-5,atol=2e-6)
    assert torch.isfinite(whole[0]['endpoint_variance']).all()


def test_training_save_resume_matches_continuous_run(tmp_path):
    rng=torch.Generator().manual_seed(71)
    dataset=TensorDataset(torch.randn(8,1,16,16,generator=rng),torch.arange(8)%3)
    config=small_config()
    continuous=train(replace(config,epochs=2),None,tmp_path/'continuous',dataset=dataset,quiet=True)
    first=train(config,None,tmp_path/'resumed',dataset=dataset,quiet=True)
    resumed=train(replace(config,epochs=2),None,tmp_path/'resumed',dataset=dataset,resume=first,quiet=True)
    a,b=[torch.load(p,weights_only=True) for p in [continuous,resumed]]
    assert a['step']==b['step']==4
    for section in ['model','ema']:
        for key,value in a[section].items():
            torch.testing.assert_close(value,b[section][key],rtol=0,atol=0)
    torch.testing.assert_close(a['rng_states'][0]['cpu'],b['rng_states'][0]['cpu'],rtol=0,atol=0)
    model,loaded_config=load_model(resumed)
    output=sample(model,torch.tensor([0,1]),loaded_config,SamplingConfig(solver='sde',steps=2,diffusion_scale=.1),'cpu')
    assert torch.isfinite(output['samples']).all()
    result=posterior_diagnostics(model,loaded_config,None,torch.device('cpu'),num_samples=4,times=[.25,.5],dataset=dataset)
    for key,values in result.items():
        t=float(key)
        assert values['endpoint_mean_variance']==pytest.approx((1-t)**2*values['velocity_mean_variance'])
        assert values['endpoint_nll_per_coordinate']==pytest.approx(values['velocity_nll_per_coordinate']+np.log(1-t))
        assert sum(row['count'] for row in values['variance_bins'])==values['component_count']


def test_streaming_feature_covariance_and_inception_score():
    features=torch.randn(31,5,dtype=torch.float64,generator=torch.Generator().manual_seed(17))
    stats=_OnlineFeatureStatistics()
    stats.update(features[:11]);stats.update(features[11:])
    result=stats.finalize()
    np.testing.assert_allclose(result['mu'],features.numpy().mean(0),atol=1e-14)
    np.testing.assert_allclose(result['sigma'],np.cov(features.numpy(),rowvar=False),atol=1e-14)
    metric=_OnlineInceptionScore(total_samples=20,splits=10,rng_seed=2020)
    logits=torch.tensor([[1.,2.,3.]]).expand(20,-1)
    metric.update(logits[:7],start_index=0);metric.update(logits[7:],start_index=7)
    assert metric.finalize()['mean']==pytest.approx(1.)
