from argparse import Namespace
import json
from pathlib import Path

import h5py
import numpy as np
import pytest
import torch
import yaml

from pfm_pointcloud.generator import PFMGenerator
from pfm_pointcloud.pfm import PointwisePFM, delta_loss, pointwise_divergence, sample
from pfm_pointcloud.evaluate import bbox_normalize, cd_metrics, main as evaluate
from pfm_pointcloud.generate import main as generate
from pfm_pointcloud.diagnostics import main as diagnostics
from pfm_pointcloud.train import main as train


def model_args():
    return Namespace(model='flow', latent_dim=8, latent_flow_depth=2,
                     latent_flow_hidden_dim=16, residual=True,
                     categories=['chair'], scale_mode='shape_unit')


def test_nll_and_variance_gradient():
    target = torch.tensor([[[2., 3., 4.]]])
    velocity = torch.zeros_like(target, requires_grad=True)
    ell = target.square().log().requires_grad_()
    loss = delta_loss(velocity, ell, torch.zeros_like(target), target)
    loss.backward()
    torch.testing.assert_close(ell.grad, torch.zeros_like(ell), atol=1e-7, rtol=0)
    assert torch.isfinite(velocity.grad).all()


def test_divergence_exact_xyz_blocks():
    x = torch.randn(2, 5, 3, requires_grad=True)
    gamma = torch.stack((x[...,0]**2+x[...,1], 3*x[...,1]+x[...,2], x[...,2].exp()+x[...,0]), -1)
    expected = torch.stack((2*x[...,0], torch.full_like(x[...,1],3), x[...,2].exp()), -1)
    torch.testing.assert_close(pointwise_divergence(gamma,x), expected)


def test_variance_bounds_joint_gradients_and_prior_roundtrip():
    model = PFMGenerator(model_args())
    x = torch.randn(2, 16, 3)
    z, kl = model.encode_with_kl(x)
    velocity, ell = model.net(x, torch.tensor([.2,.7]), z)
    assert (ell == 0).all()
    loss = delta_loss(velocity,ell,torch.zeros_like(x),x)+.001*kl
    loss.backward()
    for module in [model.encoder,model.prior,model.net]:
        assert any(p.grad is not None and p.grad.abs().sum()>0 for p in module.parameters())
    w = torch.randn(2,8)
    torch.testing.assert_close(model.prior(model.prior_sample(w)), w, atol=2e-6,rtol=2e-6)


def test_official_spectral_weights_are_materialized_once(tmp_path):
    model = PFMGenerator(model_args())
    prior = {key: value.clone() for key, value in model.prior.state_dict().items()}
    with torch.no_grad():
        for name, module in model.prior.named_modules():
            if isinstance(module, torch.nn.Linear):
                weight = module.weight.clone()
                u, singular, vh = torch.linalg.svd(weight, full_matrices=False)
                prior[name + '.weight_orig'] = weight
                prior[name + '.weight_u'] = u[:, 0]
                prior[name + '.weight_v'] = vh[0]
                prior[name + '.weight'].zero_()  # A stale cached weight is not authoritative.
                module.weight.copy_(weight / singular[0])
    path = tmp_path / 'spectral.pt'
    torch.save({'args': vars(model_args()), 'state_dict': {
        **{'encoder.' + key: value for key, value in model.encoder.state_dict().items()},
        **{'flow.' + key: value for key, value in prior.items()},
    }}, path)
    loaded = PFMGenerator.from_official(path)
    x = torch.randn(2, 8, requires_grad=True)
    expected, expected_logp = model.prior(x, x.new_zeros((2, 1)))
    actual, actual_logp = loaded.prior(x, x.new_zeros((2, 1)))
    torch.testing.assert_close(actual, expected)
    torch.testing.assert_close(actual_logp, expected_logp)
    torch.testing.assert_close(loaded.prior(actual, reverse=True), x)
    torch.testing.assert_close(torch.autograd.grad(actual.sum(), x)[0],
                               torch.autograd.grad(expected.sum(), x)[0])
    assert all(not module._forward_pre_hooks for module in loaded.prior.modules())
    assert all(not key.endswith(('weight_orig', 'weight_u', 'weight_v'))
               for key in loaded.prior.state_dict())


def test_zero_diffusion_matches_ode_and_readout():
    net=PointwisePFM(8)
    x,z=torch.randn(2,16,3),torch.randn(2,8)
    a,stats=sample(net,z,x,steps=2,capture=True)
    b,_=sample(net,z,x,steps=2,solver='sde',diffusion_scale=0)
    torch.testing.assert_close(a,b,rtol=0,atol=0)
    assert stats['nfe']==3 and stats['vjp']==0
    for entry in stats['snapshots']:
        torch.testing.assert_close(entry['gamma_x'],(1-entry['t'])**2*entry['gamma_v'])
    c,stats=sample(net,z,x,steps=2,solver='sde',brownian=torch.Generator().manual_seed(1))
    assert torch.isfinite(c).all() and stats['vjp']==6
    assert all(p.requires_grad for p in net.parameters())


def test_reference_direction_mmd_and_bbox():
    from pfm_pointcloud.metrics import lgan_mmd_cov
    distances=torch.tensor([[1.,4.,9.],[2.,5.,3.]])
    metrics=lgan_mmd_cov(distances)
    assert metrics['lgan_mmd'].item()==pytest.approx(8/3)
    assert metrics['lgan_cov'].item()==pytest.approx(1/3)
    x=bbox_normalize(torch.randn(3,8,3))
    metrics=cd_metrics(x,x,1)
    assert metrics['lgan_mmd-CD']<1e-6
    assert metrics['lgan_cov-CD']==1


def synthetic_assets(tmp_path):
    """Test fixture only: synthetic shapes and random small conditioning model."""
    rng=np.random.default_rng(23)
    dataset=tmp_path/'synthetic.hdf5'
    with h5py.File(dataset,'w') as handle:
        group=handle.create_group('03001627')
        group['train']=rng.normal(size=(4,16,3)).astype('float32')
        group['test']=rng.normal(size=(2,16,3)).astype('float32')
    args=model_args(); torch.manual_seed(23);model=PFMGenerator(args)
    state={**{'encoder.'+k:v for k,v in model.encoder.state_dict().items()},
           **{'flow.'+k:v for k,v in model.prior.state_dict().items()}}
    condition=tmp_path/'condition.pt'
    torch.save({'args':vars(args),'state_dict':state,'origin':'synthetic-test-only'},condition)
    config=yaml.safe_load((Path(__file__).parents[1]/'configs/cpu_smoke.yaml').read_text())
    config['num_points']=16
    cfg=tmp_path/'smoke.yaml';cfg.write_text(yaml.safe_dump(config))
    return dataset,condition,cfg


def test_train_resume_sample_and_evaluate(tmp_path):
    dataset,condition,config=synthetic_assets(tmp_path)
    def args(output):
        return ['--config',str(config),'--dataset',str(dataset),'--condition',str(condition),'--output',str(output)]
    full=tmp_path/'full'; resumed=tmp_path/'resumed'
    train(args(full))
    train(args(resumed)+['--stop-after','2'])
    moved=tmp_path/'copied.pt'
    moved.write_bytes((resumed/'latest.pt').read_bytes())
    train(['--resume',str(moved),'--dataset',str(dataset),'--output',str(resumed)])
    a=torch.load(full/'latest.pt',weights_only=False)
    b=torch.load(resumed/'latest.pt',weights_only=False)
    assert a['iteration']==b['iteration']==4
    assert a['scheduler']==b['scheduler']
    for key,value in a['state_dict'].items():
        torch.testing.assert_close(value,b['state_dict'][key],rtol=0,atol=0)
    assert torch.equal(a['rng']['torch'],b['rng']['torch'])
    for key, state in a['optimizer']['state'].items():
        for name,value in state.items():
            torch.testing.assert_close(value,b['optimizer']['state'][key][name],rtol=0,atol=0)
    out=tmp_path/'eval'
    evaluate(['--checkpoint',str(full/'latest.pt'),'--dataset',str(dataset),'--output',str(out)])
    results=json.loads((out/'results.json').read_text())['results']
    assert set(results)=={'ode','sde'}
    diag=tmp_path/'diagnostics.json'
    diagnostics(['--checkpoint',str(full/'latest.pt'),'--dataset',str(dataset),'--output',str(diag)])
    assert 'random_time' in json.loads(diag.read_text())
    for solver in ['ode','sde']:
        output=tmp_path/solver
        generate(['--checkpoint',str(full/'latest.pt'),'--output',str(output),'--solver',solver,'--count','2','--steps','2'])
        assert np.load(output/'points.npy').shape==(2,16,3)
    assert (out/'ode.png').is_file()
