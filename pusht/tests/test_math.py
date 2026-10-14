import torch
from diffusion_policy.model.pfm import delta_loss, log_variance, variance_diagonal, sample_delta
from diffusion_policy.model.diffusion.conditional_unet1d import ConditionalUnet1D


def test_unit_variance_is_fm_with_identical_gradient():
    v = torch.randn(3,16,2,requires_grad=True)
    target = torch.randn_like(v)
    nll = delta_loss(v, torch.zeros_like(v), target)
    fm = .5*(v-target).square().mean()
    torch.testing.assert_close(nll, fm)
    torch.testing.assert_close(torch.autograd.grad(nll,v)[0], torch.autograd.grad(fm,v)[0])


def test_effective_variance_derivative_contains_bounds_and_exp():
    x = torch.tensor([[[.2,-.4]]],requires_grad=True)
    raw = 2*x + x.flip(-1)
    gamma = log_variance(raw).exp()
    derivative,_ = variance_diagonal(gamma,x,probes=1,exact=True)
    expected = gamma.detach()*12*raw.detach().sigmoid()*(1-raw.detach().sigmoid())
    torch.testing.assert_close(derivative,expected)


def test_random_diagonal_handles_dense_jacobian():
    x = torch.tensor([[[.2,-.4]]],requires_grad=True)
    gamma = torch.stack((3*x[...,0]+5*x[...,1],7*x[...,0]+11*x[...,1]),dim=-1)
    result,_ = variance_diagonal(gamma,x,probes=4096,generator=torch.Generator().manual_seed(2))
    torch.testing.assert_close(result,torch.tensor([[[3.,11.]]]),atol=.3,rtol=0)


def test_zero_diffusion_is_ode_and_keeps_brownian_stream():
    x=torch.randn(2,16,2)
    field=lambda x,t:(-.3*x,torch.zeros_like(x))
    args=dict(steps=20,terminal_time=.99,diffusion_scale=0.,probes=2,noise_scale=1.,low=-1.,high=1.)
    generator=torch.Generator().manual_seed(3)
    before=generator.get_state().clone()
    ode,a=sample_delta(field,x,solver='ode',**args)
    sde,b=sample_delta(field,x,solver='sde',brownian_generator=generator,**args)
    torch.testing.assert_close(ode,sde,rtol=0,atol=0)
    assert a['nfe']==21 and b['vjp']==0
    assert torch.equal(before,generator.get_state())


def test_analytic_deterministic_endpoint():
    # Conditional target exactly y: v(x,t)=(y-x)/(1-t).
    x=torch.randn(2,16,2);y=torch.full_like(x,.25)
    field=lambda x,t:((y-x)/(1-t[:,None,None]),torch.zeros_like(x))
    result,_=sample_delta(field,x,steps=5,terminal_time=.99,solver='ode',
                         diffusion_scale=0,probes=2,noise_scale=1,low=-1,high=1)
    torch.testing.assert_close(result,y)


def test_actual_unet_has_continuous_time_and_action_gradients():
    torch.manual_seed(0)
    model=ConditionalUnet1D(input_dim=2,global_cond_dim=6,down_dims=(16,32),
                           diffusion_step_embed_dim=16,n_groups=8)
    model.variance_head=torch.nn.Conv1d(16,2,1)
    x=torch.randn(2,16,2,requires_grad=True);c=torch.randn(2,6)
    v,raw=model(x,torch.tensor([.25,.75]),global_cond=c)
    assert v.shape==raw.shape==x.shape
    assert not torch.equal(model(x,torch.full((2,),.25),global_cond=c)[0],
                           model(x,torch.full((2,),.75),global_cond=c)[0])
    d,_=variance_diagonal(log_variance(raw).exp(),x,probes=2,exact=True)
    assert torch.isfinite(d).all() and d.abs().max()>0
    assert all(p.grad is None for p in model.parameters())


def test_corrected_sde_matches_analytic_gaussian_moments():
    torch.manual_seed(12)
    initial=torch.randn(10000,1,1)
    mean=.7;variance=.5
    def field(x,t):
        t=t[:,None,None]
        q=(1-t)**2+t*t*variance
        v=mean+(t*variance-(1-t))/q*(x-t*mean)
        gamma=variance/q
        probability=(gamma.log()+1)/2
        raw=(probability/(1-probability)).log().expand_as(x)
        return v,raw
    result,_=sample_delta(field,initial,steps=200,terminal_time=.99,solver='sde',
                         diffusion_scale=.1,probes=2,noise_scale=1.,low=-1.,high=1.,
                         brownian_generator=torch.Generator().manual_seed(13))
    assert abs(float(result.mean())-mean)<.03
    assert abs(float(result.var())-variance)<.04
