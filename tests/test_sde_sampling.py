from dataclasses import replace
import itertools
import math

import pytest
import torch

from pfm.core.config import ExperimentConfig
from pfm.core.sampling_config import SamplingConfig
from pfm.probability.logvar import bound_normalized_logvar
from pfm.probability.posterior import posterior_from_outputs, score_from_velocity, sde_coefficients
from pfm.sampling.derivatives import estimate_variance_diagonal
from pfm.sampling.randomness import indexed_step_random
from pfm.sampling.samplers import sample

CPU = torch.device('cpu')


def test_random_vjp_has_correct_mean_variance_and_includes_sigmoid_bound():
    cfg = ExperimentConfig(logvar_min=-3, logvar_max=1)
    matrix = torch.tensor([[1., 2., -.3], [.5, -1., 3.], [-.2, .7, 2.]])
    point = torch.tensor([.2, -.4, .8])
    signs = torch.tensor(list(itertools.product([-1., 1.], repeat=3)))
    estimates = []
    for z in signs:
        x = point.clone().requires_grad_()
        raw = matrix @ x
        gamma = bound_normalized_logvar(raw, cfg).exp()
        estimates.append(estimate_variance_diagonal(gamma, x, [z], 1).value)
    p = torch.sigmoid(matrix @ point)
    chain = (-3 + 4 * p).exp() * 4 * p * (1-p)
    jacobian = chain[:, None] * matrix
    expected = jacobian.diag()
    stack = torch.stack(estimates)
    torch.testing.assert_close(stack.mean(0), expected, atol=3e-7, rtol=3e-6)
    column_offdiag = jacobian.square().sum(0) - expected.square()
    torch.testing.assert_close((stack - expected).square().mean(0), column_offdiag)
    # Summing the variance outputs would return column sums, not this diagonal.
    assert not torch.allclose(jacobian.sum(0), expected)


def test_diagonal_field_needs_one_probe_and_streaming_error_is_zero():
    x = torch.tensor([.1, -.5, .7], requires_grad=True)
    gamma = (x * .3).exp()
    probes = [torch.tensor([1., -1., 1.]), torch.tensor([-1., -1., 1.])]
    estimate = estimate_variance_diagonal(gamma, x, probes, 2)
    torch.testing.assert_close(estimate.value, .3 * gamma.detach())
    assert estimate.mean_squared_error_estimate == 0
    assert estimate.actual_vjp == 2
    assert not estimate.value.requires_grad


def test_fokker_planck_correction_preserves_gaussian_interpolant_with_spatial_variance():
    # A one-dimensional analytic law with deliberately nonconstant diffusion.
    x = torch.linspace(-2, 2, 13, dtype=torch.float64, requires_grad=True)
    t = torch.tensor(.37, dtype=torch.float64, requires_grad=True)
    q = (1-t)**2 + 2*t**2
    density = torch.exp(-x*x/(2*q)) / torch.sqrt(2*torch.pi*q)
    velocity = (2*t-(1-t))*x/q
    gamma = torch.exp(.4*torch.tanh(x))
    divergence = .4*(1-torch.tanh(x)**2)*gamma
    score = -x/q
    drift, rate = sde_coefficients(velocity, gamma, score, divergence, .7)
    def dx(y):
        return torch.autograd.grad(y.sum(), x, create_graph=True, retain_graph=True)[0]
    rhs = -dx(drift*density) + .5*dx(dx(rate*density))
    dq = 2*(2*t-(1-t))
    expected = density * dq * (x*x/q**2 - 1/q) / 2
    torch.testing.assert_close(rhs, expected, rtol=1e-10, atol=1e-10)
    wrong_drift, _ = sde_coefficients(velocity, gamma, score, torch.zeros_like(x), .7)
    wrong_rhs = -dx(wrong_drift*density) + .5*dx(dx(rate*density))
    assert (wrong_rhs-expected).abs().max() > .01


class DiagonalModel(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.weight = torch.nn.Parameter(torch.tensor(.3))
        self.calls = 0
    def forward(self, x, t, labels):
        self.calls += 1
        return {'delta': .2*x, 'logvar': self.weight*x}


def test_sde_step_matches_analytic_complete_drift_and_brownian_scaling():
    cfg = ExperimentConfig(image_channels=1, image_size=2, logvar_min=-3, logvar_max=1)
    x = torch.tensor([[[[.2, -.5], [.7, 1.]]]])
    protocol = SamplingConfig(solver='sde', steps=1, terminal_time=.05, diffusion_scale=.7, derivative_probes=1)
    model = DiagonalModel()
    result = sample(model, 'pfm_delta', torch.tensor([0]), cfg, protocol, CPU, initial_noise=x, seed=42)
    p = torch.sigmoid(.3*x)
    gamma = (-3+4*p).exp()
    divergence = gamma*4*p*(1-p)*.3
    drift = .2*x + .7/2*(divergence-gamma*x)
    xi = indexed_step_random(torch.tensor([0]), (1,2,2), seed=42, step=0, stream='brownian', device=CPU)
    expected = x + .05*drift + (.05*.7*gamma).sqrt()*xi
    torch.testing.assert_close(result.samples, expected)
    assert model.calls == result.actual_nfe == result.actual_vjp == 1
    assert model.weight.requires_grad and model.weight.grad is None
    assert not result.samples.requires_grad
    torch.testing.assert_close(result.velocity_var, gamma)
    torch.testing.assert_close(result.diffusion_rate, .7*gamma)
    torch.testing.assert_close(result.noise_covariance, .05*.7*gamma)


def test_lambda_zero_matches_ode_and_never_draws_sde_randomness(monkeypatch):
    cfg = ExperimentConfig(image_channels=1, image_size=2)
    x = torch.randn(2,1,2,2)
    ode = SamplingConfig(steps=4, terminal_time=.8, terminal_readout='endpoint_mean')
    def forbidden(*args, **kwargs):
        raise AssertionError('zero diffusion must skip derivatives and random streams')
    monkeypatch.setattr('pfm.sampling.samplers.indexed_step_random', forbidden)
    monkeypatch.setattr('pfm.sampling.samplers.estimate_variance_diagonal', forbidden)
    a = sample(DiagonalModel(), 'pfm_delta', torch.tensor([0,1]), cfg, ode, CPU, initial_noise=x)
    b = sample(DiagonalModel(), 'pfm_delta', torch.tensor([0,1]), cfg, replace(ode, solver='sde'), CPU, initial_noise=x)
    assert torch.equal(a.samples, b.samples)
    assert a.actual_nfe == b.actual_nfe == 5
    assert b.actual_vjp == 0


def test_random_streams_are_reproducible_independent_and_batch_invariant():
    indices = torch.tensor([0,1,63,64,70])
    kwargs = dict(sample_shape=(1,2,2), seed=17, step=3, device=CPU)
    a = indexed_step_random(indices, stream='brownian', **kwargs)
    chunks = torch.cat([indexed_step_random(chunk, stream='brownian', **kwargs) for chunk in indices.split(2)])
    assert torch.equal(a, chunks)
    for p in range(8):
        indexed_step_random(indices, stream='derivative', probe=p, **kwargs)
    assert torch.equal(a, indexed_step_random(indices, stream='brownian', **kwargs))
    z = indexed_step_random(indices, stream='derivative', **kwargs)
    assert set(z.flatten().tolist()) == {-1.,1.}
    assert not torch.equal(z, indexed_step_random(indices, stream='derivative', probe=1, **kwargs))


def test_sde_sampling_batch_invariance_and_endpoint_capture_semantics():
    cfg = ExperimentConfig(image_channels=1, image_size=2)
    protocol = SamplingConfig(solver='sde', steps=4, terminal_time=.8, terminal_readout='endpoint_mean', diffusion_scale=.3, derivative_probes=2)
    labels = torch.tensor([0,1,2])
    initial = torch.arange(12).reshape(3,1,2,2).float()/12
    whole = sample(DiagonalModel(), 'pfm_delta', labels, cfg, protocol, CPU, initial_noise=initial, seed=91, uncertainty_capture_t=.4)
    chunks = [sample(DiagonalModel(), 'pfm_delta', labels[i:i+1], cfg, protocol, CPU, initial_noise=initial[i:i+1], sample_indices=torch.tensor([i]), seed=91).samples for i in range(3)]
    assert torch.equal(whole.samples, torch.cat(chunks))
    assert whole.uncertainty_source_t == .4
    assert whole.uncertainty_source_state == 'pre_update'
    torch.testing.assert_close(whole.endpoint_var, (1-.4)**2*whole.velocity_var)
    assert whole.actual_nfe == 5 and whole.actual_vjp == 8


def test_nonfinite_sampler_fails_and_restores_parameter_flags():
    cfg = ExperimentConfig(image_channels=1, image_size=2)
    model = DiagonalModel()
    protocol = SamplingConfig(solver='sde', steps=2, terminal_time=.8, diffusion_scale=1)
    with pytest.raises(FloatingPointError):
        sample(model, 'pfm_delta', torch.tensor([0]), cfg, protocol, CPU, initial_noise=torch.full((1,1,2,2), float('nan')))
    assert model.weight.requires_grad


@pytest.mark.parametrize('kwargs', [dict(steps=True), dict(derivative_probes=0), dict(terminal_time=0), dict(terminal_time=float('nan')), dict(terminal_time=1-1e-9), dict(diffusion_scale=-1), dict(solver='sde',diffusion_scale=1), dict(terminal_readout='endpoint_mean'), dict(precision='fp16')])
def test_sampling_protocol_rejects_invalid_fields(kwargs):
    with pytest.raises(ValueError):
        SamplingConfig(**kwargs)


def test_both_pfm_parameterizations_share_score_and_variance():
    cfg = ExperimentConfig(logvar_min=-3, logvar_max=1)
    x = torch.tensor([[[[.4, -.3]]]])
    t = torch.tensor([.7])
    v = .2*x
    raw = .3*x
    delta = posterior_from_outputs({'delta':v,'logvar':raw}, 'pfm_delta', x,t,cfg)
    full = posterior_from_outputs({'mu':x+(1-.7)*v,'logvar':raw}, 'pfm_full', x,t,cfg)
    torch.testing.assert_close(delta.velocity,full.velocity)
    torch.testing.assert_close(delta.velocity_var,full.velocity_var)
    torch.testing.assert_close(score_from_velocity(x,t,v,2), (.7*v-x)/(4*.3))
