import itertools
import pytest
import torch
from etflow.models import pfm_probability as p
from etflow.models.pfm_sampling import sample


@pytest.mark.parametrize("n", [2, 3, 7, 20])
def test_projected_gaussian_against_orthogonal_basis(n):
    torch.manual_seed(n)
    b = torch.zeros(n, dtype=torch.long)
    eye = torch.eye(n, dtype=torch.double)
    P = eye - torch.ones_like(eye) / n
    vals, Q = torch.linalg.eigh(P)
    Q = Q[:, vals > 0.5]
    lv = torch.randn(n, dtype=torch.double, requires_grad=True)
    r = p.center(torch.randn(n, 3, dtype=torch.double), b)
    K = Q.T @ torch.diag(lv.exp()) @ Q
    y = Q.T @ r
    expected = 0.5 * ((y * torch.linalg.solve(K, y)).sum() + 3 * torch.logdet(K)) / (3 * (n - 1))
    actual = p.projected_nll(r, lv, b).mean()
    torch.testing.assert_close(actual, expected)
    torch.testing.assert_close(
        torch.autograd.grad(actual, lv, retain_graph=True)[0], torch.autograd.grad(expected, lv)[0]
    )
    torch.testing.assert_close(
        p.marginal_variance(lv.exp(), b), torch.diag(P @ torch.diag(lv.exp()) @ P)
    )


def test_batched_rotation_permutation():
    torch.manual_seed(1)
    b = torch.tensor([0] * 4 + [1] * 6)
    r = p.center(torch.randn(10, 3, dtype=torch.double), b)
    lv = torch.randn(10, dtype=torch.double)
    Q = torch.linalg.qr(torch.randn(3, 3, dtype=torch.double))[0]
    perm = torch.tensor([2, 0, 3, 1, 7, 4, 9, 5, 6, 8])
    torch.testing.assert_close(p.projected_nll(r, lv, b), p.projected_nll(r @ Q, lv, b))
    torch.testing.assert_close(p.projected_nll(r, lv, b), p.projected_nll(r[perm], lv[perm], b))
    torch.testing.assert_close(p.center(r, b), r)


def test_divergence_matches_full_covariance_jacobian():
    torch.manual_seed(4)
    n = 3
    b = torch.zeros(n, dtype=torch.long)
    x = torch.randn(n, 3, dtype=torch.double, requires_grad=True)

    def scales(x):
        h = p.center(x, b)
        return torch.exp(0.2 * h.square().sum(-1) + 0.1 * h.sum())

    def matrix(flat):
        g = scales(flat.reshape(n, 3))
        P = torch.eye(n, dtype=x.dtype) - torch.ones(n, n, dtype=x.dtype) / n
        return torch.kron(P @ torch.diag(g) @ P, torch.eye(3, dtype=x.dtype))

    jac = torch.autograd.functional.jacobian(matrix, x.flatten())
    expected = torch.einsum("ijj->i", jac).reshape(n, 3)
    actual = p.projected_divergence(scales(x), x, b, exact=True)
    torch.testing.assert_close(actual, expected)
    # Enumerate signs to validate the stochastic estimator without sampling error.
    estimates = []
    for signs in itertools.product([-1.0, 1.0], repeat=n):
        s = x.new_tensor(signs)
        grad = torch.autograd.grad(scales(x), x, grad_outputs=s)[0]
        estimates.append(p.center(s[:, None] * p.center(grad, b), b))
    torch.testing.assert_close(torch.stack(estimates).mean(0), expected)


def test_sde_zero_diffusion_is_ode_and_centered_noise_covariance():
    b = torch.zeros(4, dtype=torch.long)
    x = torch.randn(4, 3)

    def predict(x, t):
        return -x, torch.ones(4)

    a, _ = sample(predict, x, b, steps=4, mode="ode")
    c, _ = sample(predict, x, b, steps=4, mode="sde", diffusion_scale=0)
    torch.testing.assert_close(a, c, rtol=0, atol=0)
    torch.testing.assert_close(a.mean(0), torch.zeros(3), atol=1e-7, rtol=0)
    g = torch.tensor([0.3, 0.7, 1.2, 2.0], dtype=torch.double)
    P = torch.eye(4, dtype=g.dtype) - torch.ones(4, 4, dtype=g.dtype) / 4
    L = P @ torch.diag(g.sqrt())
    torch.testing.assert_close(L @ L.T, P @ torch.diag(g) @ P)


def test_corrected_sde_independent_gaussian_linear_path():
    # Independent Gaussian endpoints: marginal covariance q(t)P.
    torch.manual_seed(4)
    n_mols = 2048
    b = torch.arange(n_mols).repeat_interleave(3)
    x = p.center(torch.randn(len(b), 3), b)

    def predict(x, t):
        q = (1 - t) ** 2 + t**2
        return ((2 * t - 1) / q) * x, torch.ones(len(x))

    y, _ = sample(predict, x, b, steps=60, terminal_time=0.8, mode="sde", diffusion_scale=0.5)
    expected = (2 / 3) * 0.8**2 / ((1 - 0.8) ** 2 + 0.8**2)
    assert abs(y.square().mean().item() - expected) < 0.04
    assert p.group_sum(y, b).abs().max() < 2e-6


def test_linear_schedule_scales_both_drift_and_noise():
    b = torch.zeros(3, dtype=torch.long)
    x = p.center(torch.arange(9, dtype=torch.float32).reshape(3, 3)/10, b)
    scale, terminal, steps, seed = .3, .8, 4, 14
    def predict(x, t):
        return -.2*x, torch.ones(3)
    actual, _ = sample(predict, x, b, steps=steps, terminal_time=terminal,
                       mode="sde", diffusion_scale=scale, seed=seed)
    expected = x.clone()
    generator = torch.Generator().manual_seed(seed)
    dt = terminal/steps
    for k in range(steps):
        t = k*dt
        v = -.2*expected
        lam = scale*(1-t)
        score = (t*v-expected)/(1-t)
        noise = p.center(torch.randn(expected.shape, generator=generator), b)
        expected = p.center(expected+dt*(v+.5*lam*score)+(lam*dt)**.5*noise, b)
    expected = p.center(expected+(1-terminal)*(-.2*expected), b)
    torch.testing.assert_close(actual, expected)


def test_projected_divergence_jvp_vjp_duality():
    torch.manual_seed(18)
    b = torch.tensor([0, 0, 0, 1, 1])
    x = torch.randn(5, 3, dtype=torch.double, requires_grad=True)
    tangent = p.center(torch.randn_like(x), b)
    signs = torch.randn(5, dtype=x.dtype)
    def g(state):
        return (.2*p.center(state, b).square().sum(-1)).exp()
    _, jvp = torch.autograd.functional.jvp(g, x, tangent)
    vjp = torch.autograd.grad(g(x), x, signs)[0]
    torch.testing.assert_close((jvp*signs).sum(), (tangent*vjp).sum())
