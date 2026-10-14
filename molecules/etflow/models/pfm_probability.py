"""Gaussian posterior on each molecule's arithmetic zero-centroid subspace.

Scales are pre-projection velocity variances. Never treat them as terminal
execution variance, or use an ambient diagonal NLL for projected coordinates.
"""

import torch


def group_sum(x, batch):
    out = x.new_zeros((int(batch.max()) + 1,) + x.shape[1:])
    return out.index_add(0, batch, x)


def counts(batch, like):
    return torch.bincount(batch).to(like)


def center(x, batch):
    n = counts(batch, x)
    shape = (len(n),) + (1,) * (x.ndim - 1)
    return x - (group_sum(x, batch) / n.reshape(shape))[batch]


def bounded_logvar(raw, lower, upper):
    if not lower < upper:
        raise ValueError("log-variance bounds must be ordered")
    return lower + (upper - lower) * torch.sigmoid(raw)


def covariance_apply(g, x, batch):
    return center(g[:, None] * center(x, batch), batch)


def noise_from_standard(g, eps, batch):
    return center(g.sqrt()[:, None] * eps, batch)


def marginal_variance(g, batch):
    n = counts(batch, g)
    return (1 - 2 / n[batch]) * g + group_sum(g, batch)[batch] / n[batch].square()


def projected_nll(residual, logvar, batch):
    """Per-molecule NLL / 3(N-1), omitting parameter-independent constants."""
    n = counts(batch, residual)
    if (n < 2).any():
        raise ValueError("A centered molecule needs at least two atoms")
    r = center(residual, batch)
    w = torch.exp(-logvar)
    sw = group_sum(w, batch)
    weighted_r = group_sum(w[:, None] * r, batch)
    quadratic = group_sum(w * r.square().sum(-1), batch)
    quadratic = quadratic - weighted_r.square().sum(-1) / sw
    logdet = 3 * (group_sum(logvar, batch) + torch.log(sw / n))
    return 0.5 * (quadratic + logdet) / (3 * (n - 1))


def posterior(x, t, v, g, batch):
    remaining = 1 - t
    return x + remaining * v, remaining.square() * marginal_variance(g, batch)[:, None]


def score_from_velocity(x, t, v, source_scale):
    return (t * v - x) / (source_scale**2 * (1 - t))


def projected_divergence(g, x, batch, probes=2, generator=None, exact=False):
    """div_H(P diag(g_i I3) P), using atom-wise Rademacher VJPs.

    q[j,a] = sum_k P[j,k] d g[j]/d x[k,a]; divergence = P q.
    Independent atom signs estimate the required diagonal after projecting the
    VJP. The final projection must occur AFTER multiplying by those signs.
    """
    if not g.requires_grad:
        return torch.zeros_like(x)
    if exact:
        q = torch.zeros_like(x)
        for j in range(len(g)):
            grad = torch.autograd.grad(g[j], x, retain_graph=j < len(g) - 1)[0]
            q[j] = center(grad, batch)[j]
        return center(q, batch)
    if probes < 1:
        raise ValueError("Positive diffusion requires at least one probe")
    result = torch.zeros_like(x)
    for k in range(probes):
        signs = torch.randint(0, 2, g.shape, device=g.device, generator=generator).to(g) * 2 - 1
        grad = torch.autograd.grad(g, x, grad_outputs=signs, retain_graph=k < probes - 1)[0]
        result += center(signs[:, None] * center(grad, batch), batch)
    return result / probes
