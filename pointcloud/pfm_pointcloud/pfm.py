"""Velocity PFM on the native, pointwise coordinate network.

No cross-point operations may be added without replacing pointwise_divergence.
"""
from contextlib import nullcontext
import math

import torch
from torch import nn

from .common import ConcatSquashLinear
from .pointwise import PointwiseNet

class PointwisePFM(PointwiseNet):
    def __init__(self, context_dim, residual=True, logvar_min=-3.0, logvar_max=3.0):
        super().__init__(3, context_dim, residual)
        if not (math.isfinite(logvar_min) and math.isfinite(logvar_max) and logvar_min < logvar_max):
            raise ValueError('log-variance bounds must be finite and ordered')
        self.logvar_min, self.logvar_max = float(logvar_min), float(logvar_max)
        self.logvar_head = ConcatSquashLinear(128, 3, context_dim + 3)
        # Zero raw output starts at the bounds midpoint (ell=0 for [-3,3]).
        for parameter in self.logvar_head.parameters():
            nn.init.zeros_(parameter)

    def forward(self, x, t, context):
        hidden, ctx = self.hidden(x, t, context)
        velocity = self.layers[-1](ctx, hidden)
        if self.residual:
            velocity = velocity + x
        raw = self.logvar_head(ctx, hidden)
        return velocity.float(), self.logvar_min + (self.logvar_max-self.logvar_min) * torch.sigmoid(raw.float())


def network_context(device, precision):
    if precision not in ('fp32', 'bf16'):
        raise ValueError('precision must be fp32 or bf16')
    if precision == 'bf16' and device.type != 'cuda':
        raise ValueError('BF16 execution requires CUDA; use fp32 for CPU tests')
    return torch.autocast('cuda', dtype=torch.bfloat16) if precision == 'bf16' else nullcontext()


def delta_loss(velocity, logvar, x0, x1):
    residual = x1.float() - x0.float() - velocity.float()
    return 0.5 * (logvar.float() + residual.square() * torch.exp(-logvar.float())).mean()


def pointwise_divergence(gamma, x):
    """Exact diagonal of the block-3x3 Jacobian, with three VJPs.

At fixed latent/time, output [b,n,:] depends only on input [b,n,:].
Within each XYZ block, the Jacobian is generally dense.
"""
    if not gamma.requires_grad:
        return torch.zeros_like(x)
    diagonals = []
    for axis in range(3):
        grad = torch.autograd.grad(gamma[..., axis].sum(), x,
                                   retain_graph=axis < 2, create_graph=False,
                                   allow_unused=True)[0]
        diagonals.append(torch.zeros_like(x[..., axis]) if grad is None else grad[..., axis])
    return torch.stack(diagonals, dim=-1).detach()


@torch.no_grad()
def sample(net, z, initial_noise, *, solver='ode', steps=100, terminal_time=0.99,
           diffusion_scale=0.1, precision='fp32', brownian=None, capture=False):
    """Euler / fully corrected Ito Euler-Maruyama; endpoint mean readout.

Source scale is fixed to one. Only the current state retains input AD.
The caller supplies independent generators for initialization and Brownian noise.
"""
    if solver not in ('ode', 'sde') or type(steps) is not int or steps < 1:
        raise ValueError('invalid solver/steps')
    if not 0 < terminal_time < 1 or not math.isfinite(diffusion_scale) or diffusion_scale < 0:
        raise ValueError('invalid terminal time/diffusion scale')
    if initial_noise.ndim != 3 or initial_noise.shape[-1] != 3 or z.shape[0] != initial_noise.shape[0]:
        raise ValueError('expected [B,N,3] state and [B,F] latent')
    active_sde = solver == 'sde' and diffusion_scale > 0
    if active_sde and torch.is_inference_mode_enabled():
        raise RuntimeError('SDE requires no_grad/enable_grad, not inference_mode')
    net.eval()
    z, x = z.detach().float(), initial_noise.detach().float().clone()
    h = terminal_time / steps
    snapshots, actual_vjp = [], 0
    flags = [p.requires_grad for p in net.parameters()]
    net.requires_grad_(False)
    try:
        for k in range(steps):
            ts = k * h
            t = x.new_full((len(x),), ts)
            if active_sde:
                with torch.enable_grad():
                    current = x.detach().requires_grad_(True)
                    with network_context(x.device, precision):
                        velocity, ell = net(current, t, z)
                    gamma = ell.exp()
                    divergence = pointwise_divergence(gamma, current)
                velocity, gamma = velocity.detach(), gamma.detach()
                score = (ts * velocity - x) / (1.0 - ts)
                lam = diffusion_scale * (1.0 - ts)
                drift = velocity + 0.5 * lam * (divergence + gamma * score)
                xi = torch.randn(x.shape, device=x.device, generator=brownian)
                next_x = x + h * drift + (h * lam * gamma).sqrt() * xi
                actual_vjp += 3
                del current, divergence, ell, score, drift
            else:
                with network_context(x.device, precision):
                    velocity, ell = net(x, t, z)
                gamma = ell.exp()
                next_x = x + h * velocity
            if capture and k in {0, steps // 2, steps - 1}:
                snapshots.append({'t': ts, 'x': x.cpu(), 'mu_x': (x+(1-ts)*velocity).cpu(), 'gamma_v': gamma.cpu(),
                                  'gamma_x': ((1-ts)**2 * gamma).cpu()})
            x = next_x.detach()
            del velocity, gamma, next_x
        with network_context(x.device, precision):
            velocity, ell = net(x, x.new_full((len(x),), terminal_time), z)
        result = x + (1.0 - terminal_time) * velocity
        if not torch.isfinite(result).all():
            raise FloatingPointError('nonfinite generated point cloud')
        return result, {'nfe': steps + 1, 'vjp': actual_vjp, 'snapshots': snapshots}
    finally:
        for parameter, flag in zip(net.parameters(), flags):
            parameter.requires_grad_(flag)
