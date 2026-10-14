"""PFM in normalized action coordinates."""
import math
import torch


def log_variance(raw, low=-3.0, high=3.0):
    if not math.isfinite(low) or not math.isfinite(high) or not low < high:
        raise ValueError('invalid log variance bounds')
    return low + (high - low) * torch.sigmoid(raw.float())


def delta_loss(velocity, raw_logvar, target, low=-3.0, high=3.0):
    ell = log_variance(raw_logvar, low, high)
    residual = target.float() - velocity.float()
    return 0.5 * (ell + residual.square() * torch.exp(-ell)).mean()


def variance_diagonal(gamma, state, *, probes, generator=None, exact=False):
    """diag(d gamma / d state), with batch-independent network outputs."""
    count = state[0].numel() if exact else probes
    if count < 1:
        raise ValueError('positive derivative budget required')
    result = torch.zeros_like(state)
    if not gamma.requires_grad:
        return result, 0
    for i in range(count):
        if exact:
            z = torch.zeros(gamma.shape, device=gamma.device, dtype=gamma.dtype)
            z.reshape(len(z), -1)[:, i] = 1
        else:
            z = torch.randint(0, 2, gamma.shape, device=state.device,
                              generator=generator).to(state.dtype).mul_(2).sub_(1)
        grad = torch.autograd.grad(gamma, state, grad_outputs=z,
                                  retain_graph=i < count - 1,
                                  create_graph=False, allow_unused=True)[0]
        if grad is not None:
            result += z * grad
    return (result if exact else result / count).detach(), count


@torch.no_grad()
def sample_delta(field, initial, *, steps, terminal_time, solver,
                 diffusion_scale, probes, noise_scale, low, high,
                 brownian_generator=None, probe_generator=None, exact=False,
                 trace=None):
    """Matched-grid Euler / corrected Ito Euler-Maruyama, endpoint readout."""
    if type(steps) is not int or steps < 1 or not 0 < terminal_time < 1:
        raise ValueError('invalid sampling grid')
    if solver not in ('ode', 'sde') or not math.isfinite(diffusion_scale) or diffusion_scale < 0:
        raise ValueError('invalid solver or diffusion scale')
    if solver == 'ode' and diffusion_scale != 0:
        raise ValueError('ODE requires zero diffusion scale')
    if not math.isfinite(noise_scale) or noise_scale <= 0:
        raise ValueError('invalid source scale')
    if torch.is_inference_mode_enabled() and solver == 'sde' and diffusion_scale:
        raise RuntimeError('SDE requires no_grad/enable_grad, not inference_mode')
    x = initial.detach().float().clone()
    h = terminal_time / steps
    nvjp = 0
    def capture(state, time, velocity, raw, index):
        if trace is not None:
            trace.append({'t': float(time), 'call': index,
                          'x': state.detach().float().cpu().clone(),
                          'velocity': velocity.detach().float().cpu().clone(),
                          'logvar': log_variance(raw.detach(), low, high).cpu().clone()})
    finite = torch.isfinite(x).all()
    for k in range(steps):
        t = torch.full((len(x),), k * h, device=x.device, dtype=torch.float32)
        lam = diffusion_scale * (1 - k * h) if solver == 'sde' else 0.
        if lam:
            with torch.enable_grad():
                state = x.detach().requires_grad_(True)
                v, raw = field(state, t)
                gamma = log_variance(raw, low, high).exp()
                div, calls = variance_diagonal(gamma, state, probes=probes,
                                               generator=probe_generator, exact=exact)
            nvjp += calls
            capture(x, k * h, v, raw, k)
            v, gamma = v.detach().float(), gamma.detach()
            score = (t[:, None, None] * v - x) / (noise_scale**2 * (1-t[:, None, None]))
            drift = v + 0.5 * lam * (div + gamma * score)
            xi = torch.randn(x.shape, device=x.device, generator=brownian_generator)
            x = x + h * drift + (h * lam * gamma).sqrt() * xi
            finite &= torch.isfinite(div).all() & torch.isfinite(score).all()
        else:
            v, raw = field(x, t)
            capture(x, k * h, v, raw, k)
            x = x + h * v.float()
        x = x.detach()
        finite &= torch.isfinite(x).all() & torch.isfinite(raw).all()
        del v, raw
    t = torch.full((len(x),), terminal_time, device=x.device, dtype=torch.float32)
    v, raw = field(x, t)
    capture(x, terminal_time, v, raw, steps)
    gamma = log_variance(raw, low, high).exp()
    endpoint = x + (1-terminal_time) * v.float()
    if not bool(finite & torch.isfinite(endpoint).all() & torch.isfinite(gamma).all()):
        raise FloatingPointError('nonfinite PFM sample or coefficient')
    return endpoint, {'nfe': steps + 1, 'vjp': nvjp,
                      'endpoint_var': (1-terminal_time)**2 * gamma,
                      'velocity_var': gamma, 'source_t': terminal_time}
