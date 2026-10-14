"""Joint PFM with the native PointNet encoder and invertible latent prior."""
from argparse import Namespace
import math

import torch
from torch import nn
from .encoder import PointNetEncoder
from .flow import build_latent_flow, add_spectral_norm, SpectralNorm
from .pfm import PointwisePFM, sample
from .common import gaussian_entropy


class PFMGenerator(nn.Module):
    def __init__(self, native_args, logvar_min=-3.0, logvar_max=3.0):
        super().__init__()
        if native_args.model != 'flow':
            raise ValueError('PFM uses the native flow-prior condition model')
        self.native_args = vars(native_args).copy()
        self.encoder = PointNetEncoder(native_args.latent_dim)
        self.prior = build_latent_flow(native_args)
        self.net = PointwisePFM(native_args.latent_dim, native_args.residual, logvar_min, logvar_max)

    @classmethod
    def from_official(cls, path, logvar_min=-3.0, logvar_max=3.0):
        # Official assets contain argparse.Namespace: only load trusted files.
        checkpoint = torch.load(path, map_location='cpu', weights_only=False)
        args = checkpoint['args']
        if isinstance(args, dict):
            args = Namespace(**args)
        model = cls(args, logvar_min, logvar_max)
        state = checkpoint['state_dict']
        encoder = {k.removeprefix('encoder.'): v for k, v in state.items() if k.startswith('encoder.')}
        prior = {k.removeprefix('flow.'): v for k, v in state.items() if k.startswith('flow.')}
        model.encoder.load_state_dict(encoder, strict=True)
        if any(k.endswith('weight_orig') for k in prior):
            add_spectral_norm(model.prior)
        model.prior.load_state_dict(prior, strict=True)
        # Preserve effective upstream weights, including spectral-normalized assets.
        for module in model.prior.modules():
            for key, hook in list(module._forward_pre_hooks.items()):
                if isinstance(hook, SpectralNorm):
                    hook.compute_weight(module, 0)
                    weight = module.weight.detach().clone()
                    del module._forward_pre_hooks[key]
                    for name in ('weight', 'weight_u', 'weight_v', 'weight_orig'):
                        delattr(module, name)
                    module.register_parameter('weight', nn.Parameter(weight))
        return model

    @classmethod
    def from_checkpoint(cls, checkpoint):
        config = checkpoint['config']
        model = cls(Namespace(**checkpoint['native_args']), config['logvar_min'], config['logvar_max'])
        model.load_state_dict(checkpoint['state_dict'], strict=True)
        return model

    def encode(self, x, generator=None):
        return self.encode_with_kl(x, generator, compute_kl=False)[0]

    def encode_with_kl(self, x, generator=None, compute_kl=True):
        mu, logvar = self.encoder(x.float())
        noise = torch.randn(mu.shape, device=mu.device, generator=generator)
        z = mu + (0.5 * logvar).exp() * noise
        kl = z.new_zeros(())
        if compute_kl:
            w, delta_logp = self.prior(z, z.new_zeros((len(z), 1)))
            log_pz = (-.5 * (w.square() + math.log(2 * math.pi))).sum(1) - delta_logp.flatten()
            kl = (-gaussian_entropy(logvar) - log_pz).mean()
        return z, kl

    def training_parameters(self):
        return tuple(p for p in self.parameters() if p.requires_grad)

    @torch.no_grad()
    def prior_sample(self, w):
        return self.prior(w.float(), reverse=True)


def generate_clouds(model, config, count, *, solver='ode', steps=None,
                    diffusion_scale=None, seed=9988):
    """Sample the learned latent prior and then the coordinate PFM."""
    device = next(model.parameters()).device
    protocol = config['evaluation']
    rng = torch.Generator(device=device).manual_seed(seed)
    brownian = torch.Generator(device=device).manual_seed(seed + 100)
    clouds, snapshots = [], []
    for start in range(0, count, protocol['batch_size']):
        size = min(protocol['batch_size'], count - start)
        w = torch.randn((size, model.native_args['latent_dim']), device=device, generator=rng)
        z = model.prior_sample(w)
        noise = torch.randn((size, config['num_points'], 3), device=device, generator=rng)
        points, stats = sample(
            model.net, z, noise, solver=solver,
            steps=protocol['steps'] if steps is None else steps,
            terminal_time=protocol['terminal_time'],
            diffusion_scale=protocol['diffusion_scale'] if diffusion_scale is None else diffusion_scale,
            precision=protocol[solver + '_precision'], brownian=brownian, capture=start == 0)
        clouds.append(points.cpu())
        if start == 0:
            snapshots = stats['snapshots']
    return torch.cat(clouds), snapshots
