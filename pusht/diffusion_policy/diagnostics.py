"""Posterior readouts on held-out reference interpolations."""
import torch
from torch.utils.data import DataLoader
from diffusion_policy.common.pytorch_util import dict_apply
from diffusion_policy.model.pfm import log_variance


@torch.no_grad()
def reference_diagnostics(policy, dataset, *, batches=2, batch_size=2,
                          times=(.1, .5, .9, .99)):
    """Coordinate variance, residual second moment and Gaussian velocity NLL."""
    loader = DataLoader(dataset.get_validation_dataset(), batch_size=batch_size, shuffle=False)
    records = []
    for time in times:
        totals = torch.zeros(3, dtype=torch.float64, device=policy.device)
        count = 0
        for index, batch in enumerate(loader):
            if index >= batches:
                break
            batch = dict_apply(batch, lambda x: x.to(policy.device))
            condition = policy.encode_observation(batch['obs'])
            x1 = policy.normalizer['action'].normalize(batch['action']).float()
            x0 = policy.noise_scale * torch.randn_like(x1)
            t = torch.full((len(x1),), time, device=policy.device)
            velocity, raw = policy._field((1-time)*x0+time*x1, t, condition)
            ell = log_variance(raw, policy.logvar_min, policy.logvar_max)
            residual = (x1-x0-velocity.float()).square()
            nll = .5*(ell+residual*ell.neg().exp())
            totals += torch.stack((ell.exp().sum(), residual.sum(), nll.sum())).double()
            count += x1.numel()
        variance, mse, nll = (totals/count).cpu().tolist()
        records.append(dict(time=time, velocity_variance=variance, residual_mse=mse,
                            gaussian_nll_without_constant=nll,
                            endpoint_variance=(1-time)**2*variance))
    return records
