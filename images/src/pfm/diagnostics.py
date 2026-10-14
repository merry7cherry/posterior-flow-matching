"""Held-out reference-state raw residual moments and Gaussian NLL."""
import math
import torch
from torch.utils.data import DataLoader, Subset
from pfm.data import make_dataset
from pfm.sampling.samplers import predict


@torch.no_grad()
def posterior_diagnostics(model, config, data_root, device, *, num_samples=None,
                          batch_size=64, times=(.05, .1, .25, .5, .75, .9), bins=15,
                          seed=321, dataset=None):
    dataset = make_dataset(config, data_root, train=False) if dataset is None else dataset
    if num_samples is not None:
        dataset = Subset(dataset, range(min(num_samples, len(dataset))))
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=False)
    result = {}
    model.eval()
    for time_index, time in enumerate(times):
        if not 0 < time < 1:
            raise ValueError("diagnostic reference times must lie inside (0, 1)")
        rng = torch.Generator(device=device).manual_seed(seed + time_index)
        # Per-bin columns: count, residual, squared residual, predicted variance.
        grouped = torch.zeros(bins, 4, dtype=torch.float64, device=device)
        total = torch.zeros(5, dtype=torch.float64, device=device)
        edges = torch.linspace(math.exp(config.logvar_min), math.exp(config.logvar_max), bins + 1, device=device)
        images_seen = 0
        for x1, labels in loader:
            x1, labels = x1.to(device), labels.to(device)
            x0 = config.noise_scale * torch.randn(x1.shape, generator=rng, device=device)
            t = torch.full((len(x1), 1, 1, 1), time, device=device)
            xt = (1 - t) * x0 + t * x1
            pred = predict(model, xt, t, labels, config)
            residual = x1 - x0 - pred.velocity
            nll = 0.5 * (residual.square() * (-pred.logvar).exp() + pred.logvar + math.log(2 * math.pi))
            r, v = residual.flatten().double(), pred.velocity_var.flatten().double()
            total += torch.stack((torch.ones_like(r).sum(), r.sum(), r.square().sum(), v.sum(), nll.double().sum()))
            indices = torch.bucketize(v.float(), edges[1:-1]).clamp(0, bins - 1)
            for column, values in enumerate((torch.ones_like(r), r, r.square(), v)):
                grouped[:, column].scatter_add_(0, indices, values)
            images_seen += len(x1)
        count = total[0].item()
        if count == 0:
            raise ValueError("diagnostics need held-out images")
        summary = {"num_images": images_seen, "component_count": int(count),
                   "velocity_mean_residual": (total[1] / count).item(),
                   "velocity_mean_squared_residual": (total[2] / count).item(),
                   "velocity_mean_variance": (total[3] / count).item(),
                   "velocity_nll_per_coordinate": (total[4] / count).item(),
                   "endpoint_mean_residual": ((1 - time) * total[1] / count).item(),
                   "endpoint_mean_squared_residual": ((1 - time)**2 * total[2] / count).item(),
                   "endpoint_mean_variance": ((1 - time)**2 * total[3] / count).item(),
                   "endpoint_nll_per_coordinate": (total[4] / count).item() + math.log(1 - time),
                   "variance_bins": []}
        for index, row in enumerate(grouped.cpu()):
            if row[0] > 0:
                summary["variance_bins"].append({"lower": edges[index].item(), "upper": edges[index + 1].item(),
                    "count": int(row[0]), "mean_residual": float(row[1] / row[0]),
                    "mean_squared_residual": float(row[2] / row[0]), "mean_variance": float(row[3] / row[0])})
        result[str(time)] = summary
    return result
