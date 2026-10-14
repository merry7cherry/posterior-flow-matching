"""Posterior moments on reference interpolations."""
import math
import torch


@torch.no_grad()
def posterior_diagnostics(model, points, times, batch_size, seed, device):
    results = {}
    low, high = model.net.logvar_min, model.net.logvar_max
    for ti, time_value in enumerate([*times, None]):
        # One independent, reproducible stream per diagnostic time and suite.
        rng = torch.Generator(device=device).manual_seed(seed + 1000 + ti)
        totals = torch.zeros(8, dtype=torch.float64, device=device)
        bins = torch.zeros(10, 3, dtype=torch.float64, device=device)
        for start in range(0, len(points), batch_size):
            x1 = points[start:start+batch_size].to(device)
            z = model.encode(x1, rng)
            x0 = torch.randn(x1.shape, device=device, generator=rng)
            t = (0.001+0.998*torch.rand(len(x1), device=device, generator=rng)
                 if time_value is None else x1.new_full((len(x1),), time_value))
            xt = (1-t[:, None, None])*x0 + t[:, None, None]*x1
            velocity, ell = model.net(xt, t, z)
            gamma = ell.exp()
            residual = x1-x0-velocity
            error = residual.square()
            nll = 0.5 * (math.log(2*math.pi)+ell+error/gamma)
            endpoint_scale = (1-t[:, None, None]).square()
            entries = [error.sum(), gamma.sum(), nll.sum(), (endpoint_scale*error).sum(),
                       (endpoint_scale*gamma).sum(), (nll+torch.log1p(-t[:, None, None])).sum(),
                       (ell <= low+.01).sum(), (ell >= high-.01).sum()]
            totals += torch.stack(entries).double()
            indexes = ((ell-low)*10/(high-low)).long().clamp(0,9).flatten()
            bins[:,0].scatter_add_(0,indexes,torch.ones_like(error.flatten(),dtype=torch.float64))
            bins[:,1].scatter_add_(0,indexes,gamma.flatten().double())
            bins[:,2].scatter_add_(0,indexes,error.flatten().double())
        count = points.numel()
        values = (totals/count).cpu().tolist()
        keys = ['velocity_mse','velocity_variance','velocity_nll','endpoint_mse',
                'endpoint_variance','endpoint_nll','near_lower_bound_fraction','near_upper_bound_fraction']
        record = dict(zip(keys,values))
        record['logvar_bounds'] = [low, high]
        record['error_variance_ratio'] = values[0]/values[1]
        record['calibration_bins_count_variance_sum_error_sum'] = bins.cpu().tolist()
        record['coordinate_count'] = count
        results['random_time' if time_value is None else str(time_value)] = record
    return results


def main(argv=None):
    import argparse
    from pathlib import Path
    from .generator import PFMGenerator
    from .dataset import ShapeNetCore
    from .runtime import write_json
    parser = argparse.ArgumentParser(description='PFM moments and NLL on held-out reference states')
    parser.add_argument('--checkpoint', required=True)
    parser.add_argument('--dataset', required=True)
    parser.add_argument('--output', required=True)
    parser.add_argument('--limit', type=int)
    parser.add_argument('--device', default='cpu')
    parser.add_argument('--threads', type=int, default=2)
    args = parser.parse_args(argv)
    torch.set_num_threads(args.threads)
    ckpt = torch.load(args.checkpoint, map_location='cpu', weights_only=False)
    config = ckpt['config']; protocol = config['evaluation']
    device = torch.device(args.device)
    model = PFMGenerator.from_checkpoint(ckpt).to(device).eval()
    dataset = ShapeNetCore(args.dataset, [config['category']], 'test', config['scale_mode'], limit=args.limit)
    points = torch.stack([item['pointcloud'] for item in dataset])
    result = posterior_diagnostics(model, points, protocol['diagnostic_times'], protocol['batch_size'], protocol['seed'], device)
    Path(args.output).parent.mkdir(parents=True, exist_ok=True)
    write_json(args.output, result)


if __name__ == '__main__':
    main()
