"""Generate and evaluate matched ODE/SDE point clouds with native Chamfer metrics."""
import argparse
import json
from pathlib import Path
import numpy as np
import torch
from .generator import PFMGenerator, generate_clouds
from .dataset import ShapeNetCore
from .runtime import write_json
from .metrics import distChamfer, lgan_mmd_cov, knn
from .visualize import save_plot


def bbox_normalize(x):
    low, high = x.amin(1, keepdim=True), x.amax(1, keepdim=True)
    scale = (high-low).amax(-1, keepdim=True)/2
    if not torch.isfinite(x).all() or (scale <= 0).any():
        raise ValueError('nonfinite or degenerate point cloud')
    return (x-(low+high)/2)/scale


@torch.no_grad()
def pairwise_cd(a, b, batch_size):
    rows = []
    for index in range(len(a)):
        row = []
        for start in range(0, len(b), batch_size):
            refs = b[start:start+batch_size]
            left, right = distChamfer(a[index:index+1].expand(len(refs), -1, -1).contiguous(), refs)
            # Squared distances have small roundoff negatives at coincident points.
            row.append(left.clamp_min(0).mean(1) + right.clamp_min(0).mean(1))
        rows.append(torch.cat(row))
    return torch.stack(rows)


@torch.no_grad()
def cd_metrics(generated, reference, batch_size, reference_distances=None):
    cross = pairwise_cd(generated, reference, batch_size)
    gg = pairwise_cd(generated, generated, batch_size)
    rr = reference_distances if reference_distances is not None else pairwise_cd(reference, reference, batch_size)
    metrics = lgan_mmd_cov(cross)
    nearest = knn(gg, cross, rr, 1)
    result = {key + '-CD': value.item() for key, value in metrics.items()}
    result.update({'1-NNA-CD': nearest['acc'].item(),
                   '1-NNA-generated-CD': nearest['acc_t'].item(),
                   '1-NNA-reference-CD': nearest['acc_f'].item()})
    return result


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--checkpoint', required=True)
    parser.add_argument('--dataset', required=True)
    parser.add_argument('--output', required=True)
    parser.add_argument('--device', default='cpu')
    parser.add_argument('--limit', type=int)
    parser.add_argument('--steps', type=int)
    parser.add_argument('--diffusion-scale', type=float)
    parser.add_argument('--threads', type=int, default=2)
    args = parser.parse_args(argv)
    torch.set_num_threads(args.threads)
    device = torch.device(args.device)
    checkpoint = torch.load(args.checkpoint, map_location='cpu', weights_only=False)
    config = checkpoint['config']
    protocol = config['evaluation'].copy()
    if args.steps is not None: protocol['steps'] = args.steps
    if args.diffusion_scale is not None: protocol['diffusion_scale'] = args.diffusion_scale
    config = dict(config, evaluation=protocol)
    model = PFMGenerator.from_checkpoint(checkpoint).to(device).eval().requires_grad_(False)
    dataset = ShapeNetCore(args.dataset, [config['category']], 'test', config['scale_mode'], limit=args.limit)
    points = torch.stack([item['pointcloud'] for item in dataset])
    reference = bbox_normalize(points).to(device)
    output = Path(args.output)
    output.mkdir(parents=True, exist_ok=True)
    results = {}
    rr = pairwise_cd(reference, reference, protocol['metric_batch_size'])
    for solver in ('ode', 'sde'):
        raw, snapshots = generate_clouds(model, config, len(points), solver=solver, seed=protocol['seed'])
        normalized = bbox_normalize(raw)
        np.save(output / (solver + '_raw.npy'), raw.numpy())
        torch.save(snapshots, output / (solver + '_trajectory.pt'))
        save_plot(normalized, output / (solver + '.png'))
        results[solver] = cd_metrics(normalized.to(device), reference, protocol['metric_batch_size'], rr)
        print(json.dumps({solver: results[solver]}), flush=True)
    write_json(output / 'results.json', {'config': protocol, 'count': len(points), 'results': results})


if __name__ == '__main__':
    main()
