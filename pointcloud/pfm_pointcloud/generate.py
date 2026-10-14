"""Generate shapes from a PFM checkpoint."""
import argparse
from pathlib import Path
import numpy as np
import torch
from .generator import PFMGenerator, generate_clouds
from .visualize import save_plot
from .runtime import write_json


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--checkpoint', required=True)
    parser.add_argument('--output', required=True)
    parser.add_argument('--solver', choices=['ode', 'sde'], default='ode')
    parser.add_argument('--count', type=int, default=8)
    parser.add_argument('--steps', type=int)
    parser.add_argument('--diffusion-scale', type=float)
    parser.add_argument('--seed', type=int, default=9988)
    parser.add_argument('--device', default='cpu')
    parser.add_argument('--threads', type=int, default=2)
    args = parser.parse_args(argv)
    torch.set_num_threads(args.threads)
    checkpoint = torch.load(args.checkpoint, map_location='cpu', weights_only=False)
    model = PFMGenerator.from_checkpoint(checkpoint).to(args.device).eval().requires_grad_(False)
    points, snapshots = generate_clouds(model, checkpoint['config'], args.count,
                                       solver=args.solver, steps=args.steps,
                                       diffusion_scale=args.diffusion_scale, seed=args.seed)
    output = Path(args.output)
    output.mkdir(parents=True, exist_ok=True)
    np.save(output / 'points.npy', points.numpy())
    torch.save(snapshots, output / 'trajectory.pt')
    save_plot(points, output / 'samples.png')
    write_json(output / 'sampling.json', {**checkpoint['config']['evaluation'],
               **{key: value for key, value in vars(args).items() if value is not None}})


if __name__ == '__main__':
    main()
