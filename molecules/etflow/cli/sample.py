"""Sample molecular conformers and save a directly readable list of PyG Data."""
import argparse
from pathlib import Path

import numpy as np
import torch
from torch_geometric.data import Batch, Data
from etflow.data.dataset import EuclideanDataset
from etflow.models.pfm_model import PosteriorFlow


def generate(args):
    ckpt = torch.load(args.checkpoint, map_location='cpu', weights_only=False)
    cfg = ckpt['pfm_config']
    sampling = cfg['sampling'].copy()
    if args.steps is not None:
        sampling['n_timesteps'] = args.steps
    if args.diffusion_scale is not None:
        sampling['diffusion_scale'] = args.diffusion_scale
    chunk = sampling.pop('batch_size')
    dataset = EuclideanDataset(Path(args.data_dir or cfg['data_dir']), partition='qm9', split='test')
    model = PosteriorFlow(**cfg['model_args']).to(args.device).eval()
    model.load_state_dict(ckpt['state_dict'])
    packed = []
    count = min(len(dataset), args.max_molecules or len(dataset))
    for i in range(count):
        data = dataset[i]
        refs = torch.load(dataset.data_files[i], map_location='cpu', weights_only=False).pos.numpy()
        generated, variance = [], None
        for start in range(0, 2 * len(refs), chunk):
            size = min(chunk, 2 * len(refs) - start)
            b = Batch.from_data_list([data] * size).to(args.device)
            result = model.sample(b.atomic_numbers, b.edge_index, b.batch, b.node_attr, b.edge_attr,
                sampler_type=args.mode, seed=cfg['seed'] + i * 100000 + start,
                return_trajectory=start == 0, **sampling)
            if start == 0:
                result, frames = result
                variance = frames[-1]['endpoint_variance'][:len(data.pos)].numpy()
            generated.append(result.reshape(size, -1, 3).cpu().numpy())
        packed.append(Data(smiles=data.smiles, rdmol=data.mol, pos_ref=refs,
                           pos_gen=np.concatenate(generated), endpoint_variance=variance))
        print(f'{args.mode}: {i+1}/{count} molecules', flush=True)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    torch.save(packed, args.output)
    return packed


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--checkpoint', required=True, type=Path)
    parser.add_argument('--data-dir')
    parser.add_argument('--output', required=True, type=Path)
    parser.add_argument('--mode', choices=['ode', 'sde'], default='ode')
    parser.add_argument('--device', choices=['cpu', 'cuda'], default='cpu')
    parser.add_argument('--threads', type=int, default=2)
    parser.add_argument('--max-molecules', type=int)
    parser.add_argument('--steps', type=int)
    parser.add_argument('--diffusion-scale', type=float)
    args = parser.parse_args()
    torch.set_num_threads(args.threads)
    torch.set_float32_matmul_precision('highest')
    generate(args)


if __name__ == '__main__':
    main()
