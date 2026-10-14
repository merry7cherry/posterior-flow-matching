"""Plot one generated conformer colored by endpoint posterior covariance trace."""
import argparse
from pathlib import Path
import numpy as np
import torch


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--input', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--molecule', type=int, default=0, help='Molecule index in generated.pt')
    args = parser.parse_args()
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    data = torch.load(args.input, map_location='cpu', weights_only=False)[args.molecule]
    xyz = np.asarray(data.pos_gen[0])
    trace = 3 * np.asarray(data.endpoint_variance)
    fig = plt.figure(figsize=(6, 5))
    ax = fig.add_subplot(111, projection='3d')
    for bond in data.rdmol.GetBonds():
        points = xyz[[bond.GetBeginAtomIdx(), bond.GetEndAtomIdx()]]
        ax.plot(*points.T, color='0.6', linewidth=2)
    im = ax.scatter(*xyz.T, c=trace, cmap='magma', s=65)
    ax.set_box_aspect([1, 1, 1]); ax.set_axis_off()
    fig.colorbar(im, ax=ax, shrink=.6, label='Endpoint covariance trace (angstrom²)')
    label = 'Generated conformer; model posterior readout'
    if 'chirality' in data:
        label += '\nRaw-model variance after stereo correction'
    ax.set_title(label, fontsize=10)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(args.output, dpi=160, bbox_inches='tight')
    plt.close(fig)


if __name__ == '__main__':
    main()
