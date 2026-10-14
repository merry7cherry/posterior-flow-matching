"""Evaluate generated conformers with native symmetry-aware heavy-atom RMSD."""
import argparse
import json
from pathlib import Path

import numpy as np
import torch
from etflow.commons.covmat import CovMatEvaluator, print_covmat_results


def evaluate_metrics(input_path, output_dir, workers=1):
    packed = torch.load(input_path, map_location='cpu', weights_only=False)
    evaluator = CovMatEvaluator(num_workers=workers, use_force_field=False)
    results, _ = evaluator(packed)
    coverage, matching = print_covmat_results(results)
    row = int(np.argmin(abs(results['thresholds'] - 0.5)))
    summary = {**{k: float(v) for k, v in matching.items()},
        'COV-R_mean': float(results['CoverageR'][:, row].mean()),
        'COV-P_mean': float(results['CoverageP'][:, row].mean()),
        'molecules': len(results['CoverageR']), 'input_molecules': len(packed),
        'threshold_angstrom': 0.5}
    output_dir.mkdir(parents=True, exist_ok=True)
    coverage.to_csv(output_dir/'coverage.csv', index=False)
    (output_dir/'metrics.json').write_text(json.dumps(summary, indent=2))
    print(json.dumps(summary))
    return summary


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--input', type=Path, required=True)
    parser.add_argument('--output-dir', type=Path, required=True)
    parser.add_argument('--workers', type=int, default=1)
    args = parser.parse_args()
    torch.set_num_threads(2)
    evaluate_metrics(args.input, args.output_dir, args.workers)


if __name__ == '__main__':
    main()
