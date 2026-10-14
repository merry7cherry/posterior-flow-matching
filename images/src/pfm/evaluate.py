"""Evaluate generated-image FID/IS or held-out PFM posterior diagnostics."""
import argparse
import json
from dataclasses import asdict
from pathlib import Path
import torch
from pfm.checkpoint import load_model
from pfm.config import SamplingConfig, load_config
from pfm.diagnostics import posterior_diagnostics
from pfm.generate import generate_batches


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("mode", choices=["generation", "posterior"])
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--output", required=True, help="JSON result path")
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--sampling")
    parser.add_argument("--fid-stats")
    parser.add_argument("--inception-weights")
    parser.add_argument("--inception-score", action="store_true")
    parser.add_argument("--data-root")
    parser.add_argument("--count", type=int)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--times", type=float, nargs="+", default=[.05, .1, .25, .5, .75, .9])
    parser.add_argument("--bins", type=int, default=15)
    parser.add_argument("--seed", type=int, default=321)
    parser.add_argument("--set", action="append", default=[])
    args = parser.parse_args()
    device = torch.device(args.device)
    model, config = load_model(args.checkpoint, device)
    if args.mode == "posterior":
        if args.data_root is None:
            parser.error("posterior diagnostics require --data-root")
        result = posterior_diagnostics(model, config, args.data_root, device, num_samples=args.count,
                    batch_size=args.batch_size, times=args.times, bins=args.bins, seed=args.seed)
    else:
        if not args.sampling or not args.fid_stats:
            parser.error("generation requires --sampling and --fid-stats")
        if config.image_channels != 3:
            parser.error("FID/IS use RGB CIFAR-10 or ImageNet; use posterior diagnostics for MNIST")
        from pfm.metrics import GenerationMetrics
        sampling = load_config(args.sampling, args.set, SamplingConfig)
        count = args.count or 50000
        metrics = GenerationMetrics(args.fid_stats, count, device, inception_score=args.inception_score, weights=args.inception_weights)
        for chunk in generate_batches(model, config, sampling, device, count=count, batch_size=args.batch_size, seed=args.seed):
            metrics.update(chunk["samples"])
        result = {**metrics.finalize(), "count": count, "sampling": asdict(sampling)}
    path = Path(args.output)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps({"seed": args.seed, "results": result}, indent=2))
    print(path)


if __name__ == "__main__":
    main()
