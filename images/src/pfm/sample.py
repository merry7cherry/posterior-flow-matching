"""Generate images and one on-trajectory velocity/endpoint variance readout."""
import argparse
import json
from dataclasses import asdict
from pathlib import Path
import torch
from torchvision.utils import save_image
from pfm.checkpoint import load_model
from pfm.config import SamplingConfig, load_config
from pfm.generate import generate_batches


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--sampling", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--count", type=int, default=20)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--seed", type=int, default=321)
    parser.add_argument("--capture-time", type=float)
    parser.add_argument("--set", action="append", default=[])
    args = parser.parse_args()
    protocol = load_config(args.sampling, args.set, SamplingConfig)
    model, config = load_model(args.checkpoint, args.device)
    chunks = list(generate_batches(model, config, protocol, args.device, count=args.count,
                  batch_size=args.batch_size, seed=args.seed, capture_time=args.capture_time))
    values = {key: torch.cat([chunk[key] for chunk in chunks]) for key in ("samples", "labels", "velocity_variance", "endpoint_variance")}
    values["capture_time"] = chunks[0]["capture_time"]
    output = Path(args.output)
    output.mkdir(parents=True, exist_ok=True)
    torch.save(values, output / "samples.pt")
    save_image((values["samples"].clamp(-1, 1) + 1) / 2, output / "samples.png", nrow=min(10, args.count))
    # A fixed bound-based color scale makes different samples comparable.
    variance = values["endpoint_variance"].mean(1, keepdim=True)
    scale = (1 - values["capture_time"])**2
    lower, upper = scale * torch.exp(torch.tensor(config.logvar_min)), scale * torch.exp(torch.tensor(config.logvar_max))
    save_image(((variance - lower) / (upper - lower)).clamp(0, 1), output / "endpoint_variance.png", nrow=min(10, args.count))
    (output / "sampling.json").write_text(json.dumps({**asdict(protocol), "seed": args.seed, "capture_time": values["capture_time"]}, indent=2))
    print(output)


if __name__ == "__main__":
    main()
