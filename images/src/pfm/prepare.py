"""Download MNIST/CIFAR-10 or compute a plain FID reference statistics file."""
import argparse
from pathlib import Path
import numpy as np
import torch
from torch.utils.data import DataLoader, Subset
from torchvision import datasets, transforms
from pfm.config import ImageConfig, load_config
from pfm.data import make_dataset, CenterCropArr


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("mode", choices=["data", "fid"])
    parser.add_argument("--dataset", choices=["mnist", "cifar10"])
    parser.add_argument("--config")
    parser.add_argument("--data-root", required=True)
    parser.add_argument("--output", help="FID .npz output")
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--count", type=int, help="Optional small reference subset for software checks")
    parser.add_argument("--inception-weights")
    args = parser.parse_args()
    if args.mode == "data":
        if not args.dataset:
            parser.error("data preparation requires --dataset")
        config = ImageConfig(dataset=args.dataset)
        for train in [True, False]:
            dataset = make_dataset(config, args.data_root, train=train, download=True)
            print(f"{'train' if train else 'test'}: {len(dataset)} images")
        return
    if not args.config or not args.output:
        parser.error("FID preparation requires --config and --output")
    config = load_config(args.config)
    if config.dataset == "cifar10":
        dataset = datasets.CIFAR10(args.data_root, train=True, download=False, transform=transforms.PILToTensor())
    elif config.dataset == "imagenet":
        transform = transforms.Compose([CenterCropArr(config.image_size), transforms.PILToTensor()])
        dataset = datasets.ImageFolder(Path(args.data_root) / "train", transform=transform)
    else:
        parser.error("FID reference preparation supports CIFAR-10 and ImageNet")
    if args.count:
        dataset = Subset(dataset, range(min(args.count, len(dataset))))
    from pfm.metrics import feature_extractor, _OnlineFeatureStatistics
    extractor, layer, _ = feature_extractor(args.device, weights=args.inception_weights)
    stats = _OnlineFeatureStatistics()
    with torch.no_grad():
        for images, _ in DataLoader(dataset, batch_size=args.batch_size):
            values = extractor.convert_features_tuple_to_dict(extractor(images.to(args.device)))
            stats.update(values[layer])
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    np.savez(output, **stats.finalize())
    print(output)


if __name__ == "__main__":
    main()
