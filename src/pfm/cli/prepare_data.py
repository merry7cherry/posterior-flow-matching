from __future__ import annotations

import argparse
from collections.abc import Sequence

from pfm.core.execution import DATASET_DIR_NAMES, runtime_paths
from pfm.data.preparation import prepare_dataset


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Prepare official dataset splits and their PFM identity manifest.",
        allow_abbrev=False,
    )
    parser.add_argument("--dataset", required=True, choices=sorted(DATASET_DIR_NAMES))
    parser.add_argument("--data-root", default="data",
                        help="Parent of mnist/, cifar10/, and imagenet1k-ilsvrc2012/.")
    parser.add_argument("--download", action="store_true",
                        help="Explicitly download MNIST/CIFAR-10 with torchvision.")
    return parser


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    return build_parser().parse_args(argv)


def main(argv: Sequence[str] | None = None) -> int:
    args = parse_args(argv)
    root = runtime_paths(data_root=args.data_root).dataset_root(args.dataset)
    print(prepare_dataset(args.dataset, root, download=args.download))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
