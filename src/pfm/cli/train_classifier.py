from __future__ import annotations

import argparse
from collections.abc import Sequence

from pfm.cli.common import add_runtime_arguments, context_from_args, print_json, require_cpu_fp32
from pfm.cli.config_inputs import add_config_overrides, execution_config
from pfm.core.runtime import execution_device
from pfm.models.classifier import load_classifier_config
from pfm.training.classifier import train_classifier


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Train one immutable feature-classifier asset.",
        allow_abbrev=False,
    )
    parser.add_argument("--config", required=True)
    add_config_overrides(parser)
    parser.add_argument("--classifier-asset-id", required=True)
    add_runtime_arguments(parser)
    return parser


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    return build_parser().parse_args(argv)


def main(argv: Sequence[str] | None = None) -> int:
    args = parse_args(argv)
    job = context_from_args(args)
    config_path = execution_config(job.runtime, args.config, args.set, name="classifier config", category="train_classifier")
    config = load_classifier_config(config_path)
    require_cpu_fp32(job, config, category="train")
    device = execution_device(job)
    metrics = train_classifier(
        config,
        classifier_asset_id=args.classifier_asset_id,
        job_context=job,
        device=device,
    )
    print_json(metrics)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
