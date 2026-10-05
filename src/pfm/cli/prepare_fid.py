from __future__ import annotations

import argparse
from collections.abc import Sequence

from pfm.cli.common import add_runtime_arguments, context_from_args
from pfm.cli.config_inputs import add_config_overrides, execution_config

from pfm.core.runtime import execution_device
from pfm.evaluation.fid_reference import load_fid_config, prepare_fid_asset


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Prepare one immutable FID statistics asset.",
        allow_abbrev=False,
    )
    parser.add_argument("--config", required=True)
    add_config_overrides(parser)
    parser.add_argument("--fid-asset-id", required=True)
    parser.add_argument("--download-weights", action="store_true",
                        help="Download and verify official Inception weights before preparing statistics.")
    add_runtime_arguments(parser)
    return parser


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    return build_parser().parse_args(argv)


def main(argv: Sequence[str] | None = None) -> int:
    args = parse_args(argv)
    job = context_from_args(args)
    config_path = execution_config(job.runtime, args.config, args.set, name="FID config", category="prepare_fid")
    config = load_fid_config(config_path)
    device = execution_device(job)
    if args.download_weights:
        from pfm.evaluation.fid_metrics import prepare_inception_weights
        prepare_inception_weights()
    output_path = prepare_fid_asset(
        config,
        fid_asset_id=args.fid_asset_id,
        job_context=job,
        device=device,
    )
    print(output_path)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
