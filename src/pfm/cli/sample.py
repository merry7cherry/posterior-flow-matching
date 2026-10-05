from __future__ import annotations

import argparse
from collections.abc import Sequence

from pfm.cli.common import add_runtime_arguments, context_from_args, require_cpu_fp32
from pfm.cli.config_inputs import add_config_overrides, execution_config, execution_config_mapping

from pfm.core.layout import parse_operator_epoch
from pfm.core.runtime import execution_device
from pfm.core.sample_config import load_sample_visualization_config, sample_visualization_config_from_mapping
from pfm.evaluation.sample_grid import generate_sample_artifact


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Generate one checkpoint-bound visualization artifact.",
        allow_abbrev=False,
    )
    source = parser.add_mutually_exclusive_group(required=True)
    source.add_argument("--checkpoint", help="Portable training checkpoint file; output-dir must be new.")
    source.add_argument("--variant-root", help="Existing experiment variant directory.")
    parser.add_argument("--epoch", type=parse_operator_epoch)
    parser.add_argument("--sample-id", default="samples")
    parser.add_argument("--config", "--sample-config", dest="sample_config", required=True)
    add_config_overrides(parser)
    add_runtime_arguments(parser, checkpoint_output=True)
    return parser


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = build_parser()
    args = parser.parse_args(argv)
    if args.variant_root is not None and args.epoch is None:
        parser.error("--variant-root requires --epoch")
    if args.checkpoint is not None and args.epoch is not None:
        parser.error("--checkpoint contains its epoch; do not pass --epoch")
    return args


def main(argv: Sequence[str] | None = None) -> int:
    args = parse_args(argv)
    job = context_from_args(args)
    device = execution_device(job)
    if args.checkpoint is not None:
        from pfm.evaluation.sample_grid import sample_checkpoint_file
        config = sample_visualization_config_from_mapping(execution_config_mapping(
            job.runtime, args.sample_config, args.set, name="sample config", category="sample",
        ))
        require_cpu_fp32(job, config, category="sample")
        print(sample_checkpoint_file(args.checkpoint, config,
                                     output_dir=job.runtime.runs_root, device=device,
                                     consumer_revision=job.git_commit))
        return 0
    config_path = execution_config(job.runtime, args.sample_config, args.set, name="sample config", category="sample")
    sample_config = load_sample_visualization_config(config_path)
    require_cpu_fp32(job, sample_config, category="sample")
    artifact_dir = generate_sample_artifact(
        args.variant_root,
        args.epoch,
        args.sample_id,
        sample_config,
        sample_config_path=config_path,
        job_context=job,
        device=device,
    )
    print(artifact_dir)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
