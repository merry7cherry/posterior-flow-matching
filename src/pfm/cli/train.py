from __future__ import annotations

import argparse
from collections.abc import Sequence

from pfm.cli.common import add_runtime_arguments, context_from_args, require_cpu_fp32

from pfm.core.config import load_config
from pfm.cli.config_inputs import execution_config
from pfm.core.distributed import cleanup_distributed, distributed_context_from_job
from pfm.training.cli import (
    add_training_arguments,
    resolve_training_variant_id,
)
from pfm.training.runner import train_variant


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Train one PFM variant.",
        allow_abbrev=False,
    )
    add_training_arguments(parser)
    add_runtime_arguments(parser)
    return parser


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    return build_parser().parse_args(argv)


def main(argv: Sequence[str] | None = None) -> int:
    args = parse_args(argv)
    job = context_from_args(args, distributed=True)
    config_path = execution_config(job.runtime, args.config, args.set, name="training config", category="train")
    config = load_config(config_path)
    require_cpu_fp32(job, config, category="train")
    distributed = distributed_context_from_job(job, initialize=True)
    try:
        variant_id = resolve_training_variant_id(args.variant, config)
        paths = train_variant(
            args.variant,
            config,
            job_context=job,
            distributed_context=distributed,
            task_id=args.task_id,
            variant_id=variant_id,
            epochs=None,
            show_progress=not args.quiet and distributed.is_main_process,
            resume_from=args.resume,
        )
        if distributed.is_main_process:
            for name, path in paths.items():
                print(f"{name}: {path}")
        return 0
    finally:
        cleanup_distributed()


if __name__ == "__main__":
    raise SystemExit(main())
