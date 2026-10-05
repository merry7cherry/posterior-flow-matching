from __future__ import annotations

import argparse
from collections.abc import Sequence

from pfm.cli.common import add_runtime_arguments, context_from_args

from pfm.cli.config_inputs import add_config_overrides, execution_config
from pfm.core.layout import parse_operator_epoch, validate_path_component
from pfm.core.sde_acceptance_config import load_sde_acceptance_config


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Run bounded checkpoint-bound PFM SDE accuracy/cost checks on CUDA.",
        allow_abbrev=False,
    )
    parser.add_argument("--variant-root", required=True)
    parser.add_argument("--epoch", required=True, type=parse_operator_epoch)
    parser.add_argument("--diagnostic-id", required=True,
                        type=lambda value: validate_path_component(value, "diagnostic_id"))
    parser.add_argument("--config", required=True)
    add_config_overrides(parser)
    add_runtime_arguments(parser)
    return parser


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    return build_parser().parse_args(argv)


def main(argv: Sequence[str] | None = None) -> int:
    args = parse_args(argv)
    job = context_from_args(args)
    path = execution_config(job.runtime, args.config, args.set, name="SDE acceptance config", category="sde_acceptance")
    config = load_sde_acceptance_config(path)
    from pfm.evaluation.sde_acceptance import run_sde_acceptance
    return run_sde_acceptance(args, config, config_path=path, job_context=job)


if __name__ == "__main__":
    raise SystemExit(main())
