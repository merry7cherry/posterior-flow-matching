from __future__ import annotations

import argparse
from collections.abc import Sequence

from pfm.cli.common import add_runtime_arguments, context_from_args, print_json, require_cpu_fp32
from pfm.cli.config_inputs import add_config_overrides, execution_config, execution_config_mapping
from pfm.core.evaluation_config import load_metric_evaluation_config, metric_evaluation_config_from_mapping
from pfm.core.layout import parse_operator_epoch, validate_path_component
from pfm.core.runtime import execution_device
from pfm.evaluation.checkpoint_evaluation import evaluate_bound_checkpoint


def _explicit_epoch(value: str) -> int:
    try:
        return parse_operator_epoch(value)
    except (TypeError, ValueError) as exc:
        raise argparse.ArgumentTypeError(str(exc)) from exc


def _artifact_id(value: str) -> str:
    try:
        return validate_path_component(value, "evaluation_id")
    except (TypeError, ValueError) as exc:
        raise argparse.ArgumentTypeError(str(exc)) from exc


class _StoreOnceAction(argparse.Action):
    def __call__(
        self,
        parser: argparse.ArgumentParser,
        namespace: argparse.Namespace,
        values: str,
        option_string: str | None = None,
    ) -> None:
        if getattr(namespace, self.dest, None) is not None:
            parser.error(f"{option_string} may be specified only once")
        setattr(namespace, self.dest, values)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Publish one explicit checkpoint-bound evaluation artifact.",
        allow_abbrev=False,
    )
    source = parser.add_mutually_exclusive_group(required=True)
    source.add_argument("--checkpoint", help="Portable training checkpoint file; output-dir must be new.")
    source.add_argument("--variant-root", help="Existing experiment variant directory.")
    parser.add_argument("--epoch", type=_explicit_epoch, action=_StoreOnceAction)
    parser.add_argument(
        "--evaluation-id",
        type=_artifact_id,
        action=_StoreOnceAction,
    )
    parser.add_argument("--config", "--eval-config", dest="eval_config", required=True)
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
    if args.evaluation_id is None:
        args.evaluation_id = "evaluation"
    return args


def main(argv: Sequence[str] | None = None) -> int:
    args = parse_args(argv)
    job = context_from_args(args)
    device = execution_device(job)
    if args.checkpoint is not None:
        from pfm.evaluation.checkpoint_evaluation import evaluate_checkpoint_file
        config = metric_evaluation_config_from_mapping(execution_config_mapping(
            job.runtime, args.eval_config, args.set, name="evaluation config", category="evaluate",
        ))
        require_cpu_fp32(job, config, category="evaluate")
        result = evaluate_checkpoint_file(args.checkpoint, config,
                                          output_dir=job.runtime.runs_root, job_context=job, device=device)
        print_json(result)
        return 0
    config_path = execution_config(job.runtime, args.eval_config, args.set, name="evaluation config", category="evaluate")
    evaluation_config = load_metric_evaluation_config(config_path)
    require_cpu_fp32(job, evaluation_config, category="evaluate")
    result = evaluate_bound_checkpoint(
        args.variant_root,
        args.epoch,
        args.evaluation_id,
        evaluation_config,
        evaluation_config_path=config_path,
        job_context=job,
        device=device,
    )
    print_json(result)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
