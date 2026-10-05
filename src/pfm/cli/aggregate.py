from __future__ import annotations

import argparse
from collections.abc import Sequence
from pathlib import Path, PurePosixPath

from pfm.cli.common import add_runtime_arguments, context_from_args, print_json
from pfm.cli.config_inputs import add_config_overrides, execution_config
from pfm.core.layout import parse_operator_epoch, validate_path_component
from pfm.core.execution import require_canonical_path
from pfm.core.yaml_config import load_yaml_mapping
from pfm.evaluation.evaluation_summary import (
    publish_aggregate_result,
    validate_comparison_spec,
)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Publish one explicit metadata-only comparison aggregate.",
        allow_abbrev=False,
    )
    parser.add_argument("--task-id", required=True)
    parser.add_argument("--comparison-config", required=True)
    add_config_overrides(parser)
    parser.add_argument("--epoch", required=True, type=parse_operator_epoch)
    parser.add_argument("--aggregate-id", required=True)
    parser.add_argument(
        "--input-manifest",
        action="append",
        required=True,
        help="Task-relative consumer_manifest.json; repeat for every required input.",
    )
    add_runtime_arguments(parser, device=False)
    return parser


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    return build_parser().parse_args(argv)


def main(argv: Sequence[str] | None = None) -> int:
    args = parse_args(argv)
    job = context_from_args(args)
    task_id = validate_path_component(args.task_id, "task_id")
    aggregate_id = validate_path_component(args.aggregate_id, "aggregate_id")
    comparison_path = execution_config(
        job.runtime, args.comparison_config, args.set,
        name="comparison config", category="aggregate",
    )
    comparison = validate_comparison_spec(load_yaml_mapping(comparison_path))
    task_root = require_canonical_path(
        job.runtime.experiments_root / task_id,
        job.runtime.experiments_root,
        name="task root",
    )
    inputs = [
        _resolve_task_relative_manifest(task_root, value)
        for value in args.input_manifest
    ]
    published = publish_aggregate_result(
        parent=task_root / "aggregates",
        aggregate_id=aggregate_id,
        comparison=comparison,
        manifest_paths=inputs,
        requested_epoch=args.epoch,
        aggregate_revision=job.git_commit,
    )
    print_json(
        {
            "aggregate_id": aggregate_id,
            "artifact_dir": str(published),
            "manifest": str(published / "aggregate_manifest.json"),
            "result": str(published / "aggregate.json"),
        }
    )
    return 0


def _resolve_task_relative_manifest(task_root: Path, value: object) -> Path:
    if not isinstance(value, str) or not value or "\\" in value:
        raise ValueError("input manifest must be a task-relative POSIX path")
    relative = PurePosixPath(value)
    if (
        relative.is_absolute()
        or any(part in {"", ".", ".."} for part in relative.parts)
        or relative.as_posix() != value
    ):
        raise ValueError("input manifest must not be absolute or escape the task root")
    path = require_canonical_path(
        task_root.joinpath(*relative.parts),
        task_root,
        name="aggregate input manifest",
    )
    if path.name != "consumer_manifest.json" or not path.is_file():
        raise FileNotFoundError(
            f"aggregate input must be an explicit consumer_manifest.json: {path}"
        )
    return path


if __name__ == "__main__":
    raise SystemExit(main())
