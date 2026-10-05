from __future__ import annotations

import argparse

from pfm.core import variant_metadata
from pfm.core.config import ExperimentConfig
from pfm.core.layout import default_variant_id


def add_training_arguments(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--config", required=True)
    parser.add_argument("--set", action="append", default=[], metavar="FIELD=VALUE")
    parser.add_argument(
        "--variant",
        required=True,
        choices=sorted(variant_metadata.SUPPORTED_VARIANTS),
    )
    parser.add_argument("--task-id", default="run", help="Run id under output-dir/experiments (default: run).")
    parser.add_argument(
        "--resume",
        default=None,
        help="Resume training from an explicit checkpoint_manifest.json.",
    )
    parser.add_argument("--quiet", action="store_true")


def resolve_training_variant_id(
    variant: str,
    config: ExperimentConfig,
) -> str:
    return default_variant_id(variant, config)
