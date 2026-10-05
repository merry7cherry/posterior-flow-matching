"""Shared CLI config resolution. Execution consumes the resolved bytes it records."""
from __future__ import annotations

import argparse
from dataclasses import asdict
import hashlib
import os
import tempfile
from pathlib import Path


from pfm.core.yaml_config import load_yaml_mapping, merge_config, parse_overrides, canonical_config_bytes, parse_yaml_mapping
from pfm.core.execution import require_canonical_path


def add_config_overrides(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--set", action="append", default=[], metavar="FIELD=VALUE",
                        help="Override a field (including sampling.steps); repeat for distinct fields.")


def normalized_config(category: str, raw: dict) -> dict:
    """Delegate semantics to the existing owner, without experiment defaults."""
    if category == "train":
        from pfm.core.config import experiment_config_from_mapping, experiment_config_to_mapping
        return experiment_config_to_mapping(experiment_config_from_mapping(raw))
    if category == "evaluate":
        from pfm.core.evaluation_config import metric_evaluation_config_from_mapping
        return asdict(metric_evaluation_config_from_mapping(raw))
    if category == "sample":
        from pfm.core.sample_config import sample_visualization_config_from_mapping
        return asdict(sample_visualization_config_from_mapping(raw))
    if category == "sde_acceptance":
        from pfm.core.sde_acceptance_config import sde_acceptance_config_from_mapping
        return asdict(sde_acceptance_config_from_mapping(raw))
    if category == "prepare_fid":
        from pfm.evaluation.assets import fid_config_from_mapping, fid_config_to_mapping
        return fid_config_to_mapping(fid_config_from_mapping(raw))
    if category == "train_classifier":
        from pfm.models.classifier import validate_classifier_config_mapping
        return validate_classifier_config_mapping(raw, context="classifier config")
    if category == "aggregate":
        from pfm.evaluation.evaluation_summary import validate_comparison_spec
        return validate_comparison_spec(raw)
    raise ValueError(f"unsupported config category {category}")


def resolved_config_bytes(path: Path, items: list[str], *, category: str,
                          runtime=None, project_root: Path | None = None,
                          dependencies: dict[Path, bytes] | None = None) -> bytes:
    raw = merge_config(load_yaml_mapping(path), parse_overrides(items))
    if category == "aggregate":
        if runtime is None or dependencies is None:
            raise ValueError("comparison preparation requires runtime and dependency snapshots")
        from pfm.evaluation.evaluation_summary import prepare_comparison_config
        raw, snapshots = prepare_comparison_config(raw, runtime, project_root or runtime.project_root)
        dependencies.update(snapshots)
    return canonical_config_bytes(normalized_config(category, raw))


def execution_config(runtime, value: str, items: list[str], *, name: str, category: str) -> Path:
    # Freeze the complete effective input before any training or evaluation.
    path = runtime.config_file(value, name=name)
    dependencies = {}
    contents = resolved_config_bytes(path, items, category=category, runtime=runtime, dependencies=dependencies)
    # Already-frozen canonical inputs retain their original reference.
    if not items and path.is_relative_to(runtime.runs_root / "inputs") and path.read_bytes() == contents:
        return path
    digest = hashlib.sha256(contents).hexdigest()
    destination = runtime.runs_root / "inputs" / "configs" / f"{digest}.yaml"
    for target, payload in dependencies.items():
        publish_config_snapshot(runtime, target, payload)
    publish_config_snapshot(runtime, destination, contents)
    return destination


def execution_config_mapping(runtime, value: str, items: list[str], *, name: str, category: str) -> dict:
    """Resolve an independent checkpoint consumer without creating its output.

    These consumers publish the effective config with their complete artifact;
    pre-creating that final directory would undermine exclusive publication.
    """
    path = runtime.config_file(value, name=name)
    contents = resolved_config_bytes(path, items, category=category)
    return parse_yaml_mapping(contents, context=name)


def publish_config_snapshot(runtime, destination: Path, contents: bytes) -> None:
    require_canonical_path(destination, runtime.runs_root / "inputs", name="config snapshot")
    destination.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile(dir=destination.parent) as handle:
        handle.write(contents)
        handle.flush()
        os.fsync(handle.fileno())
        try:
            os.link(handle.name, destination)
        except FileExistsError:
            if destination.is_symlink() or destination.read_bytes() != contents:
                raise ValueError("resolved config collision")
