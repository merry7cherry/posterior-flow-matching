from __future__ import annotations

import hashlib
import json
import os
import shutil
from collections.abc import Mapping, Sequence
from pathlib import Path

from pfm.artifacts.consumer_artifact import (
    load_consumer_artifact,
    validate_relative_reference,
)
from pfm.core.identities import validate_dataset_identity
from pfm.core.immutable_artifact import (
    prepare_immutable_directory,
    publish_immutable_directory,
)
from pfm.core.layout import validate_path_component
from pfm.core.variant_metadata import CANONICAL_METHOD_PROFILES
from pfm.core.execution import DATASET_DIR_NAMES
from pfm.data import identity as dataset_identity


AGGREGATE_RESULT_NAME = "aggregate.json"
AGGREGATE_MANIFEST_NAME = "aggregate_manifest.json"
_NUMERICAL_COMPARISON_PROFILES = frozenset(
    {"generated_common", "posterior_diagnostics"}
)
_AGGREGATE_MANIFEST_KEYS = frozenset(
    {
        "schema_version",
        "aggregate_revision",
        "comparison_id",
        "comparison_sha256",
        "aggregate_result",
    }
)
_AGGREGATE_RESULT_REFERENCE_KEYS = frozenset({"path", "sha256"})
_COMPARISON_KEYS = frozenset(
    {
        "comparison_id",
        "dataset",
        "required_methods",
        "required_consumers",
        "consumer_protocols",
    }
)


def aggregate_consumer_results(
    comparison: Mapping[str, object],
    manifest_paths: Sequence[str | Path],
    *,
    task_root: str | Path,
    requested_epoch: int,
) -> dict[str, object]:
    spec = validate_comparison_spec(comparison)
    if type(requested_epoch) is not int or requested_epoch <= 0:
        raise ValueError("aggregate requested epoch must be a positive integer")
    if isinstance(manifest_paths, (str, bytes)) or not isinstance(manifest_paths, Sequence):
        raise TypeError("aggregate manifest_paths must be an explicit sequence")
    if not manifest_paths:
        raise ValueError("aggregate requires at least one explicit consumer manifest")
    resolved_task_root = Path(os.path.abspath(Path(task_root)))

    expected = {
        (method, profile)
        for method, profiles in spec["required_consumers"].items()
        for profile in profiles
    }
    observed: dict[tuple[str, str], dict[str, object]] = {}
    source_bindings: list[dict[str, object]] = []
    training_protocols: set[str] = set()
    dataset_identities: set[str] = set()
    dataset_references: set[str] = set()
    consumer_revisions: set[str | None] = set()
    checkpoint_revisions: set[str | None] = set()
    checkpoint_progresses: set[tuple[int, int]] = set()
    checkpoints_by_method: dict[str, set[str]] = {}
    generated_assets: set[str] = set()
    consumer_protocols: dict[str, set[str]] = {}
    for manifest_path in manifest_paths:
        resolved_manifest_path = Path(os.path.abspath(Path(manifest_path)))
        manifest_reference = _task_relative_reference(
            resolved_manifest_path,
            task_root=resolved_task_root,
            sha256=None,
        )
        manifest, numerical = load_consumer_artifact(resolved_manifest_path)
        binding = manifest["binding"]
        numerical_reference = manifest["numerical_result"]
        assert isinstance(binding, dict)
        assert isinstance(numerical_reference, dict)
        method = str(binding["method"])
        profile = str(binding["profile"])
        key = (method, profile)
        if key not in expected:
            raise ValueError(
                f"aggregate received unexpected method/consumer coverage: {method}/{profile}"
            )
        if key in observed:
            raise ValueError(
                f"aggregate received duplicate method/consumer coverage: {method}/{profile}"
            )
        training_protocols.add(str(binding["shared_training_protocol_sha256"]))
        dataset_identities.add(str(binding["dataset_scientific_identity_sha256"]))
        dataset_reference = binding["dataset"]
        assert isinstance(dataset_reference, dict)
        expected_dataset_path = (
            f"{DATASET_DIR_NAMES[str(spec['dataset'])]}/"
            f"{dataset_identity.DATASET_MANIFEST_NAME}"
        )
        if (
            dataset_reference["owner"] != "datasets"
            or dataset_reference["path"] != expected_dataset_path
        ):
            raise ValueError(
                "consumer dataset reference does not match comparison dataset"
            )
        dataset_references.add(_canonical_json_text(dataset_reference))
        consumer_revisions.add(binding["consumer_revision"])
        checkpoint_revisions.add(binding["checkpoint_revision"])
        checkpoint_progress = binding["checkpoint_progress"]
        assert isinstance(checkpoint_progress, dict)
        checkpoint_progresses.add(
            (int(checkpoint_progress["epoch"]), int(checkpoint_progress["step"]))
        )
        checkpoint_reference = binding["checkpoint"]
        assert isinstance(checkpoint_reference, dict)
        checkpoints_by_method.setdefault(method, set()).add(
            str(checkpoint_reference["sha256"])
        )
        if profile == "generated_common":
            generated_assets.add(_canonical_json_text(binding["assets"]))
        consumer_protocols.setdefault(profile, set()).add(
            str(binding["config"]["sha256"])
        )
        if binding["config"]["sha256"] != spec["consumer_protocols"][profile]["sha256"]:
            raise ValueError(
                f"consumer config does not match comparison protocol for {profile}"
            )
        aggregate_summary = numerical.get("aggregate")
        if not isinstance(aggregate_summary, dict):
            raise ValueError("canonical numerical result aggregate summary must be a mapping")
        result_reference = _task_relative_reference(
            resolved_manifest_path.parent / str(numerical_reference["path"]),
            task_root=resolved_task_root,
            sha256=str(numerical_reference["sha256"]),
        )
        record = {
            "method": method,
            "profile": profile,
            "checkpoint": binding["checkpoint"],
            "dataset": binding["dataset"],
            "config": binding["config"],
            "assets": binding["assets"],
            "numerical_result_sha256": numerical_reference["sha256"],
            "source": {
                "consumer_manifest": manifest_reference,
                "numerical_result": result_reference,
            },
            "summary": aggregate_summary,
        }
        observed[key] = record
        source_bindings.append(
            {
                "method": method,
                "profile": profile,
                "checkpoint": binding["checkpoint"],
                "dataset": binding["dataset"],
                "config": binding["config"],
                "assets": binding["assets"],
                "numerical_result_sha256": numerical_reference["sha256"],
                "consumer_manifest": manifest_reference,
                "numerical_result": result_reference,
            }
        )

    missing = sorted(expected - set(observed))
    if missing:
        formatted = ", ".join(f"{method}/{profile}" for method, profile in missing)
        raise ValueError(f"aggregate coverage is missing required results: {formatted}")

    if len(training_protocols) != 1:
        raise ValueError("consumer results have a shared training protocol mismatch")
    if len(dataset_identities) != 1:
        raise ValueError("consumer results have a dataset protocol binding mismatch")
    if len(dataset_references) != 1:
        raise ValueError("consumer results have a dataset reference mismatch")
    if len(consumer_revisions) != 1:
        raise ValueError("consumer results have a consumer revision mismatch")
    if len(checkpoint_revisions) != 1:
        raise ValueError("consumer results have a checkpoint revision mismatch")
    mixed_checkpoints = sorted(
        method for method, digests in checkpoints_by_method.items() if len(digests) != 1
    )
    if mixed_checkpoints:
        raise ValueError(
            "consumer results mix checkpoints within method: "
            + ", ".join(mixed_checkpoints)
        )
    if len(checkpoint_progresses) != 1:
        raise ValueError("consumer results have a checkpoint progress mismatch")
    checkpoint_epoch, checkpoint_step = next(iter(checkpoint_progresses))
    if checkpoint_epoch != requested_epoch:
        raise ValueError(
            f"aggregate requested epoch {requested_epoch}, "
            f"but consumer results bind epoch {checkpoint_epoch}"
        )
    if "generated_common" in spec["consumer_protocols"] and len(generated_assets) != 1:
        raise ValueError("generated consumer results have an asset binding mismatch")
    mismatched_profiles = sorted(
        profile for profile, digests in consumer_protocols.items() if len(digests) != 1
    )
    if mismatched_profiles:
        raise ValueError(
            "consumer results have a profile protocol mismatch: "
            + ", ".join(mismatched_profiles)
        )

    results = [observed[key] for key in sorted(observed)]
    source_bindings.sort(key=lambda item: (str(item["method"]), str(item["profile"])))
    return {
        "comparison_id": spec["comparison_id"],
        "dataset": spec["dataset"],
        "shared_training_protocol_sha256": next(iter(training_protocols)),
        "consumer_config_sha256": {
            profile: next(iter(digests))
            for profile, digests in sorted(consumer_protocols.items())
        },
        "dataset_scientific_identity_sha256": next(iter(dataset_identities)),
        "consumer_revision": next(iter(consumer_revisions)),
        "checkpoint_revision": next(iter(checkpoint_revisions)),
        "checkpoint_progress": {"epoch": checkpoint_epoch, "step": checkpoint_step},
        "source_set_sha256": _canonical_json_sha256(source_bindings),
        "results": results,
    }


def publish_aggregate_result(
    *,
    parent: str | Path,
    aggregate_id: str,
    comparison: Mapping[str, object],
    manifest_paths: Sequence[str | Path],
    requested_epoch: int,
    aggregate_revision: str,
) -> Path:
    clean_revision = _require_revision(aggregate_revision, "aggregate")
    clean_id = validate_path_component(aggregate_id, "aggregate_id")
    spec = validate_comparison_spec(comparison)
    result = aggregate_consumer_results(
        spec,
        manifest_paths,
        task_root=Path(parent).parent,
        requested_epoch=requested_epoch,
    )
    staging, target = prepare_immutable_directory(Path(parent), clean_id)
    try:
        result_path = staging / AGGREGATE_RESULT_NAME
        result_bytes = _write_canonical_json(result_path, result)
        manifest = _validate_aggregate_manifest(
            {
                "schema_version": 3,
                "aggregate_revision": clean_revision,
                "comparison_id": spec["comparison_id"],
                "comparison_sha256": _canonical_json_sha256(spec),
                "aggregate_result": {
                    "path": AGGREGATE_RESULT_NAME,
                    "sha256": hashlib.sha256(result_bytes).hexdigest(),
                },
            }
        )
        _write_canonical_json(staging / AGGREGATE_MANIFEST_NAME, manifest)
        return publish_immutable_directory(staging, target)
    except BaseException:
        if staging.exists():
            shutil.rmtree(staging)
        raise


def validate_comparison_spec(comparison: Mapping[str, object]) -> dict[str, object]:
    if not isinstance(comparison, Mapping) or set(comparison) != _COMPARISON_KEYS:
        raise ValueError(f"comparison keys must be exactly {sorted(_COMPARISON_KEYS)}")
    comparison_id = validate_path_component(
        comparison["comparison_id"], "comparison_id"
    )
    dataset = validate_dataset_identity(comparison["dataset"])
    methods = _unique_string_list(
        comparison["required_methods"],
        allowed=CANONICAL_METHOD_PROFILES,
        context="required_methods",
    )
    coverage_value = comparison["required_consumers"]
    if not isinstance(coverage_value, Mapping) or set(coverage_value) != set(methods):
        raise ValueError(
            "required_consumers must contain exactly the declared required_methods"
        )
    coverage: dict[str, list[str]] = {}
    for method in methods:
        profiles = _unique_string_list(
            coverage_value[method],
            allowed=_NUMERICAL_COMPARISON_PROFILES,
            context=f"required_consumers.{method}",
        )
        coverage[method] = profiles
    required_profiles = {
        profile for profiles in coverage.values() for profile in profiles
    }
    raw_protocols = comparison["consumer_protocols"]
    if not isinstance(raw_protocols, Mapping) or set(raw_protocols) != required_profiles:
        raise ValueError(
            "consumer_protocols must bind exactly the required consumer profiles"
        )
    protocols: dict[str, dict[str, str]] = {}
    for profile in sorted(required_profiles):
        reference = validate_relative_reference(raw_protocols[profile])
        if reference["owner"] != "project" and not (
            reference["owner"] == "runs" and reference["path"].startswith("inputs/")
        ):
            raise ValueError("consumer protocol references must be project or runs/inputs")
        protocols[profile] = reference
    return {
        "comparison_id": comparison_id,
        "dataset": dataset,
        "required_methods": methods,
        "required_consumers": coverage,
        "consumer_protocols": protocols,
    }


def _validate_aggregate_manifest(
    manifest: Mapping[str, object],
) -> dict[str, object]:
    if not isinstance(manifest, Mapping) or set(manifest) != _AGGREGATE_MANIFEST_KEYS:
        raise ValueError(
            "aggregate manifest keys must be exactly "
            f"{sorted(_AGGREGATE_MANIFEST_KEYS)}"
        )
    if (
        type(manifest["schema_version"]) is not int
        or manifest["schema_version"] != 3
    ):
        raise ValueError("aggregate manifest schema_version must equal 3")
    result_reference = manifest["aggregate_result"]
    if (
        not isinstance(result_reference, Mapping)
        or set(result_reference) != _AGGREGATE_RESULT_REFERENCE_KEYS
    ):
        raise ValueError(
            "aggregate result reference keys must be exactly "
            f"{sorted(_AGGREGATE_RESULT_REFERENCE_KEYS)}"
        )
    if result_reference["path"] != AGGREGATE_RESULT_NAME:
        raise ValueError(
            f"aggregate result reference path must be {AGGREGATE_RESULT_NAME}"
        )
    return {
        "schema_version": 3,
        "aggregate_revision": _require_revision(
            manifest["aggregate_revision"],
            "aggregate",
        ),
        "comparison_id": validate_path_component(
            manifest["comparison_id"],
            "comparison_id",
        ),
        "comparison_sha256": _require_sha256(
            manifest["comparison_sha256"],
            "comparison",
        ),
        "aggregate_result": {
            "path": AGGREGATE_RESULT_NAME,
            "sha256": _require_sha256(
                result_reference["sha256"],
                "aggregate result",
            ),
        },
    }


def _unique_string_list(
    value: object,
    *,
    allowed: frozenset[str],
    context: str,
) -> list[str]:
    if not isinstance(value, list) or not value:
        raise ValueError(f"{context} must be a non-empty JSON list")
    result: list[str] = []
    for item in value:
        if not isinstance(item, str) or item not in allowed:
            raise ValueError(f"{context} contains unsupported value: {item!r}")
        if item in result:
            raise ValueError(f"{context} must not contain duplicate values")
        result.append(item)
    return result


def _write_canonical_json(path: Path, payload: object) -> bytes:
    contents = (
        json.dumps(payload, sort_keys=True, indent=2, allow_nan=False) + "\n"
    ).encode("utf-8")
    path.write_bytes(contents)
    return contents


def _canonical_json_sha256(value: object) -> str:
    return hashlib.sha256(
        json.dumps(
            value,
            sort_keys=True,
            separators=(",", ":"),
            allow_nan=False,
        ).encode("utf-8")
    ).hexdigest()


def _canonical_json_text(value: object) -> str:
    return json.dumps(
        value,
        sort_keys=True,
        separators=(",", ":"),
        allow_nan=False,
    )


def _task_relative_reference(
    path: Path,
    *,
    task_root: Path,
    sha256: str | None,
) -> dict[str, str]:
    try:
        relative = path.relative_to(task_root)
    except ValueError as exc:
        raise ValueError("aggregate input must stay within the declared task root") from exc
    if path.is_symlink() or not path.is_file():
        raise FileNotFoundError(f"aggregate source is not a regular file: {path}")
    resolved_sha256 = (
        hashlib.sha256(path.read_bytes()).hexdigest()
        if sha256 is None
        else sha256
    )
    return {
        "owner": "artifact",
        "path": relative.as_posix(),
        "sha256": _require_sha256(resolved_sha256, "aggregate source"),
    }


def _require_sha256(value: object, context: str) -> str:
    if (
        not isinstance(value, str)
        or len(value) != 64
        or any(character not in "0123456789abcdef" for character in value)
    ):
        raise ValueError(f"{context} digest must be 64 lowercase hexadecimal characters")
    return value


def _require_revision(value: object, context: str) -> str:
    if value is None:
        return None
    if (
        not isinstance(value, str)
        or len(value) != 40
        or any(character not in "0123456789abcdef" for character in value)
    ):
        raise ValueError(
            f"{context} revision must be a lowercase 40-character git commit"
        )
    return value


def prepare_comparison_config(raw, runtime, project_root):
    """Freeze explicit evaluation protocols before aggregate execution."""
    from dataclasses import asdict
    from pfm.core.evaluation_config import metric_evaluation_config_from_mapping
    from pfm.core.yaml_config import canonical_config_bytes, parse_yaml_mapping
    from pfm.core.execution import require_canonical_path
    dependencies = {}
    protocols = raw.get("consumer_protocols")
    if not isinstance(protocols, dict):
        raise ValueError("consumer_protocols must be a mapping")
    resolved = {}
    for profile, reference in protocols.items():
        if not isinstance(reference, dict) or set(reference) not in ({"owner", "path"}, {"owner", "path", "sha256"}):
            raise ValueError("authored consumer protocol requires owner/path, optionally a frozen sha256")
        checked = validate_relative_reference({**reference, "sha256": reference.get("sha256", "0" * 64)})
        owner = checked["owner"]
        if owner == "project":
            source = require_canonical_path(project_root / checked["path"], project_root, name="consumer protocol")
        elif owner == "runs" and checked["path"].startswith("inputs/"):
            source = require_canonical_path(runtime.runs_root / checked["path"], runtime.runs_root / "inputs", name="consumer protocol")
        else:
            raise ValueError("consumer protocol references must be project or runs/inputs")
        source_bytes = source.read_bytes()
        if "sha256" in reference and hashlib.sha256(source_bytes).hexdigest() != reference["sha256"]:
            raise ValueError("consumer protocol snapshot digest mismatch")
        source_config = parse_yaml_mapping(source_bytes, context="consumer protocol")
        contents = canonical_config_bytes(asdict(metric_evaluation_config_from_mapping(source_config)))
        if source_config.get("profile") != profile:
            raise ValueError("consumer protocol profile does not match its comparison key")
        digest = hashlib.sha256(contents).hexdigest()
        target = runtime.runs_root / "inputs" / "configs" / f"{digest}.yaml"
        dependencies[target] = contents
        resolved[profile] = {"owner": "runs", "path": target.relative_to(runtime.runs_root).as_posix(), "sha256": digest}
    return validate_comparison_spec({**raw, "consumer_protocols": resolved}), dependencies
