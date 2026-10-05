from __future__ import annotations

import hashlib
import json
import os
import shutil
from collections.abc import Callable, Mapping
from pathlib import Path, PurePosixPath

from pfm.core.immutable_artifact import (
    prepare_immutable_directory,
    publish_immutable_directory,
)
from pfm.core.layout import validate_path_component
from pfm.core.variant_metadata import CANONICAL_METHOD_PROFILES


CONSUMER_MANIFEST_NAME = "consumer_manifest.json"
NUMERICAL_RESULT_NAME = "result.json"
REFERENCE_OWNERS = frozenset(
    {"project", "datasets", "assets", "runs", "artifact"}
)
CONSUMER_PROFILES = frozenset(
    {"generated_common", "posterior_diagnostics", "sample_visualization"}
)
_BINDING_KEYS = frozenset(
    {
        "profile",
        "method",
        "shared_training_protocol_sha256",
        "dataset_scientific_identity_sha256",
        "consumer_revision",
        "checkpoint_revision",
        "checkpoint_progress",
        "checkpoint",
        "dataset",
        "config",
        "assets",
    }
)
_MANIFEST_KEYS = frozenset({"schema_version", "binding", "numerical_result"})
_REFERENCE_KEYS = frozenset({"owner", "path", "sha256"})
_NUMERICAL_REFERENCE_KEYS = frozenset({"path", "sha256"})
_CHECKPOINT_PROGRESS_KEYS = frozenset({"epoch", "step"})


def make_relative_reference(
    *,
    owner: str,
    path: str | Path,
    root: str | Path,
    sha256: str,
) -> dict[str, str]:
    clean_owner = _validate_owner(owner)
    lexical_root = Path(os.path.abspath(Path(root)))
    lexical_path = Path(os.path.abspath(Path(path)))
    resolved_root = lexical_root.resolve()
    resolved_path = lexical_path.resolve()
    try:
        relative = resolved_path.relative_to(resolved_root)
    except ValueError as exc:
        raise ValueError(f"reference path must stay within its {clean_owner} owner root") from exc
    relative_text = relative.as_posix()
    _validate_relative_path(relative_text)
    return {
        "owner": clean_owner,
        "path": relative_text,
        "sha256": _require_sha256(sha256, "reference"),
    }


def validate_relative_reference(reference: Mapping[str, object]) -> dict[str, str]:
    if not isinstance(reference, Mapping) or set(reference) != _REFERENCE_KEYS:
        raise ValueError(
            f"relative reference keys must be exactly {sorted(_REFERENCE_KEYS)}"
        )
    owner = _validate_owner(reference["owner"])
    relative_path = _validate_relative_path(reference["path"])
    return {
        "owner": owner,
        "path": relative_path,
        "sha256": _require_sha256(reference["sha256"], "reference"),
    }


def publish_consumer_artifact(
    *,
    parent: str | Path,
    artifact_id: str,
    binding: Mapping[str, object],
    numerical_result: Mapping[str, object] | Callable[[Path], Mapping[str, object]],
    write_presentation: Callable[[Path], None] | None = None,
) -> Path:
    clean_artifact_id = validate_path_component(artifact_id, "artifact_id")
    validated_binding = validate_consumer_binding(binding)
    staging, target = prepare_immutable_directory(Path(parent), clean_artifact_id)
    try:
        if write_presentation is not None:
            write_presentation(staging)
        produced_result = (
            numerical_result(staging)
            if callable(numerical_result)
            else numerical_result
        )
        validated_result = _validate_numerical_result(produced_result)
        result_path = staging / NUMERICAL_RESULT_NAME
        result_bytes = _write_canonical_json(result_path, validated_result)
        manifest = _validate_consumer_manifest(
            {
                "schema_version": 2,
                "binding": validated_binding,
                "numerical_result": {
                    "path": NUMERICAL_RESULT_NAME,
                    "sha256": hashlib.sha256(result_bytes).hexdigest(),
                },
            }
        )
        _write_canonical_json(staging / CONSUMER_MANIFEST_NAME, manifest)
        return publish_immutable_directory(staging, target)
    except BaseException:
        if staging.exists():
            shutil.rmtree(staging)
        raise


def load_consumer_artifact(
    manifest_path: str | Path,
) -> tuple[dict[str, object], dict[str, object]]:
    path = Path(manifest_path)
    if path.name != CONSUMER_MANIFEST_NAME:
        raise ValueError(
            f"consumer manifest path must end with {CONSUMER_MANIFEST_NAME}"
        )
    if path.is_symlink() or not path.is_file():
        raise FileNotFoundError(f"consumer manifest does not exist as a regular file: {path}")
    manifest = _read_json_object(path, context="consumer manifest")
    validated_manifest = _validate_consumer_manifest(manifest)
    numerical_reference = validated_manifest["numerical_result"]
    assert isinstance(numerical_reference, dict)
    result_path = path.parent / str(numerical_reference["path"])
    if result_path.is_symlink() or not result_path.is_file():
        raise FileNotFoundError(
            f"canonical numerical result does not exist as a regular file: {result_path}"
        )
    result_bytes = result_path.read_bytes()
    result_sha256 = hashlib.sha256(result_bytes).hexdigest()
    if result_sha256 != numerical_reference["sha256"]:
        raise ValueError("canonical numerical result digest does not match consumer manifest")
    numerical = _loads_json_object(result_bytes, context="canonical numerical result")
    return validated_manifest, _validate_numerical_result(numerical)


def validate_consumer_binding(binding: Mapping[str, object]) -> dict[str, object]:
    if not isinstance(binding, Mapping) or set(binding) != _BINDING_KEYS:
        raise ValueError(f"consumer binding keys must be exactly {sorted(_BINDING_KEYS)}")
    profile = binding["profile"]
    if profile not in CONSUMER_PROFILES:
        raise ValueError(f"consumer profile must be one of: {sorted(CONSUMER_PROFILES)}")
    method = binding["method"]
    if method not in CANONICAL_METHOD_PROFILES:
        raise ValueError(
            "method must be one of: "
            f"{sorted(CANONICAL_METHOD_PROFILES)}"
        )
    config_reference = validate_relative_reference(
        _require_mapping(binding["config"])
    )
    if config_reference["owner"] != "project" and not (
        config_reference["owner"] == "runs" and config_reference["path"].startswith("inputs/")
    ):
        raise ValueError("consumer config reference must be project or runs/inputs")
    checkpoint_reference = validate_relative_reference(
        _require_mapping(binding["checkpoint"])
    )
    if checkpoint_reference["owner"] != "runs":
        raise ValueError("consumer checkpoint reference owner must be runs")
    dataset_reference = validate_relative_reference(
        _require_mapping(binding["dataset"])
    )
    if dataset_reference["owner"] != "datasets":
        raise ValueError("consumer dataset reference owner must be datasets")
    asset_references = _validate_assets(binding["assets"])
    if any(
        reference["owner"] != "assets"
        for reference in asset_references.values()
    ):
        raise ValueError("consumer asset reference owner must be assets")
    return {
        "profile": str(profile),
        "method": str(method),
        "shared_training_protocol_sha256": _require_sha256(
            binding["shared_training_protocol_sha256"],
            "shared training protocol",
        ),
        "dataset_scientific_identity_sha256": _require_sha256(
            binding["dataset_scientific_identity_sha256"],
            "dataset scientific identity",
        ),
        "consumer_revision": _require_revision(binding["consumer_revision"]),
        "checkpoint_revision": _require_revision(binding["checkpoint_revision"]),
        "checkpoint_progress": _validate_checkpoint_progress(
            binding["checkpoint_progress"]
        ),
        "checkpoint": checkpoint_reference,
        "dataset": dataset_reference,
        "config": config_reference,
        "assets": asset_references,
    }


def _validate_consumer_manifest(manifest: Mapping[str, object]) -> dict[str, object]:
    if set(manifest) != _MANIFEST_KEYS:
        raise ValueError(f"consumer manifest keys must be exactly {sorted(_MANIFEST_KEYS)}")
    if manifest["schema_version"] != 2:
        raise ValueError("consumer manifest schema_version must equal 2")
    numerical = manifest["numerical_result"]
    if not isinstance(numerical, Mapping) or set(numerical) != _NUMERICAL_REFERENCE_KEYS:
        raise ValueError(
            "consumer numerical result reference must contain exactly path and sha256"
        )
    if numerical["path"] != NUMERICAL_RESULT_NAME:
        raise ValueError(f"canonical numerical result path must be {NUMERICAL_RESULT_NAME}")
    return {
        "schema_version": 2,
        "binding": validate_consumer_binding(_require_mapping(manifest["binding"])),
        "numerical_result": {
            "path": NUMERICAL_RESULT_NAME,
            "sha256": _require_sha256(numerical["sha256"], "canonical numerical result"),
        },
    }


def _validate_numerical_result(value: Mapping[str, object]) -> dict[str, object]:
    if not isinstance(value, Mapping):
        raise ValueError("canonical numerical result must be a mapping")
    result = _json_round_trip(value, context="canonical numerical result")
    aggregate = result.get("aggregate")
    if not isinstance(aggregate, dict):
        raise ValueError("canonical numerical result must contain a compact aggregate mapping")
    return result


def _validate_owner(value: object) -> str:
    if not isinstance(value, str) or value not in REFERENCE_OWNERS:
        raise ValueError(f"reference owner must be one of: {sorted(REFERENCE_OWNERS)}")
    return value


def _validate_assets(value: object) -> dict[str, dict[str, str]]:
    if not isinstance(value, Mapping):
        raise ValueError("consumer asset bindings must be a mapping")
    allowed_roles = {"classifier", "fid"}
    if not set(value).issubset(allowed_roles):
        raise ValueError("consumer asset bindings contain an unsupported role")
    return {
        role: validate_relative_reference(_require_mapping(value[role]))
        for role in sorted(value)
    }


def _validate_checkpoint_progress(value: object) -> dict[str, int]:
    if not isinstance(value, Mapping) or set(value) != _CHECKPOINT_PROGRESS_KEYS:
        raise ValueError(
            "checkpoint progress keys must be exactly "
            f"{sorted(_CHECKPOINT_PROGRESS_KEYS)}"
        )
    epoch = value["epoch"]
    step = value["step"]
    if type(epoch) is not int or epoch < 1:
        raise ValueError("checkpoint progress epoch must be a positive integer")
    if type(step) is not int or step < 0:
        raise ValueError("checkpoint progress step must be a non-negative integer")
    return {"epoch": epoch, "step": step}


def _validate_relative_path(value: object) -> str:
    if not isinstance(value, str) or not value or "\\" in value:
        raise ValueError("reference path must be a non-empty POSIX relative path")
    path = PurePosixPath(value)
    if path.is_absolute() or any(part in {"", ".", ".."} for part in path.parts):
        raise ValueError("reference path must be relative and must not escape its owner root")
    if value != path.as_posix():
        raise ValueError("reference path must be a canonical POSIX relative path")
    return value


def _require_sha256(value: object, context: str) -> str:
    if (
        not isinstance(value, str)
        or len(value) != 64
        or any(character not in "0123456789abcdef" for character in value)
    ):
        raise ValueError(f"{context} digest must be 64 lowercase hexadecimal characters")
    return value


def _require_revision(value: object) -> str:
    if value is None:
        return None
    if (
        not isinstance(value, str)
        or len(value) != 40
        or any(character not in "0123456789abcdef" for character in value)
    ):
        raise ValueError("consumer revision must be a lowercase 40-character git commit")
    return value


def _require_mapping(value: object) -> Mapping[str, object]:
    if not isinstance(value, Mapping):
        raise ValueError("consumer binding references must be mappings")
    return value


def _write_canonical_json(path: Path, payload: object) -> bytes:
    path.parent.mkdir(parents=True, exist_ok=True)
    contents = (
        json.dumps(payload, sort_keys=True, indent=2, allow_nan=False) + "\n"
    ).encode("utf-8")
    path.write_bytes(contents)
    return contents


def _read_json_object(path: Path, *, context: str) -> dict[str, object]:
    return _loads_json_object(path.read_bytes(), context=context)


def _loads_json_object(contents: bytes, *, context: str) -> dict[str, object]:
    try:
        value = json.loads(contents, parse_constant=_reject_nonfinite)
    except (UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise ValueError(f"{context} must contain strict UTF-8 JSON") from exc
    if not isinstance(value, dict):
        raise ValueError(f"{context} must contain a JSON object")
    return value


def _json_round_trip(value: Mapping[str, object], *, context: str) -> dict[str, object]:
    try:
        encoded = json.dumps(value, sort_keys=True, allow_nan=False).encode("utf-8")
    except (TypeError, ValueError) as exc:
        raise ValueError(f"{context} must contain strict JSON values") from exc
    return _loads_json_object(encoded, context=context)


def _reject_nonfinite(value: str) -> None:
    raise ValueError(f"non-finite JSON number is forbidden: {value}")
