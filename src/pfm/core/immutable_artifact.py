from __future__ import annotations

import os
import tempfile
from pathlib import Path

from pfm.core.layout import validate_path_component


def prepare_immutable_directory(parent: Path, artifact_id: str) -> tuple[Path, Path]:
    clean_artifact_id = validate_path_component(artifact_id, "artifact_id")
    artifact_parent = _absolute_lexical_path(parent)
    if artifact_parent.is_symlink():
        raise ValueError(f"parent must not be a symlink: {artifact_parent}")
    artifact_parent.mkdir(parents=True, exist_ok=True)
    if not artifact_parent.is_dir():
        raise NotADirectoryError(f"parent is not a directory: {artifact_parent}")

    target_dir = artifact_parent / clean_artifact_id
    if target_dir.is_symlink():
        raise ValueError(f"target_dir must not be a symlink: {target_dir}")
    if os.path.lexists(target_dir):
        raise FileExistsError(f"immutable target already exists: {target_dir}")

    staging_dir = Path(
        tempfile.mkdtemp(
            prefix=f".{clean_artifact_id}.staging-",
            dir=artifact_parent,
        )
    )
    if staging_dir.stat().st_dev != artifact_parent.stat().st_dev:
        raise ValueError("staging_dir and target_dir must be on the same filesystem")
    return staging_dir, target_dir


def publish_immutable_directory(staging_dir: Path, target_dir: Path) -> Path:
    staging = _absolute_lexical_path(staging_dir)
    target = _absolute_lexical_path(target_dir)
    if staging.parent != target.parent:
        raise ValueError("staging_dir and target_dir must have the same parent")

    parent = staging.parent
    if parent.is_symlink():
        raise ValueError(f"publication parent must not be a symlink: {parent}")
    validate_path_component(staging.name, "staging_dir")
    validate_path_component(target.name, "target_dir")
    if staging.is_symlink():
        raise ValueError(f"staging_dir must not be a symlink: {staging}")
    if not os.path.lexists(staging):
        raise FileNotFoundError(f"staging_dir does not exist: {staging}")
    if not staging.is_dir():
        raise NotADirectoryError(f"staging_dir is not a directory: {staging}")
    if target.is_symlink():
        raise ValueError(f"target_dir must not be a symlink: {target}")
    if os.path.lexists(target):
        raise FileExistsError(f"immutable target already exists: {target}")
    if next(staging.iterdir(), None) is None:
        raise ValueError(f"staging_dir must be nonempty: {staging}")
    if staging.stat().st_dev != parent.stat().st_dev:
        raise ValueError("staging_dir and target_dir must be on the same filesystem")

    os.rename(staging, target)
    return target


def _absolute_lexical_path(path: str | Path) -> Path:
    return Path(os.path.abspath(Path(path)))
