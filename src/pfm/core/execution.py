"""Portable execution paths and process identity.

Paths are supplied by the caller; only the standard torchrun rank variables
are read from the environment. Scientific configuration lives in its own YAML.
"""
from __future__ import annotations

import os
import re
import subprocess
from dataclasses import dataclass
from pathlib import Path
from typing import Mapping


DATASET_DIR_NAMES = {
    "mnist": "mnist",
    "cifar10": "cifar10",
    "imagenet": "imagenet1k-ilsvrc2012",
}
_TORCHRUN_ENV_NAMES = ("RANK", "LOCAL_RANK", "WORLD_SIZE")
_EXACT_COMMIT_RE = re.compile(r"[0-9a-f]{40}")


@dataclass(frozen=True)
class RuntimePaths:
    project_name: str
    project_root: Path
    datasets_root: Path
    assets_root: Path
    runs_root: Path
    cache_root: Path

    def __post_init__(self) -> None:
        object.__setattr__(self, "project_name", _path_component(self.project_name, "project_name"))
        for attr in ("project_root", "datasets_root", "assets_root", "runs_root", "cache_root"):
            object.__setattr__(self, attr, Path(getattr(self, attr)).expanduser().resolve())

    @property
    def experiments_root(self) -> Path:
        return self.runs_root / "experiments"

    @property
    def diagnostics_root(self) -> Path:
        return self.runs_root / "diagnostics"

    @property
    def fid_stats_root(self) -> Path:
        return self.assets_root / "fid_stats"

    @property
    def classifier_root(self) -> Path:
        return self.assets_root / "classifiers"

    def dataset_root(self, dataset: str) -> Path:
        try:
            dirname = DATASET_DIR_NAMES[dataset]
        except (KeyError, TypeError) as exc:
            raise ValueError(f"dataset must be one of: {', '.join(sorted(DATASET_DIR_NAMES))}") from exc
        return self.datasets_root / dirname

    def project_file(self, value: str | Path, *, name: str) -> Path:
        raw = Path(value).expanduser()
        path = require_path_within(raw if raw.is_absolute() else self.project_root / raw,
                                   self.project_root, name=name)
        if not path.is_file():
            raise FileNotFoundError(f"{name} file does not exist: {path}")
        return path

    def config_file(self, value: str | Path, *, name: str) -> Path:
        # User-authored configurations may be anywhere. The CLI freezes resolved
        # contents into this run's inputs directory before invoking a service.
        raw = Path(value).expanduser()
        path = (raw if raw.is_absolute() else self.project_root / raw).resolve()
        if not path.is_file():
            raise FileNotFoundError(f"{name} file does not exist: {path}")
        return path


@dataclass(frozen=True)
class ExecutionContext:
    runtime: RuntimePaths
    git_commit: str | None
    rank: int = 0
    local_rank: int = 0
    world_size: int = 1
    device: str = "cuda"

    def __post_init__(self) -> None:
        if self.git_commit is not None and (
            not isinstance(self.git_commit, str) or _EXACT_COMMIT_RE.fullmatch(self.git_commit) is None
        ):
            raise ValueError("git_commit must be a lowercase 40-character hex commit or None")
        rank = _non_negative_int(self.rank, "RANK")
        local_rank = _non_negative_int(self.local_rank, "LOCAL_RANK")
        world_size = _positive_int(self.world_size, "WORLD_SIZE")
        if rank >= world_size:
            raise ValueError("RANK must be less than WORLD_SIZE")
        if self.device not in {"cuda", "cpu"}:
            raise ValueError("device must be cuda or cpu")
        object.__setattr__(self, "rank", rank)
        object.__setattr__(self, "local_rank", local_rank)
        object.__setattr__(self, "world_size", world_size)


def runtime_paths(
    *,
    project_root: str | Path | None = None,
    data_root: str | Path | None = None,
    output_dir: str | Path | None = None,
    assets_root: str | Path | None = None,
    cache_root: str | Path | None = None,
) -> RuntimePaths:
    project = Path.cwd() if project_root is None else Path(project_root).expanduser().resolve()
    def resolve(value, default):
        path = Path(default if value is None else value).expanduser()
        return (path if path.is_absolute() else project / path).resolve()
    return RuntimePaths(
        project_name="pfm",
        project_root=project,
        datasets_root=resolve(data_root, "data"),
        assets_root=resolve(assets_root, "assets"),
        runs_root=resolve(output_dir, "outputs"),
        cache_root=resolve(cache_root, ".cache/pfm"),
    )


def source_git_commit() -> str | None:
    """Return an exact clean source revision, or None for unversioned/dirty code."""
    source_root = Path(__file__).resolve().parents[3]
    if not (source_root / ".git").exists():
        return None
    try:
        # User-selected output directories can be anywhere in the checkout.
        # Their creation must not change the code identity or break resume.
        status = subprocess.run(
            ["git", "-C", str(source_root), "status", "--porcelain", "--untracked-files=normal",
             "--", "src", "configs", "pyproject.toml", "requirements", ".python-version"],
            check=True, capture_output=True, text=True,
        ).stdout.strip()
        if status:
            return None
        result = subprocess.run(
            ["git", "-C", str(source_root), "rev-parse", "--verify", "HEAD"],
            check=True, capture_output=True, text=True,
        ).stdout.strip()
    except (OSError, subprocess.CalledProcessError):
        return None
    return result if _EXACT_COMMIT_RE.fullmatch(result) else None


def execution_context_from_env(
    env: Mapping[str, str] | None = None,
    *,
    project_root: str | Path | None = None,
    data_root: str | Path | None = None,
    output_dir: str | Path | None = None,
    assets_root: str | Path | None = None,
    cache_root: str | Path | None = None,
    device: str = "cuda",
) -> ExecutionContext:
    rank, local_rank, world_size = _torchrun_identity(os.environ if env is None else env)
    return ExecutionContext(
        runtime=runtime_paths(project_root=project_root, data_root=data_root,
                              output_dir=output_dir, assets_root=assets_root, cache_root=cache_root),
        git_commit=source_git_commit(), rank=rank, local_rank=local_rank,
        world_size=world_size, device=device,
    )


def runtime_identity(context: ExecutionContext) -> dict[str, object]:
    return {
        "project_name": context.runtime.project_name,
        "git_commit": context.git_commit,
        "rank": context.rank,
        "world_size": context.world_size,
    }


def classifier_asset_dir(runtime: RuntimePaths, dataset: str, asset_id: str) -> Path:
    try:
        dataset_dirname = DATASET_DIR_NAMES[dataset]
    except (KeyError, TypeError) as exc:
        valid = ", ".join(sorted(DATASET_DIR_NAMES))
        raise ValueError(f"dataset must be one of: {valid}") from exc
    clean_asset_id = _strict_path_component(asset_id, "classifier_asset_id")
    root = require_canonical_path(
        runtime.classifier_root,
        runtime.classifier_root,
        name="classifier_root",
    )
    return require_canonical_path(
        root / dataset_dirname / clean_asset_id,
        root,
        name="classifier_asset_dir",
    )


def fid_asset_dir(runtime: RuntimePaths, asset_id: str) -> Path:
    clean_asset_id = _strict_path_component(asset_id, "fid_asset_id")
    root = require_canonical_path(
        runtime.fid_stats_root,
        runtime.fid_stats_root,
        name="fid_stats_root",
    )
    return require_canonical_path(
        root / clean_asset_id,
        root,
        name="fid_asset_dir",
    )


def require_path_within(path: str | Path, root: str | Path, *, name: str) -> Path:
    resolved_root = Path(root).resolve()
    resolved_path = Path(path).resolve()
    try:
        resolved_path.relative_to(resolved_root)
    except ValueError as exc:
        raise ValueError(f"{name} must be within {resolved_root}: {resolved_path}") from exc
    return resolved_path


def require_canonical_path(path: str | Path, root: str | Path, *, name: str) -> Path:
    """Validate under one lexical root, resolving only to reject physical escapes."""
    lexical_root = Path(os.path.abspath(Path(root)))
    resolved_root = lexical_root.resolve()
    lexical_path = Path(os.path.abspath(Path(path)))
    try:
        relative_parts = lexical_path.relative_to(lexical_root).parts
    except ValueError as exc:
        raise ValueError(f"{name} must be within {lexical_root}: {lexical_path}") from exc
    current = lexical_root
    for part in relative_parts:
        current = current / part
        if current.is_symlink():
            raise ValueError(f"{name} must not contain a symlink component: {current}")
    require_path_within(lexical_path, resolved_root, name=name)
    return lexical_path


def _path_component(value: str, name: str) -> str:
    component = str(value)
    if (
        component in {"", ".", ".."}
        or component != component.strip()
        or "/" in component
        or "\\" in component
    ):
        raise ValueError(f"{name} must be a non-empty single path component")
    return component


def _strict_path_component(value: object, name: str) -> str:
    if not isinstance(value, str):
        raise TypeError(f"{name} must be a string path component")
    if (
        value in {"", ".", ".."}
        or value != value.strip()
        or "/" in value
        or "\\" in value
    ):
        raise ValueError(f"{name} must be a non-empty single path component")
    return value


def _torchrun_identity(env: Mapping[str, str]) -> tuple[int, int, int]:
    present = [name for name in _TORCHRUN_ENV_NAMES if name in env and env[name] != ""]
    if not present:
        return 0, 0, 1
    if len(present) != len(_TORCHRUN_ENV_NAMES):
        raise ValueError("RANK, LOCAL_RANK, and WORLD_SIZE must be provided together")
    return (
        _non_negative_int(env["RANK"], "RANK"),
        _non_negative_int(env["LOCAL_RANK"], "LOCAL_RANK"),
        _positive_int(env["WORLD_SIZE"], "WORLD_SIZE"),
    )


def _non_negative_int(value: object, name: str) -> int:
    try:
        parsed = int(value)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"{name} must be a non-negative integer") from exc
    if parsed < 0:
        raise ValueError(f"{name} must be a non-negative integer")
    return parsed


def _positive_int(value: object, name: str) -> int:
    parsed = _non_negative_int(value, name)
    if parsed == 0:
        raise ValueError(f"{name} must be a positive integer")
    return parsed
