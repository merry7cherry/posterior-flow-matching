from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

from pfm.core.execution import RuntimePaths, require_canonical_path


def epoch_id(epoch: int) -> str:
    if isinstance(epoch, bool) or not isinstance(epoch, int):
        raise TypeError("epoch must be an integer")
    if epoch <= 0:
        raise ValueError("epoch must be positive")
    return f"epoch_{epoch:04d}"


def parse_operator_epoch(value: str) -> int:
    if not isinstance(value, str):
        raise TypeError("operator epoch must be a string")
    if not value.isascii() or not value.isdigit() or value.startswith("0"):
        raise ValueError("epoch must be an unpadded positive ASCII decimal")
    return int(value)


def validate_path_component(value: str, field_name: str) -> str:
    if not isinstance(value, str):
        raise TypeError(f"{field_name} must be a string path component")
    component = value
    if component != component.strip():
        raise ValueError(f"{field_name} path component must not have leading or trailing whitespace")
    if component in {"", ".", ".."} or "/" in component or "\\" in component:
        raise ValueError(f"{field_name} must be a non-empty single path component")
    return component


@dataclass(frozen=True)
class SharedRunLayout:
    output_root: Path
    task_id: str
    variant_id: str
    task_root: Path
    variant_root: Path
    epochs_root: Path
    recovery_root: Path
    recovery_generations_root: Path
    recovery_latest_path: Path
    train_metrics_jsonl: Path
    variant_manifest_path: Path
    task_manifest_path: Path

    def _path(self, path: Path, name: str) -> Path:
        return require_canonical_path(path, self.output_root, name=name)

    def epoch_dir(self, epoch: int) -> Path:
        return self._path(self.epochs_root / epoch_id(epoch), "epoch_dir")

    def checkpoint_path(self, epoch: int) -> Path:
        return self._path(self.epoch_dir(epoch) / "checkpoint.pt", "checkpoint_path")

    def checkpoint_manifest_path(self, epoch: int) -> Path:
        return self._path(
            self.epoch_dir(epoch) / "checkpoint_manifest.json",
            "checkpoint_manifest_path",
        )

    def evaluation_artifact_dir(self, epoch: int, evaluation_id: str) -> Path:
        clean_evaluation_id = validate_path_component(evaluation_id, "evaluation_id")
        return self._path(
            self.epoch_dir(epoch) / "evaluations" / clean_evaluation_id,
            "evaluation_artifact_dir",
        )

    def sample_artifact_dir(self, epoch: int, sample_id: str) -> Path:
        clean_sample_id = validate_path_component(sample_id, "sample_id")
        return self._path(
            self.epoch_dir(epoch) / "samples" / clean_sample_id,
            "sample_artifact_dir",
        )

def default_variant_id(variant: str, config) -> str:
    from pfm.core.variant_metadata import method_profile_name, validate_base_model_config

    validate_base_model_config(variant, config)
    return validate_path_component(method_profile_name(variant), "variant_id")


def training_run_layout(
    runtime: RuntimePaths,
    *,
    task_id: str,
    variant_id: str,
) -> SharedRunLayout:
    root = require_canonical_path(
        runtime.experiments_root,
        runtime.experiments_root,
        name="experiments_root",
    )
    clean_task_id = validate_path_component(task_id, "task_id")
    clean_variant_id = validate_path_component(variant_id, "variant_id")
    task_root = require_canonical_path(root / clean_task_id, root, name="task_root")
    variant_root = require_canonical_path(
        task_root / clean_variant_id,
        root,
        name="variant_root",
    )
    return SharedRunLayout(
        output_root=root,
        task_id=clean_task_id,
        variant_id=clean_variant_id,
        task_root=task_root,
        variant_root=variant_root,
        epochs_root=variant_root / "epochs",
        recovery_root=variant_root / "recovery",
        recovery_generations_root=variant_root / "recovery" / "generations",
        recovery_latest_path=variant_root / "recovery" / "latest.json",
        train_metrics_jsonl=variant_root / "train_metrics.jsonl",
        variant_manifest_path=variant_root / "variant_manifest.json",
        task_manifest_path=task_root / "task_manifest.json",
    )
