from pathlib import Path

import pytest

import pfm.core.immutable_artifact as immutable_artifact
from pfm.core.immutable_artifact import (
    prepare_immutable_directory,
    publish_immutable_directory,
)


def test_prepare_returns_one_empty_same_parent_staging_directory(tmp_path: Path) -> None:
    staging, target = prepare_immutable_directory(tmp_path / "artifacts", "result-321")

    assert staging.parent == target.parent == tmp_path / "artifacts"
    assert staging.name.startswith(".result-321.staging-")
    assert list(staging.iterdir()) == []
    assert target.name == "result-321"


@pytest.mark.parametrize("artifact_id", ["", ".", "..", "nested/id", r"nested\id"])
def test_prepare_rejects_non_component_identity(tmp_path: Path, artifact_id: str) -> None:
    with pytest.raises(ValueError, match="component"):
        prepare_immutable_directory(tmp_path, artifact_id)


def test_prepare_rejects_an_existing_immutable_target_without_staging(tmp_path: Path) -> None:
    target = tmp_path / "result-321"
    target.mkdir()

    with pytest.raises(FileExistsError, match="already exists"):
        prepare_immutable_directory(tmp_path, target.name)

    assert list(tmp_path.iterdir()) == [target]


def test_publish_performs_one_directory_rename(tmp_path: Path, monkeypatch) -> None:
    staging, target = prepare_immutable_directory(tmp_path, "result-321")
    (staging / "result.json").write_text("{}\n", encoding="utf-8")
    calls = []
    real_rename = immutable_artifact.os.rename

    def record(source, destination):
        calls.append((Path(source), Path(destination)))
        real_rename(source, destination)

    monkeypatch.setattr(immutable_artifact.os, "rename", record)

    assert publish_immutable_directory(staging, target) == target
    assert calls == [(staging, target)]
    assert (target / "result.json").is_file()


def test_publish_rejects_empty_staging_and_existing_target(tmp_path: Path) -> None:
    empty, target = prepare_immutable_directory(tmp_path, "empty")
    with pytest.raises(ValueError, match="nonempty"):
        publish_immutable_directory(empty, target)

    staging = tmp_path / ".result.staging-one"
    staging.mkdir()
    (staging / "result.json").write_text("{}\n", encoding="utf-8")
    existing = tmp_path / "result"
    existing.mkdir()
    with pytest.raises(FileExistsError, match="already exists"):
        publish_immutable_directory(staging, existing)
