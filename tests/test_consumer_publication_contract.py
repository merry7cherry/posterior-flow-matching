from __future__ import annotations

import hashlib
import importlib
import json
from pathlib import Path

import pytest


SHA_A = "a" * 64
SHA_B = "b" * 64
SHA_C = "c" * 64
SHA_D = "d" * 64


def _module():
    return importlib.import_module("pfm.artifacts.consumer_artifact")


def _binding(checkpoint_ref: dict[str, object]) -> dict[str, object]:
    return {
        "profile": "generated_common",
        "method": "velocity_fm",
        "shared_training_protocol_sha256": SHA_A,
        "dataset_scientific_identity_sha256": SHA_C,
        "consumer_revision": "f" * 40,
        "checkpoint_revision": "0" * 40,
        "checkpoint_progress": {"epoch": 1, "step": 100},
        "checkpoint": checkpoint_ref,
        "dataset": {
            "owner": "datasets",
            "path": "mnist/dataset_identity.json",
            "sha256": SHA_C,
        },
        "config": {
            "owner": "project",
            "path": "configs/evaluation/mnist_generation.yaml",
            "sha256": SHA_B,
        },
        "assets": {},
    }


def test_relative_reference_is_stored_under_its_declared_owner(tmp_path: Path) -> None:
    module = _module()
    old_runs = tmp_path / "old" / "runs"
    checkpoint = old_runs / "experiments" / "task" / "velocity" / "checkpoint.pt"
    checkpoint.parent.mkdir(parents=True)
    checkpoint.write_bytes(b"checkpoint")

    reference = module.make_relative_reference(
        owner="runs",
        path=checkpoint,
        root=old_runs,
        sha256=hashlib.sha256(b"checkpoint").hexdigest(),
    )

    assert reference == {
        "owner": "runs",
        "path": "experiments/task/velocity/checkpoint.pt",
        "sha256": hashlib.sha256(b"checkpoint").hexdigest(),
    }


@pytest.mark.parametrize("owner", ["datasets", "assets"])
def test_dataset_and_asset_reference_owners_are_distinct(owner: str, tmp_path: Path) -> None:
    module = _module()
    root = tmp_path / owner
    path = root / "identity.json"
    path.parent.mkdir(parents=True)
    path.write_bytes(b"identity")

    reference = module.make_relative_reference(
        owner=owner,
        path=path,
        root=root,
        sha256=hashlib.sha256(path.read_bytes()).hexdigest(),
    )

    assert reference["owner"] == owner


def test_legacy_ambiguous_data_owner_is_rejected(tmp_path: Path) -> None:
    module = _module()
    root = tmp_path / "data"
    path = root / "identity.json"
    path.parent.mkdir(parents=True)
    path.write_bytes(b"identity")

    with pytest.raises(ValueError, match="reference owner"):
        module.make_relative_reference(
            owner="data",
            path=path,
            root=root,
            sha256=hashlib.sha256(path.read_bytes()).hexdigest(),
        )


@pytest.mark.parametrize(
    ("field", "wrong_owner", "message"),
    [
        ("checkpoint", "project", "checkpoint reference owner must be runs"),
        ("dataset", "assets", "dataset reference owner must be datasets"),
        ("config", "runs", "config reference must be project or runs/inputs"),
        ("assets", "datasets", "asset reference owner must be assets"),
    ],
)
def test_consumer_binding_rejects_wrong_field_owner(
    field: str,
    wrong_owner: str,
    message: str,
) -> None:
    module = _module()
    checkpoint = {"owner": "runs", "path": "task/checkpoint.pt", "sha256": SHA_A}
    binding = _binding(checkpoint)
    if field == "assets":
        binding["assets"] = {
            "fid": {
                "owner": wrong_owner,
                "path": "fid/manifest.json",
                "sha256": SHA_D,
            }
        }
    else:
        reference = dict(binding[field])
        reference["owner"] = wrong_owner
        binding[field] = reference

    with pytest.raises(ValueError, match=message):
        module.validate_consumer_binding(binding)


def test_relative_reference_accepts_physical_path_under_logical_owner_alias(
    tmp_path: Path,
) -> None:
    module = _module()
    physical_project = tmp_path / "physical-project"
    config = physical_project / "configs" / "sample.yaml"
    config.parent.mkdir(parents=True)
    config.write_text("profile: sample_visualization\n", encoding="utf-8")
    logical_project = tmp_path / "logical-project"
    logical_project.symlink_to(physical_project, target_is_directory=True)

    reference = module.make_relative_reference(
        owner="project",
        path=config.resolve(),
        root=logical_project,
        sha256=hashlib.sha256(config.read_bytes()).hexdigest(),
    )

    assert reference["path"] == "configs/sample.yaml"


def test_publication_has_one_numerical_owner_and_no_tree_inventory(tmp_path: Path) -> None:
    module = _module()
    runs = tmp_path / "runs"
    checkpoint = runs / "task" / "checkpoint.pt"
    checkpoint.parent.mkdir(parents=True)
    checkpoint.write_bytes(b"checkpoint")
    checkpoint_ref = module.make_relative_reference(
        owner="runs",
        path=checkpoint,
        root=runs,
        sha256=hashlib.sha256(checkpoint.read_bytes()).hexdigest(),
    )

    def write_presentation(staging: Path) -> None:
        figures = staging / "figures"
        figures.mkdir()
        (figures / "quality.png").write_bytes(b"not-a-real-png")

    target = module.publish_consumer_artifact(
        parent=tmp_path / "artifacts",
        artifact_id="quality-epoch-1",
        binding=_binding(checkpoint_ref),
        numerical_result={"aggregate": {"conditional_accuracy": 0.9}},
        write_presentation=write_presentation,
    )

    manifest = json.loads((target / "consumer_manifest.json").read_text(encoding="utf-8"))
    assert manifest["binding"] == _binding(checkpoint_ref)
    assert set(manifest["numerical_result"]) == {"path", "sha256"}
    assert manifest["numerical_result"]["path"] == "result.json"
    assert "output_members" not in manifest
    assert "checkpoint_manifest" not in json.dumps(manifest)
    assert ".staging-" not in json.dumps(manifest)
    assert manifest["numerical_result"]["sha256"] == hashlib.sha256(
        (target / "result.json").read_bytes()
    ).hexdigest()
    assert (target / "figures" / "quality.png").is_file()


def test_publication_is_atomic_on_failure_and_collision_is_immutable(tmp_path: Path) -> None:
    module = _module()
    checkpoint_ref = {"owner": "runs", "path": "task/checkpoint.pt", "sha256": SHA_A}
    parent = tmp_path / "artifacts"

    def fail_after_partial_write(staging: Path) -> None:
        (staging / "partial.txt").write_text("partial", encoding="utf-8")
        raise RuntimeError("render failed")

    with pytest.raises(RuntimeError, match="render failed"):
        module.publish_consumer_artifact(
            parent=parent,
            artifact_id="result",
            binding=_binding(checkpoint_ref),
            numerical_result={"aggregate": {"value": 1}},
            write_presentation=fail_after_partial_write,
        )
    assert not (parent / "result").exists()

    published = module.publish_consumer_artifact(
        parent=parent,
        artifact_id="result",
        binding=_binding(checkpoint_ref),
        numerical_result={"aggregate": {"value": 1}},
    )
    before = (published / "result.json").read_bytes()
    with pytest.raises(FileExistsError, match="immutable target"):
        module.publish_consumer_artifact(
            parent=parent,
            artifact_id="result",
            binding=_binding(checkpoint_ref),
            numerical_result={"aggregate": {"value": 2}},
        )
    assert (published / "result.json").read_bytes() == before


def test_reader_digest_binds_only_the_canonical_numerical_result(tmp_path: Path) -> None:
    module = _module()
    target = module.publish_consumer_artifact(
        parent=tmp_path,
        artifact_id="result",
        binding=_binding(
            {"owner": "runs", "path": "task/checkpoint.pt", "sha256": SHA_A}
        ),
        numerical_result={"aggregate": {"value": 1}},
    )
    manifest, numerical = module.load_consumer_artifact(
        target / "consumer_manifest.json"
    )
    assert manifest["binding"]["profile"] == "generated_common"
    assert numerical == {"aggregate": {"value": 1}}

    (target / "result.json").write_text(
        json.dumps({"aggregate": {"value": 2}}),
        encoding="utf-8",
    )
    with pytest.raises(ValueError, match="digest"):
        module.load_consumer_artifact(target / "consumer_manifest.json")
