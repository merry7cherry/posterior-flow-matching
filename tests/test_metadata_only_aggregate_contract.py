from __future__ import annotations

import ast
import hashlib
import importlib
import json
from pathlib import Path

import pytest


SHA_A = "a" * 64
SHA_B = "b" * 64
SHA_C = "c" * 64


def _publication_module():
    return importlib.import_module("pfm.artifacts.consumer_artifact")


def _summary_module():
    return importlib.import_module("pfm.evaluation.evaluation_summary")


def _spec() -> dict[str, object]:
    return {
        "comparison_id": "two-method-quality",
        "dataset": "mnist",
        "required_methods": ["velocity_fm", "pfm_full"],
        "required_consumers": {
            "velocity_fm": ["generated_common"],
            "pfm_full": ["generated_common", "posterior_diagnostics"],
        },
        "consumer_protocols": {
            "generated_common": {
                "owner": "project",
                "path": "configs/generated_common.yaml",
                "sha256": SHA_B,
            },
            "posterior_diagnostics": {
                "owner": "project",
                "path": "configs/posterior_diagnostics.yaml",
                "sha256": SHA_C,
            },
        },
    }


def _aggregate_manifest_payload() -> dict[str, object]:
    return {
        "schema_version": 3,
        "aggregate_revision": "9" * 40,
        "comparison_id": "two-method-quality",
        "comparison_sha256": "1" * 64,
        "aggregate_result": {
            "path": "aggregate.json",
            "sha256": "2" * 64,
        },
    }


def _publish(
    root: Path,
    *,
    method: str,
    profile: str,
    value: float,
    training_protocol: str = SHA_A,
    consumer_protocol: str | None = None,
    checkpoint_progress: tuple[int, int] = (1, 100),
    checkpoint_revision: str = "0" * 40,
    checkpoint_sha256: str | None = None,
    assets: dict[str, object] | None = None,
    config_path: str | None = None,
    dataset_path: str = "mnist/dataset_identity.json",
) -> Path:
    publication = _publication_module()
    expected_protocols = {
        "generated_common": SHA_B,
        "posterior_diagnostics": SHA_C,
    }
    protocol = consumer_protocol or str(expected_protocols[profile])
    target = publication.publish_consumer_artifact(
        parent=root,
        artifact_id=f"{method}-{profile}",
        binding={
            "profile": profile,
            "method": method,
            "shared_training_protocol_sha256": training_protocol,
            "dataset_scientific_identity_sha256": "e" * 64,
            "consumer_revision": "f" * 40,
            "checkpoint_revision": checkpoint_revision,
            "checkpoint_progress": {
                "epoch": checkpoint_progress[0],
                "step": checkpoint_progress[1],
            },
            "checkpoint": {
                "owner": "runs",
                "path": f"task/{method}/epochs/epoch_0001/checkpoint.pt",
                "sha256": checkpoint_sha256 or hashlib.sha256(method.encode()).hexdigest(),
            },
            "dataset": {
                "owner": "datasets",
                "path": dataset_path,
                "sha256": "e" * 64,
            },
            "config": {
                "owner": "project",
                "path": config_path or f"configs/{profile}.yaml",
                "sha256": protocol,
            },
            "assets": assets or {},
        },
        numerical_result={
            "aggregate": {"score": value},
            "detail_not_copied_to_aggregate": {"bins": list(range(30))},
        },
    )
    return target / "consumer_manifest.json"


def _complete_inputs(
    root: Path,
    *,
    checkpoint_progress: tuple[int, int] = (1, 100),
) -> list[Path]:
    inputs: list[Path] = []
    value = 0.0
    for method, profiles in _spec()["required_consumers"].items():
        for profile in profiles:
            value += 1.0
            inputs.append(
                _publish(
                    root,
                    method=method,
                    profile=profile,
                    value=value,
                    checkpoint_progress=checkpoint_progress,
                )
            )
    return inputs


def test_aggregate_source_has_no_checkpoint_model_data_or_discovery_dependencies() -> None:
    module = _summary_module()
    source_path = Path(module.__file__ or "")
    source = source_path.read_text(encoding="utf-8")
    tree = ast.parse(source)
    imported = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            imported.update(alias.name for alias in node.names)
        elif isinstance(node, ast.ImportFrom) and node.module:
            imported.add(node.module)

    forbidden_imports = {
        "torch",
        "pfm.core.checkpointing",
        "pfm.data.datasets",
        "pfm.evaluation.checkpoint_loading",
        "pfm.models.factory",
        "pfm.visualization.plots",
    }
    assert imported.isdisjoint(forbidden_imports)
    assert ".rglob(" not in source
    assert ".glob(" not in source
    assert ".iterdir(" not in source
    assert "os.walk(" not in source
    assert "torch.load(" not in source


def test_comparison_rejects_qualitative_sample_as_a_numerical_dependency() -> None:
    module = _summary_module()
    spec = _spec()
    spec["required_consumers"] = {
        "velocity_fm": ["generated_common", "sample_visualization"],
        "pfm_full": ["generated_common", "posterior_diagnostics"],
    }
    spec["consumer_protocols"] = {
        **spec["consumer_protocols"],
        "sample_visualization": {
            "owner": "project",
            "path": "configs/sample_visualization.yaml",
            "sha256": "d" * 64,
        },
    }

    with pytest.raises(ValueError, match="required_consumers.velocity_fm"):
        module.validate_comparison_spec(spec)


def test_explicit_aggregate_is_order_independent_and_compact(tmp_path: Path) -> None:
    module = _summary_module()
    inputs = _complete_inputs(tmp_path / "inputs")
    first = module.aggregate_consumer_results(
        _spec(), inputs, task_root=tmp_path / "inputs", requested_epoch=1
    )
    second = module.aggregate_consumer_results(
        _spec(),
        list(reversed(inputs)),
        task_root=tmp_path / "inputs",
        requested_epoch=1,
    )

    assert first == second
    assert first["comparison_id"] == "two-method-quality"
    assert [
        (entry["method"], entry["profile"])
        for entry in first["results"]
    ] == sorted(
        (method, profile)
        for method, profiles in _spec()["required_consumers"].items()
        for profile in profiles
    )
    serialized = json.dumps(first, sort_keys=True)
    assert "detail_not_copied_to_aggregate" not in serialized
    assert "bins" not in serialized
    assert "checkpoint_manifest" not in serialized
    assert all(
        entry["source"][role]["owner"] == "artifact"
        for entry in first["results"]
        for role in ("consumer_manifest", "numerical_result")
    )


@pytest.mark.parametrize("case", ["missing", "duplicate", "unexpected"])
def test_aggregate_fails_closed_on_incomplete_or_extra_coverage(
    tmp_path: Path,
    case: str,
) -> None:
    module = _summary_module()
    inputs = _complete_inputs(tmp_path / "inputs")
    if case == "missing":
        inputs.pop()
    elif case == "duplicate":
        inputs.append(inputs[0])
    else:
        inputs.append(
            _publish(
                tmp_path / "inputs",
                method="endpoint_fm",
                profile="generated_common",
                value=99.0,
            )
        )
    with pytest.raises(ValueError, match="missing|duplicate|unexpected|coverage"):
        module.aggregate_consumer_results(
            _spec(), inputs, task_root=tmp_path / "inputs", requested_epoch=1
        )


@pytest.mark.parametrize("mismatch", ["training", "consumer"])
def test_aggregate_fails_on_protocol_mismatch(tmp_path: Path, mismatch: str) -> None:
    module = _summary_module()
    inputs = _complete_inputs(tmp_path / "inputs")
    bad = _publish(
        tmp_path / "inputs" / "bad",
        method="velocity_fm",
        profile="generated_common",
        value=1.0,
        training_protocol="f" * 64 if mismatch == "training" else SHA_A,
        consumer_protocol="f" * 64 if mismatch == "consumer" else SHA_B,
    )
    inputs[0] = bad
    with pytest.raises(ValueError, match="protocol"):
        module.aggregate_consumer_results(
            _spec(), inputs, task_root=tmp_path / "inputs", requested_epoch=1
        )


def test_aggregate_fails_on_checkpoint_progress_mismatch(tmp_path: Path) -> None:
    module = _summary_module()
    inputs = _complete_inputs(tmp_path / "inputs")
    inputs[0] = _publish(
        tmp_path / "inputs" / "bad",
        method="velocity_fm",
        profile="generated_common",
        value=1.0,
        checkpoint_progress=(50, 5000),
    )

    with pytest.raises(ValueError, match="checkpoint progress mismatch"):
        module.aggregate_consumer_results(
            _spec(), inputs, task_root=tmp_path / "inputs", requested_epoch=1
        )


def test_aggregate_rejects_consistent_inputs_from_the_wrong_requested_epoch(
    tmp_path: Path,
) -> None:
    module = _summary_module()
    inputs = _complete_inputs(
        tmp_path / "inputs",
        checkpoint_progress=(50, 5000),
    )

    with pytest.raises(
        ValueError,
        match="requested epoch 200, but consumer results bind epoch 50",
    ):
        module.aggregate_consumer_results(
            _spec(),
            inputs,
            task_root=tmp_path / "inputs",
            requested_epoch=200,
        )


def test_aggregate_rejects_mixed_checkpoint_revisions(tmp_path: Path) -> None:
    module = _summary_module()
    inputs = _complete_inputs(tmp_path / "inputs")
    inputs[0] = _publish(
        tmp_path / "inputs" / "bad-revision",
        method="velocity_fm",
        profile="generated_common",
        value=1.0,
        checkpoint_revision="1" * 40,
    )

    with pytest.raises(ValueError, match="checkpoint revision mismatch"):
        module.aggregate_consumer_results(
            _spec(), inputs, task_root=tmp_path / "inputs", requested_epoch=1
        )


def test_aggregate_rejects_mixed_checkpoint_within_method(tmp_path: Path) -> None:
    module = _summary_module()
    inputs = _complete_inputs(tmp_path / "inputs")
    inputs[1] = _publish(
        tmp_path / "inputs" / "bad-checkpoint",
        method="pfm_full",
        profile="generated_common",
        value=1.0,
        checkpoint_sha256="1" * 64,
    )

    with pytest.raises(ValueError, match="mix checkpoints within method"):
        module.aggregate_consumer_results(
            _spec(), inputs, task_root=tmp_path / "inputs", requested_epoch=1
        )


def test_aggregate_rejects_generated_asset_mismatch(tmp_path: Path) -> None:
    module = _summary_module()
    inputs = _complete_inputs(tmp_path / "inputs")
    inputs[0] = _publish(
        tmp_path / "inputs" / "bad-assets",
        method="velocity_fm",
        profile="generated_common",
        value=1.0,
        assets={
            "classifier": {
                "owner": "assets",
                "path": "classifiers/mnist/manifest.json",
                "sha256": "2" * 64,
            }
        },
    )

    with pytest.raises(ValueError, match="asset binding mismatch"):
        module.aggregate_consumer_results(
            _spec(), inputs, task_root=tmp_path / "inputs", requested_epoch=1
        )


def test_primary_comparison_rejects_all_smoke_protocols(tmp_path: Path) -> None:
    module = _summary_module()
    inputs = _complete_inputs(tmp_path / "inputs")
    smoke_inputs = []
    for index, path in enumerate(inputs):
        manifest = json.loads(path.read_text(encoding="utf-8"))
        binding = manifest["binding"]
        smoke_inputs.append(
            _publish(
                tmp_path / "inputs" / f"smoke-{index}",
                method=binding["method"],
                profile=binding["profile"],
                value=float(index),
                consumer_protocol="3" * 64,
                config_path="configs/evaluation/smoke.yaml",
            )
        )

    with pytest.raises(ValueError, match="comparison protocol"):
        module.aggregate_consumer_results(
            _spec(), smoke_inputs, task_root=tmp_path / "inputs", requested_epoch=1
        )


def test_comparison_dataset_rejects_complete_wrong_dataset_set(tmp_path: Path) -> None:
    module = _summary_module()
    inputs = _complete_inputs(tmp_path / "inputs")
    wrong = []
    for index, path in enumerate(inputs):
        binding = json.loads(path.read_text(encoding="utf-8"))["binding"]
        wrong.append(
            _publish(
                tmp_path / "inputs" / f"wrong-dataset-{index}",
                method=binding["method"],
                profile=binding["profile"],
                value=float(index),
                dataset_path="cifar10/dataset_identity.json",
            )
        )

    with pytest.raises(ValueError, match="comparison dataset"):
        module.aggregate_consumer_results(
            _spec(), wrong, task_root=tmp_path / "inputs", requested_epoch=1
        )


def test_aggregate_uses_only_explicit_files_and_reads_each_numerical_owner_once(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    module = _summary_module()
    inputs = _complete_inputs(tmp_path / "inputs")
    original_read_bytes = Path.read_bytes
    reads: dict[Path, int] = {}

    def counted_read_bytes(path: Path) -> bytes:
        resolved = path.resolve()
        reads[resolved] = reads.get(resolved, 0) + 1
        return original_read_bytes(path)

    def forbidden_discovery(*_args, **_kwargs):
        raise AssertionError("aggregate attempted run-tree discovery")

    monkeypatch.setattr(Path, "read_bytes", counted_read_bytes)
    monkeypatch.setattr(Path, "rglob", forbidden_discovery)
    monkeypatch.setattr(Path, "glob", forbidden_discovery)
    monkeypatch.setattr(Path, "iterdir", forbidden_discovery)

    module.aggregate_consumer_results(
        _spec(), inputs, task_root=tmp_path / "inputs", requested_epoch=1
    )

    numerical_paths = [path.parent / "result.json" for path in inputs]
    assert {reads[path.resolve()] for path in numerical_paths} == {1}


def test_aggregate_manifest_records_executor_revision_without_changing_result(
    tmp_path: Path,
) -> None:
    module = _summary_module()
    task_root = tmp_path / "task"
    aggregate_revision = "9" * 40

    published = module.publish_aggregate_result(
        parent=task_root / "aggregates",
        aggregate_id="primary-v2",
        comparison=_spec(),
        manifest_paths=_complete_inputs(task_root),
        requested_epoch=1,
        aggregate_revision=aggregate_revision,
    )

    manifest = json.loads(
        (published / "aggregate_manifest.json").read_text(encoding="utf-8")
    )
    result = json.loads((published / "aggregate.json").read_text(encoding="utf-8"))
    result_bytes = (published / "aggregate.json").read_bytes()
    assert set(manifest) == {
        "schema_version",
        "aggregate_revision",
        "comparison_id",
        "comparison_sha256",
        "aggregate_result",
    }
    assert manifest["schema_version"] == 3
    assert manifest["aggregate_revision"] == aggregate_revision
    assert manifest["comparison_id"] == "two-method-quality"
    assert manifest["comparison_sha256"] == module._canonical_json_sha256(
        module.validate_comparison_spec(_spec())
    )
    assert manifest["aggregate_result"] == {
        "path": "aggregate.json",
        "sha256": hashlib.sha256(result_bytes).hexdigest(),
    }
    assert result["consumer_revision"] == "f" * 40
    assert result["checkpoint_revision"] == "0" * 40
    assert result["checkpoint_progress"] == {"epoch": 1, "step": 100}
    assert "requested_epoch" not in result
    assert aggregate_revision != result["consumer_revision"]
    assert "aggregate_revision" not in result


def test_current_aggregate_manifest_validator_accepts_exact_shape() -> None:
    module = _summary_module()
    manifest = _aggregate_manifest_payload()

    assert module._validate_aggregate_manifest(manifest) == manifest


@pytest.mark.parametrize(
    ("case", "message"),
    [
        ("unknown", "keys"),
        ("missing", "keys"),
        ("schema", "schema_version"),
        ("revision", "aggregate revision"),
        ("comparison_id", "comparison_id"),
        ("comparison_digest", "comparison"),
        ("result_mapping", "aggregate result"),
        ("result_keys", "aggregate result"),
        ("result_path", "aggregate.json"),
        ("result_digest", "aggregate result"),
    ],
)
def test_current_aggregate_manifest_validator_rejects_malformed_or_unknown_fields(
    case: str,
    message: str,
) -> None:
    module = _summary_module()
    manifest = _aggregate_manifest_payload()
    if case == "unknown":
        manifest["unexpected"] = True
    elif case == "missing":
        manifest.pop("comparison_id")
    elif case == "schema":
        manifest["schema_version"] = 1
    elif case == "revision":
        manifest["aggregate_revision"] = "A" * 40
    elif case == "comparison_id":
        manifest["comparison_id"] = "../escape"
    elif case == "comparison_digest":
        manifest["comparison_sha256"] = "not-a-digest"
    elif case == "result_mapping":
        manifest["aggregate_result"] = []
    else:
        result_reference = manifest["aggregate_result"]
        assert isinstance(result_reference, dict)
        if case == "result_keys":
            result_reference["unexpected"] = True
        elif case == "result_path":
            result_reference["path"] = "other.json"
        else:
            result_reference["sha256"] = "not-a-digest"

    with pytest.raises(ValueError, match=message):
        module._validate_aggregate_manifest(manifest)


@pytest.mark.parametrize(
    "aggregate_revision",
    ["a" * 39, "A" * 40, "g" * 40, 123],
)
def test_aggregate_publication_rejects_invalid_executor_revision_before_reading_inputs(
    tmp_path: Path,
    aggregate_revision: object,
) -> None:
    module = _summary_module()

    with pytest.raises(ValueError, match="aggregate revision"):
        module.publish_aggregate_result(
            parent=tmp_path / "task" / "aggregates",
            aggregate_id="primary-v2",
            comparison=_spec(),
            manifest_paths=[],
            requested_epoch=1,
            aggregate_revision=aggregate_revision,
        )
