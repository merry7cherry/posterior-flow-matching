from __future__ import annotations

import ast
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]
PFM_ROOT = PROJECT_ROOT / "src" / "pfm"


def _public_functions(path: Path) -> set[str]:
    tree = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
    return {
        node.name
        for node in tree.body
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef))
        and not node.name.startswith("_")
    }


def _imported_names(path: Path, module: str) -> set[str]:
    tree = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
    return {
        alias.name
        for node in tree.body
        if isinstance(node, ast.ImportFrom) and node.module == module
        for alias in node.names
    }


def test_checkpoint_lifecycle_has_explicit_layer_owners() -> None:
    core_path = PFM_ROOT / "core" / "checkpointing.py"
    training_path = PFM_ROOT / "training" / "checkpointing.py"
    evaluation_path = PFM_ROOT / "evaluation" / "checkpoint_loading.py"

    assert training_path.is_file()
    core_functions = _public_functions(core_path)
    training_functions = _public_functions(training_path)
    evaluation_functions = _public_functions(evaluation_path)

    producer_functions = {
        "build_training_checkpoint",
        "load_canonical_resume_checkpoint",
        "publish_training_checkpoint",
        "validate_resume_runtime_identity",
        "write_checkpoint_manifest",
    }
    assert producer_functions <= training_functions
    assert producer_functions.isdisjoint(core_functions)
    assert "load_model_from_bound_checkpoint" in evaluation_functions


def test_checkpoint_consumers_share_preflight_and_binding_owner() -> None:
    loading_path = PFM_ROOT / "evaluation" / "checkpoint_loading.py"
    loading_functions = _public_functions(loading_path)

    assert {
        "resolve_bound_epoch_checkpoint",
        "build_checkpoint_consumer_binding",
    } <= loading_functions

    for relative in ("checkpoint_evaluation.py", "sample_grid.py"):
        consumer_path = PFM_ROOT / "evaluation" / relative
        assert "build_checkpoint_consumer_binding" in _imported_names(
            consumer_path,
            "pfm.evaluation.checkpoint_loading",
        )
        assert "make_relative_reference" not in _imported_names(
            consumer_path,
            "pfm.artifacts.consumer_artifact",
        )
