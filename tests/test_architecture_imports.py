from __future__ import annotations

import ast
import tomllib
from collections.abc import Mapping
from pathlib import Path

import yaml


PROJECT_ROOT = Path(__file__).resolve().parents[1]
SOURCE_ROOT = PROJECT_ROOT / "src" / "pfm"

SUPPORTED_PACKAGES = {
    "artifacts",
    "cli",
    "core",
    "data",
    "diagnostics",
    "evaluation",
    "models",
    "probability",
    "sampling",
    "training",
    "visualization",
}
EXPECTED_CONSOLE_SCRIPTS = {
    "pfm-aggregate": "pfm.cli.aggregate:main",
    "pfm-evaluate": "pfm.cli.evaluate:main",
    "pfm-prepare-fid": "pfm.cli.prepare_fid:main",
    "pfm-prepare-data": "pfm.cli.prepare_data:main",
    "pfm-sample": "pfm.cli.sample:main",
    "pfm-train": "pfm.cli.train:main",
    "pfm-train-classifier": "pfm.cli.train_classifier:main",
}
FORBIDDEN_CONFIG_KEYS = {"device", "data_root", "output_root"}


def _python_files(*roots: Path) -> list[Path]:
    return sorted(path for root in roots for path in root.rglob("*.py"))


def _nested_forbidden_keys(value: object, prefix: str = "") -> list[str]:
    if isinstance(value, Mapping):
        offenders = []
        for key, nested in value.items():
            path = f"{prefix}.{key}" if prefix else str(key)
            if key in FORBIDDEN_CONFIG_KEYS:
                offenders.append(path)
            offenders.extend(_nested_forbidden_keys(nested, path))
        return offenders
    if isinstance(value, list):
        return [
            offender
            for index, nested in enumerate(value)
            for offender in _nested_forbidden_keys(nested, f"{prefix}[{index}]")
        ]
    return []


def _current_docs() -> list[Path]:
    files = [PROJECT_ROOT / "README.md"]
    for path in (PROJECT_ROOT / "docs").rglob("*"):
        if not path.is_file():
            continue
        try:
            text = path.read_text(encoding="utf-8")
        except UnicodeDecodeError:
            continue
        if "\0" not in text:
            files.append(path)
    return sorted(set(files))


def _module_name(path: Path) -> str:
    relative = path.relative_to(SOURCE_ROOT)
    parts = relative.parent.parts if relative.name == "__init__.py" else relative.with_suffix("").parts
    return ".".join(("pfm", *parts))


def _resolved_import_module(path: Path, node: ast.ImportFrom) -> str:
    if node.level == 0:
        return node.module or ""
    source = _module_name(path)
    package = source if path.name == "__init__.py" else source.rsplit(".", 1)[0]
    parts = package.split(".")[: -(node.level - 1) or None]
    if node.module:
        parts.extend(node.module.split("."))
    return ".".join(parts)


def _defined_function_names(path: Path) -> set[str]:
    tree = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
    return {
        node.name
        for node in tree.body
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef))
    }


def _imports_name_from(path: Path, module: str, name: str) -> bool:
    tree = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
    return any(
        isinstance(node, ast.ImportFrom)
        and _resolved_import_module(path, node) == module
        and any(alias.name == name for alias in node.names)
        for node in tree.body
    )


def _top_level_assigned_names(path: Path) -> set[str]:
    tree = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
    names = set()
    for node in tree.body:
        if isinstance(node, ast.Assign):
            names.update(
                target.id for target in node.targets if isinstance(target, ast.Name)
            )
        elif isinstance(node, ast.AnnAssign) and isinstance(node.target, ast.Name):
            names.add(node.target.id)
    return names


def _top_level_bound_names(path: Path) -> set[str]:
    tree = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
    names = _top_level_assigned_names(path)
    for node in tree.body:
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef)):
            names.add(node.name)
        elif isinstance(node, ast.Import):
            names.update(alias.asname or alias.name.split(".", 1)[0] for alias in node.names)
        elif isinstance(node, ast.ImportFrom):
            names.update(alias.asname or alias.name for alias in node.names)
    return names


def test_supported_package_and_entrypoint_inventory() -> None:
    packages = {
        path.name
        for path in SOURCE_ROOT.iterdir()
        if path.is_dir() and not path.name.startswith("__")
    }
    flat_modules = {
        path.name for path in SOURCE_ROOT.glob("*.py") if path.name != "__init__.py"
    }
    pyproject = tomllib.loads((PROJECT_ROOT / "pyproject.toml").read_text(encoding="utf-8"))

    assert packages == SUPPORTED_PACKAGES
    assert all((SOURCE_ROOT / package / "__init__.py").is_file() for package in packages)
    assert flat_modules == set()
    assert pyproject["project"]["scripts"] == EXPECTED_CONSOLE_SCRIPTS


def test_dataset_download_is_only_available_in_explicit_preparation() -> None:
    offenders = []
    for path in _python_files(SOURCE_ROOT, PROJECT_ROOT / "scripts"):
        if path in {SOURCE_ROOT / "data" / "preparation.py", SOURCE_ROOT / "cli" / "prepare_data.py"}:
            continue
        tree = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
        for node in ast.walk(tree):
            if not isinstance(node, ast.Call):
                continue
            downloads = [
                keyword.value
                for keyword in node.keywords
                if keyword.arg == "download"
            ]
            if any(not isinstance(value, ast.Constant) or value.value is not False for value in downloads):
                offenders.append(f"{path.relative_to(PROJECT_ROOT)}:{node.lineno}")
    assert offenders == []


def test_dataset_readers_have_no_publication_io() -> None:
    forbidden_imports = {
        "pfm.core.atomic_io",
        "pfm.core.immutable_artifact",
    }
    forbidden_calls = {
        "mkdir",
        "rename",
        "replace",
        "rmtree",
        "unlink",
        "write_bytes",
        "write_text",
        "write_json_atomic",
        "write_json_exclusive",
    }
    offenders = []
    for path in _python_files(SOURCE_ROOT / "data"):
        if path.name == "preparation.py":
            continue
        tree = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
        for node in ast.walk(tree):
            if isinstance(node, ast.ImportFrom) and node.module in forbidden_imports:
                offenders.append(f"{path.relative_to(PROJECT_ROOT)}:{node.lineno}:import")
            if isinstance(node, ast.Import):
                offenders.extend(
                    f"{path.relative_to(PROJECT_ROOT)}:{node.lineno}:import"
                    for alias in node.names
                    if alias.name in forbidden_imports
                )
            if not isinstance(node, ast.Call):
                continue
            function = node.func
            name = (
                function.attr
                if isinstance(function, ast.Attribute)
                else function.id if isinstance(function, ast.Name) else ""
            )
            if name in forbidden_calls:
                offenders.append(
                    f"{path.relative_to(PROJECT_ROOT)}:{node.lineno}:{name}"
                )
    assert offenders == []


def test_authored_configs_do_not_own_runtime_paths_or_devices() -> None:
    offenders = []
    for path in sorted((PROJECT_ROOT / "configs").rglob("*.yaml")):
        payload = yaml.safe_load(path.read_text(encoding="utf-8"))
        offenders.extend(
            f"{path.relative_to(PROJECT_ROOT)}:{key}"
            for key in _nested_forbidden_keys(payload)
        )
    assert offenders == []


def test_current_docs_do_not_require_cluster_deployment() -> None:
    offenders = []
    for path in _current_docs():
        text = path.read_text(encoding="utf-8")
        offenders.extend(
            f"{path.relative_to(PROJECT_ROOT)}:{token}"
            for token in ("PFM_ENV_ROOT", "SLURM_JOB_ID", "scripts/uva/", "UVA_HPC.md")
            if token in text
        )
    assert offenders == []


def test_production_cross_module_imports_use_public_names() -> None:
    offenders = []
    for path in _python_files(SOURCE_ROOT, PROJECT_ROOT / "scripts"):
        source_module = _module_name(path) if path.is_relative_to(SOURCE_ROOT) else None
        tree = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
        for node in ast.walk(tree):
            if isinstance(node, ast.ImportFrom):
                target = _resolved_import_module(path, node) if source_module else (node.module or "")
                if not (target == "pfm" or target.startswith("pfm.")) or target == source_module:
                    continue
                if any(part.startswith("_") for part in target.split(".")[1:]):
                    offenders.append(f"{path.relative_to(PROJECT_ROOT)}:{node.lineno}:{target}")
                offenders.extend(
                    f"{path.relative_to(PROJECT_ROOT)}:{node.lineno}:{alias.name}"
                    for alias in node.names
                    if alias.name.startswith("_")
                )
            elif isinstance(node, ast.Import):
                offenders.extend(
                    f"{path.relative_to(PROJECT_ROOT)}:{node.lineno}:{alias.name}"
                    for alias in node.names
                    if alias.name.startswith("pfm.")
                    and any(part.startswith("_") for part in alias.name.split(".")[1:])
                )
    assert offenders == []


def test_probability_targets_owns_interpolant_time_broadcasting() -> None:
    owner = SOURCE_ROOT / "probability" / "targets.py"
    consumers = [
        SOURCE_ROOT / "models" / "variants.py",
        SOURCE_ROOT / "diagnostics" / "posterior_residuals.py",
    ]

    assert "time_like" in _defined_function_names(owner)
    for path in consumers:
        assert "time_like" not in _top_level_bound_names(path)
        assert not _imports_name_from(path, "pfm.probability.targets", "time_like")
        assert _imports_name_from(path, "pfm.probability", "targets")


def test_dataset_manifest_name_has_one_module_owner() -> None:
    owner = SOURCE_ROOT / "data" / "identity.py"
    consumers = [
        SOURCE_ROOT / "data" / "datasets.py",
        SOURCE_ROOT / "evaluation" / "assets.py",
        SOURCE_ROOT / "evaluation" / "checkpoint_loading.py",
        SOURCE_ROOT / "evaluation" / "evaluation_summary.py",
        SOURCE_ROOT / "evaluation" / "fid_reference.py",
        SOURCE_ROOT / "training" / "classifier.py",
    ]

    assert "DATASET_MANIFEST_NAME" in _top_level_bound_names(owner)
    for path in consumers:
        assert "DATASET_MANIFEST_NAME" not in _top_level_bound_names(path)
        assert _imports_name_from(path, "pfm.data", "identity")


def test_variant_metadata_owns_method_and_capability_facts() -> None:
    owner_names = _top_level_assigned_names(
        SOURCE_ROOT / "core" / "variant_metadata.py"
    )
    assert {"VARIANT_CAPABILITIES", "SUPPORTED_VARIANTS"}.issubset(owner_names)
    assert "SELF_CONDITIONING_MODES" not in _top_level_bound_names(
        SOURCE_ROOT / "core" / "variant_metadata.py"
    )


def test_retired_self_conditioning_has_no_runtime_owners() -> None:
    for package in ("core", "probability", "sampling"):
        assert not (SOURCE_ROOT / package / "self_conditioning.py").exists()
    assert "validate_base_model_config" in _defined_function_names(
        SOURCE_ROOT / "core" / "variant_metadata.py"
    )
    assert "training_forward_with_self_conditioning" not in _defined_function_names(
        SOURCE_ROOT / "training" / "step.py"
    )

    retired_duplicates = {
        SOURCE_ROOT / "models" / "variants.py": {"VALID_VARIANTS", "PFM_LOGVAR_VARIANTS"},
        SOURCE_ROOT / "diagnostics" / "posterior_schema.py": {"PFM_CALIBRATION_VARIANTS"},
    }
    for path, names in retired_duplicates.items():
        assert _top_level_bound_names(path).isdisjoint(names)
