from __future__ import annotations

import ast
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]
SOURCE_ROOT = PROJECT_ROOT / "src" / "pfm"
LAYERS = {
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
ALLOWED_IMPORTS = {
    "core": {"core"},
    "data": {"core", "data"},
    "models": {"core", "models", "probability"},
    "probability": {"core", "probability"},
    "sampling": {"core", "models", "probability", "sampling"},
    "training": {"core", "data", "models", "probability", "sampling", "training"},
    "evaluation": {"artifacts", "core", "data", "diagnostics", "evaluation", "models", "probability", "sampling", "visualization"},
    "artifacts": {"artifacts", "core", "diagnostics", "visualization"},
    "diagnostics": {"core", "diagnostics", "models", "probability", "sampling", "visualization"},
    "visualization": {"core", "data", "visualization"},
    "cli": {"cli", "core", "data", "evaluation", "models", "training"},
}


def _module_layer(path: Path) -> str | None:
    rel = path.relative_to(SOURCE_ROOT)
    if len(rel.parts) < 2:
        return None
    layer = rel.parts[0]
    return layer if layer in LAYERS else None


def _imported_layer(module: str) -> str | None:
    parts = module.split(".")
    if not parts or parts[0] != "pfm" or len(parts) < 2:
        return None
    layer = parts[1]
    return layer if layer in LAYERS else None


def test_layer_import_direction_is_enforced() -> None:
    offenders = []
    for path in SOURCE_ROOT.rglob("*.py"):
        source_layer = _module_layer(path)
        if source_layer is None:
            continue
        tree = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
        allowed = ALLOWED_IMPORTS[source_layer]
        for node in ast.walk(tree):
            modules: list[str] = []
            if isinstance(node, ast.ImportFrom) and node.module:
                modules.append(node.module)
            elif isinstance(node, ast.Import):
                modules.extend(alias.name for alias in node.names)
            for module in modules:
                imported_layer = _imported_layer(module)
                if imported_layer is not None and imported_layer not in allowed:
                    offenders.append(
                        f"{path.relative_to(PROJECT_ROOT)}:{node.lineno}: "
                        f"{source_layer} cannot import {imported_layer} via {module}"
                    )
    assert offenders == []
