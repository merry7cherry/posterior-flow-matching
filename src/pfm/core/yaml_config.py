from __future__ import annotations

from collections.abc import Mapping
from pathlib import Path
from typing import Any
import re
import math
from copy import deepcopy

import yaml


class _UniqueKeyLoader(yaml.SafeLoader):
    pass


def _construct_unique_mapping(
    loader: _UniqueKeyLoader,
    node: yaml.MappingNode,
    deep: bool = False,
) -> dict[Any, Any]:
    mapping: dict[Any, Any] = {}
    for key_node, value_node in node.value:
        key = loader.construct_object(key_node, deep=deep)
        if key in mapping:
            raise ValueError(f"duplicate YAML key {key!r} at line {key_node.start_mark.line + 1}")
        mapping[key] = loader.construct_object(value_node, deep=deep)
    return mapping


_UniqueKeyLoader.add_constructor(
    yaml.resolver.BaseResolver.DEFAULT_MAPPING_TAG,
    _construct_unique_mapping,
)


def load_yaml_mapping(path: str | Path) -> Mapping[str, Any]:
    config_path = Path(path)
    return parse_yaml_mapping(config_path.read_bytes(), context=str(config_path))


def parse_yaml_mapping(contents: str | bytes, *, context: str) -> Mapping[str, Any]:
    raw = yaml.load(contents, Loader=_UniqueKeyLoader)
    if raw is None:
        return {}
    if not isinstance(raw, Mapping):
        raise ValueError(f"{context} YAML document must be a mapping")
    return raw


def parse_overrides(items: list[str] | None) -> dict[str, Any]:
    """One explicit assignment per known config field; null is a value."""
    result: dict[str, Any] = {}
    for item in items or []:
        key, separator, value = item.partition("=")
        if not separator or not re.fullmatch(r"[a-z][a-z0-9_]*(?:\.[a-z][a-z0-9_]*)*", key):
            raise ValueError(f"override must be field=value: {item!r}")
        if any(key == prior or key.startswith(prior + ".") or prior.startswith(key + ".") for prior in result):
            raise ValueError(f"duplicate or overlapping override: {key}")
        if not value.strip():
            raise ValueError(f"empty override for {key}; use null explicitly")
        parsed = yaml.load(value, Loader=_UniqueKeyLoader)
        # YAML 1.1 treats exponent-only numbers such as 1e-4 as strings.
        if isinstance(parsed, str) and re.fullmatch(r"[-+]?\d+(?:\.\d*)?[eE][-+]?\d+", value):
            parsed = float(value)
        result[key] = parsed
    return result


def merge_config(raw: Mapping[str, Any], overrides: Mapping[str, Any] | None) -> dict[str, Any]:
    result = deepcopy(dict(raw))
    for key, value in (overrides or {}).items():
        parts = key.split(".")
        target = result
        for part in parts[:-1]:
            child = target.get(part)
            if not isinstance(child, Mapping):
                raise ValueError(f"nested override parent must be an existing mapping: {key}")
            if not isinstance(child, dict):
                child = target[part] = dict(child)
            target = child
        # The owning schema checks the final field, including unknown keys.
        target[parts[-1]] = value
    return result


def canonical_config_bytes(raw: dict) -> bytes:
    def plain(value):
        if isinstance(value, dict):
            return {key: plain(item) for key, item in value.items()}
        if isinstance(value, (tuple, list)):
            return [plain(item) for item in value]
        if isinstance(value, float):
            if not math.isfinite(value):
                raise ValueError("configuration numbers must be finite")
            # Equivalent numeric spellings have one identity after schema validation.
            return int(value) if value.is_integer() else value
        return value
    return yaml.safe_dump(plain(raw), sort_keys=True, allow_unicode=True).encode()
