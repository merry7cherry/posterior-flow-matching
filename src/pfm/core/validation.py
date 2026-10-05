from __future__ import annotations

import math
from collections.abc import Mapping
from numbers import Integral, Real


_MISSING = object()


def require_mapping_field(mapping: Mapping[str, object], key: str, *, context: str) -> object:
    if key not in mapping:
        raise ValueError(f"{context} {key} is required")
    return mapping[key]


def require_positive_integer(value: object, name: str, *, context: str) -> int:
    resolved = _require_integral(value, name, context=context)
    if resolved < 1:
        raise ValueError(f"{context} {name} must be a positive integer")
    return resolved


def require_non_negative_integer(value: object, name: str, *, context: str) -> int:
    resolved = _require_integral(value, name, context=context)
    if resolved < 0:
        raise ValueError(f"{context} {name} must be a non-negative integer")
    return resolved


def require_finite_number(value: object, name: str, *, context: str) -> float:
    if value is _MISSING:
        raise ValueError(f"{context} {name} is required")
    if isinstance(value, bool) or not isinstance(value, Real):
        raise ValueError(f"{context} {name} must be a finite number")
    resolved = float(value)
    if not math.isfinite(resolved):
        raise ValueError(f"{context} {name} must be finite")
    return resolved


def require_positive_integer_attribute(source: object, attribute: str, *, context: str) -> int:
    return require_positive_integer(
        getattr(source, attribute, _MISSING),
        attribute,
        context=context,
    )


def _require_integral(value: object, name: str, *, context: str) -> int:
    if value is _MISSING:
        raise ValueError(f"{context} {name} is required")
    if isinstance(value, bool) or not isinstance(value, Integral):
        raise ValueError(f"{context} {name} must be an integer")
    return int(value)
