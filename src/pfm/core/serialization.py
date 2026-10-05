from __future__ import annotations

from collections.abc import Mapping
import math
from pathlib import Path

import numpy as np
import torch


def to_jsonable(value: object) -> object:
    if isinstance(value, float):
        if not math.isfinite(value):
            raise ValueError("JSON numeric values must be finite")
        return value
    if isinstance(value, (str, int, bool)) or value is None:
        return value
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, torch.Tensor):
        tensor = value.detach().cpu()
        if torch.is_complex(tensor):
            raise TypeError("complex tensor values are not JSON serializable")
        if tensor.ndim == 0:
            return to_jsonable(tensor.item())
        return to_jsonable(tensor.tolist())
    if isinstance(value, np.generic):
        if np.issubdtype(value.dtype, np.complexfloating):
            raise TypeError("complex numpy scalar values are not JSON serializable")
        if value.dtype == np.dtype("O"):
            raise TypeError("object numpy scalar values are not JSON serializable")
        return to_jsonable(value.item())
    if isinstance(value, np.ndarray):
        if np.issubdtype(value.dtype, np.complexfloating):
            raise TypeError("complex numpy array values are not JSON serializable")
        if value.dtype == np.dtype("O"):
            raise TypeError("object numpy array values are not JSON serializable")
        return to_jsonable(value.tolist())
    if isinstance(value, Mapping):
        converted: dict[str, object] = {}
        for key, item in value.items():
            if not isinstance(key, str):
                raise TypeError(f"JSON object keys must be strings, got {type(key).__name__}")
            converted[key] = to_jsonable(item)
        return converted
    if isinstance(value, (list, tuple)):
        return [to_jsonable(item) for item in value]
    raise TypeError(f"object of type {type(value).__name__} is not JSON serializable")
