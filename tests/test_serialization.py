from pathlib import Path

import numpy as np
import pytest
import torch

from pfm.core.serialization import to_jsonable


def test_to_jsonable_converts_project_metric_values():
    payload = {
        "path": Path("runs/task"),
        "scalar_tensor": torch.tensor(3.0),
        "vector_tensor": torch.tensor([1, 2]),
        "numpy_scalar": np.float32(0.25),
        "numpy_array": np.array([[1, 2], [3, 4]]),
        "tuple_value": ("a", 1),
    }

    assert to_jsonable(payload) == {
        "path": "runs/task",
        "scalar_tensor": 3.0,
        "vector_tensor": [1, 2],
        "numpy_scalar": np.float32(0.25).item(),
        "numpy_array": [[1, 2], [3, 4]],
        "tuple_value": ["a", 1],
    }


def test_to_jsonable_rejects_non_string_mapping_keys():
    with pytest.raises(TypeError, match="JSON object keys"):
        to_jsonable({1: "value"})


@pytest.mark.parametrize("value", [float("nan"), float("inf"), float("-inf")])
def test_to_jsonable_rejects_nonfinite_python_floats(value):
    with pytest.raises(ValueError, match="finite"):
        to_jsonable({"loss": value})


def test_write_json_atomic_rejects_nonfinite_values(tmp_path):
    from pfm.core.atomic_io import write_json_atomic

    with pytest.raises(ValueError, match="Out of range float values"):
        write_json_atomic(tmp_path / "metrics.json", {"loss": float("inf")})


def test_to_jsonable_rejects_unknown_objects():
    class Unknown:
        pass

    with pytest.raises(TypeError, match="not JSON serializable"):
        to_jsonable(Unknown())


@pytest.mark.parametrize(
    "value",
    [
        torch.tensor(1 + 2j),
        torch.tensor([1 + 2j]),
        np.complex64(1 + 2j),
        np.array([1 + 2j]),
    ],
)
def test_to_jsonable_rejects_complex_tensor_and_numpy_values(value):
    with pytest.raises(TypeError, match="complex"):
        to_jsonable(value)


def test_to_jsonable_rejects_numpy_object_arrays_before_recursive_conversion():
    with pytest.raises(TypeError, match="object"):
        to_jsonable(np.array([{"nested": "jsonable"}], dtype=object))
