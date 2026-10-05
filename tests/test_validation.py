import pytest

from pfm.core.validation import (
    require_finite_number,
    require_non_negative_integer,
    require_positive_integer,
)


@pytest.mark.parametrize("value", [1.0, 1.5, "1", True, False, None])
def test_require_positive_integer_rejects_non_integral_protocol_values(value):
    with pytest.raises(ValueError):
        require_positive_integer(value, "sampling_steps", context="evaluation")


def test_require_positive_integer_accepts_plain_int():
    assert require_positive_integer(3, "sampling_steps", context="evaluation") == 3


@pytest.mark.parametrize("value", [-1, 1.0, "0", True, None])
def test_require_non_negative_integer_rejects_invalid_values(value):
    with pytest.raises(ValueError):
        require_non_negative_integer(value, "step", context="training metrics")


def test_require_non_negative_integer_accepts_zero():
    assert require_non_negative_integer(0, "step", context="training metrics") == 0


@pytest.mark.parametrize("value", [float("nan"), float("inf"), True, "1.0", None])
def test_require_finite_number_rejects_invalid_values(value):
    with pytest.raises(ValueError):
        require_finite_number(value, "loss", context="training checkpoint")
