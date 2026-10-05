import pytest

from pfm.diagnostics.sampling_numerics import SamplingNumericalAccumulator


def _summary(value, *, error=0.5, nonfinite_count=0):
    return {
        "score_mean_square": value,
        "divergence_mean_square": value,
        "correction_mean_square": value,
        "expected_brownian_increment_mean_square": value,
        "derivative_update_mse_estimate": error,
        "nonfinite_count": nonfinite_count,
    }


def test_unequal_generation_chunks_use_sample_weights_not_chunk_weights():
    accumulator = SamplingNumericalAccumulator()
    accumulator.update(_summary(1.0, error=2.0, nonfinite_count=2), 3)
    accumulator.update(_summary(5.0, error=6.0, nonfinite_count=1), 1)
    assert accumulator.finalize() == _summary(2.0, error=3.0, nonfinite_count=3)


def test_unavailable_probe_error_does_not_become_zero_or_partial_population_average():
    accumulator = SamplingNumericalAccumulator()
    accumulator.update(_summary(1.0, error=2.0), 3)
    accumulator.update(_summary(5.0, error=None), 1)
    assert accumulator.finalize() == _summary(2.0, error=None)


def test_empty_summary_can_only_combine_with_empty_summaries():
    accumulator = SamplingNumericalAccumulator()
    assert accumulator.finalize() == {}
    accumulator.update({}, 2)
    assert accumulator.finalize() == {}
    with pytest.raises(ValueError, match="availability must be consistent"):
        accumulator.update(_summary(1.0), 1)


@pytest.mark.parametrize("invalid", [-1.0, float("inf"), float("nan"), True])
def test_invalid_mean_square_is_rejected_before_accumulation(invalid):
    accumulator = SamplingNumericalAccumulator()
    with pytest.raises(ValueError):
        accumulator.update(_summary(invalid), 2)
    assert accumulator.finalize() == {}
