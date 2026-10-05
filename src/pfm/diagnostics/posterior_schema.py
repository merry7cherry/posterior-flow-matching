from __future__ import annotations

NLL_TENSOR_KEYS = (
    "velocity_constant_free_nll_per_dim",
    "endpoint_constant_free_nll_per_dim",
    "velocity_complete_nll_per_dim",
    "endpoint_complete_nll_per_dim",
    "velocity_complete_nll_total",
    "endpoint_complete_nll_total",
)

POSTERIOR_SPACE_NAMES = ("velocity", "endpoint")

CONTROLLED_POSTERIOR_ESTIMAND = "fixed_time_marginal_interpolant_calibration"
CONTROLLED_POSTERIOR_SAMPLING = "independent_x0_x1_pairs_per_t"
