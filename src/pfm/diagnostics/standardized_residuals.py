from __future__ import annotations

import math

import torch


STANDARDIZED_RESIDUAL_DEFINITION = "component_standardized_velocity_residual"
GAUSSIAN_COVERAGE_THRESHOLDS = {
    "90": 1.6448536269514722,
    "95": 1.959963984540054,
}


class StandardizedResidualAccumulator:
    """Reduce component residuals once; retain only image totals for time bins.

    Endpoint and velocity standardized residuals are the same random variable
    for the linear interpolant. Coverage describes marginal component intervals,
    not joint image coverage or independent observations across image pixels.
    """

    def __init__(self, *, include_time: bool) -> None:
        self._include_time = bool(include_time)
        self._totals = torch.zeros(5, dtype=torch.float64)
        self._sample_count = 0
        self._image_totals: list[torch.Tensor] = []
        self._image_times: list[torch.Tensor] = []

    def update(self, standardized_residual: torch.Tensor, time: torch.Tensor) -> None:
        z = standardized_residual.detach().float()
        if z.ndim < 2 or z.numel() == 0:
            raise ValueError("standardized residual requires non-empty batch and component dimensions")
        if not bool(torch.isfinite(z).all()):
            raise ValueError("standardized residual must contain only finite values")
        flattened = z.flatten(1)
        absolute = flattened.abs()
        totals = torch.stack(
            (
                torch.full(
                    (z.shape[0],),
                    flattened.shape[1],
                    dtype=torch.float64,
                    device=z.device,
                ),
                flattened.sum(dim=1, dtype=torch.float64),
                flattened.square().sum(dim=1, dtype=torch.float64),
                absolute.le(GAUSSIAN_COVERAGE_THRESHOLDS["90"]).sum(dim=1),
                absolute.le(GAUSSIAN_COVERAGE_THRESHOLDS["95"]).sum(dim=1),
            ),
            dim=1,
        ).cpu()
        if not bool(torch.isfinite(totals).all()):
            raise ValueError("standardized residual moments must contain only finite values")
        self._totals += totals.sum(dim=0)
        self._sample_count += int(z.shape[0])
        if self._include_time:
            times = time.detach().expand_as(z).flatten(1)
            if not bool(torch.isfinite(times).all()):
                raise ValueError("standardized residual time must contain only finite values")
            if not bool(times.eq(times[:, :1]).all()):
                raise ValueError("standardized residual time must be constant within each image")
            self._image_times.append(times[:, 0].to(dtype=torch.float64).cpu())
            self._image_totals.append(totals)

    def summary(
        self,
        *,
        num_bins: int,
        t_min: float,
        t_max: float,
    ) -> dict[str, object]:
        if self._sample_count == 0:
            raise ValueError("standardized residual accumulator has no component values")
        result: dict[str, object] = {
            "definition": STANDARDIZED_RESIDUAL_DEFINITION,
            "shared_spaces": ["velocity", "endpoint"],
            **self._summary_totals(self._totals),
            **{
                f"gaussian_threshold_{level}": threshold
                for level, threshold in GAUSSIAN_COVERAGE_THRESHOLDS.items()
            },
        }
        if not self._include_time:
            return result
        lower, upper = float(t_min), float(t_max)
        if not math.isfinite(lower) or not math.isfinite(upper) or not 0 <= lower < upper <= 1:
            raise ValueError("standardized residual time bounds must satisfy 0 <= t_min < t_max <= 1")
        times = torch.cat(self._image_times)
        if bool((times < lower - 1e-12).any()) or bool((times > upper + 1e-12).any()):
            raise ValueError("standardized residual times fall outside the authored bounded interval")
        totals = torch.cat(self._image_totals)
        bin_count = max(1, int(num_bins))
        indices = ((times - lower) / (upper - lower) * bin_count).floor().long()
        indices = indices.clamp(0, bin_count - 1)
        bins = []
        for index in range(bin_count):
            selected = indices == index
            if not bool(selected.any()):
                continue
            bin_time = times[selected]
            bins.append(
                {
                    "bin": index,
                    "t_min": float(bin_time.min()),
                    "t_max": float(bin_time.max()),
                    "t_mean": float(bin_time.mean()),
                    "sample_count": int(selected.sum()),
                    **self._summary_totals(totals[selected].sum(dim=0)),
                }
            )
        if sum(int(item["component_count"]) for item in bins) != result["component_count"]:
            raise ValueError("standardized residual time-bin counts do not reconcile")
        result.update(
            {
                "num_time_bins": bin_count,
                "time_binning": "equal_width_bounded_interval",
                "time_bounds": [lower, upper],
                "time_bins": bins,
            }
        )
        return result

    @staticmethod
    def _summary_totals(totals: torch.Tensor) -> dict[str, object]:
        count = int(totals[0])
        return {
            "component_count": count,
            "mean": float(totals[1] / count),
            "second_moment": float(totals[2] / count),
            "gaussian_coverage_90": float(totals[3] / count),
            "gaussian_coverage_95": float(totals[4] / count),
            "coverage_count_90": int(totals[3]),
            "coverage_count_95": int(totals[4]),
        }
