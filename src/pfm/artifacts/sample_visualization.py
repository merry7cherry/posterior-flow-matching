from __future__ import annotations

import math
from pathlib import Path

import torch

from pfm.core.config import ExperimentConfig
from pfm.core.sampling_config import SamplingConfig
from pfm.diagnostics.sampling_numerics import SamplingNumericalAccumulator
from pfm.core.variant_metadata import variant_capabilities
from pfm.visualization.sample_figures import (
    save_raw_variance_grid_for_labels,
    save_sample_grid_for_labels,
    save_variance_position_grid_for_labels,
)


class SampleVisualizationWriter:
    """Stream one bounded, fixed-scale visualization product into paged grids."""

    def __init__(
        self,
        *,
        artifact_root: Path,
        config: ExperimentConfig,
        variant: str,
        num_per_class: int,
        class_start: int,
        class_count: int,
        expected_count: int,
        sampling: SamplingConfig,
        capture_t: float,
    ) -> None:
        self.artifact_root = Path(artifact_root)
        self.presentation_root = self.artifact_root / "presentation"
        self.config = config
        self.variant = str(variant)
        self.num_per_class = int(num_per_class)
        self.class_start = int(class_start)
        self.class_count = int(class_count)
        self.expected_count = int(expected_count)
        self.sampling = sampling
        self.capture_t = float(capture_t)
        self.has_variance = variant_capabilities(self.variant).predicts_logvar
        self.has_diffusion = sampling.solver == "sde" and sampling.diffusion_scale > 0
        self.classes_per_page = 10 if int(config.num_classes) <= 10 else 50
        self.items_per_page = self.classes_per_page * self.num_per_class
        self._labels: list[torch.Tensor] = []
        self._samples: list[torch.Tensor] = []
        self._endpoint: list[torch.Tensor] = []
        self._velocity: list[torch.Tensor] = []
        self._diffusion: list[torch.Tensor] = []
        self._noise_covariance: list[torch.Tensor] = []
        self._buffer_count = 0
        self._seen = 0
        self._page_index = 0
        self._actual_nfe: int | None = None
        self._capture_step: int | None = None
        self._capture_observed_t: float | None = None
        self._capture_state: str | None = None
        self._numerics = SamplingNumericalAccumulator()
        self._actual_vjp: int | None = None
        self._sampling_elapsed_seconds = 0.0
        self._total_forward_calls = 0
        self._total_vjp_calls = 0
        velocity_range = (
            math.exp(float(config.logvar_min)),
            math.exp(float(config.logvar_max)),
        )
        self.velocity_range = velocity_range
        self.endpoint_scale = (1.0 - self.capture_t) ** 2
        self.endpoint_range = tuple(self.endpoint_scale * value for value in velocity_range)
        self.diffusion_range = tuple(sampling.lambda_at(self.capture_t) * value for value in velocity_range)
        self.noise_covariance_range = tuple(sampling.terminal_time / sampling.steps * value for value in self.diffusion_range)

    def write_chunk(self, labels: torch.Tensor, sample_result) -> None:
        labels_cpu = labels.detach().cpu().long()
        samples_cpu = sample_result.samples.detach().cpu()
        if int(labels_cpu.numel()) != int(samples_cpu.shape[0]):
            raise ValueError("sample visualization labels and samples must have matching counts")
        self._set_consistent("_actual_nfe", int(sample_result.actual_nfe), "actual_nfe")
        if sample_result.sampling != self.sampling:
            raise ValueError("sample visualization observed a different sampling protocol")
        self._set_consistent("_actual_vjp", int(sample_result.actual_vjp), "actual_vjp")
        self._numerics.update(sample_result.numerical_summary, int(labels_cpu.numel()))
        self._sampling_elapsed_seconds += float(sample_result.elapsed_seconds)
        self._total_forward_calls += int(sample_result.actual_nfe)
        self._total_vjp_calls += int(sample_result.actual_vjp)
        if self.has_variance:
            if not sample_result.variance_available:
                raise ValueError("PFM sample visualization requires velocity and endpoint variance")
            if sample_result.velocity_var is None or sample_result.endpoint_var is None:
                raise ValueError("PFM sample visualization variance tensors are missing")
            observed_t = float(sample_result.uncertainty_source_t)
            if not math.isclose(observed_t, self.capture_t, rel_tol=0, abs_tol=1e-12):
                raise ValueError("PFM sample visualization did not capture the exact requested time")
            if sample_result.uncertainty_source_state != "pre_update":
                raise ValueError("PFM sample visualization variance must come from pre-update state")
            self._set_consistent(
                "_capture_step",
                int(sample_result.uncertainty_source_step),
                "uncertainty_source_step",
            )
            self._set_consistent(
                "_capture_observed_t", observed_t, "uncertainty_source_t"
            )
            self._set_consistent(
                "_capture_state",
                str(sample_result.uncertainty_source_state),
                "uncertainty_source_state",
            )
            self._velocity.append(sample_result.velocity_var.detach().cpu())
            self._endpoint.append(sample_result.endpoint_var.detach().cpu())
            if self.has_diffusion:
                if sample_result.diffusion_rate is None or sample_result.noise_covariance is None:
                    raise ValueError("SDE sample visualization requires diffusion capture maps")
                self._diffusion.append(sample_result.diffusion_rate.detach().cpu())
                self._noise_covariance.append(sample_result.noise_covariance.detach().cpu())
        elif sample_result.variance_available:
            raise ValueError("baseline sample visualization must not expose variance maps")
        self._labels.append(labels_cpu)
        self._samples.append(samples_cpu)
        count = int(labels_cpu.numel())
        self._buffer_count += count
        self._seen += count
        while self._buffer_count >= self.items_per_page:
            self._render_next_page(self.items_per_page)

    def finalize(self) -> dict[str, object]:
        if self._seen != self.expected_count:
            raise ValueError(
                f"sample visualization received {self._seen} items, expected {self.expected_count}"
            )
        if self._buffer_count:
            self._render_next_page(self._buffer_count)
        expected_pages = math.ceil(self.class_count / self.classes_per_page)
        if self._page_index != expected_pages:
            raise ValueError("sample visualization page count does not cover all classes")
        actual_nfe = self._actual_nfe
        if actual_nfe is None:
            raise ValueError("sample visualization cannot finalize an empty stream")
        pages: list[dict[str, object]] = []
        for page_index in range(expected_pages):
            class_min = self.class_start + page_index * self.classes_per_page
            class_max = min(
                self.class_start + self.class_count - 1,
                class_min + self.classes_per_page - 1,
            )
            page_id = f"classes_{class_min:04d}_{class_max:04d}"
            entry: dict[str, object] = {
                "class_min": class_min,
                "class_max": class_max,
                "samples": f"presentation/samples/{page_id}.png",
            }
            if self.has_variance:
                entry["velocity_variance"] = (
                    f"presentation/velocity_variance/{page_id}.png"
                )
                entry["endpoint_variance"] = (
                    f"presentation/endpoint_variance/{page_id}.png"
                )
                entry["velocity_variance_position"] = (
                    f"presentation/velocity_variance_position/{page_id}.png"
                )
                entry["endpoint_variance_position"] = (
                    f"presentation/endpoint_variance_position/{page_id}.png"
                )
            if self.has_diffusion:
                entry["diffusion_rate"] = f"presentation/diffusion_rate/{page_id}.png"
                entry["brownian_step_covariance"] = f"presentation/brownian_step_covariance/{page_id}.png"
            pages.append(entry)
        presentation: dict[str, object] = {"pages": pages}
        if self.has_variance:
            presentation["variance_scales"] = {
                "velocity": list(self.velocity_range),
                "endpoint": list(self.endpoint_range),
                "endpoint_scale_at_t_vis": self.endpoint_scale,
            }
            presentation["variance_position_scale"] = {
                "definition": "(variance - lower_bound) / (upper_bound - lower_bound)",
                "display_transform": "log10(max(position, 1e-8))",
                "display_range": [-8.0, -2.0],
            }
        if self.has_diffusion:
            presentation["diffusion_scales"] = {
                "diffusion_rate": list(self.diffusion_range),
                "brownian_step_covariance": list(self.noise_covariance_range),
                "definition": "lambda(t) * velocity_variance and h * lambda(t) * velocity_variance",
            }
        return {
            "profile": "sample_visualization",
            "sampling": {
                "protocol": self.sampling.to_dict(),
                "actual_nfe": actual_nfe,
                "uncertainty_capture_requested_t": self.capture_t,
                "uncertainty_capture_observed_t": self._capture_observed_t,
                "uncertainty_source_step": self._capture_step,
                "uncertainty_source_state": self._capture_state,
                "actual_vjp": self._actual_vjp,
                "total_forward_calls": self._total_forward_calls,
                "total_vjp_calls": self._total_vjp_calls,
                "elapsed_seconds": self._sampling_elapsed_seconds,
                "numerical_summary": self._numerics.finalize(),
            },
            "population": self.expected_count,
            "num_per_class": self.num_per_class,
            "presentation": presentation,
            "aggregate": {
                "population": self.expected_count,
                "actual_nfe": actual_nfe,
                "actual_vjp": self._actual_vjp,
                "variance_maps": self.has_variance,
                "page_count": expected_pages,
            },
        }

    def _render_next_page(self, count: int) -> None:
        labels, self._labels = _take_prefix(self._labels, count)
        samples, self._samples = _take_prefix(self._samples, count)
        velocity = endpoint = diffusion = noise_covariance = None
        if self.has_variance:
            velocity, self._velocity = _take_prefix(self._velocity, count)
            endpoint, self._endpoint = _take_prefix(self._endpoint, count)
        if self.has_diffusion:
            diffusion, self._diffusion = _take_prefix(self._diffusion, count)
            noise_covariance, self._noise_covariance = _take_prefix(self._noise_covariance, count)
        self._buffer_count -= count
        class_min = self.class_start + self._page_index * self.classes_per_page
        class_max = min(
            self.class_start + self.class_count - 1,
            class_min + self.classes_per_page - 1,
        )
        ordered_labels = list(range(class_min, class_max + 1))
        page_id = f"classes_{class_min:04d}_{class_max:04d}"
        save_sample_grid_for_labels(
            samples,
            labels,
            self.presentation_root / "samples" / f"{page_id}.png",
            ordered_labels,
            self.num_per_class,
        )
        if self.has_variance:
            assert velocity is not None and endpoint is not None
            save_raw_variance_grid_for_labels(
                velocity,
                labels,
                self.presentation_root / "velocity_variance" / f"{page_id}.png",
                ordered_labels,
                self.num_per_class,
                value_range=self.velocity_range,
                colorbar_label="Posterior velocity variance",
            )
            save_raw_variance_grid_for_labels(
                endpoint,
                labels,
                self.presentation_root / "endpoint_variance" / f"{page_id}.png",
                ordered_labels,
                self.num_per_class,
                value_range=self.endpoint_range,
                colorbar_label="Posterior endpoint variance",
            )
            save_variance_position_grid_for_labels(
                velocity,
                labels,
                self.presentation_root
                / "velocity_variance_position"
                / f"{page_id}.png",
                ordered_labels,
                self.num_per_class,
                value_range=self.velocity_range,
            )
            save_variance_position_grid_for_labels(
                endpoint,
                labels,
                self.presentation_root
                / "endpoint_variance_position"
                / f"{page_id}.png",
                ordered_labels,
                self.num_per_class,
                value_range=self.endpoint_range,
            )
        if self.has_diffusion:
            for values, name, value_range, colorbar_label in (
                (diffusion, "diffusion_rate", self.diffusion_range, "Diffusion rate"),
                (noise_covariance, "brownian_step_covariance", self.noise_covariance_range, "Brownian step variance"),
            ):
                assert values is not None
                save_raw_variance_grid_for_labels(
                    values, labels, self.presentation_root / name / f"{page_id}.png",
                    ordered_labels, self.num_per_class, value_range=value_range,
                    colorbar_label=colorbar_label,
                )
        self._page_index += 1

    def _set_consistent(self, attribute: str, value: object, name: str) -> None:
        current = getattr(self, attribute)
        if current is None:
            setattr(self, attribute, value)
        elif current != value:
            raise ValueError(f"sample visualization chunks have inconsistent {name}")


def _take_prefix(
    chunks: list[torch.Tensor],
    count: int,
) -> tuple[torch.Tensor, list[torch.Tensor]]:
    joined = torch.cat(chunks, dim=0)
    if int(joined.shape[0]) < count:
        raise ValueError("sample visualization buffer is shorter than requested page")
    prefix = joined[:count]
    remainder = joined[count:]
    return prefix, ([remainder] if int(remainder.shape[0]) else [])
