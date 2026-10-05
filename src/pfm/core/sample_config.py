from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
from pathlib import Path
import math

from pfm.core.yaml_config import load_yaml_mapping
from pfm.core.sampling_config import SamplingConfig


@dataclass(frozen=True)
class SampleVisualizationConfig:
    profile: str
    sampling: SamplingConfig
    batch_size: int
    class_start: int
    class_count: int
    num_per_class: int
    seed: int
    uncertainty_capture_t: float

    def __post_init__(self) -> None:
        if self.profile != "sample_visualization":
            raise ValueError("sample profile must equal sample_visualization")
        for name in ("batch_size", "class_count", "num_per_class"):
            value = getattr(self, name)
            if isinstance(value, bool) or not isinstance(value, int) or value < 1:
                raise ValueError(f"{name} must be a positive integer")
        if isinstance(self.class_start, bool) or not isinstance(self.class_start, int) or self.class_start < 0:
            raise ValueError("class_start must be a non-negative integer")
        if type(self.seed) is not int or not 0 <= self.seed < 2**32:
            raise ValueError("seed must be an integer in [0, 2**32)")
        value = self.uncertainty_capture_t
        if isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(value) or not 0 <= value < 1:
            raise ValueError("uncertainty_capture_t must be finite and in [0, 1)")
        if not isinstance(self.sampling, SamplingConfig):
            raise TypeError("sampling must be a SamplingConfig")
        h = self.sampling.terminal_time / self.sampling.steps
        step = round(self.uncertainty_capture_t / h)
        if step >= self.sampling.steps or abs(step * h - self.uncertainty_capture_t) > 1e-12:
            raise ValueError("uncertainty_capture_t must lie on the pre-update sampling grid")



def load_sample_visualization_config(
    path: str | Path,
) -> SampleVisualizationConfig:
    return sample_visualization_config_from_mapping(load_yaml_mapping(Path(path)))


def sample_visualization_config_from_mapping(raw: Mapping) -> SampleVisualizationConfig:
    required = {
        "profile",
        "sampling",
        "batch_size",
        "class_start",
        "class_count",
        "num_per_class",
        "seed",
        "uncertainty_capture_t",
    }
    if set(raw) != required:
        missing = sorted(required - set(raw))
        unknown = sorted(set(raw) - required)
        raise ValueError(
            f"sample_visualization config has missing={missing} unknown={unknown}"
        )
    data = dict(raw)
    data["sampling"] = SamplingConfig.from_mapping(data["sampling"])
    return SampleVisualizationConfig(**data)
