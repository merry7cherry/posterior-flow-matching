from __future__ import annotations

from collections.abc import Mapping
from dataclasses import asdict, dataclass, fields
import math
import struct

from pfm.core.inference_config import InferenceExecutionConfig


@dataclass(frozen=True)
class SamplingConfig:
    """One inference protocol shared by generation and visualization."""

    solver: str = "ode"
    steps: int = 100
    terminal_time: float = 1.0
    terminal_readout: str = "state"
    diffusion_scale: float = 0.0
    diffusion_schedule: str = "constant"
    derivative_probes: int = 1
    precision: str = "fp32"
    compile: bool = False
    allow_tf32: bool = False
    cudnn_benchmark: bool = True

    def __post_init__(self) -> None:
        if self.solver not in {"ode", "sde"}:
            raise ValueError("sampling solver must be ode or sde")
        for name in ("steps", "derivative_probes"):
            value = getattr(self, name)
            if type(value) is not int or value < 1:
                raise ValueError(f"sampling {name} must be a positive integer")
        for name in ("terminal_time", "diffusion_scale"):
            value = getattr(self, name)
            if isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(value):
                raise ValueError(f"sampling {name} must be finite numeric")
        if not 0 < self.terminal_time <= 1:
            raise ValueError("sampling terminal_time must lie in (0, 1]")
        if self.diffusion_scale < 0:
            raise ValueError("sampling diffusion_scale must be nonnegative")
        if self.solver == "ode" and self.diffusion_scale != 0:
            raise ValueError("ODE requires diffusion_scale=0")
        if self.solver == "sde" and self.diffusion_scale > 0 and self.terminal_time >= 1:
            raise ValueError("SDE with diffusion requires terminal_time < 1")
        if self.diffusion_schedule not in {"constant", "linear_decay"}:
            raise ValueError("diffusion_schedule must be constant or linear_decay")
        if self.terminal_readout not in {"state", "endpoint_mean"}:
            raise ValueError("terminal_readout must be state or endpoint_mean")
        if self.terminal_readout == "endpoint_mean" and self.terminal_time >= 1:
            raise ValueError("endpoint_mean readout requires terminal_time < 1")
        fp32 = lambda value: struct.unpack("f", struct.pack("f", value))[0]
        terminal = fp32(self.terminal_time)
        last_evaluation = fp32((self.steps - 1) * self.terminal_time / self.steps)
        if terminal <= 0 or fp32(self.terminal_time / self.steps) <= 0 or last_evaluation >= terminal:
            raise ValueError("sampling grid endpoints must remain distinct in FP32")
        if last_evaluation >= 1 or (self.terminal_time < 1 and terminal >= 1):
            raise ValueError("sampling cutoff and coefficient times must remain below 1 in FP32")
        _ = self.execution  # validate the shared execution choices

    @property
    def execution(self) -> InferenceExecutionConfig:
        return InferenceExecutionConfig(self.precision, self.compile, self.allow_tf32, self.cudnn_benchmark)

    def lambda_at(self, time: float) -> float:
        return float(self.diffusion_scale) * (1.0 - time if self.diffusion_schedule == "linear_decay" else 1.0)

    def to_dict(self) -> dict[str, object]:
        return asdict(self)

    @classmethod
    def from_mapping(cls, raw: object) -> SamplingConfig:
        if not isinstance(raw, Mapping):
            raise TypeError("sampling must be a mapping")
        required = {field.name for field in fields(cls)}
        if set(raw) != required:
            raise ValueError(f"sampling fields missing={sorted(required - set(raw))} unknown={sorted(set(raw) - required)}")
        return cls(**dict(raw))
