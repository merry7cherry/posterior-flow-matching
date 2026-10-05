"""Bounded numerical workload; checkpoint metadata owns the model and data."""
from dataclasses import dataclass, fields
from pathlib import Path
import math

from pfm.core.inference_config import InferenceExecutionConfig
from pfm.core.sampling_config import SamplingConfig
from pfm.core.yaml_config import load_yaml_mapping


@dataclass(frozen=True)
class SDEAcceptanceConfig:
    batch_size: int
    seed: int
    diffusion_scale: float
    terminal_time: float
    probe_repeats: int
    reference_times: tuple[float, ...]
    probe_counts: tuple[int, ...]
    precisions: tuple[str, ...]
    compile: bool
    allow_tf32: bool
    cudnn_benchmark: bool
    sampler_steps: tuple[int, ...]
    sampler_probes: tuple[int, ...]
    cutoff_times: tuple[float, ...]

    def __post_init__(self):
        for name, maximum in (("batch_size", 16), ("probe_repeats", 4)):
            value = getattr(self, name)
            if type(value) is not int or not 1 <= value <= maximum:
                raise ValueError(f"{name} must lie in [1,{maximum}]")
        if type(self.seed) is not int or not 0 <= self.seed < 2**32:
            raise ValueError("seed must lie in [0,2**32)")
        for name, maximum, length in (("probe_counts", 16, 5), ("sampler_probes", 16, 4),
                                      ("sampler_steps", 1000, 4)):
            values = getattr(self, name)
            if (not isinstance(values, tuple) or not 1 <= len(values) <= length
                    or any(type(v) is not int or not 1 <= v <= maximum for v in values)
                    or tuple(sorted(set(values))) != values):
                raise ValueError(f"{name} must be bounded, positive, unique and increasing")
        if any(max(self.sampler_steps) % k for k in self.sampler_steps):
            raise ValueError("sampler_steps must divide the finest Brownian grid")
        for name, length in (("reference_times", 9), ("cutoff_times", 4)):
            values = getattr(self, name)
            if (not isinstance(values, tuple) or not 1 <= len(values) <= length
                    or any(type(v) not in (float, int) or not math.isfinite(v) or not 0 < v < 1 for v in values)
                    or tuple(sorted(set(values))) != values):
                raise ValueError(f"{name} must be unique increasing times strictly inside (0,1)")
        if self.terminal_time not in self.cutoff_times or max(self.reference_times) >= min(self.cutoff_times):
            raise ValueError("terminal_time must be a cutoff and reference_times must precede all cutoffs")
        if (not isinstance(self.precisions, tuple) or not 1 <= len(self.precisions) <= 2
                or len(set(self.precisions)) != len(self.precisions)):
            raise ValueError("precisions must select unique fp32/bf16 execution paths")
        if type(self.diffusion_scale) not in (float, int) or not math.isfinite(self.diffusion_scale) or self.diffusion_scale <= 0:
            raise ValueError("diffusion_scale must be positive and finite")
        for precision in self.precisions:
            InferenceExecutionConfig(precision, self.compile, self.allow_tf32, self.cudnn_benchmark)
            for cutoff in self.cutoff_times:
                for steps in self.sampler_steps:
                    SamplingConfig(solver="sde", steps=steps, terminal_time=cutoff,
                                   terminal_readout="endpoint_mean", diffusion_scale=self.diffusion_scale,
                                   precision=precision, compile=self.compile,
                                   allow_tf32=self.allow_tf32, cudnn_benchmark=self.cudnn_benchmark)


def load_sde_acceptance_config(path: str | Path) -> SDEAcceptanceConfig:
    return sde_acceptance_config_from_mapping(load_yaml_mapping(Path(path)))


def sde_acceptance_config_from_mapping(raw: dict) -> SDEAcceptanceConfig:
    raw = dict(raw)
    required = {field.name for field in fields(SDEAcceptanceConfig)}
    if set(raw) != required:
        raise ValueError(f"SDE acceptance fields missing={sorted(required-set(raw))} unknown={sorted(set(raw)-required)}")
    for name in ("reference_times", "probe_counts", "precisions", "sampler_steps", "sampler_probes", "cutoff_times"):
        if not isinstance(raw[name], (list, tuple)):
            raise ValueError(f"{name} must be a sequence")
        raw[name] = tuple(raw[name])
    return SDEAcceptanceConfig(**raw)
