"""Execution choices shared by checkpoint consumers, separate from model identity."""
from dataclasses import dataclass, replace

from pfm.core.config import ExperimentConfig


@dataclass(frozen=True)
class InferenceExecutionConfig:
    precision: str = "fp32"
    compile: bool = False
    allow_tf32: bool = False
    cudnn_benchmark: bool = True

    def __post_init__(self) -> None:
        if self.precision not in {"fp32", "bf16"}:
            raise ValueError("inference precision must be fp32 or bf16")
        for name in ("compile", "allow_tf32", "cudnn_benchmark"):
            if type(getattr(self, name)) is not bool:
                raise ValueError(f"inference {name} must be boolean")

    def apply(self, config: ExperimentConfig) -> ExperimentConfig:
        if self.compile and config.architecture not in {"jit_transformer", "edm_ddpmpp"}:
            raise ValueError("inference compile is supported only for JiT and EDM backbones")
        return replace(config, mixed_precision="none" if self.precision == "fp32" else "bf16",
                       jit_compile=self.compile,
                       edm_compile=self.compile if config.architecture == "edm_ddpmpp" else config.edm_compile,
                       allow_tf32=self.allow_tf32,
                       cudnn_benchmark=self.cudnn_benchmark)
