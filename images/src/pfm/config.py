"""Image model/training settings and the ODE/SDE protocol."""
from dataclasses import asdict, dataclass
from pathlib import Path
import yaml


@dataclass
class ImageConfig:
    seed: int = 42
    dataset: str = "mnist"
    architecture: str = "conditional_unet"
    num_classes: int = 10
    image_channels: int = 1
    image_size: int = 28
    time_embed_dim: int = 128
    time_embed_scale: float = 1000.0
    time_embed_max_period: float = 10000.0
    time_proj_dim: int = 128
    class_embed_dim: int = 128
    class_proj_dim: int = 128
    unet_base_channels: int = 32
    unet_channel_mult: tuple[int, ...] = (1, 2, 4)
    unet_num_res_blocks: int = 2
    unet_attention_resolutions: tuple[int, ...] = ()
    unet_num_heads: int = 1
    unet_dropout: float = 0.05
    unet_norm_groups: int = 8
    edm_model_channels: int = 128
    edm_channel_mult: tuple[int, ...] = (1, 2, 2, 2)
    edm_channel_mult_emb: int = 4
    edm_num_blocks: int = 4
    edm_attn_resolutions: tuple[int, ...] = (16,)
    edm_dropout: float = 0.10
    edm_label_dropout: float = 0.0
    edm_norm_groups: int = 32
    jit_model: str = "JiT-B/16"
    jit_attn_dropout: float = 0.0
    jit_proj_dropout: float = 0.0
    mixed_precision: str = "bf16"
    allow_tf32: bool = False
    cudnn_benchmark: bool = True
    drop_last: bool = True
    batch_size: int = 256
    num_workers: int = 2
    epochs: int = 200
    limit_train_batches: int | None = None
    lr: float = 2e-4
    lr_reference_batch_size: int | None = None
    weight_decay: float = 1e-4
    adamw_beta1: float = 0.9
    adamw_beta2: float = 0.999
    warmup_epochs: int = 0
    max_grad_norm: float = 1.0
    ema_decay: float = 0.999
    eps_t: float = 1e-3
    time_sampling: str = "uniform"
    logit_normal_p_mean: float = -0.8
    logit_normal_p_std: float = 0.8
    logvar_min: float = -1.0
    logvar_max: float = 1.0
    logvar_init: float = 0.0
    noise_scale: float = 1.0
    checkpoint_every_epochs: int = 50
    mean_parameterization: str = "velocity"
    compile_model: bool = False

    @property
    def image_shape(self):
        return (self.image_channels, self.image_size, self.image_size)

    def __post_init__(self):
        if self.mean_parameterization not in {"velocity", "endpoint"}:
            raise ValueError("mean_parameterization must be velocity or endpoint")
        if not self.logvar_min < self.logvar_max:
            raise ValueError("logvar_min must be below logvar_max")
        if not 0 < self.eps_t < 0.5:
            raise ValueError("eps_t must be in (0, 0.5)")
        if self.batch_size < 1 or self.epochs < 1 or self.checkpoint_every_epochs < 1:
            raise ValueError("batch size, epochs and checkpoint interval must be positive")
        if self.limit_train_batches is not None and self.limit_train_batches < 1:
            raise ValueError("limit_train_batches must be positive or null")
        if self.mixed_precision not in {"none", "bf16"}:
            raise ValueError("mixed_precision must be none or bf16")


@dataclass
class SamplingConfig:
    solver: str = "ode"
    steps: int = 50
    terminal_time: float = 0.99
    terminal_readout: str = "endpoint_mean"
    diffusion_scale: float = 0.0
    diffusion_schedule: str = "linear_decay"
    derivative_probes: int = 2
    precision: str = "fp32"

    def __post_init__(self):
        if self.solver not in {"ode", "sde"} or self.steps < 1:
            raise ValueError("use ode/sde with a positive number of steps")
        if not 0 < self.terminal_time <= 1:
            raise ValueError("terminal_time must be in (0, 1]")
        if self.terminal_readout not in {"state", "endpoint_mean"}:
            raise ValueError("terminal_readout must be state or endpoint_mean")
        if self.diffusion_scale < 0 or self.derivative_probes < 1:
            raise ValueError("diffusion scale must be nonnegative and probes positive")
        if self.solver == "ode" and self.diffusion_scale != 0:
            raise ValueError("ODE requires zero diffusion_scale")
        if (self.terminal_readout == "endpoint_mean" or self.diffusion_scale > 0) and self.terminal_time >= 1:
            raise ValueError("endpoint readout and positive SDE diffusion require terminal_time < 1")
        if self.diffusion_schedule not in {"constant", "linear_decay"}:
            raise ValueError("diffusion_schedule must be constant or linear_decay")
        if self.precision not in {"fp32", "bf16"}:
            raise ValueError("precision must be fp32 or bf16")

    def lambda_at(self, t):
        return self.diffusion_scale * (1 - t if self.diffusion_schedule == "linear_decay" else 1)


def load_config(path, overrides=(), cls=ImageConfig):
    values = yaml.safe_load(Path(path).read_text()) or {}
    for item in overrides:
        key, value = item.split("=", 1)
        values[key] = yaml.safe_load(value)
    return cls(**values)


def save_config(config, path):
    Path(path).write_text(yaml.safe_dump(asdict(config), sort_keys=False))
