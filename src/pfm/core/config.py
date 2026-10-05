from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass, fields
import hashlib
import json
import math
import struct
from pathlib import Path
from typing import Any, Optional

from pfm.core import variant_metadata
from pfm.core.identities import (
    validate_architecture_identity,
    validate_dataset_attributes,
    validate_dataset_identity,
)
from pfm.core.yaml_config import load_yaml_mapping, merge_config


MIXED_PRECISION_MODES = ("none", "bf16")

COMMON_REQUIRED_KEYS = frozenset(
    {
        "seed", "dataset", "architecture", "num_classes", "image_channels", "image_size",
        "mixed_precision", "allow_tf32",
        "cudnn_benchmark", "drop_last", "batch_size", "num_workers", "epochs", "lr",
        "base_lr", "lr_reference_batch_size", "scale_lr_by_effective_batch", "weight_decay",
        "adamw_beta1", "adamw_beta2", "warmup_epochs", "lr_schedule", "max_grad_norm",
        "ema_decay", "eps_t", "time_sampling", "logit_normal_p_mean", "logit_normal_p_std",
        "logvar_min", "logvar_max", "logvar_init", "self_conditioning_mode",
        "noise_scale", "checkpoint_every_epochs", "limit_train_batches",
    }
)
ARCHITECTURE_REQUIRED_KEYS = {
    "conditional_unet": frozenset(
        {
            "time_embed_dim", "time_embed_scale", "time_embed_max_period", "time_proj_dim",
            "class_embed_dim", "class_proj_dim", "unet_base_channels", "unet_channel_mult",
            "unet_num_res_blocks", "unet_attention_resolutions", "unet_num_heads",
            "unet_dropout", "unet_norm_groups",
        }
    ),
    "edm_ddpmpp": frozenset(
        {
            "time_embed_scale", "time_embed_max_period", "edm_model_channels",
            "edm_channel_mult", "edm_channel_mult_emb", "edm_num_blocks", "edm_compile",
            "edm_attn_resolutions", "edm_dropout", "edm_label_dropout", "edm_norm_groups",
        }
    ),
    "jit_transformer": frozenset(
        {
            "jit_model", "jit_attn_dropout", "jit_proj_dropout", "jit_compile",
            "jit_attention_implementation",
        }
    ),
}

_BOOLEAN_FIELDS = frozenset(
    {
        "jit_compile", "edm_compile", "allow_tf32", "cudnn_benchmark", "drop_last",
        "scale_lr_by_effective_batch",
    }
)
_INTEGER_FIELDS = frozenset(
    {
        "seed", "num_classes", "image_channels", "image_size", "time_embed_dim", "time_proj_dim",
        "class_embed_dim", "class_proj_dim", "unet_base_channels", "unet_num_res_blocks",
        "unet_num_heads", "unet_norm_groups", "edm_model_channels", "edm_channel_mult_emb",
        "edm_num_blocks", "edm_norm_groups", "batch_size", "num_workers", "epochs",
        "lr_reference_batch_size", "warmup_epochs", "checkpoint_every_epochs",
    }
)
_FLOAT_FIELDS = frozenset(
    {
        "time_embed_scale", "time_embed_max_period", "unet_dropout", "edm_dropout",
        "edm_label_dropout", "jit_attn_dropout", "jit_proj_dropout", "weight_decay", "ema_decay",
        "adamw_beta1", "adamw_beta2", "max_grad_norm",
        "eps_t", "logit_normal_p_mean", "logit_normal_p_std", "logvar_min", "logvar_max",
        "logvar_init", "noise_scale",
    }
)
_OPTIONAL_FLOAT_FIELDS = frozenset({"lr", "base_lr"})
_INTEGER_SEQUENCE_FIELDS = frozenset(
    {"unet_channel_mult", "unet_attention_resolutions", "edm_channel_mult", "edm_attn_resolutions"}
)


@dataclass
class ExperimentConfig:
    """Typed internal values; authored runs must supply all active YAML fields.

    Constructor conveniences also cover inactive architecture fields. They are
    not merged into production input: experiment_config_from_mapping validates
    completeness before constructing this object.
    """

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
    edm_compile: bool = False
    edm_model_channels: int = 128
    edm_channel_mult: tuple[int, ...] = (1, 2, 2, 2)
    edm_channel_mult_emb: int = 4
    edm_num_blocks: int = 4
    edm_attn_resolutions: tuple[int, ...] = (16,)
    edm_dropout: float = 0.10
    edm_label_dropout: float = 0.0
    edm_norm_groups: int = 32
    jit_model: str = "JiT-L/16"
    jit_attn_dropout: float = 0.0
    jit_proj_dropout: float = 0.0
    jit_compile: bool = True
    # Fixed checkpoint identity; JiT has one supported attention implementation.
    jit_attention_implementation: str = "upstream_fp32_qk"
    mixed_precision: str = "bf16"
    allow_tf32: bool = False
    cudnn_benchmark: bool = True
    drop_last: bool = True
    batch_size: int = 256
    num_workers: int = 2
    epochs: int = 200
    limit_train_batches: Optional[int] = None
    lr: Optional[float] = 2e-4
    base_lr: Optional[float] = None
    lr_reference_batch_size: int = 256
    scale_lr_by_effective_batch: bool = False
    weight_decay: float = 1e-4
    adamw_beta1: float = 0.9
    adamw_beta2: float = 0.999
    warmup_epochs: int = 0
    lr_schedule: str = "constant"
    max_grad_norm: float = 1.0
    ema_decay: float = 0.999
    eps_t: float = 1e-3
    time_sampling: str = "uniform"
    logit_normal_p_mean: float = -0.8
    logit_normal_p_std: float = 0.8
    logvar_min: float = -1.0
    logvar_max: float = 1.0
    logvar_init: float = 0.0
    # Fixed identity retained so existing baseline checkpoints remain loadable.
    self_conditioning_mode: str = "none"
    noise_scale: float = 1.0
    checkpoint_every_epochs: int = 50

    @property
    def image_shape(self) -> tuple[int, int, int]:
        return (self.image_channels, self.image_size, self.image_size)


def validate_mixed_precision(value: object) -> str:
    if not isinstance(value, str) or value not in MIXED_PRECISION_MODES:
        raise ValueError("mixed_precision must be one of {'none', 'bf16'}")
    return value


def experiment_config_from_mapping(raw: Mapping[str, Any]) -> ExperimentConfig:
    return ExperimentConfig(**_coerce_experiment_config_values(raw))


def checkpoint_config_from_mapping(raw: Mapping[str, Any]) -> ExperimentConfig:
    """Decode the one prior EDM payload whose execution was necessarily eager.

    This is checkpoint-only: authored YAML must still supply edm_compile.
    No tensor layout or recorded provenance is rewritten.
    """
    data = dict(raw)
    if data.get("architecture") == "edm_ddpmpp" and "edm_compile" not in data:
        data["edm_compile"] = False
    return experiment_config_from_mapping(data)


def experiment_config_to_mapping(config: ExperimentConfig) -> dict[str, Any]:
    if not isinstance(config, ExperimentConfig):
        raise TypeError("config must be an ExperimentConfig")
    architecture = validate_architecture_identity(config.architecture)
    keys = COMMON_REQUIRED_KEYS | ARCHITECTURE_REQUIRED_KEYS[architecture]
    return _coerce_experiment_config_values(
        {field_name: getattr(config, field_name) for field_name in keys}
    )


def shared_training_protocol_sha256(config: ExperimentConfig, *, legacy_edm: bool = False) -> str:
    """Digest the common training protocol, excluding approved method dimensions."""
    payload = experiment_config_to_mapping(config)
    if legacy_edm:
        if config.architecture != "edm_ddpmpp" or config.edm_compile:
            raise ValueError("legacy EDM protocol requires eager EDM execution")
        payload.pop("edm_compile")
    # Preserve the existing baseline protocol digest after retiring the SC dimension.
    for field_name in ("self_conditioning_mode", "epochs", "checkpoint_every_epochs"):
        payload.pop(field_name)
    return hashlib.sha256(
        json.dumps(
            payload,
            sort_keys=True,
            separators=(",", ":"),
            allow_nan=False,
        ).encode("utf-8")
    ).hexdigest()


def _coerce_experiment_config_values(raw: Mapping[str, Any]) -> dict[str, Any]:
    data = dict(raw)
    valid = {field.name for field in fields(ExperimentConfig)}
    unknown = sorted(set(data) - valid)
    if unknown:
        raise ValueError(f"Unknown config keys: {unknown}")
    missing_common = sorted(COMMON_REQUIRED_KEYS - set(data))
    if missing_common:
        raise ValueError(f"Missing required authored config keys: {missing_common}")
    data["dataset"] = validate_dataset_identity(data["dataset"])
    data["architecture"] = validate_architecture_identity(data["architecture"])
    allowed = COMMON_REQUIRED_KEYS | ARCHITECTURE_REQUIRED_KEYS[data["architecture"]]
    inactive = sorted(set(data) - allowed)
    if inactive:
        raise ValueError(
            f"Config fields are not valid for architecture {data['architecture']!r}: {inactive}"
        )
    missing_architecture = sorted(ARCHITECTURE_REQUIRED_KEYS[data["architecture"]] - set(data))
    if missing_architecture:
        raise ValueError(f"Missing required authored config keys: {missing_architecture}")
    for field_name in _BOOLEAN_FIELDS:
        if field_name in data:
            data[field_name] = _require_bool(data[field_name], field_name)
    for field_name in _INTEGER_FIELDS:
        if field_name in data:
            data[field_name] = _require_int(data[field_name], field_name)
    for field_name in _FLOAT_FIELDS:
        if field_name in data:
            data[field_name] = _require_number(data[field_name], field_name)
    for field_name in _OPTIONAL_FLOAT_FIELDS:
        if field_name in data:
            data[field_name] = _require_optional_number(data[field_name], field_name)
    if data["limit_train_batches"] is not None:
        data["limit_train_batches"] = _require_int(data["limit_train_batches"], "limit_train_batches")
        if data["limit_train_batches"] < 1:
            raise ValueError("limit_train_batches must be null or a positive integer")
    for field_name in _INTEGER_SEQUENCE_FIELDS:
        if field_name in data:
            data[field_name] = _require_integer_sequence(data[field_name], field_name)
    if "mixed_precision" in data:
        data["mixed_precision"] = validate_mixed_precision(data["mixed_precision"])
    for field_name in (
        "jit_model", "jit_attention_implementation", "time_sampling", "lr_schedule",
        "self_conditioning_mode",
    ):
        if field_name in data and not isinstance(data[field_name], str):
            raise ValueError(f"{field_name} must be a string")
    if data["time_sampling"] not in {"uniform", "logit_normal"}:
        raise ValueError("time_sampling must be exactly 'uniform' or 'logit_normal'")
    if data.get("jit_attention_implementation") not in {
        None,
        "upstream_fp32_qk",
    }:
        raise ValueError(
            "jit_attention_implementation must be 'upstream_fp32_qk'; SDPA is retired"
        )
    if data["lr_schedule"] != "constant":
        raise ValueError("lr_schedule must be exactly 'constant'")
    _validate_config_data(data)
    validate_dataset_attributes(data["dataset"], data)
    return data


def load_config(path: str | Path, overrides: Optional[Mapping[str, Any]] = None) -> ExperimentConfig:
    config_path = Path(path)
    raw = merge_config(load_yaml_mapping(config_path), overrides)
    config = experiment_config_from_mapping(raw)
    return config



def _require_bool(value: Any, field_name: str) -> bool:
    if isinstance(value, bool):
        return value
    raise ValueError(f"{field_name} must be a boolean")


def _require_int(value: Any, field_name: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int):
        raise ValueError(f"{field_name} must be an integer")
    return value


def _require_number(value: Any, field_name: str) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ValueError(f"{field_name} must be a finite number")
    resolved = float(value)
    if not math.isfinite(resolved):
        raise ValueError(f"{field_name} must be finite")
    return resolved


def _require_optional_number(value: Any, field_name: str) -> Optional[float]:
    if value is None:
        return None
    return _require_number(value, field_name)


def _require_integer_sequence(value: Any, field_name: str) -> tuple[int, ...]:
    if not isinstance(value, (list, tuple)):
        raise ValueError(f"{field_name} must be a sequence of integers")
    if any(isinstance(item, bool) or not isinstance(item, int) for item in value):
        raise ValueError(f"{field_name} must be a sequence of integers")
    return tuple(value)


def _validate_config_data(data: Mapping[str, Any]) -> None:
    positive_integer_fields = {
        "num_classes", "image_channels", "image_size", "batch_size", "epochs",
        "lr_reference_batch_size",
    }
    architecture = str(data["architecture"])
    if architecture == "conditional_unet":
        positive_integer_fields.update(
            {
                "time_embed_dim", "time_proj_dim", "class_embed_dim", "class_proj_dim",
                "unet_base_channels", "unet_num_res_blocks", "unet_num_heads",
                "unet_norm_groups",
            }
        )
    elif architecture == "edm_ddpmpp":
        positive_integer_fields.update(
            {
                "edm_model_channels", "edm_channel_mult_emb", "edm_num_blocks",
                "edm_norm_groups",
            }
        )
    for field_name in positive_integer_fields:
        if int(data[field_name]) < 1:
            raise ValueError(f"{field_name} must be positive")
    if not 0 <= int(data["seed"]) < 2**32:
        raise ValueError("seed must be an integer in [0, 2**32)")
    if int(data["num_workers"]) < 0:
        raise ValueError("num_workers must be non-negative")
    if int(data["warmup_epochs"]) < 0:
        raise ValueError("warmup_epochs must be non-negative")

    for field_name in ("time_embed_scale", "time_embed_max_period", "logit_normal_p_std", "noise_scale"):
        if field_name in data and float(data[field_name]) <= 0.0:
            raise ValueError(f"{field_name} must be positive")
    if float(data["weight_decay"]) < 0.0:
        raise ValueError("weight_decay must be non-negative")
    for field_name in ("adamw_beta1", "adamw_beta2"):
        if not 0.0 < float(data[field_name]) < 1.0:
            raise ValueError(f"{field_name} must be in (0, 1)")
    if float(data["max_grad_norm"]) <= 0.0:
        raise ValueError("max_grad_norm must be positive")
    if not 0.0 < float(data["ema_decay"]) < 1.0:
        raise ValueError("ema_decay must be in (0, 1)")
    if not 0.0 < float(data["eps_t"]) < 0.5:
        raise ValueError("eps_t must be in (0, 0.5)")
    # Training time is FP32, including under BF16 model autocast. Check the
    # actual endpoints, not just their real-number bounds; never silently
    # change the configured time law to repair an unrepresentable interval.
    lower = struct.unpack("f", struct.pack("f", float(data["eps_t"])))[0]
    upper = struct.unpack("f", struct.pack("f", 1.0 - float(data["eps_t"])))[0]
    if not 0.0 < lower < upper < 1.0:
        raise ValueError("eps_t must define distinct FP32 endpoints strictly inside (0, 1)")
    for field_name in (
        "unet_dropout", "edm_dropout", "edm_label_dropout",
        "jit_attn_dropout", "jit_proj_dropout",
    ):
        if field_name in data and not 0.0 <= float(data[field_name]) < 1.0:
            raise ValueError(f"{field_name} must be in [0, 1)")
    for field_name in ("unet_channel_mult", "edm_channel_mult"):
        if field_name in data and (
            not data[field_name] or any(int(value) < 1 for value in data[field_name])
        ):
            raise ValueError(f"{field_name} must contain positive integers")
    for field_name in ("unet_attention_resolutions", "edm_attn_resolutions"):
        if field_name in data and any(int(value) < 1 for value in data[field_name]):
            raise ValueError(f"{field_name} must contain positive integers")

    logvar_min = float(data["logvar_min"])
    logvar_init = float(data["logvar_init"])
    logvar_max = float(data["logvar_max"])
    if not logvar_min < logvar_init < logvar_max:
        raise ValueError("logvar bounds must satisfy logvar_min < logvar_init < logvar_max")

    lr = data.get("lr", ExperimentConfig.lr)
    base_lr = data.get("base_lr", ExperimentConfig.base_lr)
    lr_reference_batch_size = int(data.get("lr_reference_batch_size", ExperimentConfig.lr_reference_batch_size))
    scale_lr_by_effective_batch = bool(
        data.get("scale_lr_by_effective_batch", ExperimentConfig.scale_lr_by_effective_batch)
    )
    if lr is not None and float(lr) <= 0.0:
        raise ValueError("lr must be positive when set")
    if base_lr is not None and float(base_lr) <= 0.0:
        raise ValueError("base_lr must be positive when set")
    if lr_reference_batch_size < 1:
        raise ValueError("lr_reference_batch_size must be >= 1")
    if scale_lr_by_effective_batch and base_lr is None:
        raise ValueError("base_lr is required when scale_lr_by_effective_batch=True")
    if scale_lr_by_effective_batch and lr is not None:
        raise ValueError("lr must be null when scale_lr_by_effective_batch=True")
    if not scale_lr_by_effective_batch and lr is None:
        raise ValueError("lr is required unless scale_lr_by_effective_batch=True")
    if not scale_lr_by_effective_batch and base_lr is not None:
        raise ValueError("base_lr must be null when scale_lr_by_effective_batch=False")

    self_conditioning_mode = str(
        data.get("self_conditioning_mode", ExperimentConfig.self_conditioning_mode)
    )
    if self_conditioning_mode != "none":
        raise ValueError("self_conditioning_mode must be 'none'; self-conditioning runtime is retired")
    checkpoint_every_epochs = int(data.get("checkpoint_every_epochs", ExperimentConfig.checkpoint_every_epochs))
    if checkpoint_every_epochs < 0:
        raise ValueError("checkpoint_every_epochs must be non-negative")
