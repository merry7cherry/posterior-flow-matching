"""Shape relations required by the native image architectures."""
from __future__ import annotations

from typing import TYPE_CHECKING

from pfm.models.jit_config import jit_spec_for

if TYPE_CHECKING:
    from pfm.config import ImageConfig


def validate_model_configuration(config: ImageConfig) -> None:
    if config.architecture == "jit_transformer":
        jit_spec_for(config.jit_model, config.image_size)
        return
    if config.architecture == "conditional_unet":
        if config.time_proj_dim != config.class_proj_dim:
            raise ValueError("time_proj_dim and class_proj_dim must match for additive conditioning")
        if config.time_embed_dim < 2:
            raise ValueError("time_embed_dim must be at least 2")
        base, multipliers, groups = (
            config.unet_base_channels, config.unet_channel_mult, config.unet_norm_groups
        )
    elif config.architecture == "edm_ddpmpp":
        base, multipliers, groups = (
            config.edm_model_channels, config.edm_channel_mult, config.edm_norm_groups
        )
        if base < 2:
            raise ValueError("edm_model_channels must be at least 2 for positional embedding")
    else:
        raise ValueError(f"Unsupported architecture {config.architecture!r}")

    if not multipliers or base < 1 or any(mult < 1 for mult in multipliers):
        raise ValueError("model channels and channel multipliers must be positive")
    if type(groups) is not int:
        raise ValueError(f"GroupNorm requested_groups must be an integer, got {type(groups).__name__}")
    if groups < 1:
        raise ValueError("GroupNorm requested_groups must be positive")
    # Encoder widths, decoder skip concatenations (sums of these widths),
    # and output heads all use this same fixed GroupNorm group count.
    for channels in {base, *(base * mult for mult in multipliers)}:
        if channels % groups:
            raise ValueError(f"GroupNorm requested_groups={groups} must divide num_channels={channels}")
    if config.architecture == "conditional_unet":
        heads = config.unet_num_heads
        if heads < 1:
            raise ValueError("unet_num_heads must be positive")
        # The middle block always has attention; other levels opt in by
        # downsampling factor, not by image resolution.
        attended = {base * multipliers[-1]}
        attended.update(base * mult for level, mult in enumerate(multipliers)
                        if 2**level in config.unet_attention_resolutions)
        for channels in attended:
            if channels % heads:
                raise ValueError(f"channels={channels} must be divisible by num_heads={heads}")
