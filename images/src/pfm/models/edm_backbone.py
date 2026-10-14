from __future__ import annotations

import math

import torch
from torch import nn
from torch.nn import functional as F

from pfm.config import ImageConfig


# Architecture reference: NVlabs/edm training/networks.py.
# Copyright (c) 2022 NVIDIA CORPORATION & AFFILIATES, CC BY-NC-SA 4.0.
# See THIRD_PARTY_NOTICES.md and third_party/licenses/EDM-CC-BY-NC-SA-4.0.txt.
# This file ports only the DDPM++ backbone shape into the local PFM/FM model API.


class EDMPositionalEmbedding(nn.Module):
    """Sinusoidal embedding matching the DDPM++ time-conditioning style."""

    def __init__(self, dim: int, scale: float, max_period: float) -> None:
        super().__init__()
        if dim < 2:
            raise ValueError("EDM positional embedding dim must be at least 2")
        self.dim = int(dim)
        self.scale = float(scale)
        self.max_period = float(max_period)

    def forward(self, t: torch.Tensor) -> torch.Tensor:
        if t.ndim == 2 and t.shape[1] == 1:
            t = t[:, 0]
        if t.ndim != 1:
            raise ValueError(f"Expected t shape [B] or [B,1], got {tuple(t.shape)}")
        half_dim = self.dim // 2
        exponent = torch.arange(half_dim, device=t.device, dtype=t.dtype)
        exponent = -math.log(self.max_period) * exponent / max(half_dim - 1, 1)
        angles = (t * self.scale)[:, None] * torch.exp(exponent)[None, :]
        embedding = torch.cat([torch.sin(angles), torch.cos(angles)], dim=-1)
        if self.dim % 2 == 1:
            embedding = torch.cat([embedding, torch.zeros_like(embedding[:, :1])], dim=-1)
        return embedding


class EDMConditioning(nn.Module):
    def __init__(self, config: ImageConfig) -> None:
        super().__init__()
        model_channels = int(config.edm_model_channels)
        emb_channels = model_channels * int(config.edm_channel_mult_emb)
        self.num_classes = int(config.num_classes)
        self.label_dropout = float(config.edm_label_dropout)
        self.time_embedding = EDMPositionalEmbedding(
            model_channels,
            scale=config.time_embed_scale,
            max_period=config.time_embed_max_period,
        )
        self.time_map = nn.Sequential(
            nn.Linear(model_channels, emb_channels),
            nn.SiLU(),
            nn.Linear(emb_channels, emb_channels),
        )
        self.label_map = nn.Linear(self.num_classes, emb_channels, bias=False)
        self.out_dim = emb_channels

    def forward(self, t: torch.Tensor, labels: torch.Tensor) -> torch.Tensor:
        if labels.ndim != 1:
            raise ValueError(f"Expected labels shape [B], got {tuple(labels.shape)}")
        time_embedding = self.time_map(self.time_embedding(t))
        class_onehot = F.one_hot(labels.long(), num_classes=self.num_classes).to(dtype=time_embedding.dtype)
        if self.training and self.label_dropout > 0.0:
            keep = torch.rand(labels.shape[0], device=labels.device) >= self.label_dropout
            class_onehot = class_onehot * keep[:, None].to(class_onehot.dtype)
        return time_embedding + self.label_map(class_onehot)


class EDMResidualBlock(nn.Module):
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        emb_channels: int,
        norm_groups: int,
        dropout: float,
    ) -> None:
        super().__init__()
        self.norm1 = _make_group_norm(in_channels, norm_groups)
        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1)
        self.emb_proj = nn.Linear(emb_channels, out_channels * 2)
        self.norm2 = _make_group_norm(out_channels, norm_groups)
        self.dropout = nn.Dropout(float(dropout))
        self.conv2 = _zero_module(nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1))
        self.skip = nn.Identity() if in_channels == out_channels else nn.Conv2d(in_channels, out_channels, kernel_size=1)
        self.skip_scale = math.sqrt(0.5)

    def forward(self, x: torch.Tensor, emb: torch.Tensor) -> torch.Tensor:
        h = self.conv1(F.silu(self.norm1(x)))
        scale, shift = self.emb_proj(F.silu(emb)).chunk(2, dim=1)
        h = self.norm2(h)
        h = h * (1.0 + scale[:, :, None, None]) + shift[:, :, None, None]
        h = self.conv2(self.dropout(F.silu(h)))
        return (h + self.skip(x)) * self.skip_scale


class EDMAttentionBlock(nn.Module):
    def __init__(self, channels: int, norm_groups: int) -> None:
        super().__init__()
        self.channels = int(channels)
        self.norm = _make_group_norm(channels, norm_groups)
        self.qkv = nn.Conv1d(channels, channels * 3, kernel_size=1)
        self.proj = _zero_module(nn.Conv1d(channels, channels, kernel_size=1))
        self.skip_scale = math.sqrt(0.5)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        batch, channels, height, width = x.shape
        h = self.norm(x).reshape(batch, channels, height * width)
        q, k, v = self.qkv(h).chunk(3, dim=1)
        scale = channels ** -0.25
        attention = torch.einsum("bct,bcs->bts", q * scale, k * scale)
        attention = torch.softmax(attention.float(), dim=-1).to(dtype=x.dtype)
        h = torch.einsum("bts,bcs->bct", attention, v)
        h = self.proj(h).reshape(batch, channels, height, width)
        return (x + h) * self.skip_scale


class EDMDownsample(nn.Module):
    def __init__(self, channels: int) -> None:
        super().__init__()
        self.conv = nn.Conv2d(channels, channels, kernel_size=3, stride=2, padding=1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.conv(x)


class EDMUpsample(nn.Module):
    def __init__(self, channels: int) -> None:
        super().__init__()
        self.conv = nn.Conv2d(channels, channels, kernel_size=3, padding=1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.conv(F.interpolate(x, scale_factor=2, mode="nearest"))


class EDMBlockSequence(nn.Module):
    def __init__(self, *layers: nn.Module) -> None:
        super().__init__()
        self.layers = nn.ModuleList(layers)

    def forward(self, x: torch.Tensor, emb: torch.Tensor) -> torch.Tensor:
        for layer in self.layers:
            if isinstance(layer, EDMResidualBlock):
                x = layer(x, emb)
            else:
                x = layer(x)
        return x


class EDMDDPMppBackbone(nn.Module):
    """DDPM++ style class-conditional U-Net backbone for the project heads.

    The structure follows the CIFAR10 DDPM++ configuration used by NVlabs EDM:
    model channels 128, channel multipliers 1-2-2-2, four residual blocks per
    resolution, attention at 16x16, and 0.10 dropout. This module deliberately
    exposes only a feature map and conditioning vector; PFM/FM losses, samplers,
    and output-head semantics stay in the local codebase.
    """

    def __init__(self, config: ImageConfig) -> None:
        super().__init__()
        self.image_channels = int(config.image_channels)
        self.image_size = int(config.image_size)
        self.model_channels = int(config.edm_model_channels)
        self.channel_mult = tuple(int(mult) for mult in config.edm_channel_mult)
        self.num_blocks = int(config.edm_num_blocks)
        self.attn_resolutions = tuple(int(resolution) for resolution in config.edm_attn_resolutions)
        self.dropout = float(config.edm_dropout)
        self.norm_groups = _validate_group_norm_groups(config.edm_norm_groups)
        if self.image_channels <= 0:
            raise ValueError("image_channels must be positive")
        if self.image_size <= 0:
            raise ValueError("image_size must be positive")
        if self.model_channels <= 0:
            raise ValueError("edm_model_channels must be positive")
        if not self.channel_mult:
            raise ValueError("edm_channel_mult must contain at least one level")
        if self.num_blocks <= 0:
            raise ValueError("edm_num_blocks must be positive")
        if not 0.0 <= self.dropout < 1.0:
            raise ValueError("edm_dropout must be in [0, 1)")
        if not 0.0 <= float(config.edm_label_dropout) < 1.0:
            raise ValueError("edm_label_dropout must be in [0, 1)")

        self.conditioning = EDMConditioning(config)
        emb_channels = self.conditioning.out_dim
        current_channels = self.model_channels
        current_resolution = self.image_size
        first_conv = nn.Conv2d(
            in_channels=self.image_channels,
            out_channels=current_channels,
            kernel_size=3,
            padding=1,
        )
        self.input_blocks = nn.ModuleList([EDMBlockSequence(first_conv)])
        input_channels = [current_channels]

        for level, mult in enumerate(self.channel_mult):
            out_channels = self.model_channels * mult
            for _ in range(self.num_blocks):
                layers: list[nn.Module] = [
                    EDMResidualBlock(
                        current_channels,
                        out_channels,
                        emb_channels,
                        self.norm_groups,
                        self.dropout,
                    )
                ]
                current_channels = out_channels
                if current_resolution in self.attn_resolutions:
                    layers.append(EDMAttentionBlock(current_channels, self.norm_groups))
                self.input_blocks.append(EDMBlockSequence(*layers))
                input_channels.append(current_channels)
            if level != len(self.channel_mult) - 1:
                self.input_blocks.append(EDMBlockSequence(EDMDownsample(current_channels)))
                input_channels.append(current_channels)
                current_resolution //= 2

        self.middle_block = EDMBlockSequence(
            EDMResidualBlock(current_channels, current_channels, emb_channels, self.norm_groups, self.dropout),
            EDMAttentionBlock(current_channels, self.norm_groups),
            EDMResidualBlock(current_channels, current_channels, emb_channels, self.norm_groups, self.dropout),
        )

        self.output_blocks = nn.ModuleList()
        for level, mult in reversed(list(enumerate(self.channel_mult))):
            out_channels = self.model_channels * mult
            for block_idx in range(self.num_blocks + 1):
                skip_channels = input_channels.pop()
                layers = [
                    EDMResidualBlock(
                        current_channels + skip_channels,
                        out_channels,
                        emb_channels,
                        self.norm_groups,
                        self.dropout,
                    )
                ]
                current_channels = out_channels
                if current_resolution in self.attn_resolutions:
                    layers.append(EDMAttentionBlock(current_channels, self.norm_groups))
                if level != 0 and block_idx == self.num_blocks:
                    layers.append(EDMUpsample(current_channels))
                    current_resolution *= 2
                self.output_blocks.append(EDMBlockSequence(*layers))

        self.out_channels = current_channels
        self.condition_dim = emb_channels

    def forward_features(self, xt: torch.Tensor, t: torch.Tensor, labels: torch.Tensor):
        expected_shape = (self.image_channels, self.image_size, self.image_size)
        if xt.ndim != 4 or tuple(xt.shape[1:]) != expected_shape:
            raise ValueError(
                f"Expected xt shape [B,{expected_shape[0]},{expected_shape[1]},{expected_shape[2]}], "
                f"got {tuple(xt.shape)}"
            )
        emb = self.conditioning(t, labels)
        skips = []
        h = xt
        for block in self.input_blocks:
            h = block(h, emb)
            skips.append(h)
        h = self.middle_block(h, emb)
        for block in self.output_blocks:
            skip = skips.pop()
            if h.shape[-2:] != skip.shape[-2:]:
                h = F.interpolate(h, size=skip.shape[-2:], mode="nearest")
            h = block(torch.cat([h, skip], dim=1), emb)
        return h, emb


def _make_group_norm(num_channels: int, requested_groups: int) -> nn.GroupNorm:
    channels = int(num_channels)
    groups = _validate_group_norm_groups(requested_groups)
    if channels < 1:
        raise ValueError("GroupNorm num_channels must be positive")
    if channels % groups != 0:
        raise ValueError(f"GroupNorm requested_groups={groups} must divide num_channels={channels}")
    return nn.GroupNorm(groups, channels, eps=1e-6)


def _validate_group_norm_groups(requested_groups: int) -> int:
    if not isinstance(requested_groups, int) or isinstance(requested_groups, bool):
        actual_type = type(requested_groups).__name__
        raise ValueError(f"GroupNorm requested_groups must be an integer, got {actual_type}")
    if requested_groups < 1:
        raise ValueError("GroupNorm requested_groups must be positive")
    return requested_groups


def _zero_module(module: nn.Module) -> nn.Module:
    for parameter in module.parameters():
        nn.init.zeros_(parameter)
    return module
