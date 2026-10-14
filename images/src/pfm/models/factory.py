from __future__ import annotations

from typing import Callable, Dict, Optional

import torch
from torch import nn
from torch.nn import functional as F

from pfm.config import ImageConfig
from pfm.models.edm_backbone import EDMDDPMppBackbone
from pfm.models.config_validation import validate_model_configuration
from pfm.models.embeddings import ConditionEmbeddings
from pfm.models.jit_backbone import JiTBackbone, JiTImageHead
from pfm.models.heads import PosteriorHeads
from pfm.probability import inverse_smooth_logvar_bound


class TimestepBlock(nn.Module):
    def forward(self, x: torch.Tensor, condition: torch.Tensor) -> torch.Tensor:
        raise NotImplementedError


class TimestepEmbedSequential(nn.Sequential):
    def forward(self, x: torch.Tensor, condition: torch.Tensor) -> torch.Tensor:
        for module in self:
            if isinstance(module, TimestepBlock):
                x = module(x, condition)
            else:
                x = module(x)
        return x


class ResidualBlock(TimestepBlock):
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        condition_dim: int,
        norm_groups: int,
        dropout: float,
    ) -> None:
        super().__init__()
        self.norm1 = _make_group_norm(in_channels, norm_groups)
        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1)
        self.condition = nn.Linear(condition_dim, out_channels * 2)
        self.norm2 = _make_group_norm(out_channels, norm_groups)
        self.dropout = nn.Dropout(float(dropout))
        self.conv2 = _zero_module(nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1))
        self.skip = (
            nn.Identity()
            if in_channels == out_channels
            else nn.Conv2d(in_channels, out_channels, kernel_size=1)
        )

    def forward(self, x: torch.Tensor, condition: torch.Tensor) -> torch.Tensor:
        h = self.conv1(F.silu(self.norm1(x)))
        scale, shift = self.condition(F.silu(condition)).chunk(2, dim=-1)
        h = self.norm2(h)
        h = h * (1.0 + scale[:, :, None, None]) + shift[:, :, None, None]
        h = self.conv2(self.dropout(F.silu(h)))
        return h + self.skip(x)


class AttentionBlock(nn.Module):
    def __init__(self, channels: int, norm_groups: int, num_heads: int = 1) -> None:
        super().__init__()
        if channels % num_heads != 0:
            raise ValueError(f"channels={channels} must be divisible by num_heads={num_heads}")
        self.channels = channels
        self.num_heads = num_heads
        self.norm = _make_group_norm(channels, norm_groups)
        self.qkv = nn.Conv1d(channels, channels * 3, kernel_size=1)
        self.proj = _zero_module(nn.Conv1d(channels, channels, kernel_size=1))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        bsz, channels, height, width = x.shape
        h = self.norm(x).view(bsz, channels, height * width)
        q, k, v = self.qkv(h).chunk(3, dim=1)
        head_channels = channels // self.num_heads
        q = q.view(bsz * self.num_heads, head_channels, height * width)
        k = k.view(bsz * self.num_heads, head_channels, height * width)
        v = v.view(bsz * self.num_heads, head_channels, height * width)
        scale = head_channels ** -0.5
        weights = torch.einsum("bct,bcs->bts", q * scale, k)
        weights = torch.softmax(weights.float(), dim=-1).type_as(weights)
        h = torch.einsum("bts,bcs->bct", weights, v)
        h = h.view(bsz, channels, height * width)
        h = self.proj(h).view(bsz, channels, height, width)
        return x + h


class Downsample(nn.Module):
    def __init__(self, channels: int) -> None:
        super().__init__()
        self.op = nn.Conv2d(channels, channels, kernel_size=3, stride=2, padding=1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.op(x)


class Upsample(nn.Module):
    def __init__(self, channels: int) -> None:
        super().__init__()
        self.op = nn.Conv2d(channels, channels, kernel_size=3, padding=1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = F.interpolate(x, scale_factor=2, mode="nearest")
        return self.op(x)


class OutputHead(nn.Module):
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        norm_groups: int,
        init_bias: float = 0.0,
        init_std: Optional[float] = None,
    ) -> None:
        super().__init__()
        self.norm = _make_group_norm(in_channels, norm_groups)
        self.proj = nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1)
        if init_std is None:
            _zero_module(self.proj)
        else:
            nn.init.normal_(self.proj.weight, mean=0.0, std=float(init_std))
            nn.init.zeros_(self.proj.bias)
        nn.init.constant_(self.proj.bias, float(init_bias))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.proj(F.silu(self.norm(x)))


class PredictionHead(nn.Module):
    def __init__(
        self,
        channels: int,
        out_channels: int,
        condition_dim: int,
        norm_groups: int,
        dropout: float,
        init_bias: float = 0.0,
        init_std: Optional[float] = None,
    ) -> None:
        super().__init__()
        self.private_blocks = nn.ModuleList(
            [
                ResidualBlock(channels, channels, condition_dim, norm_groups, dropout),
                ResidualBlock(channels, channels, condition_dim, norm_groups, dropout),
            ]
        )
        self.output = OutputHead(channels, out_channels, norm_groups, init_bias, init_std)

    def forward(self, x: torch.Tensor, condition: torch.Tensor) -> torch.Tensor:
        for block in self.private_blocks:
            x = block(x, condition)
        return self.output(x)


class ConditionalUNet(nn.Module):
    def __init__(self, config: ImageConfig) -> None:
        super().__init__()
        validate_model_configuration(config)
        self.config = config
        self.image_channels = int(config.image_channels)
        self.image_size = int(config.image_size)
        if self.image_channels <= 0:
            raise ValueError("image_channels must be positive")
        if self.image_size <= 0:
            raise ValueError("image_size must be positive")
        base_channels = int(config.unet_base_channels)
        if base_channels <= 0:
            raise ValueError("unet_base_channels must be positive")
        if config.unet_num_res_blocks <= 0:
            raise ValueError("unet_num_res_blocks must be positive")
        if config.unet_num_heads <= 0:
            raise ValueError("unet_num_heads must be positive")
        if not config.unet_channel_mult:
            raise ValueError("unet_channel_mult must contain at least one level")
        if not 0.0 <= config.unet_dropout < 1.0:
            raise ValueError("unet_dropout must be in [0, 1)")
        embedding_dim = base_channels * 4
        condition_dim = config.time_proj_dim
        self.embeddings = ConditionEmbeddings(
            num_classes=config.num_classes,
            time_embed_dim=config.time_embed_dim,
            time_embed_scale=config.time_embed_scale,
            time_embed_max_period=config.time_embed_max_period,
            time_proj_dim=config.time_proj_dim,
            class_embed_dim=config.class_embed_dim,
            class_proj_dim=config.class_proj_dim,
        )
        self.condition_projector = nn.Sequential(
            nn.SiLU(),
            nn.Linear(condition_dim, embedding_dim),
        )

        self.model_channels = base_channels
        self.num_res_blocks = int(config.unet_num_res_blocks)
        self.channel_mult = tuple(int(mult) for mult in config.unet_channel_mult)
        self.attention_resolutions = tuple(int(resolution) for resolution in config.unet_attention_resolutions)

        current_channels = base_channels
        ds = 1
        first_conv = nn.Conv2d(
            in_channels=self.image_channels,
            out_channels=current_channels,
            kernel_size=3,
            padding=1,
        )
        self.input_blocks = nn.ModuleList(
            [TimestepEmbedSequential(first_conv)]
        )
        input_block_channels = [current_channels]
        for level, mult in enumerate(self.channel_mult):
            out_channels = base_channels * mult
            for _ in range(self.num_res_blocks):
                layers: list[nn.Module] = [
                    ResidualBlock(
                        current_channels,
                        out_channels,
                        embedding_dim,
                        config.unet_norm_groups,
                        config.unet_dropout,
                    )
                ]
                current_channels = out_channels
                if ds in self.attention_resolutions:
                    layers.append(
                        AttentionBlock(current_channels, config.unet_norm_groups, config.unet_num_heads)
                    )
                self.input_blocks.append(TimestepEmbedSequential(*layers))
                input_block_channels.append(current_channels)
            if level != len(self.channel_mult) - 1:
                self.input_blocks.append(TimestepEmbedSequential(Downsample(current_channels)))
                input_block_channels.append(current_channels)
                ds *= 2

        self.middle_block = TimestepEmbedSequential(
            ResidualBlock(
                current_channels,
                current_channels,
                embedding_dim,
                config.unet_norm_groups,
                config.unet_dropout,
            ),
            AttentionBlock(current_channels, config.unet_norm_groups, config.unet_num_heads),
            ResidualBlock(
                current_channels,
                current_channels,
                embedding_dim,
                config.unet_norm_groups,
                config.unet_dropout,
            ),
        )

        self.output_blocks = nn.ModuleList()
        for level, mult in reversed(list(enumerate(self.channel_mult))):
            out_channels = base_channels * mult
            for block_idx in range(self.num_res_blocks + 1):
                skip_channels = input_block_channels.pop()
                layers: list[nn.Module] = [
                    ResidualBlock(
                        current_channels + skip_channels,
                        out_channels,
                        embedding_dim,
                        config.unet_norm_groups,
                        config.unet_dropout,
                    )
                ]
                current_channels = out_channels
                if ds in self.attention_resolutions:
                    layers.append(
                        AttentionBlock(current_channels, config.unet_norm_groups, config.unet_num_heads)
                    )
                if level != 0 and block_idx == self.num_res_blocks:
                    layers.append(Upsample(current_channels))
                    ds //= 2
                self.output_blocks.append(TimestepEmbedSequential(*layers))

        self.posterior_heads = PosteriorHeads(
            prediction_head_factory=lambda init_bias: PredictionHead(
                current_channels,
                self.image_channels,
                embedding_dim,
                config.unet_norm_groups,
                config.unet_dropout,
                init_bias,
            ),
            logvar_init_bias=inverse_smooth_logvar_bound(config.logvar_init, config),
        )

    def forward(
        self,
        xt: torch.Tensor,
        t: torch.Tensor,
        labels: torch.Tensor,
    ) -> Dict[str, Optional[torch.Tensor]]:
        expected_shape = (self.image_channels, self.image_size, self.image_size)
        if xt.ndim != 4 or tuple(xt.shape[1:]) != expected_shape:
            raise ValueError(
                f"Expected xt shape [B,{expected_shape[0]},{expected_shape[1]},{expected_shape[2]}], "
                f"got {tuple(xt.shape)}"
            )
        time_features, class_features = self.embeddings(t, labels)
        condition = self.condition_projector(time_features + class_features)

        skips = []
        h = xt
        for block in self.input_blocks:
            h = block(h, condition)
            skips.append(h)
        h = self.middle_block(h, condition)
        for block in self.output_blocks:
            skip = skips.pop()
            if h.shape[-2:] != skip.shape[-2:]:
                h = F.interpolate(h, size=skip.shape[-2:], mode="nearest")
            h = block(torch.cat([h, skip], dim=1), condition)

        return self.posterior_heads(h, condition)


class EDMDDPMppModel(nn.Module):
    def __init__(self, config: ImageConfig) -> None:
        super().__init__()
        validate_model_configuration(config)
        self.config = config
        self.image_channels = int(config.image_channels)
        self.image_size = int(config.image_size)
        self.backbone = EDMDDPMppBackbone(config)
        feature_channels = self.backbone.out_channels
        condition_dim = self.backbone.condition_dim
        norm_groups = _validate_group_norm_groups(config.edm_norm_groups)
        dropout = float(config.edm_dropout)

        self.posterior_heads = PosteriorHeads(
            prediction_head_factory=lambda init_bias: PredictionHead(
                feature_channels,
                self.image_channels,
                condition_dim,
                norm_groups,
                dropout,
                init_bias,
            ),
            logvar_init_bias=inverse_smooth_logvar_bound(config.logvar_init, config),
        )

    def forward(
        self,
        xt: torch.Tensor,
        t: torch.Tensor,
        labels: torch.Tensor,
    ) -> Dict[str, Optional[torch.Tensor]]:
        h, condition = self.backbone.forward_features(xt, t, labels)
        return self.posterior_heads(h, condition)


class JiTPFMModel(nn.Module):
    def __init__(self, config: ImageConfig) -> None:
        super().__init__()
        validate_model_configuration(config)
        self.config = config
        self.image_channels = int(config.image_channels)
        self.image_size = int(config.image_size)
        self.backbone = JiTBackbone(config)
        hidden_size = int(self.backbone.hidden_size)
        spec = self.backbone.spec

        self.posterior_heads = PosteriorHeads(
            prediction_head_factory=lambda init_bias: JiTImageHead(
                spec,
                self.image_size,
                hidden_size,
                self.image_channels,
                init_bias=init_bias,
            ),
            logvar_init_bias=inverse_smooth_logvar_bound(config.logvar_init, config),
        )

    def forward(
        self,
        xt: torch.Tensor,
        t: torch.Tensor,
        labels: torch.Tensor,
    ) -> Dict[str, Optional[torch.Tensor]]:
        tokens, condition = self.backbone.forward_tokens(xt, t, labels)
        return self.posterior_heads(tokens, condition)


def _make_group_norm(num_channels: int, requested_groups: int) -> nn.GroupNorm:
    channels = int(num_channels)
    groups = _validate_group_norm_groups(requested_groups)
    if channels < 1:
        raise ValueError("GroupNorm num_channels must be positive")
    if channels % groups != 0:
        raise ValueError(f"GroupNorm requested_groups={groups} must divide num_channels={channels}")
    return nn.GroupNorm(groups, channels)


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



def build_model(config: ImageConfig, device=None) -> nn.Module:
    """Native task backbone with one mean head and one velocity-variance head."""
    builders = {"conditional_unet": ConditionalUNet, "edm_ddpmpp": EDMDDPMppModel,
                "jit_transformer": JiTPFMModel}
    model = builders[config.architecture](config)
    return model.to(device) if device is not None else model
