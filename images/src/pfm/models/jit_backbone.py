# Adapted from LTH14/JiT, copyright (c) 2025 Tianhong Li, MIT License.
# See THIRD_PARTY_NOTICES.md and third_party/licenses/JiT-MIT.txt.
from __future__ import annotations

import math
from typing import Optional

import torch
from torch import nn
from torch.nn import functional as F

from pfm.config import ImageConfig
from pfm.models.jit_config import JiTModelSpec, jit_spec_for


JIT_UPSTREAM_REPOSITORY = "https://github.com/LTH14/JiT"
JIT_UPSTREAM_COMMIT = "cbc743a2ada5e9762697da2c83f8c4f8379e8c17"


class RMSNorm(nn.Module):
    def __init__(self, hidden_size: int, eps: float = 1e-6) -> None:
        super().__init__()
        self.weight = nn.Parameter(torch.ones(hidden_size))
        self.eps = float(eps)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        dtype = x.dtype
        x_float = x.float()
        variance = x_float.square().mean(dim=-1, keepdim=True)
        return (self.weight * x_float * torch.rsqrt(variance + self.eps)).to(dtype)


class TimestepEmbedder(nn.Module):
    def __init__(self, hidden_size: int, frequency_embedding_size: int = 256) -> None:
        super().__init__()
        self.frequency_embedding_size = int(frequency_embedding_size)
        self.mlp = nn.Sequential(
            nn.Linear(self.frequency_embedding_size, hidden_size),
            nn.SiLU(),
            nn.Linear(hidden_size, hidden_size),
        )

    def forward(self, t: torch.Tensor) -> torch.Tensor:
        half = self.frequency_embedding_size // 2
        freqs = torch.exp(
            -math.log(10000.0) * torch.arange(0, half, dtype=torch.float32, device=t.device) / float(half)
        )
        args = t[:, None].float() * freqs[None]
        emb = torch.cat([torch.cos(args), torch.sin(args)], dim=-1)
        if self.frequency_embedding_size % 2:
            emb = torch.cat([emb, torch.zeros_like(emb[:, :1])], dim=-1)
        return self.mlp(emb)


class BottleneckPatchEmbed(nn.Module):
    def __init__(
        self,
        image_size: int,
        patch_size: int,
        in_channels: int,
        bottleneck_dim: int,
        hidden_size: int,
    ) -> None:
        super().__init__()
        self.image_size = int(image_size)
        self.patch_size = int(patch_size)
        self.grid_size = self.image_size // self.patch_size
        self.num_patches = self.grid_size * self.grid_size
        self.proj1 = nn.Conv2d(
            in_channels=in_channels,
            out_channels=bottleneck_dim,
            kernel_size=self.patch_size,
            stride=self.patch_size,
            bias=False,
        )
        self.proj2 = nn.Conv2d(bottleneck_dim, hidden_size, kernel_size=1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if x.shape[-2:] != (self.image_size, self.image_size):
            raise ValueError(f"expected image size {self.image_size}, got {tuple(x.shape[-2:])}")
        return self.proj2(self.proj1(x)).flatten(2).transpose(1, 2)


class SwiGLUFFN(nn.Module):
    def __init__(self, hidden_size: int, mlp_ratio: float, dropout: float) -> None:
        super().__init__()
        hidden = int(hidden_size * float(mlp_ratio) * 2.0 / 3.0)
        self.w12 = nn.Linear(hidden_size, 2 * hidden)
        self.w3 = nn.Linear(hidden, hidden_size)
        self.dropout = nn.Dropout(float(dropout))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x1, x2 = self.w12(x).chunk(2, dim=-1)
        return self.w3(self.dropout(F.silu(x1) * x2))


def _modulate(x: torch.Tensor, shift: torch.Tensor, scale: torch.Tensor) -> torch.Tensor:
    return x * (1.0 + scale.unsqueeze(1)) + shift.unsqueeze(1)


def _rotate_half(x: torch.Tensor) -> torch.Tensor:
    paired = x.reshape(*x.shape[:-1], -1, 2)
    first, second = paired.unbind(dim=-1)
    return torch.stack((-second, first), dim=-1).flatten(-2)


class VisionRotaryEmbeddingFast(nn.Module):
    """JiT's 2D vision RoPE, with identity rotations for in-context tokens."""

    def __init__(
        self,
        dim: int,
        pt_seq_len: int,
        num_cls_token: int = 0,
        theta: float = 10000.0,
    ) -> None:
        super().__init__()
        dim = int(dim)
        pt_seq_len = int(pt_seq_len)
        num_cls_token = int(num_cls_token)
        if dim < 2 or dim % 2 != 0:
            raise ValueError("JiT RoPE dim must be a positive even integer")

        frequencies = 1.0 / (
            float(theta)
            ** (torch.arange(0, dim, 2, dtype=torch.float32) / float(dim))
        )
        positions = torch.arange(pt_seq_len, dtype=torch.float32)
        angles = torch.einsum("i,j->ij", positions, frequencies).repeat_interleave(2, dim=-1)
        angles_2d = torch.cat(
            (
                angles[:, None, :].expand(pt_seq_len, pt_seq_len, dim),
                angles[None, :, :].expand(pt_seq_len, pt_seq_len, dim),
            ),
            dim=-1,
        ).reshape(pt_seq_len * pt_seq_len, 2 * dim)

        cos = angles_2d.cos()
        sin = angles_2d.sin()
        if num_cls_token > 0:
            cos = torch.cat((torch.ones(num_cls_token, 2 * dim), cos), dim=0)
            sin = torch.cat((torch.zeros(num_cls_token, 2 * dim), sin), dim=0)
        self.register_buffer("freqs_cos", cos, persistent=False)
        self.register_buffer("freqs_sin", sin, persistent=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x * self.freqs_cos + _rotate_half(x) * self.freqs_sin


def _upstream_scaled_dot_product_attention(
    query: torch.Tensor,
    key: torch.Tensor,
    value: torch.Tensor,
    dropout_p: float,
) -> torch.Tensor:
    """JiT's pinned explicit FP32-QK attention reference path."""
    scale_factor = 1.0 / math.sqrt(query.size(-1))
    with torch.amp.autocast(query.device.type, enabled=False):
        attention_weight = query.float() @ key.float().transpose(-2, -1)
        attention_weight = attention_weight * scale_factor
    attention_weight = torch.softmax(attention_weight, dim=-1)
    attention_weight = torch.dropout(
        attention_weight,
        float(dropout_p),
        train=True,
    )
    return attention_weight @ value


class Attention(nn.Module):
    """Upstream JiT qkv/projection attention with per-head Q/K RMSNorm."""

    def __init__(
        self,
        dim: int,
        num_heads: int,
        attn_dropout: float,
        proj_dropout: float,
    ) -> None:
        super().__init__()
        if int(dim) % int(num_heads) != 0:
            raise ValueError("JiT attention dim must be divisible by num_heads")
        self.num_heads = int(num_heads)
        head_dim = int(dim) // self.num_heads
        self.q_norm = RMSNorm(head_dim)
        self.k_norm = RMSNorm(head_dim)
        self.qkv = nn.Linear(int(dim), 3 * int(dim), bias=True)
        self.attn_drop = nn.Dropout(float(attn_dropout))
        self.proj = nn.Linear(int(dim), int(dim))
        self.proj_drop = nn.Dropout(float(proj_dropout))

    def forward(self, x: torch.Tensor, rope: VisionRotaryEmbeddingFast) -> torch.Tensor:
        batch_size, num_tokens, channels = x.shape
        qkv = self.qkv(x).reshape(
            batch_size,
            num_tokens,
            3,
            self.num_heads,
            channels // self.num_heads,
        )
        qkv = qkv.permute(2, 0, 3, 1, 4)
        query, key, value = qkv.unbind(dim=0)
        query = rope(self.q_norm(query))
        key = rope(self.k_norm(key))
        dropout_p = self.attn_drop.p if self.training else 0.0
        attended = _upstream_scaled_dot_product_attention(
            query, key, value, dropout_p,
        )
        attended = attended.transpose(1, 2).reshape(batch_size, num_tokens, channels)
        return self.proj_drop(self.proj(attended))


class JiTBlock(nn.Module):
    def __init__(
        self,
        hidden_size: int,
        num_heads: int,
        mlp_ratio: float,
        attn_dropout: float,
        proj_dropout: float,
    ) -> None:
        super().__init__()
        self.norm1 = RMSNorm(hidden_size)
        self.attn = Attention(
            hidden_size,
            num_heads,
            attn_dropout=attn_dropout,
            proj_dropout=proj_dropout,
        )
        self.norm2 = RMSNorm(hidden_size)
        self.mlp = SwiGLUFFN(hidden_size, mlp_ratio, proj_dropout)
        self.adaLN_modulation = nn.Sequential(nn.SiLU(), nn.Linear(hidden_size, 6 * hidden_size))
        nn.init.zeros_(self.adaLN_modulation[-1].weight)
        nn.init.zeros_(self.adaLN_modulation[-1].bias)

    def forward(
        self,
        x: torch.Tensor,
        condition: torch.Tensor,
        feat_rope: VisionRotaryEmbeddingFast,
    ) -> torch.Tensor:
        shift_msa, scale_msa, gate_msa, shift_mlp, scale_mlp, gate_mlp = self.adaLN_modulation(condition).chunk(6, dim=-1)
        attn_in = _modulate(self.norm1(x), shift_msa, scale_msa)
        x = x + gate_msa.unsqueeze(1) * self.attn(attn_in, feat_rope)
        x = x + gate_mlp.unsqueeze(1) * self.mlp(_modulate(self.norm2(x), shift_mlp, scale_mlp))
        return x


class JiTBackbone(nn.Module):
    """JiT architecture from LTH14/JiT at ``JIT_UPSTREAM_COMMIT``, with PFM heads."""

    def __init__(self, config: ImageConfig) -> None:
        super().__init__()
        self.config = config
        self.spec = jit_spec_for(config.jit_model, int(config.image_size))
        self.image_channels = int(config.image_channels)
        self.hidden_size = int(self.spec.hidden_size)
        self.patch_size = int(self.spec.patch_size)
        self.grid_size = int(config.image_size) // self.patch_size
        self.num_patches = self.grid_size * self.grid_size
        self.x_embedder = BottleneckPatchEmbed(
            image_size=int(config.image_size),
            patch_size=self.patch_size,
            in_channels=self.image_channels,
            bottleneck_dim=int(self.spec.bottleneck_dim),
            hidden_size=self.hidden_size,
        )
        self.t_embedder = TimestepEmbedder(self.hidden_size)
        self.y_embedder = nn.Embedding(int(config.num_classes), self.hidden_size)
        self.pos_embed = nn.Parameter(_sincos_2d(self.hidden_size, self.grid_size).unsqueeze(0), requires_grad=False)
        self.in_context_len = int(self.spec.in_context_len)
        self.in_context_start = int(self.spec.in_context_start)
        if self.in_context_len > 0:
            self.in_context_posemb = nn.Parameter(torch.empty(1, self.in_context_len, self.hidden_size))
            nn.init.normal_(self.in_context_posemb, std=0.02)
        num_heads = int(self.spec.num_heads)
        if self.hidden_size % num_heads != 0:
            raise ValueError("JiT hidden_size must be divisible by num_heads")
        head_dim = self.hidden_size // num_heads
        if head_dim % 4 != 0:
            raise ValueError("JiT attention head dimension must be divisible by 4 for 2D RoPE")
        self.feat_rope = VisionRotaryEmbeddingFast(
            dim=head_dim // 2,
            pt_seq_len=self.grid_size,
        )
        self.feat_rope_incontext = VisionRotaryEmbeddingFast(
            dim=head_dim // 2,
            pt_seq_len=self.grid_size,
            num_cls_token=self.in_context_len,
        )
        self.blocks = nn.ModuleList(
            [
                JiTBlock(
                    self.hidden_size,
                    self.spec.num_heads,
                    mlp_ratio=4.0,
                    attn_dropout=float(config.jit_attn_dropout) if (self.spec.depth // 4 * 3 > i >= self.spec.depth // 4) else 0.0,
                    proj_dropout=float(config.jit_proj_dropout) if (self.spec.depth // 4 * 3 > i >= self.spec.depth // 4) else 0.0,
                )
                for i in range(int(self.spec.depth))
            ]
        )
        self.initialize_weights()

    def initialize_weights(self) -> None:
        for module in self.modules():
            if isinstance(module, nn.Linear):
                nn.init.xavier_uniform_(module.weight)
                if module.bias is not None:
                    nn.init.zeros_(module.bias)
        nn.init.xavier_uniform_(self.x_embedder.proj1.weight.view(self.x_embedder.proj1.weight.shape[0], -1))
        nn.init.xavier_uniform_(self.x_embedder.proj2.weight.view(self.x_embedder.proj2.weight.shape[0], -1))
        nn.init.zeros_(self.x_embedder.proj2.bias)
        nn.init.normal_(self.y_embedder.weight, std=0.02)
        nn.init.normal_(self.t_embedder.mlp[0].weight, std=0.02)
        nn.init.normal_(self.t_embedder.mlp[2].weight, std=0.02)
        for block in self.blocks:
            nn.init.zeros_(block.adaLN_modulation[-1].weight)
            nn.init.zeros_(block.adaLN_modulation[-1].bias)

    def forward_tokens(self, x: torch.Tensor, t: torch.Tensor, labels: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        condition = self.t_embedder(t) + self.y_embedder(labels)
        tokens = self.x_embedder(x) + self.pos_embed.to(device=x.device, dtype=x.dtype)
        for idx, block in enumerate(self.blocks):
            if self.in_context_len > 0 and idx == self.in_context_start:
                context = self.y_embedder(labels).unsqueeze(1).repeat(1, self.in_context_len, 1)
                context = context + self.in_context_posemb.to(device=x.device, dtype=x.dtype)
                tokens = torch.cat([context, tokens], dim=1)
            feat_rope = self.feat_rope if idx < self.in_context_start else self.feat_rope_incontext
            tokens = block(tokens, condition, feat_rope)
        if self.in_context_len > 0:
            tokens = tokens[:, self.in_context_len :]
        return tokens, condition


class JiTImageHead(nn.Module):
    def __init__(
        self,
        spec: JiTModelSpec,
        image_size: int,
        hidden_size: int,
        out_channels: int,
        init_bias: float = 0.0,
        init_std: Optional[float] = None,
    ) -> None:
        super().__init__()
        self.patch_size = int(spec.patch_size)
        self.out_channels = int(out_channels)
        self.grid_size = int(image_size) // self.patch_size
        self.norm = RMSNorm(hidden_size)
        self.adaLN_modulation = nn.Sequential(nn.SiLU(), nn.Linear(hidden_size, 2 * hidden_size))
        self.linear = nn.Linear(hidden_size, self.patch_size * self.patch_size * self.out_channels)
        nn.init.zeros_(self.adaLN_modulation[-1].weight)
        nn.init.zeros_(self.adaLN_modulation[-1].bias)
        if init_std is None:
            nn.init.zeros_(self.linear.weight)
        else:
            nn.init.normal_(self.linear.weight, mean=0.0, std=float(init_std))
        nn.init.constant_(self.linear.bias, float(init_bias))

    def forward(self, tokens: torch.Tensor, condition: torch.Tensor) -> torch.Tensor:
        shift, scale = self.adaLN_modulation(condition).chunk(2, dim=-1)
        patches = self.linear(_modulate(self.norm(tokens), shift, scale))
        return self.unpatchify(patches)

    def unpatchify(self, patches: torch.Tensor) -> torch.Tensor:
        bsz, num_patches, _ = patches.shape
        grid = int(num_patches**0.5)
        if grid * grid != num_patches:
            raise ValueError(f"patch token count must be square, got {num_patches}")
        p = self.patch_size
        c = self.out_channels
        x = patches.reshape(bsz, grid, grid, p, p, c)
        x = torch.einsum("nhwpqc->nchpwq", x)
        return x.reshape(bsz, c, grid * p, grid * p)


def _sincos_2d(embed_dim: int, grid_size: int) -> torch.Tensor:
    if embed_dim % 4 != 0:
        raise ValueError("JiT hidden_size must be divisible by 4 for 2D sinusoidal embedding")
    grid_h = torch.arange(grid_size, dtype=torch.float32)
    grid_w = torch.arange(grid_size, dtype=torch.float32)
    yy, xx = torch.meshgrid(grid_h, grid_w, indexing="ij")
    pos = torch.stack([xx.reshape(-1), yy.reshape(-1)], dim=1)
    half = embed_dim // 2
    return torch.cat([_sincos_1d(half, pos[:, 0]), _sincos_1d(half, pos[:, 1])], dim=1)




def _sincos_1d(embed_dim: int, pos: torch.Tensor) -> torch.Tensor:
    omega = torch.arange(embed_dim // 2, dtype=torch.float32)
    omega = 1.0 / (10000 ** (omega / float(embed_dim // 2)))
    out = pos[:, None] * omega[None]
    return torch.cat([torch.sin(out), torch.cos(out)], dim=1)
