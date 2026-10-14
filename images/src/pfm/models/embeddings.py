from __future__ import annotations

import math

import torch
from torch import nn


class DiffusionTimestepEmbedding(nn.Module):
    def __init__(self, dim: int, scale: float = 1000.0, max_period: float = 10000.0) -> None:
        super().__init__()
        if dim < 2:
            raise ValueError("DiffusionTimestepEmbedding dim must be at least 2")
        self.dim = dim
        self.scale = float(scale)
        self.max_period = float(max_period)

    def forward(self, t: torch.Tensor) -> torch.Tensor:
        if t.ndim == 2 and t.shape[1] == 1:
            t = t[:, 0]
        if t.ndim != 1:
            raise ValueError(f"Expected t shape [B] or [B,1], got {tuple(t.shape)}")
        half_dim = self.dim // 2
        exponent = torch.arange(half_dim, device=t.device, dtype=t.dtype)
        exponent = -math.log(self.max_period) * exponent / max(half_dim, 1)
        frequencies = torch.exp(exponent)
        angles = (t * self.scale)[:, None] * frequencies[None, :]
        embedding = torch.cat([torch.cos(angles), torch.sin(angles)], dim=-1)
        if self.dim % 2 == 1:
            embedding = torch.cat([embedding, torch.zeros_like(embedding[:, :1])], dim=-1)
        return embedding


class ConditionEmbeddings(nn.Module):
    def __init__(
        self,
        num_classes: int,
        time_embed_dim: int,
        time_embed_scale: float,
        time_embed_max_period: float,
        time_proj_dim: int,
        class_embed_dim: int,
        class_proj_dim: int,
    ) -> None:
        super().__init__()
        self.time_embedding = DiffusionTimestepEmbedding(
            time_embed_dim,
            scale=time_embed_scale,
            max_period=time_embed_max_period,
        )
        self.time_projector = nn.Sequential(
            nn.Linear(time_embed_dim, time_proj_dim),
            nn.SiLU(),
            nn.Linear(time_proj_dim, time_proj_dim),
        )
        self.class_embedding = nn.Embedding(num_classes, class_embed_dim)
        self.class_projector = nn.Sequential(
            nn.Linear(class_embed_dim, class_proj_dim),
            nn.SiLU(),
            nn.Linear(class_proj_dim, class_proj_dim),
        )

    def forward(self, t: torch.Tensor, labels: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        return self.time_projector(self.time_embedding(t)), self.class_projector(self.class_embedding(labels))
