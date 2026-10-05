from __future__ import annotations

from collections.abc import Callable, Mapping

import torch
from torch import nn

from pfm.core.variant_metadata import DELTA_VARIANTS, SUPPORTED_VARIANTS
from pfm.probability import targets as probability_targets


HeadFactory = Callable[[float], nn.Module]


class VariantAdapter(nn.Module):
    def __init__(self, variant: str, heads: Mapping[str, nn.Module]) -> None:
        super().__init__()
        if variant not in SUPPORTED_VARIANTS:
            raise ValueError(
                f"Unknown variant {variant!r}; expected one of {sorted(SUPPORTED_VARIANTS)}"
            )
        self.variant = variant
        self.heads = nn.ModuleDict(dict(heads))

    def transform_heads(self, transform: Callable[[nn.Module], nn.Module]) -> None:
        self.heads = nn.ModuleDict({name: transform(head) for name, head in self.heads.items()})

    def forward(
        self,
        features: torch.Tensor,
        condition: torch.Tensor,
        *,
        xt: torch.Tensor,
        t: torch.Tensor,
    ) -> dict[str, torch.Tensor]:
        if self.variant == "pfm_full":
            return {
                "mu": self.heads["mu"](features, condition),
                "logvar": self.heads["logvar"](features, condition),
            }
        if self.variant in DELTA_VARIANTS:
            delta = self.heads["delta"](features, condition)
            return {
                "mu": xt + (1.0 - probability_targets.time_like(t, xt)) * delta,
                "delta": delta,
                "logvar": self.heads["logvar"](features, condition),
            }
        if self.variant == "endpoint_fm":
            return {"x1_hat": self.heads["x1_hat"](features, condition)}
        if self.variant == "velocity_fm":
            return {"velocity": self.heads["velocity"](features, condition)}
        raise ValueError(f"Unhandled variant {self.variant!r}")


def build_variant_adapter(
    variant: str,
    *,
    prediction_head_factory: HeadFactory,
    logvar_init_bias: float,
) -> VariantAdapter:
    if variant == "pfm_full":
        return VariantAdapter(
            variant,
            {
                "mu": prediction_head_factory(0.0),
                "logvar": prediction_head_factory(float(logvar_init_bias)),
            },
        )
    if variant in DELTA_VARIANTS:
        return VariantAdapter(
            variant,
            {
                "delta": prediction_head_factory(0.0),
                "logvar": prediction_head_factory(float(logvar_init_bias)),
            },
        )
    if variant == "endpoint_fm":
        return VariantAdapter(variant, {"x1_hat": prediction_head_factory(0.0)})
    if variant == "velocity_fm":
        return VariantAdapter(variant, {"velocity": prediction_head_factory(0.0)})
    raise ValueError(
        f"Unknown variant {variant!r}; expected one of {sorted(SUPPORTED_VARIANTS)}"
    )
