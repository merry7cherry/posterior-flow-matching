from __future__ import annotations

import copy

import torch

def build_ema_model(model: torch.nn.Module) -> torch.nn.Module:
    ema_model = copy.deepcopy(model).eval()
    for parameter in ema_model.parameters():
        parameter.requires_grad_(False)
    return ema_model


@torch.no_grad()
def update_ema_model(ema_model: torch.nn.Module, model: torch.nn.Module, decay: float) -> None:
    ema_parameters = list(ema_model.parameters())
    model_parameters = list(model.parameters())
    ema_buffers = list(ema_model.buffers())
    model_buffers = list(model.buffers())
    _require_matching_tensor_counts(
        ema_parameters,
        model_parameters,
        tensor_kind="parameter",
    )
    _require_matching_tensor_counts(
        ema_buffers,
        model_buffers,
        tensor_kind="buffer",
    )
    if ema_parameters:
        torch._foreach_mul_(ema_parameters, float(decay))
        torch._foreach_add_(
            ema_parameters,
            model_parameters,
            alpha=1.0 - float(decay),
        )
    if ema_buffers:
        torch._foreach_copy_(ema_buffers, model_buffers)


def _require_matching_tensor_counts(
    ema_tensors: list[torch.Tensor],
    model_tensors: list[torch.Tensor],
    *,
    tensor_kind: str,
) -> None:
    if len(ema_tensors) != len(model_tensors):
        raise ValueError(
            f"EMA/model {tensor_kind} count mismatch: "
            f"{len(ema_tensors)} != {len(model_tensors)}"
        )
