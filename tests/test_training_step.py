from __future__ import annotations

import pytest
import torch

from pfm.training.step import build_ema_model, update_ema_model


class _BufferedModule(torch.nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.weight = torch.nn.Parameter(torch.zeros(3))
        self.register_buffer("running", torch.zeros(2))


def test_ema_update_uses_foreach_for_parameters_and_buffers(monkeypatch):
    model = _BufferedModule()
    ema_model = build_ema_model(model)
    model.weight.detach().fill_(4.0)
    ema_model.weight.detach().fill_(2.0)
    model.running.fill_(7.0)
    calls: list[tuple[str, int]] = []
    original_mul = torch._foreach_mul_
    original_add = torch._foreach_add_
    original_copy = torch._foreach_copy_

    def traced_mul(tensors, value):
        calls.append(("mul", len(tensors)))
        return original_mul(tensors, value)

    def traced_add(tensors, others, *, alpha):
        calls.append(("add", len(tensors)))
        return original_add(tensors, others, alpha=alpha)

    def traced_copy(tensors, others):
        calls.append(("copy", len(tensors)))
        return original_copy(tensors, others)

    monkeypatch.setattr(torch, "_foreach_mul_", traced_mul)
    monkeypatch.setattr(torch, "_foreach_add_", traced_add)
    monkeypatch.setattr(torch, "_foreach_copy_", traced_copy)

    update_ema_model(ema_model, model, decay=0.25)

    assert calls == [("mul", 1), ("add", 1), ("copy", 1)]
    assert torch.equal(ema_model.weight, torch.full((3,), 3.5))
    assert torch.equal(ema_model.running, model.running)


@pytest.mark.parametrize("dtype", [torch.float32, torch.bfloat16])
def test_ema_foreach_update_matches_two_operation_reference_exactly(dtype):
    model = _BufferedModule().to(dtype=dtype)
    ema_model = build_ema_model(model)
    model.weight.detach().copy_(torch.tensor([4.0, -2.0, 0.5], dtype=dtype))
    ema_model.weight.detach().copy_(torch.tensor([2.0, 3.0, -1.0], dtype=dtype))
    model.running.copy_(torch.tensor([7.0, -5.0], dtype=dtype))
    expected_weight = ema_model.weight.detach().clone()
    expected_weight.mul_(0.25).add_(model.weight.detach(), alpha=0.75)

    update_ema_model(ema_model, model, decay=0.25)

    assert torch.equal(ema_model.weight, expected_weight)
    assert torch.equal(ema_model.running, model.running)
    assert not ema_model.weight.requires_grad


@pytest.mark.parametrize("tensor_kind", ["parameter", "buffer"])
def test_ema_update_rejects_tensor_count_mismatch_before_mutation(tensor_kind):
    model = _BufferedModule()
    ema_model = build_ema_model(model)
    original_weight = ema_model.weight.detach().clone()
    if tensor_kind == "parameter":
        model.register_parameter("extra", torch.nn.Parameter(torch.ones(())))
    else:
        model.register_buffer("extra", torch.ones(()))

    with pytest.raises(ValueError, match=f"{tensor_kind} count mismatch"):
        update_ema_model(ema_model, model, decay=0.5)

    assert torch.equal(ema_model.weight, original_weight)
